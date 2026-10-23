"""Small hand-crafted Bayesian network for MNIST digit classification.

This module implements the *Small Bayesian Network Protocol v1* of the
course assignment.  Everything (feature extraction, structure, CPT
estimation, smoothing, inference) is written from scratch with NumPy; no
probabilistic-graphical-model library is used.

Model
-----
The network has 13 nodes: the class node ``Y`` and 12 observed discrete
feature nodes ``F_1 ... F_12``.  ``Y`` is the root and is a parent of every
feature.  In addition, five *feature -> feature* edges encode structural
dependencies that the naive-Bayes assumption would ignore:

    Y                  -> every feature
    holes              -> mid_col_strokes
    holes              -> mid_row_strokes
    holes              -> center_density
    top_density        -> tb_symmetry
    right_density      -> lr_symmetry

The joint distribution therefore factorises as

    P(y, F_1..F_12) = P(y) * prod_i P(F_i | Parents(F_i))

and, because every feature is observed at prediction time, the exact
posterior is obtained by direct evaluation (no approximate inference):

    y_hat = argmax_y [ log P(y) + sum_i log P(F_i | Parents(F_i)) ].

Each feature takes 4 or 5 discrete values (levels).  All conditional
probability tables are estimated by maximum likelihood with add-alpha
smoothing (alpha = 1.0 by default, i.e. Laplace smoothing).

Features (all computed from the raw 28x28 grey image, values 0-255)
--------------------------------------------------------------------
top_density      mean ink of rows 0-8            (5 levels, train quantiles)
bottom_density   mean ink of rows 19-27          (5 levels)
left_density     mean ink of columns 0-8         (5 levels)
right_density    mean ink of columns 19-27       (5 levels)
top_row          first row containing ink        (4 levels)
bottom_row       last row containing ink         (5 levels)
center_density   mean ink of the central 12x12   (5 levels)
holes            number of enclosed background   (5 levels: 0,1,2,3,>=4)
mid_col_strokes  ink runs along central column    (5 levels: 0,1,2,3,>=4)
mid_row_strokes  ink runs along central row       (5 levels)
lr_symmetry      left-right mirror agreement     (5 levels)
tb_symmetry      top-bottom mirror agreement     (5 levels)

The split points are fixed constants calibrated once on the *training* split
(20/40/60/80 percentiles for continuous features; natural integer cut
points for the count features).  They do not use labels and are stored in
``_THRESHOLDS`` so that :func:`extract_features` is a pure, reusable
function, as required by the protocol.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

FEATURE_NAMES = (
    "top_density",
    "bottom_density",
    "left_density",
    "right_density",
    "top_row",
    "bottom_row",
    "center_density",
    "holes",
    "mid_col_strokes",
    "mid_row_strokes",
    "lr_symmetry",
    "tb_symmetry",
)

#: Directed edges between feature nodes.  ``Y`` is a parent of every feature
#: and is implicit; only the feature-to-feature edges are listed here.
#: Each child has at most one feature parent, hence at most two parents in
#: total (the class node plus one feature), as required by the handout.
FEATURE_PARENTS = {
    "mid_col_strokes": ("holes",),
    "mid_row_strokes": ("holes",),
    "center_density": ("holes",),
    "tb_symmetry": ("top_density",),
    "lr_symmetry": ("right_density",),
}

#: Digitisation thresholds, calibrated on the 60,000 training images only.
#: For a threshold tuple ``t`` a value ``v`` is mapped to
#: ``sum(v >= t_k)``, i.e. ``len(t) + 1`` levels.
_THRESHOLDS = {
    "top_density": (0.0492, 0.0685, 0.0912, 0.1190),
    "bottom_density": (0.0632, 0.0847, 0.1102, 0.1439),
    "left_density": (0.0023, 0.0226, 0.0436, 0.0720),
    "right_density": (0.0123, 0.0398, 0.0649, 0.0994),
    "top_row": (3.5, 4.5, 5.5, 6.5),
    "bottom_row": (21.5, 22.5, 23.5, 24.5),
    "center_density": (0.2741, 0.3417, 0.4035, 0.4798),
    "holes": (0.5, 1.5, 2.5, 3.5),
    "mid_col_strokes": (1.5, 2.5, 3.5),
    "mid_row_strokes": (1.5, 2.5),
    "lr_symmetry": (0.1888, 0.2536, 0.3127, 0.3892),
    "tb_symmetry": (0.1569, 0.2097, 0.2661, 0.3484),
}

N_FEATURES = len(FEATURE_NAMES)
N_CLASSES = 10


# ---------------------------------------------------------------------------
# Low-level image measurements
# ---------------------------------------------------------------------------

def _as_images(X) -> np.ndarray:
    """Validate ``X`` and return it as float images in [0, 1], shape (n,28,28)."""
    X = np.asarray(X)
    if X.ndim == 1 and X.shape[0] == 784:
        X = X.reshape(1, 784)
    if X.ndim == 3 and X.shape[1:] == (28, 28):
        X = X.reshape(len(X), 784)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_samples, 784), got shape {X.shape}")
    if X.shape[1] != 784:
        raise ValueError(f"X must have 784 columns, got {X.shape[1]}")
    X = X.astype(np.float32, copy=False)
    # The protocol passes raw 0-255 pixels.  Accept 0-1 input as a courtesy.
    if X.size and float(X.max()) <= 1.0:
        X = X * 255.0
    return (X / 255.0).reshape(-1, 28, 28)


def _mirror_overlap(img: np.ndarray, axis: int) -> np.ndarray:
    """Normalised overlap between an image and its mirror image.

    ``sum(min(a, mirror(a))) / sum(max(a, mirror(a)))`` in [0, 1], where 1
    means perfectly symmetric.  Computed on grey values, so it also rewards
    matching stroke intensities.
    """
    mirror = np.flip(img, axis=axis)
    low = np.minimum(img, mirror).sum(axis=(1, 2))
    high = np.maximum(img, mirror).sum(axis=(1, 2))
    out = np.ones(len(img), dtype=np.float32)
    nz = high > 0
    out[nz] = low[nz] / high[nz]
    return out


def euler_number(binary: np.ndarray) -> np.ndarray:
    """Euler number ``E = C - H`` of binary images, vectorised.

    Uses the classical 2x2 block formula for 8-connected foreground:

        E = (n1 - n3 - 2 * nD) / 4

    where ``n1`` / ``n3`` count 2x2 blocks with one / three ink pixels and
    ``nD`` counts blocks with two diagonally adjacent ink pixels.  The image
    is zero-padded so that pixels on the border are counted correctly.
    """
    b = binary.astype(np.int16)
    p = np.pad(b, ((0, 0), (1, 1), (1, 1)))
    blocks = p[:, :-1, :-1] + p[:, 1:, :-1] + p[:, :-1, 1:] + p[:, 1:, 1:]
    n1 = np.count_nonzero(blocks == 1, axis=(1, 2))
    n3 = np.count_nonzero(blocks == 3, axis=(1, 2))
    a, b2 = p[:, :-1, :-1], p[:, 1:, 1:]
    c, d = p[:, :-1, 1:], p[:, 1:, :-1]
    nd = np.count_nonzero(
        ((a == 1) & (b2 == 1) & (c == 0) & (d == 0))
        | ((c == 1) & (d == 1) & (a == 0) & (b2 == 0)),
        axis=(1, 2),
    )
    return (n1 - n3 - 2 * nd) // 4


def _neighbour_table() -> list[np.ndarray]:
    """8-neighbour flat indices for every pixel of a 28x28 image."""
    table = [[] for _ in range(784)]
    for r in range(28):
        for c in range(28):
            p = r * 28 + c
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < 28 and 0 <= cc < 28:
                        table[p].append(rr * 28 + cc)
    return [np.asarray(x, dtype=np.int32) for x in table]


_NEIGHBOURS = _neighbour_table()


def count_components(binary: np.ndarray) -> np.ndarray:
    """Number of 8-connected ink components per image.

    A plain depth-first search over the ~100 ink pixels of a digit is faster
    in pure Python than any array-wide relaxation and needs no SciPy.
    """
    out = np.empty(len(binary), dtype=np.int16)
    for k, image in enumerate(binary):
        flat = image.ravel()
        seen = np.zeros(784, dtype=bool)
        n = 0
        stack: list[int] = []
        for start in np.flatnonzero(flat):
            if seen[start]:
                continue
            n += 1
            seen[start] = True
            stack.append(int(start))
            while stack:
                p = stack.pop()
                for q in _NEIGHBOURS[p]:
                    if flat[q] and not seen[q]:
                        seen[q] = True
                        stack.append(int(q))
        out[k] = n
    return out


def _runs_along(line: np.ndarray) -> np.ndarray:
    """Number of maximal runs of True along axis 1 of a boolean (n, L) array."""
    padded = np.pad(line.astype(np.int8), ((0, 0), (1, 1)))
    return np.count_nonzero(np.diff(padded, axis=1) == 1, axis=1)


def _raw_features(X) -> dict[str, np.ndarray]:
    """Compute every continuous feature (before digitisation)."""
    img = _as_images(X)
    n = len(img)
    ink = img
    binary = img > 0.5

    # --- regional ink densities -------------------------------------------
    top = ink[:, :9, :].mean(axis=(1, 2))
    bottom = ink[:, 19:, :].mean(axis=(1, 2))
    left = ink[:, :, :9].mean(axis=(1, 2))
    right = ink[:, :, 19:].mean(axis=(1, 2))
    center = ink[:, 8:20, 8:20].mean(axis=(1, 2))

    # --- vertical extent ---------------------------------------------------
    rows = binary.any(axis=2)                       # (n, 28)
    any_ink = rows.any(axis=1)
    first_row = rows.argmax(axis=1).astype(np.float32)
    last_row = (27 - rows[:, ::-1].argmax(axis=1)).astype(np.float32)
    first_row[~any_ink] = 0.0
    last_row[~any_ink] = 0.0

    # --- topology ----------------------------------------------------------
    comps = count_components(binary).astype(np.float32)
    euler = euler_number(binary).astype(np.float32)
    holes = np.maximum(comps - euler, 0.0)

    # --- stroke counts along the central scanlines ------------------------
    col_runs = _runs_along(binary[:, :, 13]).astype(np.float32)
    row_runs = _runs_along(binary[:, 13, :]).astype(np.float32)

    # --- symmetries --------------------------------------------------------
    lr_sym = _mirror_overlap(ink, axis=2)   # left-right mirror
    tb_sym = _mirror_overlap(ink, axis=1)   # top-bottom mirror

    return {
        "top_density": top,
        "bottom_density": bottom,
        "left_density": left,
        "right_density": right,
        "top_row": first_row,
        "bottom_row": last_row,
        "center_density": center,
        "holes": holes,
        "mid_col_strokes": col_runs,
        "mid_row_strokes": row_runs,
        "lr_symmetry": lr_sym,
        "tb_symmetry": tb_sym,
    }


def extract_features(X) -> np.ndarray:
    """Extract the 12 discrete Bayesian-network features from raw images.

    Parameters
    ----------
    X : array-like, shape (n_samples, 784)
        Raw MNIST images flattened row-major, pixel values 0-255.

    Returns
    -------
    ndarray, shape (n_samples, 12), dtype int16
        Discrete feature levels; column order follows :data:`FEATURE_NAMES`.
    """
    raw = _raw_features(X)
    cols = []
    for name in FEATURE_NAMES:
        thresholds = np.asarray(_THRESHOLDS[name], dtype=np.float32)
        cols.append(np.digitize(raw[name], thresholds).astype(np.int16))
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# Bayesian network classifier
# ---------------------------------------------------------------------------

class BayesNetClassifier:
    """Small hand-designed Bayesian network classifier (Protocol v1).

    Parameters
    ----------
    alpha : float
        Add-alpha smoothing strength used for every conditional probability
        table (1.0 = Laplace smoothing).
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)

    # -- structure helpers --------------------------------------------------
    @staticmethod
    def _parent_indices() -> dict[int, tuple[int, ...]]:
        """Feature-parent indices for every feature node."""
        index = {name: i for i, name in enumerate(FEATURE_NAMES)}
        return {
            index[child]: tuple(index[p] for p in parents)
            for child, parents in FEATURE_PARENTS.items()
        }

    # -- fitting ------------------------------------------------------------
    def fit(self, X, y):
        """Estimate the class prior and every CPT from labelled images.

        Parameters
        ----------
        X : array-like, shape (n_samples, 784)
            Raw pixel matrix (0-255).
        y : array-like, shape (n_samples,)
            Digit labels.

        Returns
        -------
        self
        """
        F = extract_features(X)
        y = np.asarray(y).astype(np.int64).ravel()
        if len(F) != len(y):
            raise ValueError("X and y have inconsistent lengths")

        classes = np.unique(y)
        self.classes_ = classes
        class_to_row = {int(c): i for i, c in enumerate(classes)}
        C = len(classes)
        y_idx = np.array([class_to_row[int(v)] for v in y], dtype=np.int64)

        # Level counts per feature (depends only on the fixed thresholds).
        self.n_levels_ = np.array(
            [len(_THRESHOLDS[name]) + 1 for name in FEATURE_NAMES], dtype=np.int64
        )

        # Class prior with the same smoothing.
        prior_counts = np.bincount(y_idx, minlength=C).astype(np.float64)
        self.log_prior_ = np.log(prior_counts + self.alpha) - np.log(
            prior_counts.sum() + self.alpha * C
        )

        parents = self._parent_indices()
        self.log_cpt_ = []          # list of (C, parent_configs * levels_i)
        self.parent_config_size_ = []
        for i in range(N_FEATURES):
            n_lev = int(self.n_levels_[i])
            pa = parents.get(i, ())
            cfg, mult = self._config_index(F, pa, self.n_levels_)
            flat = (y_idx * mult + cfg) * n_lev + F[:, i]
            counts = np.bincount(
                flat, minlength=C * mult * n_lev
            ).astype(np.float64).reshape(C, mult, n_lev)
            prob = (counts + self.alpha) / (
                counts.sum(axis=2, keepdims=True) + self.alpha * n_lev
            )
            self.log_cpt_.append(np.log(prob).reshape(C, mult * n_lev))
            self.parent_config_size_.append(mult)
        return self

    @staticmethod
    def _config_index(F, parents, n_levels):
        """Encode the joint state of ``parents`` as a flat integer."""
        cfg = np.zeros(len(F), dtype=np.int64)
        mult = 1
        for p in parents:
            cfg += F[:, p].astype(np.int64) * mult
            mult *= int(n_levels[p])
        return cfg, mult

    # -- inference ----------------------------------------------------------
    def _log_joint(self, F: np.ndarray) -> np.ndarray:
        """Un-normalised log joint ``log P(y, F)`` for every class."""
        n = len(F)
        C = len(self.classes_)
        parents = self._parent_indices()
        scores = np.tile(self.log_prior_, (n, 1))
        for i in range(N_FEATURES):
            n_lev = int(self.n_levels_[i])
            pa = parents.get(i, ())
            cfg, _ = self._config_index(F, pa, self.n_levels_)
            flat = cfg * n_lev + F[:, i]
            table = self.log_cpt_[i]
            for c in range(C):
                scores[:, c] += table[c][flat]
        return scores

    def predict_log_proba(self, X) -> np.ndarray:
        """Normalised log posterior ``log P(y | X)`` for each class."""
        scores = self._log_joint(extract_features(X))
        scores -= scores.max(axis=1, keepdims=True)
        scores -= np.log(np.exp(scores).sum(axis=1, keepdims=True))
        return scores

    def predict_proba(self, X) -> np.ndarray:
        """Posterior class probabilities ``P(y | X)``, shape (n, n_classes)."""
        return np.exp(self.predict_log_proba(X))

    def predict(self, X) -> np.ndarray:
        """Predict digit labels for raw images ``X`` (n_samples, 784)."""
        scores = self._log_joint(extract_features(X))
        return self.classes_[np.argmax(scores, axis=1)]

    # -- introspection ------------------------------------------------------
    def structure(self) -> dict:
        """Return the DAG as ``{child: (parents...)}`` including the class root."""
        edges = {name: ("Y",) for name in FEATURE_NAMES}
        for child, parents in FEATURE_PARENTS.items():
            edges[child] = ("Y",) + parents
        return edges
