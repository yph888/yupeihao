"""Naive Bayes classifier for MNIST handwritten digits.

This module implements the *Naive Bayes Protocol v1* of the course
assignment from scratch with NumPy.  No ``sklearn.naive_bayes`` (or any
other ready-made Naive Bayes implementation) is used.

Model
-----
For an image with pixels ``x_1 ... x_784`` and class ``y``, the Naive Bayes
assumption is that the pixels are conditionally independent given the class:

    P(y | x_1..x_784)  proportional to  P(y) * prod_j P(x_j | y)

Taking logarithms turns the product into a sum, which avoids underflow:

    log P(y | x) = log P(y) + sum_j log P(x_j | y) + const

and the prediction is ``argmax_y`` of that expression.

Preprocessing and likelihood
-----------------------------
The default model treats each pixel as a **discrete** variable with three
grey levels (0-85, 86-170, 171-255), i.e. a categorical Naive Bayes.  The
conditional probability tables are estimated by counting, with add-alpha
smoothing:

    P(x_j = v | y = c) = (N_{c,j,v} + alpha) / (N_c + alpha * 3)

Two further variants are implemented for comparison:

* ``"multinomial"`` - binarise pixels (``pixel > 128``) and treat the image
  as a bag of 784 Bernoulli trials in the classic multinomial-document
  formulation, with the same Laplace smoothing over the 784 pixel counts;
* ``"bernoulli"`` - binarise pixels and explicitly model both the presence
  and the absence of every pixel (the textbook multivariate Bernoulli
  Naive Bayes), including the ``log(1 - p)`` terms.

The default ``alpha=1.0`` corresponds to Laplace smoothing.  The class
prior is smoothed with the same ``alpha``.

Notes
-----
* The input is a flat ``(n_samples, 784)`` matrix with values 0-255, as
  specified by the protocol.  Values in [0, 1] are rescaled as a courtesy.
* Prediction happens entirely in log space.  ``predict_proba`` exponentiates
  the normalised log posteriors.
"""

from __future__ import annotations

import numpy as np

N_PIXELS = 784
N_CLASSES = 10

#: Grey-level thresholds for the categorical variant (three equal bins).
_DEFAULT_BIN_EDGES = (255.0 / 3.0, 2.0 * 255.0 / 3.0)


class NaiveBayesClassifier:
    """Naive Bayes digit classifier (Protocol v1).

    Parameters
    ----------
    alpha : float
        Add-alpha smoothing strength.  ``1.0`` is Laplace smoothing.

    Notes
    -----
    The protocol fixes the constructor signature to ``__init__(alpha)``, so
    the likelihood variant is *not* a constructor argument.  The default
    model is the 3-level categorical Naive Bayes documented above; the
    multinomial / Bernoulli variants used in the report's comparison are
    available through the :meth:`with_variant` classmethod, e.g.::

        clf = NaiveBayesClassifier.with_variant(alpha=1.0, variant="bernoulli")
    """

    def __init__(self, alpha: float = 1.0):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = float(alpha)
        # Default likelihood configuration: 3-level categorical NB.
        self.variant = "categorical"
        self.n_bins = 3
        self.threshold = 128.0

    @classmethod
    def with_variant(cls, alpha: float = 1.0, variant: str = "categorical",
                     n_bins: int = 3, threshold: float = 128.0):
        """Build a classifier with a non-default likelihood configuration.

        Kept outside ``__init__`` so that the protocol constructor signature
        stays exactly ``NaiveBayesClassifier(alpha=1.0)``.

        Parameters
        ----------
        alpha : float
            Add-alpha smoothing strength.
        variant : {"categorical", "multinomial", "bernoulli"}
            Likelihood model.
        n_bins : int
            Number of grey levels for the categorical variant.
        threshold : float
            Binarisation threshold for multinomial / Bernoulli.
        """
        if variant not in ("categorical", "multinomial", "bernoulli"):
            raise ValueError(f"unknown variant {variant!r}")
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        clf = cls(alpha=alpha)
        clf.variant = variant
        clf.n_bins = int(n_bins)
        clf.threshold = float(threshold)
        return clf

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def _check_X(self, X) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1 and X.shape[0] == N_PIXELS:
            X = X.reshape(1, N_PIXELS)
        if X.ndim == 3 and X.shape[1:] == (28, 28):
            X = X.reshape(len(X), N_PIXELS)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n_samples, 784), got {X.shape}")
        if X.shape[1] != N_PIXELS:
            raise ValueError(f"X must have 784 columns, got {X.shape[1]}")
        X = X.astype(np.float32, copy=False)
        if X.size and float(X.max()) <= 1.0:
            X = X * 255.0
        return np.clip(X, 0.0, 255.0)

    def _binarize(self, X: np.ndarray) -> np.ndarray:
        return (X > self.threshold).astype(np.float32)

    def _digitize(self, X: np.ndarray) -> np.ndarray:
        if self.n_bins == 3:
            edges = np.asarray(_DEFAULT_BIN_EDGES, dtype=np.float32)
        else:
            edges = np.linspace(0.0, 255.0, self.n_bins + 1)[1:-1].astype(np.float32)
        return np.digitize(X, edges).astype(np.int16)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X, y):
        """Estimate priors and class-conditional likelihoods.

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
        X = self._check_X(X)
        y = np.asarray(y).astype(np.int64).ravel()
        if len(X) != len(y):
            raise ValueError("X and y have inconsistent lengths")

        classes = np.unique(y)
        self.classes_ = classes
        class_to_row = {int(c): i for i, c in enumerate(classes)}
        y_idx = np.array([class_to_row[int(v)] for v in y], dtype=np.int64)
        C = len(classes)

        prior = np.bincount(y_idx, minlength=C).astype(np.float64)
        self.log_prior_ = np.log(prior + self.alpha) - np.log(
            prior.sum() + self.alpha * C
        )
        self.n_features_in_ = N_PIXELS

        if self.variant == "categorical":
            self._fit_categorical(X, y_idx, C)
        elif self.variant == "multinomial":
            self._fit_multinomial(X, y_idx, C)
        else:
            self._fit_bernoulli(X, y_idx, C)
        return self

    def _fit_categorical(self, X, y_idx, C):
        bins = self._digitize(X)
        B = self.n_bins
        counts = np.zeros((C, N_PIXELS, B), dtype=np.float64)
        offsets = (np.arange(N_PIXELS, dtype=np.int32) * B)[None, :]
        for c in range(C):
            idx = (bins[y_idx == c].astype(np.int32) + offsets).ravel()
            counts[c] = np.bincount(idx, minlength=N_PIXELS * B).reshape(N_PIXELS, B)
        counts += self.alpha
        self.log_likelihood_ = np.log(counts / counts.sum(axis=2, keepdims=True))
        self.n_bins_ = B

    def _fit_multinomial(self, X, y_idx, C):
        binarized = self._binarize(X)
        counts = np.zeros((C, N_PIXELS), dtype=np.float64)
        for c in range(C):
            counts[c] = binarized[y_idx == c].sum(axis=0)
        # Token counts: N_c = total number of "on" pixels of class c.
        theta = (counts + self.alpha) / (
            counts.sum(axis=1, keepdims=True) + self.alpha * N_PIXELS
        )
        self.log_likelihood_ = np.log(theta)

    def _fit_bernoulli(self, X, y_idx, C):
        binarized = self._binarize(X)
        counts = np.zeros((C, N_PIXELS), dtype=np.float64)
        n_c = np.bincount(y_idx, minlength=C).astype(np.float64)[:, None]
        for c in range(C):
            counts[c] = binarized[y_idx == c].sum(axis=0)
        p = (counts + self.alpha) / (n_c + 2.0 * self.alpha)
        self.log_likelihood_ = np.log(p)
        self.log_neg_likelihood_ = np.log1p(-p)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_log_proba(self, X) -> np.ndarray:
        """Normalised log posterior for each class, shape (n_samples, n_classes)."""
        X = self._check_X(X)
        scores = np.tile(self.log_prior_, (len(X), 1))

        if self.variant == "categorical":
            bins = self._digitize(X)
            cols = np.arange(N_PIXELS)[None, :]
            for c in range(len(self.classes_)):
                scores[:, c] += self.log_likelihood_[c][cols, bins].sum(axis=1)
        elif self.variant == "multinomial":
            scores += self._binarize(X) @ self.log_likelihood_.T
        else:
            binarized = self._binarize(X)
            # log P(x_j=1|y) - log P(x_j=0|y) is the log-odds contribution
            # of an "on" pixel; the log P(x_j=0|y) term is added once for
            # every pixel (the (1 - x_j) part of the Bernoulli likelihood).
            log_odds = self.log_likelihood_ - self.log_neg_likelihood_
            scores += binarized @ log_odds.T
            scores += self.log_neg_likelihood_.sum(axis=1)[None, :]

        scores -= scores.max(axis=1, keepdims=True)
        scores -= np.log(np.exp(scores).sum(axis=1, keepdims=True))
        return scores

    def predict_proba(self, X) -> np.ndarray:
        """Posterior class probabilities, shape (n_samples, n_classes)."""
        return np.exp(self.predict_log_proba(X))

    def predict(self, X) -> np.ndarray:
        """Predict digit labels for raw images ``X`` (n_samples, 784)."""
        scores = self.predict_log_proba(X)
        return self.classes_[np.argmax(scores, axis=1)]
