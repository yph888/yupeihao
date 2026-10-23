"""Training and evaluation entry point for Assignment 2.

Usage
-----
    python main.py                          # auto-detect ../assignment2-data/mnist
    python main.py --data path/to/mnist     # explicit data directory
    python main.py --no-figures             # skip figure generation

The script

1. loads the four MNIST IDX files through the course-provided loader,
2. trains ``NaiveBayesClassifier`` and ``BayesNetClassifier`` on the 60,000
   training images,
3. evaluates both on the 10,000 test images (accuracy, macro precision /
   recall / F1, per-class metrics, confusion matrices),
4. writes ``results/results.json`` and the figures used in ``report.md``.

Everything is deterministic; no random seed is involved.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bayes_net import (  # noqa: E402
    BayesNetClassifier,
    FEATURE_NAMES,
    FEATURE_PARENTS,
    N_FEATURES,
)
from naive_bayes import NaiveBayesClassifier  # noqa: E402

N_CLASSES = 10
CLASS_NAMES = [str(i) for i in range(N_CLASSES)]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def locate_data_dir(explicit: str | None) -> Path:
    """Find the directory that contains the four MNIST ``*.gz`` files."""
    names = ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("ASSIGNMENT2_DATA")
    if env:
        candidates.append(Path(env))
    candidates += [
        HERE.parent / "assignment2-data" / "mnist",
        HERE.parent.parent / "assignment2-data" / "mnist",
        HERE / ".." / "assignment2-data" / "mnist",
        Path.cwd() / "assignment2-data" / "mnist",
        Path.cwd() / "data" / "mnist",
        HERE / "data" / "mnist",
    ]
    for cand in candidates:
        cand = cand.expanduser().resolve()
        if all((cand / n).exists() for n in names):
            return cand
    raise FileNotFoundError(
        "MNIST data directory not found. Pass --data <dir> pointing at the "
        "directory containing train-images-idx3-ubyte.gz etc."
    )


def load_mnist(data_dir: Path):
    """Load MNIST through the course-provided ``mnist_loader.py``."""
    loader_path = data_dir / "mnist_loader.py"
    if not loader_path.exists():
        raise FileNotFoundError(f"missing course loader: {loader_path}")
    spec = importlib.util.spec_from_file_location("course_mnist_loader", loader_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    X_train, y_train, X_test, y_test = module.load_mnist(data_dir)
    X_train = np.asarray(X_train, dtype=np.uint8).reshape(len(X_train), 784)
    X_test = np.asarray(X_test, dtype=np.uint8).reshape(len(X_test), 784)
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Metrics (implemented here to keep the evaluation dependency-free)
# ---------------------------------------------------------------------------

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     n_classes: int = N_CLASSES) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    idx = y_true.astype(np.int64) * n_classes + y_pred.astype(np.int64)
    return np.bincount(idx, minlength=n_classes ** 2).reshape(n_classes, n_classes)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy plus macro-averaged precision, recall and F1."""
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)

    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)

    accuracy = float(tp.sum() / cm.sum())
    return {
        "accuracy": accuracy,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {
            CLASS_NAMES[c]: {
                "precision": float(precision[c]),
                "recall": float(recall[c]),
                "f1": float(f1[c]),
                "support": int(support[c]),
            }
            for c in range(N_CLASSES)
        },
        "confusion_matrix": cm.tolist(),
    }


def top_confused_pairs(cm: np.ndarray, k: int = 5) -> list[dict]:
    """Most frequent off-diagonal confusion pairs ``true -> predicted``."""
    pairs = []
    for t in range(len(cm)):
        for p in range(len(cm)):
            if t != p and cm[t, p] > 0:
                pairs.append({"true": t, "pred": p, "count": int(cm[t, p])})
    pairs.sort(key=lambda d: d["count"], reverse=True)
    return pairs[:k]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    return plt


def plot_confusion(cm: np.ndarray, title: str, path: Path):
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(title)
    threshold = cm.max() * 0.55
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=7.5, color="white" if cm[i, j] > threshold else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="样本数")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_misclassified(images: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                       probability: np.ndarray, cm: np.ndarray, path: Path,
                       title: str, n: int = 12, n_pairs: int = 4):
    """Grid of *typical* errors: a few examples from each most-confused pair."""
    plt = _setup_matplotlib()
    pairs = top_confused_pairs(cm, n_pairs)
    per_pair = max(1, n // max(len(pairs), 1))
    chosen: list[int] = []
    for pair in pairs:
        cand = np.flatnonzero((y_true == pair["true"]) & (y_pred == pair["pred"]))
        chosen.extend(int(i) for i in cand[:per_pair])
    if len(chosen) < n:
        for idx in np.flatnonzero(y_true != y_pred):
            if int(idx) not in chosen:
                chosen.append(int(idx))
            if len(chosen) >= n:
                break
    chosen = np.asarray(chosen[:n])

    cols = 4
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8.6, 2.35 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, idx in zip(axes, chosen):
        ax.imshow(images[idx].reshape(28, 28), cmap="gray_r")
        conf = probability[idx, y_pred[idx]]
        ax.set_title(f"真值 {int(y_true[idx])} → 预测 {int(y_pred[idx])}\n"
                     f"P(预测类)={conf:.4f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)


def plot_metrics(nb: dict, bn: dict, path: Path):
    plt = _setup_matplotlib()
    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    labels = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    x = np.arange(len(keys))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    nb_vals = [nb[k] for k in keys]
    bn_vals = [bn[k] for k in keys]
    bars1 = ax.bar(x - width / 2, nb_vals, width, label="Naive Bayes", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, bn_vals, width, label="Bayesian Network", color="#DD8452")
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("指标值")
    ax.set_title("两个模型在测试集上的总体指标")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_class_accuracy(nb: dict, bn: dict, path: Path):
    plt = _setup_matplotlib()
    labels = CLASS_NAMES
    x = np.arange(N_CLASSES)
    width = 0.36
    nb_recall = [nb["per_class"][c]["recall"] for c in labels]
    bn_recall = [bn["per_class"][c]["recall"] for c in labels]
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.bar(x - width / 2, nb_recall, width, label="Naive Bayes", color="#4C72B0")
    ax.bar(x + width / 2, bn_recall, width, label="Bayesian Network", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("数字类别")
    ax.set_ylabel("召回率")
    ax.set_ylim(0, 1.0)
    ax.set_title("各类别召回率对比")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


_CN_NAMES = {
    "top_density": "顶部密度",
    "bottom_density": "底部密度",
    "left_density": "左侧密度",
    "right_density": "右侧密度",
    "top_row": "顶部行位置",
    "bottom_row": "底部行位置",
    "center_density": "中心密度",
    "holes": "孔洞数",
    "mid_col_strokes": "中列笔画数",
    "mid_row_strokes": "中行笔画数",
    "lr_symmetry": "左右对称度",
    "tb_symmetry": "上下对称度",
}


def plot_bn_structure(path: Path):
    """Draw the hand-designed DAG: Y root, 12 features, 5 feature edges."""
    plt = _setup_matplotlib()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
    from matplotlib.path import Path as MplPath

    # Vertical layout: the class root on the left, the 12 features in one
    # column on the right.  Feature-to-feature edges are routed on the far
    # right so that they never cross the Y -> feature edges.
    order = [
        "top_density", "tb_symmetry", "bottom_density", "left_density",
        "right_density", "lr_symmetry", "top_row", "bottom_row",
        "center_density", "mid_row_strokes", "mid_col_strokes", "holes",
    ]
    centre, top, step = 0.44, 0.895, 0.0715
    box_w, box_h = 0.235, 0.050
    pos = {name: (centre, top - i * step) for i, name in enumerate(order)}
    pos["Y"] = (0.08, 0.50)
    left, right = centre - box_w / 2, centre + box_w / 2

    groups = {
        "region": (["top_density", "bottom_density", "left_density", "right_density"], "#BBDEFB"),
        "geometry": (["top_row", "bottom_row"], "#C8E6C9"),
        "topology": (["holes"], "#FFE0B2"),
        "derived": (["center_density", "mid_col_strokes", "mid_row_strokes",
                     "lr_symmetry", "tb_symmetry"], "#E1BEE7"),
    }

    fig, ax = plt.subplots(figsize=(8.6, 8.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Y -> every feature (light grey, drawn first so boxes cover the ends).
    for name in order:
        _, y = pos[name]
        ax.annotate("", xy=(left - 0.004, y), xytext=(pos["Y"][0] + 0.045, pos["Y"][1]),
                    arrowprops=dict(arrowstyle="-|>", color="#90A4AE", lw=0.9,
                                    alpha=0.8, shrinkA=1, shrinkB=0, mutation_scale=8),
                    zorder=1)

    # Feature nodes.
    for _, (names, color) in groups.items():
        for name in names:
            x, y = pos[name]
            ax.add_patch(FancyBboxPatch((x - box_w / 2, y - box_h / 2), box_w, box_h,
                                        boxstyle="round,pad=0.002", fc=color,
                                        ec="#546E7A", lw=0.8, zorder=3))
            ax.text(x, y, _CN_NAMES[name], ha="center", va="center",
                    fontsize=10.5, zorder=4)

    # Y node.
    ax.add_patch(FancyBboxPatch((pos["Y"][0] - 0.042, pos["Y"][1] - 0.052), 0.084, 0.104,
                                boxstyle="round,pad=0.004", fc="#37474F", ec="#37474F",
                                zorder=3))
    ax.text(pos["Y"][0], pos["Y"][1] + 0.010, "Y", color="white", ha="center",
            va="center", fontsize=17, fontweight="bold", zorder=4)
    ax.text(pos["Y"][0], pos["Y"][1] - 0.032, "数字类别", color="white", ha="center",
            va="center", fontsize=7.5, zorder=4)

    # Feature -> feature edges: cubic Bezier bowing to the right of the column.
    feature_edges = [
        ("holes", "mid_col_strokes", "#E65100", 0.18),
        ("holes", "mid_row_strokes", "#E65100", 0.23),
        ("holes", "center_density", "#E65100", 0.28),
        ("top_density", "tb_symmetry", "#1565C0", 0.14),
        ("right_density", "lr_symmetry", "#2E7D32", 0.14),
    ]
    for src, dst, color, bulge in feature_edges:
        _, y1 = pos[src]
        _, y2 = pos[dst]
        ctrl_x = right + bulge
        verts = [(right, y1), (ctrl_x, y1), (ctrl_x, y2), (right, y2)]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
        ax.add_patch(FancyArrowPatch(
            path=MplPath(verts, codes), arrowstyle="-|>", mutation_scale=12,
            color=color, lw=1.7, shrinkA=0, shrinkB=0, zorder=2))

    handles = [Patch(fc=c, ec="#546E7A") for _, c in groups.values()]
    labels = ["区域墨迹密度", "几何位置", "拓扑结构", "派生/结构特征"]
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 0.0),
              fontsize=8.5, ncol=2, frameon=True)

    ax.text(0.5, 0.985, "小型贝叶斯网络结构", ha="center", va="top", fontsize=15)
    ax.text(0.5, 0.950, "Y 为所有 12 个特征节点的父节点（浅灰边）；彩色边为额外的特征间依赖",
            ha="center", va="top", fontsize=8.5, color="#37474F")
    ax.text(0.985, 0.015,
            "橙色：孔洞数 → 中列/中行笔画数、中心密度\n"
            "蓝色：顶部密度 → 上下对称度\n"
            "绿色：右侧密度 → 左右对称度",
            ha="right", va="bottom", fontsize=8, color="#37474F")
    fig.subplots_adjust(left=0.01, right=0.99, top=1.0, bottom=0.0)
    fig.savefig(path)
    plt.close(fig)


def plot_feature_example(X: np.ndarray, y: np.ndarray, path: Path, index: int | None = None):
    """Illustrate the region/scanline definitions on one test digit."""
    plt = _setup_matplotlib()
    if index is None:
        index = int(np.flatnonzero(y == 3)[0])
    image = X[index].reshape(28, 28).astype(float) / 255.0

    from bayes_net import FEATURE_NAMES, _raw_features, extract_features

    raw = _raw_features(X[index:index + 1])
    levels = extract_features(X[index:index + 1])[0]

    fig = plt.figure(figsize=(14.0, 3.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.75], wspace=0.12)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

    axes[0].imshow(image, cmap="gray_r")
    axes[0].set_title(f"原始图像（标签 {int(y[index])}）")
    axes[0].axis("off")

    axes[1].imshow(image, cmap="gray_r")
    for yline in (9, 19):
        axes[1].axhline(yline - 0.5, color="#D32F2F", lw=1.4, ls="--")
    axes[1].set_title("水平三区（顶部/中部/底部）")
    axes[1].axis("off")

    axes[2].imshow(image, cmap="gray_r")
    for xline in (9, 19):
        axes[2].axvline(xline - 0.5, color="#1565C0", lw=1.4, ls="--")
    axes[2].axvline(13, color="#2E7D32", lw=1.0, alpha=0.8)
    axes[2].axhline(13, color="#2E7D32", lw=1.0, alpha=0.8)
    axes[2].set_title("垂直三区与中央扫描线")
    axes[2].axis("off")

    axes[3].axis("off")
    lines = ["特征        原始值    离散等级"]
    for name, level in zip(FEATURE_NAMES, levels):
        value = float(raw[name][0])
        lines.append(f"{_CN_NAMES[name]}    {value:6.2f}      {int(level)}")
    axes[3].text(0.0, 0.95, "\n".join(lines), va="top", ha="left",
                 fontsize=9.0, family="Microsoft YaHei", linespacing=1.45)

    fig.suptitle("贝叶斯网络特征提取示例", fontsize=14)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.04)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=None,
                        help="directory containing the MNIST idx.gz files")
    parser.add_argument("--out", type=str, default=str(HERE / "results"),
                        help="directory for results.json (default: assignment2/results)")
    parser.add_argument("--figures", type=str, default=str(HERE / "figures"),
                        help="directory for figures (default: assignment2/figures)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="smoothing strength for both models (default 1.0)")
    parser.add_argument("--nb-variant", type=str, default="categorical",
                        choices=["categorical", "multinomial", "bernoulli"])
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    data_dir = locate_data_dir(args.data)
    print(f"[data] {data_dir}")
    X_train, y_train, X_test, y_test = load_mnist(data_dir)
    print(f"[data] train {X_train.shape}, test {X_test.shape}; "
          f"train class counts {np.bincount(y_train).tolist()}")

    out_dir = Path(args.out).resolve()
    fig_dir = Path(args.figures).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "data": {
            "dir": str(data_dir),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
        "alpha": args.alpha,
    }

    # ---------------- Naive Bayes ----------------
    print("\n[Naive Bayes] fitting ...")
    nb = NaiveBayesClassifier.with_variant(alpha=args.alpha, variant=args.nb_variant)
    t0 = time.perf_counter()
    nb.fit(X_train, y_train)
    nb_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    nb_pred = nb.predict(X_test)
    nb_predict = time.perf_counter() - t0
    nb_proba = nb.predict_proba(X_test)
    nb_metrics = classification_metrics(y_test, nb_pred)
    nb_metrics.update({"fit_seconds": nb_fit, "predict_seconds": nb_predict,
                       "variant": args.nb_variant, "alpha": args.alpha,
                       "mean_max_probability": float(nb_proba.max(axis=1).mean()),
                       "fraction_prob_gt_0.99": float((nb_proba.max(axis=1) > 0.99).mean())})
    nb_metrics["top_confused_pairs"] = top_confused_pairs(
        np.asarray(nb_metrics["confusion_matrix"]))
    results["naive_bayes"] = nb_metrics

    # ---------------- Bayesian network ----------------
    print("[Bayesian Network] fitting ...")
    bn = BayesNetClassifier(alpha=args.alpha)
    t0 = time.perf_counter()
    bn.fit(X_train, y_train)
    bn_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    bn_pred = bn.predict(X_test)
    bn_predict = time.perf_counter() - t0
    bn_proba = bn.predict_proba(X_test)
    bn_metrics = classification_metrics(y_test, bn_pred)
    bn_metrics.update({"fit_seconds": bn_fit, "predict_seconds": bn_predict,
                       "alpha": args.alpha,
                       "n_features": N_FEATURES,
                       "mean_max_probability": float(bn_proba.max(axis=1).mean()),
                       "fraction_prob_gt_0.99": float((bn_proba.max(axis=1) > 0.99).mean()),
                       "feature_edges": {k: list(v) for k, v in FEATURE_PARENTS.items()}})
    bn_metrics["top_confused_pairs"] = top_confused_pairs(
        np.asarray(bn_metrics["confusion_matrix"]))
    results["bayesian_network"] = bn_metrics

    # ---------------- Report ----------------
    print("\n" + "=" * 78)
    header = f"{'Model':<18}{'Accuracy':>10}{'Macro-P':>10}{'Macro-R':>10}{'Macro-F1':>10}"
    print(header)
    print("-" * 78)
    for name, m in (("Naive Bayes", nb_metrics), ("Bayesian Network", bn_metrics)):
        print(f"{name:<18}{m['accuracy']:>10.4f}{m['macro_precision']:>10.4f}"
              f"{m['macro_recall']:>10.4f}{m['macro_f1']:>10.4f}")
    print("=" * 78)
    print(f"fit/predict time: NB {nb_fit:.1f}s/{nb_predict:.1f}s, "
          f"BN {bn_fit:.1f}s/{bn_predict:.1f}s")

    for name, m in (("Naive Bayes", nb_metrics), ("Bayesian Network", bn_metrics)):
        print(f"\n[{name}] per-class metrics")
        print(f"{'class':<6}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}")
        for c in CLASS_NAMES:
            row = m["per_class"][c]
            print(f"{c:<6}{row['precision']:>10.4f}{row['recall']:>10.4f}"
                  f"{row['f1']:>10.4f}{row['support']:>10d}")
        pairs = ", ".join(f"{p['true']}->{p['pred']}({p['count']})"
                          for p in m["top_confused_pairs"])
        print(f"most confused pairs: {pairs}")

    # ---------------- Save ----------------
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[results] {out_dir / 'results.json'}")

    # ---------------- Figures ----------------
    if not args.no_figures:
        print("[figures] writing ...")
        plot_confusion(np.asarray(nb_metrics["confusion_matrix"]),
                       "Naive Bayes 混淆矩阵（测试集）", fig_dir / "confusion_nb.png")
        plot_confusion(np.asarray(bn_metrics["confusion_matrix"]),
                       "Bayesian Network 混淆矩阵（测试集）", fig_dir / "confusion_bn.png")

        plot_misclassified(X_test, y_test, nb_pred, nb_proba,
                           np.asarray(nb_metrics["confusion_matrix"]),
                           fig_dir / "misclassified_nb.png",
                           "Naive Bayes 典型误分类样例")
        plot_misclassified(X_test, y_test, bn_pred, bn_proba,
                           np.asarray(bn_metrics["confusion_matrix"]),
                           fig_dir / "misclassified_bn.png",
                           "Bayesian Network 典型误分类样例")

        plot_metrics(nb_metrics, bn_metrics, fig_dir / "metrics_comparison.png")
        plot_class_accuracy(nb_metrics, bn_metrics, fig_dir / "class_recall.png")
        plot_bn_structure(fig_dir / "bn_structure.png")

        # Feature illustration: pick a correctly classified '3' with one hole.
        from bayes_net import extract_features as _ef
        levels = _ef(X_test)
        candidates = np.flatnonzero((y_test == 3) & (levels[:, 7] == 1) & (bn_pred == 3))
        idx = int(candidates[0]) if len(candidates) else int(np.flatnonzero(y_test == 3)[0])
        plot_feature_example(X_test, y_test, fig_dir / "feature_extraction.png", index=idx)

        print(f"[figures] {fig_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
