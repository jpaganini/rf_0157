#!/usr/bin/env python
"""
Script: 05_evaluate_model.py

Inputs:
  --model          Path to trained .joblib model artifact.
  --features       Path to TSV file containing features, label, and group columns.
  --label          Name of the label column in the TSV.
  --group_column   Name of the group column in the TSV.
  --n_splits       Number of cross-validation folds (mutually exclusive with --no_cv). Default: 5.
  --no_cv          If set, skip cross-validation and predict on the full dataset once.
  --output_dir     Directory in which to write outputs (will be created if it does not exist).
  --name           Prefix to add to all output filenames.
  --log_level      Logging verbosity level (choices: DEBUG, INFO, WARNING, ERROR). Default: INFO.

Outputs:
  <name>_predictions.tsv            Tab-separated file with 'truth' and 'prediction' columns for each sample.
  <name>_probabilities.tsv          Tab-separated file with class probability columns indexed by sample ID.
  <name>_classification_report.tsv  Tab-separated file summarizing precision, recall, f1-score for each class.
  <name>_confusion_matrix.tsv       Tab-separated file containing the confusion matrix (actual vs. predicted).
  <name>_feature_importances.tsv    Tab-separated file of feature importance scores averaged across folds (if available).
  <name>_shap_values.npy            NumPy array file of aggregated SHAP values across folds (if computed).
  <name>_confusion_matrix.png       Heatmap of the confusion matrix saved as PNG.
  <name>_shap_summary.png           SHAP summary beeswarm plot saved as PNG (if SHAP values computed).

Instructions of Use:
  This script performs grouped cross-validation or single-run evaluation of a scikit-learn/imblearn
  pipeline model. It loads the model and features table, executes training and/or prediction,
  computes metrics and explanations, and writes results to the specified output directory.

Usage Example:
  python 05_evaluate_model.py \
    --model path/to/model.joblib \
    --features path/to/features.tsv \
    --label target_column \
    --group_column group_column \
    --n_splits 5 \
    --output_dir results/ \
    --name experiment1 \
    --log_level INFO
"""

# ────────────────────────────── standard library ─────────────────────────────
from __future__ import annotations

import argparse                      # CLI parsing
import logging                       # console logging
import sys                           # graceful exits
from collections import defaultdict  # group counting helper
from pathlib import Path             # path handling
from typing import List              # type alias
from typing import Optional, List

# ──────────────────────────────── 3rd‑party ──────────────────────────────────
import joblib                        # load joblib artefact
import matplotlib.pyplot as plt      # plots
import numpy as np                   # numerical ops
import pandas as pd                  # TSV I/O
import shap                          # SHAP explanations
from sklearn.base import clone       # deep‑copy estimator
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import compute_sample_weight

# ─────────────────────────────── global consts ───────────────────────────────
RSEED = 50
plt.rcParams.update({"figure.autolayout": True})

# ╭──────────────────────────────────────────────────────────────────────────╮
# │                               utilities                                 │
# ╰──────────────────────────────────────────────────────────────────────────╯

def safe_mkdir(path: Path) -> None:
    """Create *parent* directory of *path* if it isn’t `.`."""
    parent = path.expanduser().resolve().parent
    if parent != Path("."):
        parent.mkdir(parents=True, exist_ok=True)


def infer_problem_type(y: np.ndarray) -> str:
    """Return `'multilabel'` (>2 classes) or `'binary'`."""
    return "multilabel" if len(np.unique(y)) > 2 else "binary"


def min_groups_per_class(y: np.ndarray, groups: np.ndarray) -> int:
    """Smallest number of distinct groups containing any class."""
    c2g = defaultdict(set)
    for label, grp in zip(y, groups):
        c2g[label].add(grp)
    return min(len(g) for g in c2g.values())


def get_cv_iterator(y: np.ndarray, groups: np.ndarray, n_splits):
    """Return list of train/test indices for grouped CV; error if impossible."""
    min_g = min_groups_per_class(y, groups)
    if n_splits > min_g:
        raise ValueError(
            f"n_splits={n_splits} > available groups per rarest class ({min_g})")
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RSEED)
    return list(sgkf.split(np.zeros_like(y), y, groups))

def predict_with_pipeline(
    pipeline, X: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    return pipeline.predict(X), pipeline.predict_proba(X)

# ╭──────────────────────────────────────────────────────────────────────────╮
# │                          core evaluation loop                           │
# ╰──────────────────────────────────────────────────────────────────────────╯

def run_evaluation(
    model_path: Path,
    features_tsv: Path,
    label_col: str,
    group_col: str,
    n_splits: Optional[int],
    no_cv: bool,
    output_dir: Path,
    name: str,
) -> None:
    """Grouped‑CV evaluation: predictions, metrics, feature importances, SHAP."""

    # 0️⃣  Output directory
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1️⃣  Load artefacts & data ------------------------------------------------
    logging.info("Loading model ➜ %s", model_path)
    pipeline = joblib.load(model_path)  # imblearn.Pipeline or estimator

    logging.info("Reading feature matrix ➜ %s", features_tsv)
    df = pd.read_csv(features_tsv, sep="\t", index_col=0)
    if label_col not in df.columns or group_col not in df.columns:
        raise KeyError("label or group column missing in TSV header")

    y = df[label_col].values
    X = df.drop(columns=[label_col, group_col])
    groups = df[group_col].values

    problem_type = infer_problem_type(y)
    class_names: List[str] = np.unique(y).tolist()

    # containers for results
    oof_true, oof_pred, oof_proba = [], [], []
    feature_imps, shap_vals_all = [], []

    if no_cv:
        logging.info("Hold‑out mode (no CV)")
        preds, probas = predict_with_pipeline(pipeline, X)
        oof_true.append(y)
        oof_pred.append(preds)
        oof_proba.append(
            pd.DataFrame(probas, index=X.index, columns=class_names)
        )
        model_step = pipeline.named_steps.get('model', pipeline)
        if hasattr(model_step, 'feature_importances_'):
            feature_imps.append(
                pd.Series(model_step.feature_importances_, index=X.columns)
            )
        shap_vals_all.append(shap.TreeExplainer(model_step).shap_values(X))
    else:
        if n_splits is None:
            raise ValueError("n_splits required when CV enabled")
    # 2️⃣  Build CV iterator ----------------------------------------------------
        cv_iter = get_cv_iterator(y, groups, n_splits)

        # 3️⃣  Fold loop ------------------------------------------------------------
        for fold, (tr, te) in enumerate(cv_iter, 1):
            logging.info("Fold %d/%d", fold, n_splits)
            X_tr, X_te = X.iloc[tr], X.iloc[te]
            y_tr, y_te = y[tr], y[te]

            # clone keeps original hyper‑params but clears fitted state
            clf = clone(pipeline)

            # balanced sample‑weights for XGBoost when *no* oversampling step exists
            uses_oversampler = "oversampler" in clf.named_steps
            is_xgb = clf.named_steps.get("model").__class__.__name__.startswith("XGB")
            fit_kwargs = {}
            if is_xgb and not uses_oversampler:
                fit_kwargs["model__sample_weight"] = compute_sample_weight("balanced", y_tr)

            clf.fit(X_tr, y_tr, **fit_kwargs)

            preds = clf.predict(X_te)
            probas = clf.predict_proba(X_te)

            oof_true.append(y_te)
            oof_pred.append(preds)
            oof_proba.append(pd.DataFrame(probas, index=X_te.index, columns=class_names))

            # feature importances
            model_step = clf.named_steps.get("model", clf)
            if hasattr(model_step, "feature_importances_"):
                feature_imps.append(pd.Series(model_step.feature_importances_, index=X.columns))

            # SHAP values
            shap_vals_all.append(shap.TreeExplainer(model_step).shap_values(X_te))

    # 4️⃣  Aggregate OOF results -----------------------------------------------
    y_true = np.concatenate(oof_true)
    y_pred = np.concatenate(oof_pred)
    proba_df = pd.concat(oof_proba)

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    logging.info("OOF Accuracy %.4f | Balanced Accuracy %.4f", acc, bacc)

    report_df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).T.reset_index()
    cm_arr = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_df = pd.DataFrame(cm_arr, index=class_names, columns=class_names)

    if feature_imps:
        fi_df = pd.concat(feature_imps, axis=1)
        fi_df["average"] = fi_df.mean(axis=1)
        fi_df = fi_df.sort_values("average", ascending=False)
    else:
        fi_df = pd.DataFrame()

    # combine SHAP arrays across folds
    shap_stack = None
    if shap_vals_all:
        shap_stack = (
            [np.concatenate([fold[c] for fold in shap_vals_all], axis=0) for c in range(len(class_names))]
            if isinstance(shap_vals_all[0], list)
            else np.concatenate(shap_vals_all, axis=0)
    )

    # 5️⃣  Write artefacts ------------------------------------------------------
    files = {
        "predictions": output_dir / f"{name}_predictions.tsv",
        "probabilities": output_dir / f"{name}_probabilities.tsv",
        "class_report": output_dir / f"{name}_classification_report.tsv",
        "conf_matrix": output_dir / f"{name}_confusion_matrix.tsv",
        "feat_importances": output_dir / f"{name}_feature_importances.tsv",
        "shap_values": output_dir / f"{name}_shap_values.npy",
    }

    proba_df.to_csv(files["probabilities"], sep="\t")
    pd.DataFrame({"truth": y_true, "prediction": y_pred}).to_csv(files["predictions"], sep="\t", index=False)
    report_df.to_csv(files["class_report"], sep="\t", index=False)
    cm_df.to_csv(files["conf_matrix"], sep="\t")
    if not fi_df.empty:
        fi_df.to_csv(files["feat_importances"], sep="\t")
    if shap_stack is not None:
        np.save(files["shap_values"], shap_stack)

    # 6️⃣  Diagnostic plots -----------------------------------------------------
    # confusion matrix heat‑map
    ConfusionMatrixDisplay(confusion_matrix=cm_arr, display_labels=class_names).plot(
        cmap=plt.cm.Blues,
        values_format=".2g",
    )
    plt.savefig(output_dir / f"{name}_confusion_matrix.png", dpi=300)
    plt.close()

    # SHAP summary beeswarm
    if shap_stack is not None:
        shap.summary_plot(
            shap_stack if problem_type == "binary" else shap_stack[0],
            X.loc[proba_df.index],
            feature_names=X.columns,
            show=False,
        )
        plt.savefig(output_dir / f"{name}_shap_summary.png", dpi=300)
        plt.close()

    logging.info("All outputs written to %s", output_dir)

# ╭──────────────────────────────────────────────────────────────────────────╮
# │                                 CLI                                    │
# ╰──────────────────────────────────────────────────────────────────────────╯

def parse_args():
    p = argparse.ArgumentParser("Grouped‑CV evaluation (TSV outputs only)")
    p.add_argument("--model", type=Path, required=True, help="trained *.joblib artefact")
    p.add_argument("--features", type=Path, required=True, help="TSV with features + label + group")
    p.add_argument("--label", required=True, help="name of the label column")
    p.add_argument("--group_column", required=True, help="name of the group column")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--n_splits", type=int, help="CV folds (default 5)")
    group.add_argument('--no_cv', action='store_true', help='Skip CV and predict full dataset once')
    p.add_argument("--output_dir", type=Path, required=True, help="directory for outputs")
    p.add_argument("--name", required=True, help="prefix for output files")
    p.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        run_evaluation(
            model_path=args.model,
            features_tsv=args.features,
            label_col=args.label,
            group_col=args.group_column,
            n_splits=args.n_splits,
            no_cv=args.no_cv,
            output_dir=args.output_dir,
            name=args.name
        )
    except Exception:
        logging.exception("Evaluation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()