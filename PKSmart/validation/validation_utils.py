import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (5, 5)


def fraction_within_fold(pred, true, fold=2, log_transformed=True, fu_endpoint=False):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)

    if fu_endpoint:
        ratio = pred / np.where(true == 0, 1e-10, true)
        within = (ratio <= fold) & (ratio >= 1/fold)
    elif log_transformed:
        # In log scale, diff = pred - true. 10^diff is the ratio.
        # fold 2 means |diff| <= log10(2)
        diff = pred - true
        within = np.abs(diff) <= np.log10(fold)
    else:
        ratio = pred / np.where(true == 0, 1e-10, true)
        within = (ratio <= fold) & (ratio >= 1/fold)

    return within.mean() * 100.0


def gmfe(pred, true, log_transformed=True, fu_endpoint=False):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)

    if fu_endpoint:
        logs = np.abs(np.log10(pred / np.where(true == 0, 1e-10, true)))
    elif log_transformed:
        # pred and true are already logs
        logs = np.abs(pred - true)
    else:
        logs = np.abs(np.log10(pred / np.where(true == 0, 1e-10, true)))

    return 10 ** logs.mean()


def bias_relative(pred, true, log_transformed=True, fu_endpoint=False):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)

    if fu_endpoint or not log_transformed:
        # Avoid division by zero
        safe_true = np.where(true == 0, 1e-10, true)
        diff = (pred - true) / safe_true
    else:
        # In linear scale
        pred_lin = 10 ** pred
        true_lin = 10 ** true
        diff = (pred_lin - true_lin) / true_lin

    return float(np.mean(diff)) * 100.0


def bias_median(pred, true, log_transformed=True, fu_endpoint=False):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)

    if fu_endpoint:
        diff = pred - true
    elif log_transformed:
        diff = 10 ** pred - 10 ** true
    else:
        diff = pred - true

    return float(np.median(diff))


def eval_regression_metrics(pred, true, log_transformed=True, fu_endpoint=False):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)

    # RMSE 계산: 호환성을 위해 직접 sqrt 사용
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)

    return {
        "R2": float(r2_score(true, pred)),
        "RMSE": float(rmse),
        "GMFE": float(gmfe(pred, true, log_transformed=log_transformed, fu_endpoint=fu_endpoint)),
        "within_2fold_%": float(fraction_within_fold(pred, true, 2, log_transformed, fu_endpoint)),
        "within_3fold_%": float(fraction_within_fold(pred, true, 3, log_transformed, fu_endpoint)),
        "within_5fold_%": float(fraction_within_fold(pred, true, 5, log_transformed, fu_endpoint)),
        "bias_median": float(bias_median(pred, true, log_transformed, fu_endpoint)),
    }


def plot_pred_vs_true(pred, true, title="", log_transformed=True):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)

    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=true, y=pred, s=20, alpha=0.6)

    min_v = min(true.min(), pred.min())
    max_v = max(true.max(), pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], "k--", lw=1)

    xlabel = "Observed (log10)" if log_transformed else "Observed"
    ylabel = "Predicted (log10)" if log_transformed else "Predicted"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_fold_error_hist(pred, true, log_transformed=True, fu_endpoint=False, title="Fold error distribution"):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)

    if fu_endpoint:
        fold_err = np.abs(pred / true)
    elif log_transformed:
        fold_err = np.abs(10 ** pred / 10 ** true)
    else:
        fold_err = np.abs(pred / true)

    log_fe = np.log10(fold_err)

    plt.figure(figsize=(5, 4))
    sns.histplot(log_fe, kde=True, bins=30)
    plt.axvline(0, color="k", linestyle="--", label="1-fold")
    plt.axvline(np.log10(2), color="r", linestyle="--", label="2-fold")
    plt.axvline(-np.log10(2), color="r", linestyle="--")
    plt.axvline(np.log10(3), color="orange", linestyle="--", label="3-fold")
    plt.axvline(-np.log10(3), color="orange", linestyle="--")
    plt.xlabel("log10(fold error)")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def categorize_cl(log10_cl, low_thr=0.70, mid_thr=1.08):
    if log10_cl < low_thr:
        return "low"
    elif log10_cl < mid_thr:
        return "medium"
    else:
        return "high"


def add_cl_category_columns(df, true_col="true", pred_col="pred"):
    df = df.copy()
    df["cl_true_cat"] = df[true_col].apply(categorize_cl)
    df["cl_pred_cat"] = df[pred_col].apply(categorize_cl)
    return df


def eval_cl_classification(df, true_cat_col="cl_true_cat", pred_cat_col="cl_pred_cat"):
    y_true = df[true_cat_col].values
    y_pred = df[pred_cat_col].values

    labels = ["low", "medium", "high"]
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("CL category confusion matrix")
    plt.tight_layout()
    plt.show()

    return report, cm_df


def plot_cumulative_fold_error(pred, true, log_transformed=True, fu_endpoint=False, title="Cumulative Fold Error"):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)

    if fu_endpoint:
        fold_errors = np.maximum(pred / np.where(true == 0, 1e-10, true), true / np.where(pred == 0, 1e-10, pred))
    elif log_transformed:
        fold_errors = 10 ** np.abs(pred - true)
    else:
        fold_errors = np.maximum(pred / np.where(true == 0, 1e-10, true), true / np.where(pred == 0, 1e-10, pred))

    fold_errors = np.sort(fold_errors)
    cumulative_perc = np.arange(1, len(fold_errors) + 1) / len(fold_errors) * 100

    plt.figure(figsize=(6, 4))
    plt.plot(fold_errors, cumulative_perc, lw=2, color='royalblue')
    plt.xscale('log')
    plt.axvline(2, color='r', linestyle='--', label='2-fold')
    plt.axvline(3, color='orange', linestyle='--', label='3-fold')
    plt.xlim(1, 10)
    plt.ylim(0, 105)
    plt.xlabel("Fold Error (True/Pred or Pred/True)")
    plt.ylabel("Cumulative Percentage (%)")
    plt.title(title)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_residual(pred, true, title="Residual Plot"):
    residuals = pred - true
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=true, y=residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Observed (Log10)")
    plt.ylabel("Residual (Pred - Obs)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names, top_n=20, title="Top Feature Importance"):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'best_estimator_') and hasattr(model.best_estimator_, 'feature_importances_'):
        importances = model.best_estimator_.feature_importances_
    else:
        print("Model does not support feature_importances_")
        return

    indices = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(indices)), importances[indices], align='center', color='teal')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Importance Score")
    plt.title(title)
    plt.tight_layout()
    plt.show()

