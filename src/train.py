"""Train + compare 3 algorithms x 3 imbalance strategies with stratified CV, then evaluate on a
hold-out set. Writes models/, reports/metrics.json, reports/cv_results.csv, reports/holdout_predictions.csv."""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split

from src import config as C
from src.metrics import summarise, top_k_stats
from src.models import MODELS, STRATEGIES, make_pipeline, split_columns


def recall_top10(y, s):
    return top_k_stats(y, s, 0.10)["recall_at_top10"]


def main(n_splits: int = 3, reuse_cv: bool = False):
    df = pd.read_csv(C.DATA_PROCESSED / "model_table.csv")
    y = df[C.TARGET].values
    X = df.drop(columns=[C.TARGET])
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=C.SEED)
    X_tr, X_te, y_tr, y_te = X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]
    num, cat = split_columns(df)
    X_tr_f, X_te_f = X_tr.drop(columns=[C.ID_COL]), X_te.drop(columns=[C.ID_COL])
    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=C.SEED)
    scoring = {"pr_auc": "average_precision", "roc_auc": "roc_auc",
               "recall_top10": make_scorer(recall_top10, response_method="predict_proba")}

    rows = []
    cv_path = C.REPORTS / "cv_results.csv"
    for m in MODELS:
        for s in STRATEGIES:
            if reuse_cv and cv_path.exists():
                break
            t0 = time.time()
            pipe = make_pipeline(m, s, num, cat, pos_weight)
            r = cross_validate(pipe, X_tr_f, y_tr, cv=cv, scoring=scoring, n_jobs=1)
            rows.append({"model": m, "strategy": s,
                         "cv_pr_auc": r["test_pr_auc"].mean(), "cv_pr_auc_std": r["test_pr_auc"].std(),
                         "cv_roc_auc": r["test_roc_auc"].mean(), "cv_recall_top10": r["test_recall_top10"].mean(),
                         "fit_seconds": time.time() - t0})
            print(f"{m:14s} {s:13s} PR-AUC={rows[-1]['cv_pr_auc']:.4f} ROC={rows[-1]['cv_roc_auc']:.4f} "
                  f"R@10={rows[-1]['cv_recall_top10']:.3f} ({rows[-1]['fit_seconds']:.0f}s)", flush=True)
    if reuse_cv and cv_path.exists():
        cvdf = pd.read_csv(cv_path)
    else:
        cvdf = pd.DataFrame(rows)
        cvdf.to_csv(cv_path, index=False)

    # best strategy per family -> refit on full training set -> hold-out evaluation
    C.MODELS.mkdir(exist_ok=True)
    preds = pd.DataFrame({C.ID_COL: X_te[C.ID_COL].values, C.TARGET: y_te})
    metrics, chosen = {}, {}
    for m in MODELS:
        best = cvdf[cvdf.model == m].sort_values("cv_pr_auc", ascending=False).iloc[0]
        pipe = make_pipeline(m, best.strategy, num, cat, pos_weight).fit(X_tr_f, y_tr)
        score = pipe.predict_proba(X_te_f)[:, 1]
        preds[f"score_{m}"] = score
        metrics[m] = {"strategy": best.strategy, **summarise(y_te, score)}
        chosen[m] = pipe
        joblib.dump(pipe, C.MODELS / f"{m}.joblib")
        print(f"HOLD-OUT {m:14s} ({best.strategy}) PR-AUC={metrics[m]['pr_auc']:.4f} "
              f"ROC={metrics[m]['roc_auc']:.4f} R@10={metrics[m]['recall_at_top10']:.3f}", flush=True)

    champion = max(metrics, key=lambda k: cvdf[(cvdf.model == k) & (cvdf.strategy == metrics[k]["strategy"])]
                   .cv_pr_auc.iloc[0])
    # probability calibration (isotonic) from out-of-fold training predictions of the champion config
    pipe_c = make_pipeline(champion, metrics[champion]["strategy"], num, cat, pos_weight)
    oof = cross_val_predict(pipe_c, X_tr_f, y_tr, cv=cv, method="predict_proba")[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof, y_tr)
    joblib.dump(iso, C.MODELS / "calibrator.joblib")
    preds["prob_champion_calibrated"] = iso.predict(preds[f"score_{champion}"].values)
    metrics[champion]["brier_calibrated"] = float(np.mean((preds["prob_champion_calibrated"] - y_te) ** 2))
    preds["monthly_charge"] = X_te["monthly_charge"].values
    preds.to_csv(C.REPORTS / "holdout_predictions.csv", index=False)
    out = {"champion": champion, "holdout": metrics, "train_rows": int(len(y_tr)), "test_rows": int(len(y_te)),
           "test_churn_rate": float(y_te.mean()), "cv_folds": n_splits}
    (C.REPORTS / "metrics.json").write_text(json.dumps(out, indent=2))
    print("CHAMPION:", champion)


if __name__ == "__main__":
    import sys
    main(reuse_cv="--reuse-cv" in sys.argv)
