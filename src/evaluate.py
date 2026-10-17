"""Hold-out evaluation: PR/gains/lift/calibration, SHAP explainability, odds ratios, and the retention
cost model. Reads outputs of src.train; writes figures + reports/business_results.json."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from src import business as B
from src import config as C
from src.metrics import gains_curve
from src.models import split_columns
from src.plotting import PALETTE, plt, save, style

LABELS = {"logistic": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost"}


def main():
    style()
    met = json.loads((C.REPORTS / "metrics.json").read_text())
    champ = met["champion"]
    preds = pd.read_csv(C.REPORTS / "holdout_predictions.csv")
    y = preds[C.TARGET].values
    cv = pd.read_csv(C.REPORTS / "cv_results.csv")
    out = {"champion": champ}

    # 1. CV heatmap: algorithm x imbalance strategy
    piv = cv.pivot(index="model", columns="strategy", values="cv_pr_auc").loc[list(LABELS)][["none", "class_weight", "smote"]]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    import seaborn as sns
    sns.heatmap(piv, annot=True, fmt=".3f", cmap="Blues", ax=ax, cbar=False,
                yticklabels=[LABELS[i] for i in piv.index])
    ax.set_title("3-fold CV PR-AUC: algorithm x imbalance strategy"); ax.set_xlabel(""); ax.set_ylabel("")
    save(fig, "model_cv_comparison.png")

    # 2. Precision-recall curves
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for m, col in zip(LABELS, ["#8c8c8c", "#2a9d8f", PALETTE["accent"]]):
        p, r, _ = precision_recall_curve(y, preds[f"score_{m}"])
        ax.plot(r, p, color=col, lw=2, label=f"{LABELS[m]} (AP={met['holdout'][m]['pr_auc']:.3f})")
    ax.axhline(y.mean(), color="black", ls=":", lw=1, label=f"No skill ({y.mean():.1%})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-recall (hold-out)"); ax.legend()
    save(fig, "model_pr_curves.png")

    # 3. Cumulative gains + decile lift
    s = preds[f"score_{champ}"].values
    gx, gy = gains_curve(y, s)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(gx * 100, gy * 100, color=PALETTE["primary"], lw=2, label=LABELS[champ])
    ax[0].plot([0, 100], [0, 100], color="black", ls=":", label="Random")
    ax[0].set_xlabel("% of customers contacted (highest risk first)"); ax[0].set_ylabel("% of churners captured")
    ax[0].set_title("Cumulative gains"); ax[0].legend()
    dec = pd.DataFrame({"y": y, "s": s}); dec["decile"] = pd.qcut(dec.s.rank(method="first", ascending=False), 10, labels=range(1, 11))
    lift = dec.groupby("decile", observed=True).y.mean() / y.mean()
    ax[1].bar(lift.index.astype(str), lift.values, color=PALETTE["primary"])
    ax[1].axhline(1, color="black", ls=":"); ax[1].set_xlabel("Risk decile (1 = highest)"); ax[1].set_ylabel("Lift vs base rate")
    ax[1].set_title("Lift by decile")
    for i, v in enumerate(lift.values):
        ax[1].text(i, v + 0.1, f"{v:.1f}x", ha="center", fontsize=8)
    save(fig, "model_gains_lift.png")
    out["lift_by_decile"] = {int(k): float(v) for k, v in lift.items()}

    # 4. Calibration (raw score vs isotonic-calibrated)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for col, lab, c in (("score_" + champ, "Raw score", "#8c8c8c"), ("prob_champion_calibrated", "Isotonic-calibrated", PALETTE["accent"])):
        fr, mp = calibration_curve(y, preds[col], n_bins=10, strategy="quantile")
        ax.plot(mp, fr, "o-", color=c, label=lab)
    ax.plot([0, 1], [0, 1], "k:", label="Perfect"); ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration (hold-out)"); ax.legend()
    save(fig, "model_calibration.png")

    # 5. Cost model
    mc = preds["monthly_charge"].values
    curve = B.net_value_curve(y, s, mc)
    best = B.optimal_point(curve)
    top10 = curve.iloc[(curve.contact_fraction - 0.10).abs().argmin()]
    va = B.value_aware_rule(y, preds["prob_champion_calibrated"].values, mc)
    rnd = B.random_baseline(y, mc, best.contact_fraction)
    sens = B.sensitivity(y, s, mc)
    sens.to_csv(C.REPORTS / "sensitivity.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(curve.contact_fraction * 100, curve.net_value / 1000, color=PALETTE["primary"], lw=2, label="Model-ranked outreach")
    ax.axhline(0, color="black", lw=0.8)
    ax.scatter([best.contact_fraction * 100], [best.net_value / 1000], color=PALETTE["accent"], zorder=5,
               label=f"Optimum: top {best.contact_fraction:.0%} -> ${best.net_value/1000:,.1f}k")
    ax.set_xlabel("% of customers contacted"); ax.set_ylabel("Net value ($k per 10k customers)")
    ax.set_title("Retention campaign economics (illustrative assumptions)"); ax.legend()
    save(fig, "business_net_value_curve.png")
    pv = sens.pivot(index="save_rate", columns="cost_per_contact", values="net_value") / 1000
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    sns.heatmap(pv, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax, cbar_kws={"label": "$k"})
    ax.set_title("Net value at optimum ($k): sensitivity"); ax.set_xlabel("Cost per contact ($)"); ax.set_ylabel("Offer save rate")
    save(fig, "business_sensitivity.png")
    out["business"] = {
        "holdout_customers": int(len(y)), "holdout_churners": int(y.sum()),
        "optimal_contact_fraction": float(best.contact_fraction), "optimal_contacted": int(best.contacted),
        "optimal_churners_reached": int(best.churners_reached), "optimal_net_value": float(best.net_value),
        "optimal_precision": float(best.churners_reached / best.contacted),
        "top10_net_value": float(top10.net_value), "random_same_budget_net_value": rnd,
        "value_aware_rule": va, "assumptions": {
            "offer_cost": C.OFFER_COST, "contact_cost": C.CONTACT_COST, "save_rate": C.SAVE_RATE,
            "retained_months": C.RETAINED_MONTHS, "gross_margin": C.GROSS_MARGIN},
        "min_sensitivity_net_value": float(sens.net_value.min()), "max_sensitivity_net_value": float(sens.net_value.max())}

    # 6. SHAP on champion (tree model on transformed features)
    df = pd.read_csv(C.DATA_PROCESSED / "model_table.csv")
    num, cat = split_columns(df)
    pipe = joblib.load(C.MODELS / f"{champ}.joblib")
    test_ids = preds[C.ID_COL]
    Xs = df.set_index(C.ID_COL).loc[test_ids].drop(columns=[C.TARGET]).sample(3000, random_state=C.SEED)
    Xt = pipe.named_steps["prep"].transform(Xs)
    names = pipe.named_steps["prep"].get_feature_names_out()
    Xt = pd.DataFrame(Xt, columns=names)
    sv = shap.TreeExplainer(pipe.named_steps["clf"]).shap_values(Xt)
    plt.figure(); shap.summary_plot(sv, Xt, max_display=15, show=False); fig = plt.gcf(); fig.set_size_inches(8, 6)
    plt.title("SHAP summary: what drives predicted churn risk"); save(fig, "shap_summary.png")
    imp = pd.Series(np.abs(sv).mean(0), index=names).sort_values(ascending=False)
    imp.to_csv(C.REPORTS / "shap_importance.csv", header=["mean_abs_shap"])
    out["top_shap_features"] = imp.head(10).round(4).to_dict()

    # 7. Logistic regression odds ratios (per +1 SD) for interpretable communication
    lr = joblib.load(C.MODELS / "logistic.joblib")
    lr_names = lr.named_steps["prep"].get_feature_names_out()
    coef = pd.Series(lr.named_steps["clf"].coef_[0], index=lr_names)
    orr = np.exp(coef).sort_values(ascending=False)
    orr = orr[~orr.index.str.startswith("missingindicator")]
    pd.DataFrame({"odds_ratio_per_sd": orr}).to_csv(C.REPORTS / "logistic_odds_ratios.csv")
    out["logistic_top_odds_ratios"] = orr.head(6).round(2).to_dict()
    out["logistic_lowest_odds_ratios"] = orr.tail(6).round(2).to_dict()

    # 8. Top-risk action list (champion score, calibrated probability)
    action = preds[[C.ID_COL, "monthly_charge", "prob_champion_calibrated"]].copy()
    action["risk_decile"] = dec["decile"].values.astype(int)
    action.sort_values("prob_champion_calibrated", ascending=False).head(500).to_csv(
        C.REPORTS / "top_risk_action_list_sample.csv", index=False)

    (C.REPORTS / "business_results.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
