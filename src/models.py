"""Model factory. Resampling (SMOTE) lives INSIDE the imblearn pipeline so it is applied to
training folds only -- never to validation folds or the hold-out set (no leakage)."""
from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src import config as C
from src.features import CATEGORICAL

MODELS = ("logistic", "random_forest", "xgboost")
STRATEGIES = ("none", "class_weight", "smote")


def split_columns(df):
    drop = {C.ID_COL, C.TARGET}
    cat = [c for c in CATEGORICAL if c in df.columns]
    num = [c for c in df.columns if c not in drop and c not in cat]
    return num, cat


def make_preprocessor(num, cat, scale: bool):
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    from sklearn.pipeline import Pipeline as SkPipe
    return ColumnTransformer([
        ("num", SkPipe(num_steps), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ], verbose_feature_names_out=False)


def make_pipeline(model: str, strategy: str, num, cat, pos_weight: float = 1.0, seed: int = C.SEED):
    weighted = strategy == "class_weight"
    if model == "logistic":
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced" if weighted else None,
                                 random_state=seed)
    elif model == "random_forest":
        clf = RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=30, n_jobs=1,
                                     class_weight="balanced_subsample" if weighted else None,
                                     random_state=seed)
    elif model == "xgboost":
        clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=5, tree_method="hist",
                            scale_pos_weight=pos_weight if weighted else 1.0,
                            eval_metric="aucpr", n_jobs=1, random_state=seed)
    else:
        raise ValueError(model)
    steps = [("prep", make_preprocessor(num, cat, scale=(model == "logistic")))]
    if strategy == "smote":
        steps.append(("smote", SMOTE(random_state=seed, k_neighbors=5)))
    steps.append(("clf", clf))
    return Pipeline(steps)
