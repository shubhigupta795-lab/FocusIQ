"""Participant-disjoint evaluation; all preprocessing/resampling inside folds."""
import numpy as np, pandas as pd, joblib, time, hashlib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, roc_auc_score, roc_curve
from imblearn.pipeline import Pipeline as IPipeline
from imblearn.over_sampling import SMOTENC, RandomOverSampler
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from data_engine import FEATURES, NUM, CAT, ROOT, SOURCE, CLASS_NAMES
MODELS = ROOT / 'models'
MODELS.mkdir(exist_ok=True)

def prep():
    return ColumnTransformer([
        (
            'num',
            Pipeline([
                ('impute', SimpleImputer(strategy='median')),
                ('scale', StandardScaler())
            ]),
            NUM
        ),
        (
            'cat',
            Pipeline([
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]),
            CAT
        )
    ])

def pipeline(model, sampler=None):
    if sampler == 'smote':
        # Ordinal categories before SMOTENC, one-hot after sampling.
        first = ColumnTransformer([
            (
                'num',
                Pipeline([
                    ('impute', SimpleImputer(strategy='median')),
                    ('scale', StandardScaler())
                ]),
                NUM
            ),
            (
                'cat',
                Pipeline([
                    ('impute', SimpleImputer(strategy='most_frequent')),
                    (
                        'encode',
                        OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    )
                ]),
                CAT
            )
        ])
        inds = list(range(len(NUM), len(FEATURES)))
        after = ColumnTransformer([
            ('num', 'passthrough', list(range(len(NUM)))),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), inds)
        ])
        return IPipeline([
            ('prep', first),
            ('sample', SMOTENC(categorical_features=inds, random_state=42, k_neighbors=3)),
            ('encode', after),
            ('model', model)
        ])
    if sampler == 'random':
        return IPipeline([
            ('prep', prep()),
            ('sample', RandomOverSampler(random_state=42)),
            ('model', model)
        ])
    return Pipeline([('prep', prep()), ('model', model)])

def train(d):
    X = d[FEATURES]
    y = d.productivity_class
    groups = d.user_id
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.22, random_state=42).split(X, y, groups))
    xt, xv = (X.iloc[tr], X.iloc[te])
    yt, yv = (y.iloc[tr], y.iloc[te])
    g = groups.iloc[tr]
    cv = GroupKFold(3)
    candidates = {
        'Logistic regression': LogisticRegression(max_iter=1000),
        'KNN': KNeighborsClassifier(9),
        'Decision tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random forest': RandomForestClassifier(
            n_estimators=40,
            max_depth=7,
            random_state=42,
            n_jobs=1
        ),
        'Bagging': BaggingClassifier(n_estimators=12, random_state=42, n_jobs=1),
        'Gradient boosting': GradientBoostingClassifier(n_estimators=40, random_state=42),
        'XGBoost': XGBClassifier(
            n_estimators=40,
            max_depth=3,
            learning_rate=0.1,
            n_jobs=1,
            random_state=42,
            eval_metric='mlogloss'
        )
    }
    runs = []
    for name, m in candidates.items():
        p = pipeline(m)
        score = cross_val_score(p, xt, yt, groups=g, cv=cv, scoring='f1_macro').mean()
        runs.append((name, p, float(score)))
    for method in ['random', 'smote']:
        p = pipeline(LogisticRegression(max_iter=1000), method)
        s = cross_val_score(p, xt, yt, groups=g, cv=cv, scoring='f1_macro').mean()
        runs.append(('Logistic + ' + method, p, float(s)))
    gs = GridSearchCV(
        pipeline(DecisionTreeClassifier(random_state=42)),
        {'model__max_depth': [2, 4, 7], 'model__min_samples_leaf': [5, 15]},
        cv=cv,
        scoring='f1_macro',
        n_jobs=1
    )
    gs.fit(xt, yt, groups=g)
    runs.append(('Grid tuned tree', gs.best_estimator_, float(gs.best_score_)))
    rs = RandomizedSearchCV(
        pipeline(RandomForestClassifier(random_state=42, n_jobs=1)),
        {
            'model__n_estimators': [20, 40, 60],
            'model__max_depth': [3, 6, 10, None],
            'model__min_samples_leaf': [2, 5, 10]
        },
        n_iter=3,
        cv=cv,
        scoring='f1_macro',
        random_state=42,
        n_jobs=1
    )
    rs.fit(xt, yt, groups=g)
    runs.append(('Random tuned forest', rs.best_estimator_, float(rs.best_score_)))
    winner = max(runs, key=lambda r: r[2])[0]
    rows = []
    selected = None
    for name, p, score in runs:
        p.fit(xt, yt)
        z = p.predict(xv)
        probabilities = p.predict_proba(xv)
        curves = []
        for label, class_name in enumerate(CLASS_NAMES):
            fpr, tpr, _ = roc_curve(yv == label, probabilities[:, label])
            curves.append({
                "class": class_name, "fpr": fpr.tolist(), "tpr": tpr.tolist(),
                "auc": float(roc_auc_score(yv == label, probabilities[:, label]))
            })
        rows.append(dict(
            model=name,
            cv_f1=score,
            accuracy=accuracy_score(yv, z),
            precision=precision_score(yv, z, average='macro', zero_division=0),
            recall=recall_score(yv, z, average='macro', zero_division=0),
            f1=f1_score(yv, z, average='macro', zero_division=0),
            confusion=confusion_matrix(yv, z, labels=[0, 1, 2]).tolist(),
            roc_auc=float(roc_auc_score(yv, probabilities, multi_class="ovr", average="macro", labels=[0, 1, 2])),
            roc=curves
        ))
        if name == winner:
            selected = p
    reg = pipeline(LinearRegression())
    reg.fit(xt, d.productivity.iloc[tr])
    pred = reg.predict(xv)
    actual = d.productivity.iloc[te]
    bias = []
    for depth in [1, 2, 4, 7, 12, None]:
        p = pipeline(DecisionTreeClassifier(max_depth=depth, random_state=42))
        p.fit(xt, yt)
        bias.append({
            'depth': str(depth),
            'train': float(f1_score(yt, p.predict(xt), average='macro', zero_division=0)),
            'cv': float(cross_val_score(p, xt, yt, groups=g, cv=cv, scoring='f1_macro').mean())
        })
    info = {
        'winner': winner,
        'models': rows,
        'regression': {
            'mae': mean_absolute_error(actual, pred),
            'rmse': mean_squared_error(actual, pred) ** 0.5,
            'r2': r2_score(actual, pred)
        },
        'bias': bias,
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'version': time.strftime('%Y%m%d-%H%M%S'),
        'train_users': int(g.nunique()),
        'test_users': int(groups.iloc[te].nunique()),
        'low_focus_rate': float((y == 0).mean()),
        'class_names': CLASS_NAMES,
        'class_counts': [int((y == i).sum()) for i in range(3)],
        'classification_schema': 'three-class-v1',
        'grid_params': gs.best_params_,
        'random_params': rs.best_params_
    }
    bundle = {
        'classifier': selected,
        'regressor': reg,
        'info': info,
        'reference': xt.sample(min(45, len(xt)), random_state=42),
        'baseline': xt[NUM].mean().to_dict(),
        'sd': xt[NUM].std().to_dict()
    }
    path = MODELS / f"model-{info['version']}.joblib"
    joblib.dump(bundle, path)
    joblib.dump(bundle, MODELS / 'current.joblib')
    return bundle

def get(d):
    p = MODELS / 'current.joblib'
    if p.exists():
        b = joblib.load(p)
        if b['info'].get('classification_schema') == 'three-class-v1' and b['info'].get('source_sha256') == hashlib.sha256(SOURCE.read_bytes()).hexdigest():
            return b
    return train(d)

def explain(bundle, x):
    # Model-agnostic SHAP with numeric values, categorical values fixed to the current scenario.
    import shap
    background = bundle['reference'][NUM].iloc[:12].to_numpy()
    cat = x[CAT].iloc[0].to_dict()

    def predict(a):
        f = pd.DataFrame(a, columns=NUM)
        for k, v in cat.items():
            f[k] = v
        return bundle['classifier'].predict_proba(f[FEATURES])[:, 0]
    ex = shap.Explainer(
        predict,
        shap.maskers.Independent(background),
        algorithm='permutation',
        seed=42
    )
    v = ex(x[NUM].to_numpy(), max_evals=2 * len(NUM) + 1).values[0]
    return sorted(
        [{'feature': k, 'contribution': float(z)} for k, z in zip(NUM, v)],
        key=lambda r: abs(r['contribution']),
        reverse=True
    )
