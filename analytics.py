import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.proportion import proportions_ztest
from data_engine import FEATURES, NUM

def clean_json(v):
    if isinstance(v, dict):
        return {k: clean_json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [clean_json(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else None
    return v

def describe(d):
    cols = [
        'screen_hours',
        'productivity',
        'sleep_hours',
        'social_hours',
        'video_hours',
        'gaming_hours',
        'study_hours',
        'stress',
        'focus_score',
        'distraction_index'
    ]
    s = d[cols].describe().T
    s['median'] = d[cols].median()
    s['variance'] = d[cols].var()
    q = d[cols].quantile([0.25, 0.75])
    iqr = q.iloc[1] - q.iloc[0]
    s['outliers'] = ((d[cols] < q.iloc[0] - 1.5 * iqr) | (d[cols] > q.iloc[1] + 1.5 * iqr)).sum()
    return s.reset_index(names='feature').to_dict('records')

def research(d):
    base = d
    high = base.screen_hours >= 6
    low = base.productivity < 40
    n = int(high.sum())
    conditional = float(low[high].mean()) if n else None
    z, p = proportions_ztest(int((~low).sum()), len(base), 0.5, prop_var=0.5)
    chi, cp = stats.chi2_contingency(pd.crosstab(high, low))[:2]
    groups = [base.loc[m, 'productivity'].values for m in [
        base.screen_hours < 4,
        base.screen_hours.between(4, 6, inclusive='left'),
        base.screen_hours >= 6
    ]]
    f, fp = stats.f_oneway(*groups)
    t, tp = stats.ttest_ind(
        base.loc[high, 'productivity'],
        base.loc[~high, 'productivity'],
        equal_var=False
    )
    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(
        base.productivity.values,
        len(base),
        replace=True
    ).mean() for _ in range(500)])
    return clean_json({
        'tests': [
            {
                'test': 'One-proportion Z',
                'question': 'Students with productivity ≥40 vs 50%',
                'stat': z,
                'p': p
            },
            {
                'test': 'Welch t-test',
                'question': 'Productivity: phone usage ≥6h vs <6h',
                'stat': t,
                'p': tp
            },
            {
                'test': 'Chi-square',
                'question': 'High phone usage vs low productivity',
                'stat': chi,
                'p': cp
            },
            {
                'test': 'One-way ANOVA',
                'question': 'Productivity across three phone usage groups',
                'stat': f,
                'p': fp
            }
        ],
        'conditional': conditional,
        'conditional_n': n,
        'baseline_users': len(base),
        'bootstrap_ci': np.quantile(boot, [0.025, 0.975]),
        'group_means': [float(g.mean()) for g in groups],
        'group_counts': [len(g) for g in groups]
    })

def clusters(d):
    # Fit both methods to the same standardized sample; no target scores used.
    u = d.groupby('user_id')[NUM].mean().dropna()
    u = u.sample(n=min(2000, len(u)), random_state=42)
    a = StandardScaler().fit_transform(u)
    pca = PCA(n_components=2).fit(a)
    k = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(a)
    h = AgglomerativeClustering(n_clusters=4).fit_predict(a)
    habits = ['screen_hours', 'social_hours', 'video_hours', 'gaming_hours', 'study_hours', 'sleep_hours']
    names = ['Phone', 'Social media', 'YouTube', 'Gaming', 'Study', 'Sleep']
    profiles = {}
    quality = {}
    for method, labels in [('k', k), ('h', h)]:
        profiles[method] = []
        quality[method] = float(silhouette_score(a, labels))
        for cluster in range(4):
            group = u.loc[labels == cluster]
            means = group[habits].mean()
            # Name profiles by their largest standardized deviation from sample means.
            deviations = (means - u[habits].mean()) / u[habits].std().replace(0, 1)
            strongest = deviations.abs().nlargest(2).index
            descriptions = []
            for key in strongest:
                direction = 'Higher' if deviations[key] >= 0 else 'Lower'
                name = names[habits.index(key)]
                descriptions.append(f'{direction} {name} time')
            profiles[method].append({
                'cluster': cluster,
                'count': len(group),
                'share': len(group) / len(u),
                'name': ' / '.join(descriptions),
                **means.to_dict()
            })
    return clean_json({
        'sample_size': len(u),
        'variance': pca.explained_variance_ratio_,
        'profiles': profiles,
        'silhouette': quality,
        'overall': u[habits].mean().to_dict()
    })

def probability(kind, param, n, alpha, effect):
    rng = np.random.default_rng(42)
    n = int(n)
    if kind == 'Normal':
        values = rng.normal(60, param, n)
        label = 'Simulated focus score (unbounded model)'
    elif kind == 'Binomial':
        values = rng.binomial(7, min(param / 100, 1), n)
        label = 'Productive days out of seven'
    elif kind == 'Poisson':
        values = rng.poisson(param, n)
        label = 'Simulated notification arrivals per interval (not in CSV)'
    else:
        values = rng.uniform(0, param, n)
        label = 'Break duration in minutes'
    # One-sided Z test, standardized population SD known, H0 mean 0.
    threshold = stats.norm.ppf(1 - alpha)
    null = rng.normal(0, 1, 10000)
    alt = rng.normal(effect * np.sqrt(n), 1, 10000)
    return clean_json({
        'values': values,
        'label': label,
        'mean': np.mean(values),
        'sd': np.std(values),
        'type1': np.mean(null > threshold),
        'type2': np.mean(alt <= threshold),
        'power': np.mean(alt > threshold)
    })
