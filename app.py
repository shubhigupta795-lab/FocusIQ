from pathlib import Path
from contextlib import asynccontextmanager
from datetime import date
import io, sqlite3, threading
import numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from typing import Literal
from reportlab.pdfgen import canvas
from plotly.offline import get_plotlyjs
from data_engine import ROOT, FEATURES, NUM, AGE, STYLE, load, engineer, wellness, persona, CLASS_NAMES
import ml, analytics
D, AUDIT = load()
LOCK = threading.Lock()
BUNDLE = None
DB = ROOT / 'data' / 'local.db'

def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con
with db() as con:
    con.execute('CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY, created TEXT DEFAULT CURRENT_TIMESTAMP, version TEXT, predicted REAL, risk REAL, actual REAL, features TEXT, category TEXT)')
    if 'category' not in {r[1] for r in con.execute('PRAGMA table_info(predictions)')}:
        con.execute('ALTER TABLE predictions ADD COLUMN category TEXT')

def bundle():
    global BUNDLE
    with LOCK:
        if BUNDLE is None:
            BUNDLE = ml.get(D)
        return BUNDLE
app = FastAPI(title='FocusIQ', version='4.0.0')
app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')

@app.get('/')
def index():
    return FileResponse(ROOT / 'static' / 'index.html')

@app.get('/plotly.js', include_in_schema=False)
def plotlyjs():
    return Response(get_plotlyjs(), media_type='application/javascript')

class Habits(BaseModel):
    screen_hours: float = Field(6, ge=0, le=24)
    social_hours: float = Field(2, ge=0, le=24)
    video_hours: float = Field(1, ge=0, le=24)
    gaming_hours: float = Field(0.5, ge=0, le=24)
    study_hours: float = Field(5, ge=0, le=24)
    sleep_hours: float = Field(7, ge=0, le=24)
    breaks: int = Field(7, ge=0, le=30)
    exercise: int = Field(60, ge=0, le=300)
    stress: int = Field(5, ge=1, le=10)
    age: int = Field(23, ge=17, le=29)
    gender: Literal['Male', 'Female', 'Other'] = 'Male'
    night_minutes: float | None = Field(None, ge=0, le=180)
    notifications: int | None = Field(None, ge=0, le=2000)
    checks: int | None = Field(None, ge=0, le=2000)

    @model_validator(mode='after')
    def consistent(self):
        if self.night_minutes is not None and self.night_minutes > self.screen_hours * 60:
            raise ValueError('Night usage must not exceed total phone usage.')
        return self

class Actual(BaseModel):
    actual: float = Field(ge=0, le=100)

def frame(h):
    return engineer(pd.DataFrame([h.model_dump()]))[FEATURES]

def profile(h):
    v = h.model_dump()
    x = frame(h)
    b = bundle()
    score = float(np.clip(b['regressor'].predict(x)[0], 0, 100))
    probabilities = b['classifier'].predict_proba(x)[0]
    risk = float(probabilities[0])
    category = CLASS_NAMES[int(np.argmax(probabilities))]
    means = D.groupby('user_id').screen_hours.mean()
    percentile = float((means < v['screen_hours']).mean() * 100)
    return {
        'wellness': wellness(v),
        'persona': persona(v),
        'predicted_focus': score,
        'risk': risk,
        'percentile': percentile,
        'distraction': float(engineer(pd.DataFrame([v])).distraction_index.iloc[0]),
        'category': category,
        'probabilities': dict(zip(CLASS_NAMES, map(float, probabilities))),
        'version': b['info']['version'],
        'tips': [
            'Social: use scheduled check-in windows.',
            'Video: disable autoplay.',
            'Gaming: set a session timer.',
            'Work: protect uninterrupted focus blocks.'
        ],
        'out_of_range': [k for k in NUM if v[k] < D[k].min() or v[k] > D[k].max()]
    }

@app.post('/api/profile')
def predict(h: Habits):
    p = profile(h)
    with db() as con:
        cur = con.execute(
            'INSERT INTO predictions(version,predicted,risk,features,category) VALUES(?,?,?,?,?)',
            (p['version'], p['predicted_focus'], p['risk'], frame(h).to_json(orient='records'), p['category'])
        )
        p['prediction_id'] = cur.lastrowid
    return p

@app.post('/api/scenario')
def scenario(h: Habits):
    return profile(h)

@app.post('/api/explain')
def explain(h: Habits):
    return ml.explain(bundle(), frame(h))

@app.get('/api/data')
def data(age: str='All', style: str='All', minimum: float=0, maximum: float=24):
    d = D[D.screen_hours.between(minimum, maximum)]
    d = d if age == 'All' else d[d.age_group == age]
    d = d if style == 'All' else d[d.gender == style]
    cols = [
        'user_id',
        'age',
        'age_group',
        'gender',
        'screen_hours',
        'social_hours',
        'video_hours',
        'gaming_hours',
        'study_hours',
        'productivity',
        'focus_score',
        'sleep_hours',
        'breaks',
        'exercise',
        'stress',
        'distraction_index',
        'coffee_intake_mg',
        'assignments_completed',
        'attendance_percentage',
        'final_grade'
    ]
    return {
        'rows': d[cols].replace({np.nan: None}).to_dict('records'),
        'audit': AUDIT,
        'stats': analytics.clean_json(analytics.describe(d))
    }

@app.get('/api/research')
def research():
    return analytics.research(D)

@app.get('/api/clusters')
def clusters():
    return analytics.clusters(D)

@app.get('/api/probability')
def probability(
    kind: Literal['Normal', 'Binomial', 'Poisson', 'Uniform']='Normal',
    param: float=Query(10, gt=0, le=100),
    n: int=Query(100, ge=5, le=1000),
    alpha: float=Query(0.05, gt=0, lt=0.5),
    effect: float=Query(0.3, ge=0, le=3)
):
    return analytics.probability(kind, param, n, alpha, effect)

@app.get('/api/models')
def models():
    return bundle()['info']
IMPORTANCE_CACHE = {}

@app.get('/api/importance')
def importance():
    from sklearn.model_selection import GroupShuffleSplit
    b = bundle()
    version = b['info']['version']
    if version not in IMPORTANCE_CACHE:
        # Reproduce the held-out split used in ml.train; no training data is scored.
        _, te = next(GroupShuffleSplit(n_splits=1, test_size=0.22, random_state=42).split(D[FEATURES], D.productivity_class, D.user_id))
        sample = D.iloc[te][FEATURES].sample(n=min(24, len(te)), random_state=42)
        sums = {k: 0.0 for k in NUM}
        for i in range(len(sample)):
            for r in ml.explain(b, sample.iloc[[i]]):
                sums[r['feature']] += abs(r['contribution'])
        IMPORTANCE_CACHE[version] = {
            'sample_size': len(sample),
            'features': sorted(
                [{'feature': k, 'mean_abs_shap': v / len(sample)} for k, v in sums.items()],
                key=lambda r: r['mean_abs_shap'],
                reverse=True
            ),
            'version': version
        }
    return IMPORTANCE_CACHE[version]

@app.post('/api/retrain')
def retrain():
    global BUNDLE
    with LOCK:
        BUNDLE = ml.train(D)
        return BUNDLE['info']

@app.patch('/api/predictions/{id}')
def actual(id: int, v: Actual):
    with db() as con:
        c = con.execute('UPDATE predictions SET actual=? WHERE id=?', (v.actual, id))
        if not c.rowcount:
            raise HTTPException(404, 'Prediction not found')
    return {'ok': True}

@app.get('/api/monitor')
def monitor():
    with db() as con:
        rows = [dict(x) for x in con.execute('SELECT * FROM predictions ORDER BY id DESC')]
    b = bundle()
    current = [r for r in rows if r['version'] == b['info']['version']]
    labeled = [r for r in current if r['actual'] is not None]
    drift = {}
    if current:
        import json
        x = pd.DataFrame([json.loads(r['features'])[0] for r in current])
        drift = {k: abs(float(x[k].mean()) - b['baseline'][k]) / max(b['sd'][k], 1e-06) for k in NUM if x[k].notna().any()}
    # Use only actual outcomes, grouped by the date the prediction was made.
    daily = []
    if labeled:
        observed = pd.DataFrame(labeled)
        observed['day'] = observed.created.str[:10]
        observed['error'] = (observed.actual - observed.predicted).abs()
        trend = observed.groupby('day').error.mean().reset_index()
        trend['rolling'] = trend.error.rolling(3, min_periods=3).mean()
        daily = analytics.clean_json(trend.to_dict('records'))
    for r in rows:
        r.pop('features')
    return {
        'versions': sorted((p.stem for p in ml.MODELS.glob('model-*.joblib'))),
        'current': b['info']['version'],
        'predictions': rows[:100],
        'labeled': len(labeled),
        'mae': float(np.mean([abs(r['actual'] - r['predicted']) for r in labeled])) if labeled else None,
        'drift': analytics.clean_json(drift),
        'current_count': len(current),
        'daily_errors': daily
    }

@app.post('/api/report')
def report(h: Habits):
    p = profile(h)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(595, 842))
    c.setTitle('FocusIQ personal report')
    c.setFillColorRGB(0.08, 0.18, 0.22)
    c.rect(0, 700, 595, 142, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont('Helvetica-Bold', 26)
    c.drawString(42, 785, 'FocusIQ / Personal report')
    c.setFont('Helvetica', 13)
    c.drawString(42, 750, 'Digital habits and productivity assessment')
    c.setFillColorRGB(0.08, 0.18, 0.22)
    c.setFont('Helvetica-Bold', 23)
    c.drawString(42, 661, f"Wellness score: {p['wellness']}/100")
    c.setFont('Helvetica', 12)
    lines = [
        f"Persona: {p['persona']}",
        f'Screen time: {h.screen_hours:.1f} h/day',
        f'Sleep: {h.sleep_hours:.1f} h/day',
        f"Above {p['percentile']:.0f}% of dataset students' phone usage",
        f"Estimated productivity: {p['predicted_focus']:.0f}/100 | Low productivity risk: {p['risk']:.0%}",
        '',
        f"Predicted class: {p['category']}",
        'Personal habit suggestions:'
    ] + p['tips'] + ['', 'Wellness is a heuristic; missing optional inputs are omitted.', 'Source: uploaded student productivity CSV; provenance unverified.', 'Model version: ' + p['version']]
    y = 625
    for line in lines:
        c.drawString(42, y, line)
        y -= 25
    c.save()
    return Response(
        buf.getvalue(),
        media_type='application/pdf',
        headers={'Content-Disposition': 'attachment; filename=focusiq-report.pdf'}
    )

