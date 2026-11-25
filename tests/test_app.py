import hashlib
import sqlite3
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
from data_engine import SOURCE, load


@pytest.fixture
def client(tmp_path, monkeypatch):
    dst = tmp_path / 'test.db'
    with sqlite3.connect(app.DB) as src, sqlite3.connect(dst) as out:
        src.backup(out)
    monkeypatch.setattr(app, 'DB', dst)
    return TestClient(app.app)


def test_data(client):
    d, a = load()
    assert len(d) == d.user_id.nunique() == 20000
    assert a['missing_source_cells'] == 0
    assert a['category_sum_exceeds_phone_hours'] == 15458
    assert len(client.get('/api/data').json()['rows']) == 20000
    assert all(r['age_group'] == '17–20' for r in client.get('/api/data', params={'age': '17–20'}).json()['rows'])


def test_models(client):
    m = client.get('/api/models').json()
    assert m['source_sha256'] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert m['train_users'] + m['test_users'] == 20000
    assert m['class_names'] == ['Low', 'Normal', 'High']
    assert sum(m['class_counts']) == 20000
    assert len(m['models']) == 11
    for r in m['models']:
        assert len(r['confusion']) == 3 and all(len(row) == 3 for row in r['confusion'])
        assert sum(map(sum, r['confusion'])) == m['test_users']
        assert len(r['roc']) == 3
        assert all(0 <= r[k] <= 1 for k in ['accuracy','precision','recall','f1','roc_auc'])
        assert abs(r['roc_auc'] - sum(c['auc'] for c in r['roc'])/3) < 1e-9
        for c in r['roc']:
            assert c['fpr'][0] == c['tpr'][0] == 0
            assert c['fpr'][-1] == c['tpr'][-1] == 1


def test_clusters(client):
    d = client.get('/api/clusters').json()
    for method in ('k', 'h'):
        assert len(d['profiles'][method]) == 4
        assert sum(r['count'] for r in d['profiles'][method]) == 2000
        assert abs(sum(r['share'] for r in d['profiles'][method])-1) < 1e-9
        assert all(r['name'] for r in d['profiles'][method])
        assert -1 <= d['silhouette'][method] <= 1
    assert d['profiles']['k'] != d['profiles']['h']
    assert 'points' not in d and 'dendrogram' not in d


def test_research_and_probability(client):
    assert len(client.get('/api/research').json()['tests']) == 4
    for kind in ['Normal', 'Binomial', 'Poisson', 'Uniform']:
        assert len(client.get('/api/probability', params={'kind': kind}).json()['values']) == 100


def test_profile_monitor_report(client):
    v = client.post('/api/profile', json={}).json()
    assert 0 <= v['wellness'] <= 100
    assert v['category'] in ['Low','Normal','High']
    assert abs(sum(v['probabilities'].values())-1) < 1e-6
    assert v['risk'] == v['probabilities']['Low']
    assert client.post('/api/profile', json={'screen_hours': 1, 'night_minutes': 120}).status_code == 422
    assert client.post('/api/report', json={}).content.startswith(b'%PDF')
    assert client.patch('/api/predictions/' + str(v['prediction_id']), json={'actual': 60}).status_code == 200
    m = client.get('/api/monitor').json()
    assert m['labeled'] >= 1 and m['daily_errors']
    assert m['predictions'][0]['category'] == v['category']
    assert client.post('/api/explain', json={}).status_code == 200
    assert client.post('/api/scenario', json={}).status_code == 200


def test_dashboard(client):
    html = client.get('/').text
    for id in ['simScore','globalImportance','overviewHist','overviewScatter','overviewBox','modelChoice','roc','clusterPopulation','clusterHabits','errorTrend']:
        assert f'id="{id}"' in html
    for id in ['dayDonut','forest','detox','logForm','pca','pilotForm']:
        assert f'id="{id}"' not in html
    assert client.get('/api/logs').status_code == 404
    assert client.post('/api/pilot', json={}).status_code == 404
    assert client.get('/static/dashboard.js').status_code == 200
    d = client.get('/api/importance').json()
    assert d['sample_size'] == 24 and len(d['features']) == 10
    assert all(r['mean_abs_shap'] >= 0 for r in d['features'])
