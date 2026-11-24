"""Check the website API against the real saved checkpoints."""
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)
assert 'Telugu Next-Word Prediction' in client.get('/').text
assert all(model['available'] for model in client.get('/api/models').json())
for key in ('ensemble', 'hybrid', 'bilstm_v2', 'lstm'):
    result = client.post('/api/predict', json=dict(text='ఆరోగ్యంగా ఉండటానికి', model=key)).json()
    assert len(result['predictions']) == 5, key
assert client.post('/api/predict', json=dict(text='')).json()['predictions'] == []
assert client.post('/api/predict', json=dict(text='hello 123 !!!')).json()['predictions'] == []
assert client.post('/api/predict', json=dict(text='ఔఔఔఔఔఔఔఔఔఔ')).json()['predictions'] == []
assert client.post('/api/predict', json=dict(text='నేను', model='nope')).status_code == 404
assert set(client.post('/api/compare', json=dict(text='నేను తెలుగు')).json()) == {'ensemble', 'hybrid', 'bilstm_v2', 'lstm'}
generated = client.post('/api/generate', json=dict(text='ఆరోగ్యంగా ఉండటానికి', words=5, creative=False)).json()
assert len(generated['added']) == 5 and generated['text'].startswith('ఆరోగ్యంగా ఉండటానికి')
assert client.get('/results.json').status_code == 200
print('Website checks passed')
