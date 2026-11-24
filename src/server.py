"""Website server for Telugu, Hindi and English next-word prediction: JSON API plus the static pages in web/."""
import argparse
import json
import random
import threading
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
from telugu_ensemble_predictor import FILES as ENSEMBLE_FILES, TeluguEnsemblePredictor
from telugu_hybrid_predictor import TeluguHybridPredictor
from telugu_neural_predictor import TeluguNeuralPredictor
from text_utils import project_root

ROOT = project_root()
OUTPUTS = ROOT / 'outputs'
WEB = ROOT / 'web'
BILSTM_V2 = OUTPUTS / 'telugu_accuracy_v2' / 'bilstm_best.pt'
LSTM = OUTPUTS / 'telugu_comparison' / 'lstm_best.pt'
HYBRID = OUTPUTS / 'telugu_hybrid_v2'
FAMILIES = dict(transformer=('Transformer', 'Attention'), bilstm=('Bi-LSTM', 'Recurrent'),
                wordchar_cnn=('Word-character CNN', 'Reads spellings'), cnn=('Gated CNN', 'Convolution'),
                gru=('GRU', 'Recurrent'), lstm=('LSTM', 'Baseline'))
MODELS = {
    'ensemble': dict(name='Combined', note='Best accuracy', files=ENSEMBLE_FILES),
    'hybrid': dict(name='Hybrid v2', note='Bi-LSTM + counts', files=[HYBRID / 'counts.pkl', HYBRID / 'selection.json', BILSTM_V2]),
    'bilstm_v2': dict(name='Bi-LSTM v2', note='Neural only', files=[BILSTM_V2]),
    'lstm': dict(name='LSTM', note='Baseline', files=[LSTM]),
}
LOADERS = {
    'ensemble': TeluguEnsemblePredictor,
    'hybrid': TeluguHybridPredictor,
    'bilstm_v2': lambda: TeluguNeuralPredictor(BILSTM_V2),
    'lstm': lambda: TeluguNeuralPredictor(LSTM),
}


def family_file(language, architecture):
    path = OUTPUTS / f'{language}_families' / f'{architecture}_best.pt'
    return path if path.exists() else OUTPUTS / f'{language}_comparison' / f'{architecture}_best.pt'


def family_loader(language, architecture):
    def load():
        from family_predictor_hindi import FamilyPredictor
        return FamilyPredictor(language, architecture)
    return load


def combined_loader(language):
    def load():
        from family_predictor_hindi import CombinedPredictor
        return CombinedPredictor(language)
    return load


for language in ('hindi', 'english'):
    folder = OUTPUTS / f'{language}_families'
    MODELS[f'{language}_ensemble'] = dict(name='Combined', note='All families + counts', language=language,
                                          files=[folder / 'ensemble_selection.json', folder / 'counts.pkl'])
    LOADERS[f'{language}_ensemble'] = combined_loader(language)
    for architecture, (name, note) in FAMILIES.items():
        MODELS[f'{language}_{architecture}'] = dict(name=name, note=note, language=language, files=[family_file(language, architecture)])
        LOADERS[f'{language}_{architecture}'] = family_loader(language, architecture)
# New Telugu models go after the original four, which keep their order and default.
TELUGU_FAMILIES = OUTPUTS / 'telugu_families'
MODELS['telugu_families_ensemble'] = dict(name='Combined (all families)', note='Every family + counts',
                                          files=[TELUGU_FAMILIES / 'ensemble_selection.json', TELUGU_FAMILIES / 'counts.pkl'])
LOADERS['telugu_families_ensemble'] = combined_loader('telugu')
for architecture in ('transformer', 'gru'):
    MODELS[f'telugu_{architecture}'] = dict(name=FAMILIES[architecture][0], note=FAMILIES[architecture][1],
                                            files=[TELUGU_FAMILIES / f'{architecture}_best.pt'])
    LOADERS[f'telugu_{architecture}'] = family_loader('telugu', architecture)
loaded, load_lock = {}, threading.Lock()
app = FastAPI(title='Telugu Next-Word Prediction')


class PredictRequest(BaseModel):
    text: str = Field(max_length=2000)
    model: str = 'ensemble'
    top_k: int = Field(default=5, ge=1, le=10)


class CompareRequest(BaseModel):
    text: str = Field(max_length=2000)
    language: str = 'telugu'


class GenerateRequest(BaseModel):
    text: str = Field(max_length=2000)
    model: str = 'ensemble'
    words: int = Field(default=12, ge=1, le=30)
    creative: bool = True


def available(key):
    return all(path.exists() for path in MODELS[key]['files'])


def comparison(language):
    path = OUTPUTS / f'{language}_families' / 'comparison.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def in_language(language):
    """Telugu keeps its fixed order. Other languages list trained models only, best on validation first."""
    keys = [key for key, value in MODELS.items() if value.get('language', 'telugu') == language]
    if language == 'telugu':
        return keys
    keys = [key for key in keys if available(key)]
    order = {f"{language}_{row['key']}": row['validation_top1'] for row in (comparison(language) or dict(models=[]))['models']}
    return sorted(keys, key=lambda key: order.get(key, 0), reverse=True)


def predictor(key):
    if key not in MODELS:
        raise HTTPException(404, 'Unknown model')
    if not available(key):
        raise HTTPException(503, 'This model\'s files are missing')
    with load_lock:
        if key not in loaded:
            loaded[key] = LOADERS[key]()
        return loaded[key]


@app.get('/api/models')
def models(language: str = 'telugu'):
    rows = [dict(key=key, name=MODELS[key]['name'], note=MODELS[key]['note'], available=available(key)) for key in in_language(language)]
    if language != 'telugu' and rows:
        rows[0]['default'] = True
    return rows


@app.post('/api/predict')
def predict(request: PredictRequest):
    return predictor(request.model).predict(request.text, request.top_k)


@app.post('/api/compare')
def compare(request: CompareRequest):
    return {key: predictor(key).predict(request.text, 5) for key in in_language(request.language) if available(key)}


@app.post('/api/generate')
def generate(request: GenerateRequest):
    model, text, rng = predictor(request.model), request.text.strip(), random.Random()
    added = []
    for _ in range(request.words):
        top = model.predict(text, 5)['predictions']
        if not top:
            break
        pick = rng.choices(top, [item['probability'] for item in top])[0] if request.creative else top[0]
        text += ' ' + pick['word']
        added.append(pick['word'])
    return dict(text=text, added=added)


@app.get('/api/results/{language}')
def results(language: str):
    data = comparison(language)
    if data is None:
        raise HTTPException(503, 'Results are not ready')
    return data


@app.get('/')
def home():
    return FileResponse(WEB / 'index.html')


app.mount('/', StaticFiles(directory=WEB), name='web')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8501)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')
