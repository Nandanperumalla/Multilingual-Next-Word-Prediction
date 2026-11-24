const $ = id => document.getElementById(id);
const LANGUAGES = {
  telugu: {
    code: 'te', word: 'Telugu', model: 'ensemble', glyph: 'అ', title: 'Telugu Next-Word Prediction',
    subtitle: 'తెలుగులో తదుపరి పదాన్ని సూచించే వ్యవస్థ', textLabel: 'మీ తెలుగు వాక్యం', nextLabel: 'తదుపరి పదాలు',
    placeholder: 'ఇక్కడ తెలుగులో టైప్ చేయండి…',
    samples: ['ఆరోగ్యంగా ఉండటానికి', 'నేను తెలుగు', 'భారతదేశం ఒక', 'ఈ రోజు వాతావరణం'],
  },
  hindi: {
    code: 'hi', word: 'Hindi', model: null, glyph: 'अ', title: 'Hindi Next-Word Prediction',
    subtitle: 'हिंदी में अगला शब्द सुझाने वाली प्रणाली', textLabel: 'आपका हिंदी वाक्य', nextLabel: 'अगले शब्द',
    placeholder: 'यहाँ हिंदी में टाइप करें…',
    samples: ['मैं हिंदी', 'स्वस्थ रहने के लिए', 'भारत एक', 'आज का मौसम'],
  },
  english: {
    code: 'en', word: 'English', model: null, glyph: 'A', title: 'English Next-Word Prediction',
    subtitle: 'A system that suggests the next English word', textLabel: 'Your English sentence', nextLabel: 'Next words',
    placeholder: 'Type in English here…',
    samples: ['one of the', 'to stay healthy', 'the best way to', 'the weather today'],
  },
};
let language = 'telugu', names = {}, model = LANGUAGES.telugu.model, suggestions = [], latest = 0, timer;

function escapeHtml(value) {
  const node = document.createElement('div');
  node.textContent = value;
  return node.innerHTML;
}

async function post(path, body) {
  const response = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  if (!response.ok) throw new Error((await response.json()).detail || 'Request failed');
  return response.json();
}

function renderSuggestions(result) {
  suggestions = result.predictions;
  const box = $('suggestions'), status = $('status');
  box.replaceChildren();
  if (!result.context.length) {
    status.textContent = `Type a few ${LANGUAGES[language].word} words to see suggestions.`;
    return;
  }
  if (!suggestions.length) {
    status.textContent = `These words are new to the model. Try a more common ${LANGUAGES[language].word} word.`;
    return;
  }
  const top = suggestions[0].probability;
  suggestions.forEach(item => {
    const button = document.createElement('button');
    button.className = 'word';
    button.lang = LANGUAGES[language].code;
    button.style.setProperty('--w', `${Math.round(item.probability / top * 100)}%`);
    button.innerHTML = `<span>${escapeHtml(item.word)}</span><small>${(item.probability * 100).toFixed(1)}%</small>`;
    button.onclick = () => addWord(item.word);
    box.append(button);
  });
  status.textContent = result.unknown_words.length ? `New to the model: ${result.unknown_words.join(', ')}` : '';
}

async function refresh() {
  const request = ++latest;
  try {
    const result = await post('/api/predict', {text: $('text').value, model, top_k: 5});
    if (request === latest) renderSuggestions(result);
  } catch (error) {
    if (request === latest) $('status').textContent = error.message;
  }
}

function addWord(word) {
  const box = $('text');
  box.value = `${box.value.trimEnd()} ${word} `.trimStart();
  box.focus();
  refresh();
}

function setText(value) {
  $('text').value = `${value} `;
  $('text').focus();
  refresh();
}

async function loadModels() {
  const list = await (await fetch(`/api/models?language=${language}`)).json();
  names = Object.fromEntries(list.map(item => [item.key, item.name]));
  if (!model) model = (list.find(item => item.default) || {}).key;
  $('models').replaceChildren(...list.map(item => {
    const button = document.createElement('button');
    button.className = 'model';
    button.disabled = !item.available;
    button.setAttribute('aria-pressed', item.key === model);
    button.innerHTML = `${escapeHtml(item.name)}<span>${escapeHtml(item.note)}</span>`;
    button.onclick = () => {
      model = item.key;
      document.querySelectorAll('.model').forEach(other => other.setAttribute('aria-pressed', other === button));
      refresh();
    };
    return button;
  }));
  $('gen-run').dataset.ready = '1';
  return list;
}

async function runCompare() {
  const text = $('compare-text').value, asked = language;
  try {
    const all = await post('/api/compare', {text, language});
    if (asked !== language) return;
    $('compare-grid').replaceChildren(...Object.entries(all).map(([key, result]) => {
      const column = document.createElement('div');
      const rows = result.predictions.map(item => `<li><span>${escapeHtml(item.word)}</span><small>${(item.probability * 100).toFixed(1)}%</small></li>`).join('');
      column.innerHTML = `<h3 lang="en">${escapeHtml(names[key])}</h3><ol>${rows || '<li>No suggestions</li>'}</ol>`;
      return column;
    }));
  } catch (error) {
    $('compare-grid').textContent = error.message;
  }
}

async function runGenerate() {
  const button = $('gen-run');
  button.disabled = true;
  try {
    const result = await post('/api/generate', {text: $('gen-text').value, model, words: Number($('gen-words').value), creative: $('gen-creative').checked});
    const seed = result.text.slice(0, result.text.length - result.added.join(' ').length).trimEnd();
    $('gen-out').innerHTML = result.added.length ? `${escapeHtml(seed)} <b>${escapeHtml(result.added.join(' '))}</b>` : `Try a phrase with common ${LANGUAGES[language].word} words.`;
  } catch (error) {
    $('gen-out').textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

function table(id, columns, rows) {
  const head = columns.map(c => `<th class="${c.n ? 'n' : ''}">${c.label}</th>`).join('');
  const body = rows.map(row => `<tr class="${row.mine ? 'mine' : ''}">${columns.map(c => `<td class="${c.n ? 'n' : ''}">${c.cell(row)}</td>`).join('')}</tr>`).join('');
  $(id).innerHTML = `<thead><tr>${head}</tr></thead><tbody>${body}</tbody>`;
}

async function loadResults() {
  const data = await (await fetch('results.json')).json();
  const max = (key, rows = data.models) => rows.reduce((a, b) => (b[key] > a[key] ? b : a));
  const best1 = max('top1'), best5 = max('top5');
  const perplexity = data.models.filter(m => m.perplexity).reduce((a, b) => (b.perplexity < a.perplexity ? b : a));
  $('stats').innerHTML = [
    ['Best first-guess accuracy (Top-1)', `${best1.top1.toFixed(2)}%`, best1.name],
    ['Best top-five accuracy (Top-5)', `${best5.top5.toFixed(2)}%`, best5.name],
    ['Lowest perplexity', perplexity.perplexity.toFixed(2), perplexity.name],
  ].map(([label, value, name]) => `<div class="stat"><p>${label}</p><strong>${value}</strong><p>${name}</p></div>`).join('');
  const scale = 100 / best5.top5 * 0.85;
  $('chart').innerHTML = '<div class="legend"><span><i class="top1"></i>Top-1</span><span><i class="top5"></i>Top-5</span></div>' + data.models.map(m => `
    <div class="bar-row"><span>${m.name}</span><div class="bars">
      <div class="bar"><i class="top1" style="width:${m.top1 * scale}%"></i>${m.top1.toFixed(2)}%</div>
      <div class="bar"><i class="top5" style="width:${m.top5 * scale}%"></i>${m.top5.toFixed(2)}%</div>
    </div></div>`).join('');
  table('benchmark', [
    {label: 'Model', cell: r => r.name}, {label: 'Language', cell: r => r.language},
    {label: 'Top-1', n: 1, cell: r => `${r.top1}%`}, {label: 'Top-3', n: 1, cell: r => `${r.top3}%`},
  ], data.benchmark);
  table('pretrained', [
    {label: 'Model', cell: r => r.name}, {label: 'Top-1', n: 1, cell: r => `${r.top1}%`}, {label: 'Top-5', n: 1, cell: r => `${r.top5}%`},
  ], data.pretrained);
  table('papers', [
    {label: 'Paper', cell: r => `<a href="${r.url}" target="_blank" rel="noopener">${escapeHtml(r.source)}</a>`},
    {label: 'Model', cell: r => escapeHtml(r.name)}, {label: 'Language', cell: r => r.language},
    {label: 'Reported', n: 1, cell: r => r.score}, {label: 'Test method', cell: r => escapeHtml(r.test)},
  ], data.papers);
  ['telugu', 'hindi', 'english'].forEach(loadFamilyResults);
}

async function loadFamilyResults(name) {
  const response = await fetch(`/api/results/${name}`);
  if (!response.ok) return;
  const data = await response.json();
  const percent = value => (value == null ? '–' : `${(value * 100).toFixed(2)}%`);
  const number = value => (value == null ? '–' : value.toFixed(2));
  const size = value => (value == null ? '–' : `${(value / 1e6).toFixed(2)}M`);
  const word = LANGUAGES[name].word, best = data.models[0];
  $(`${name}-note`).textContent = `All rows use the same ${best.examples.toLocaleString('en-US')} unseen ${word} test examples. Best by validation Top-1: ${best.name}. One training run for each model, and the test set was reused for reporting${name === 'hindi' ? '; the text is machine-translated' : ''}${name === 'telugu' ? '. The original four Telugu models are unchanged; GRU, Transformer and both new combinations are new' : ''}. Compare languages as a rough guide only.`;
  table(name, [
    {label: 'Model', cell: r => r.name},
    {label: 'Parameters', n: 1, cell: r => size(r.parameters) + (['ensemble', 'hybrid'].includes(r.key) ? ' + counts' : '')},
    {label: 'Top-1', n: 1, cell: r => percent(r.top1_accuracy)}, {label: 'Top-3', n: 1, cell: r => percent(r.top3_accuracy)},
    {label: 'Top-5', n: 1, cell: r => percent(r.top5_accuracy)}, {label: 'Perplexity', n: 1, cell: r => number(r.vocabulary_perplexity)},
    {label: 'Unknown targets', n: 1, cell: r => percent(r.target_oov_rate)},
    {label: 'Top-1 without loops', n: 1, cell: r => percent(r.without_loops && r.without_loops.top1_accuracy)},
  ], data.models.map(row => ({...row, mine: row.key === data.best})));
  $(`${name}-sheet`).hidden = false;
  if (!data.sample.length) return;
  table(`${name}-sample`, [
    {label: 'Model', cell: r => escapeHtml(r.name)}, {label: 'Top-1', n: 1, cell: r => percent(r.top1_accuracy)},
    {label: 'Top-3', n: 1, cell: r => percent(r.top3_accuracy)}, {label: 'Top-5', n: 1, cell: r => percent(r.top5_accuracy)},
  ], data.sample);
  $(`${name}-sample-sheet`).hidden = false;
}

function showTab(name) {
  document.querySelectorAll('nav button').forEach(button => button.setAttribute('aria-selected', button.dataset.tab === name));
  document.querySelectorAll('.panel').forEach(panel => { panel.hidden = panel.id !== name; });
}

async function setLanguage(next) {
  const settings = LANGUAGES[next];
  language = next;
  model = settings.model;
  document.body.dataset.language = next;
  document.querySelectorAll('[data-language]').forEach(button => button.setAttribute('aria-pressed', button.dataset.language === next));
  $('glyph').textContent = settings.glyph;
  $('brand-title').textContent = document.title = settings.title;
  $('brand-sub').textContent = settings.subtitle;
  $('text-label').textContent = settings.textLabel;
  $('next-label').textContent = settings.nextLabel;
  $('text').placeholder = settings.placeholder;
  ['text', 'compare-text', 'gen-text', 'samples', 'compare-grid', 'gen-out'].forEach(id => { $(id).lang = settings.code; });
  $('gen-out').textContent = '';
  $('samples').replaceChildren(...settings.samples.map(sample => {
    const button = document.createElement('button');
    button.textContent = sample;
    button.onclick = () => setText(sample);
    return button;
  }));
  $('compare-text').value = $('gen-text').value = settings.samples[0];
  const list = await loadModels();
  if (!list.some(item => item.key === model && item.available)) {
    model = (list.find(item => item.available) || {key: model}).key;
    document.querySelectorAll('.model').forEach((button, index) => button.setAttribute('aria-pressed', list[index].key === model));
  }
  runCompare();
  setText(settings.samples[0]);
}

async function start() {
  document.querySelectorAll('nav button').forEach(button => { button.onclick = () => showTab(button.dataset.tab); });
  $('text').addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(refresh, 180); });
  $('text').addEventListener('keydown', event => {
    if (event.key === 'Tab' && !event.shiftKey && suggestions.length) {
      event.preventDefault();
      addWord(suggestions[0].word);
    }
  });
  $('gen-words').oninput = () => { $('gen-count').textContent = $('gen-words').value; };
  $('gen-run').onclick = runGenerate;
  let compareTimer;
  $('compare-text').addEventListener('input', () => { clearTimeout(compareTimer); compareTimer = setTimeout(runCompare, 250); });
  document.querySelectorAll('[data-language]').forEach(button => { button.onclick = () => setLanguage(button.dataset.language); });
  for (const name of ['hindi', 'english']) {
    const list = await (await fetch(`/api/models?language=${name}`)).json();
    document.querySelector(`[data-language=${name}]`).disabled = !list.length;
  }
  await setLanguage('telugu');
  loadResults();
}

start();
