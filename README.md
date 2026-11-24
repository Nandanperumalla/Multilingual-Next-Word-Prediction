# Multilingual Next-Word Prediction: Telugu phase

PyTorch next-word prediction with a custom website (HTML, CSS and JavaScript on a FastAPI server). Telugu is the main verified language; Hindi and English were added with six model families each and a Combined model, and Telugu gained a GRU, a Transformer and a new Combined model. Legacy multilingual and federated scripts are retained for reference, not used in the current demo.

## Latest verified results

All rows use the same 200,092 held-out test examples. Higher accuracy is better; lower perplexity is better.

| Model / experiment | Top-1 | Top-5 | Perplexity |
|---|---:|---:|---:|
| LSTM, 5 epochs | 18.03% | 33.57% | 199.02 |
| Bi-LSTM, 5 epochs | 18.82% | 35.00% | 178.60 |
| Word CNN | 17.67% | 33.53% | 193.76 |
| Fine-tuned word CNN, 10 epochs | 18.68% | 34.70% | 183.54 |
| Gated word-character CNN | 18.55% | 34.68% | 167.13 |
| Fine-tuned Bi-LSTM v2 | 19.42% | 35.56% | 176.04 |
| Regularized Bi-LSTM v3 | 19.35% | 35.72% | 169.48 |
| Hybrid v1 | 20.62% | **37.63%** | Not measured |
| Longer-context hybrid v2 | 20.84% | 37.46% | Not measured |
| Combined model (LSTM, two Bi-LSTMs and counts) | **21.15%** | **38.24%** | Not measured |

These are measured experimental results, not 80–90% accuracy. Exact metrics and validation-selection records are in `outputs/`.

### Pretrained multilingual LLM comparison

A pretrained 560M-parameter multilingual causal LM ([bigscience/bloom-560m](https://huggingface.co/bigscience/bloom-560m)) was scored with and without LoRA fine-tuning. Pretrained scoring is slow (about 5.5 s per example on an Apple M5), so every row below uses the same fixed 500-example sample of the held-out test set (seed 42). This sample is slightly easier than the full test set: Hybrid v2 scores 22.6% here versus 20.84% on all 200,092 examples.

| Model | Parameters | Top-1 | Top-5 |
|---|---:|---:|---:|
| Hybrid v2 | 6.6M + counts | **22.6%** | **38.4%** |
| Fine-tuned Bi-LSTM v2 | 6.6M | 21.4% | 37.4% |
| bloom-560m, LoRA fine-tuned | 560M | 8.6% | 18.6% |
| bloom-560m, no fine-tuning | 560M | 5.6% | 13.6% |

LoRA (r=16, 2.4M trainable parameters) ran for 2,500 steps on 20,000 training sentences, at learning rate 5e-5, in 30 minutes. Fine-tuning improves bloom-560m by 3 points top-1, but it remains far below the task-specific models. Likely reasons: BLOOM's tokenizer splits Telugu words into many sub-pieces; the pretrained model sometimes drifts into other Indic scripts; and this short fine-tune covers under 1% of the training examples the project models saw. bloom-560m ranks whole words by greedily completing its 30 most likely Telugu word-start tokens, so its scores approximate exact word probabilities. Out-of-vocabulary targets count as misses for every model. This is a single run on a 500-example sample, so differences of a point or two are not conclusive.

Reproduce (requires the rebuilt data in `data_rebuilt/`, `transformers` and `peft`):

```bash
python src/finetune_telugu_causal_lm.py eval            # no fine-tuning
python src/finetune_telugu_causal_lm.py train           # LoRA fine-tune
python src/finetune_telugu_causal_lm.py eval --adapter  # fine-tuned
python src/evaluate_telugu_slice.py                     # project models on the same sample
```

Records are in `outputs/telugu_causal_lm/`. Comparison with published papers is in [RESEARCH_COMPARISON.md](RESEARCH_COMPARISON.md).

## All three languages: model families

### Telugu: GRU, Transformer and new combinations (200,092 test examples)

The Telugu GRU and Transformer were trained as new files in `outputs/telugu_families/`, with the Telugu baseline settings (5 epochs). The Telugu rows above and every earlier Telugu checkpoint and website model are unchanged; the new Combined model only reads the earlier checkpoints. Rows are sorted by validation Top-1.

| Model | Parameters | Validation Top-1 | Top-1 | Top-3 | Top-5 | Perplexity |
|---|---:|---:|---:|---:|---:|---:|
| **Combined (all families)** | 16.72M + counts | 21.35% | **21.69%** | 33.66% | 39.12% | 118.14 |
| Hybrid (Bi-LSTM v3 + counts) | 6.62M + counts | 20.43% | 20.75% | 32.30% | 37.50% | 162.81 |
| Transformer (new) | 5.93M | 19.37% | 19.49% | 31.14% | 36.57% | 139.28 |
| Bi-LSTM v3 | 6.62M | 19.15% | 19.35% | 30.53% | 35.72% | 169.48 |
| Bi-LSTM v2 | 6.62M | 19.07% | 19.42% | 30.47% | 35.56% | 176.04 |
| Bi-LSTM | 6.62M | 18.60% | 18.82% | 29.79% | 35.00% | 178.60 |
| Gated CNN v2 | 4.16M | 18.57% | 18.68% | 29.65% | 34.70% | 183.54 |
| Word-character CNN | 4.18M | 18.36% | 18.55% | 29.59% | 34.68% | 167.13 |
| LSTM | 3.96M | 17.85% | 18.03% | 28.78% | 33.57% | 199.02 |
| GRU (new) | 3.93M | 17.79% | 17.94% | 28.62% | 33.54% | 203.14 |
| Gated CNN | 4.16M | 17.55% | 17.67% | 28.42% | 33.53% | 193.76 |

Target OOV rate: 7.21% for every row. Combined weights (chosen on validation): Transformer 0.3, Bi-LSTM v3 0.2, Gated CNN v2 0.2, counts 0.3. It beats the earlier Telugu Combined model (21.15%) by 0.54 points Top-1, with one training run each. The Transformer pilots (same grid as below) chose 4 layers and learning rate 0.001. On the website, Telugu still opens with the original Combined model; "Combined (all families)", Transformer and GRU were added after the original four.

### Hindi and English

For Hindi and English, six different model families were trained and compared, plus a Combined model and a Hybrid. Both languages use the same method as Telugu: a 90/5/5 split by SHA256(seed 42, normalized instruction), exact sentence overlap removed, a 20,000-word vocabulary fitted on training only, and an eight-word context. Every model in a language is scored on the same test examples. Telugu files, checkpoints and numbers are unchanged.

Layer choices, epochs and Combined weights were chosen on validation data only; the test split was scored once per final model. The best model is the one with the highest validation Top-1, and it is in bold. Rows are sorted by validation Top-1.

### Hindi (296,193 test examples)

| Model | Parameters | Validation Top-1 | Top-1 | Top-3 | Top-5 | Perplexity | Top-1 without repeats |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Combined** | 12.55M + counts | 34.75% | **34.15%** | 50.52% | 57.62% | 37.84 | 33.50% |
| Hybrid | 6.62M + counts | 33.65% | 32.87% | 48.71% | 55.69% | 47.76 | 32.35% |
| Transformer | 5.93M | 32.65% | 32.40% | 48.66% | 55.96% | 41.81 | 31.61% |
| Bi-LSTM | 6.62M | 31.15% | 30.57% | 46.08% | 53.18% | 53.28 | 29.97% |
| Word-character CNN | 4.18M | 30.86% | 30.26% | 45.93% | 53.12% | 53.46 | 29.59% |
| Gated CNN | 4.16M | 30.59% | 30.06% | 45.68% | 52.81% | 55.09 | 29.48% |
| GRU | 3.93M | 29.82% | 29.00% | 44.37% | 51.40% | 62.06 | 28.52% |
| LSTM | 3.96M | 29.67% | 29.21% | 44.44% | 51.42% | 61.49 | 28.73% |

Target OOV rate: 1.04% for every row. "Without repeats" drops the 11,333 test examples (3.83%) that sit in translation repetition loops. Combined weights: Transformer 0.4, Bi-LSTM 0.3, counts 0.3; the word-character CNN was the best CNN but got weight 0 on validation. Hybrid is the Bi-LSTM plus counts (counts weight 0.5).

### English (308,909 test examples)

| Model | Parameters | Validation Top-1 | Top-1 | Top-3 | Top-5 | Perplexity | Top-1 without repeats |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Combined** | 12.55M + counts | 30.21% | **30.50%** | 45.66% | 51.92% | 57.50 | 30.48% |
| Hybrid | 6.62M + counts | 29.17% | 29.46% | 44.02% | 50.18% | 73.71 | 29.44% |
| Transformer | 5.93M | 27.53% | 27.73% | 42.58% | 48.87% | 68.33 | 27.72% |
| Bi-LSTM | 6.62M | 25.71% | 26.04% | 39.62% | 45.58% | 93.65 | 26.02% |
| Gated CNN | 4.16M | 25.37% | 25.63% | 39.11% | 45.07% | 95.53 | 25.62% |
| Word-character CNN | 4.18M | 25.35% | 25.67% | 39.22% | 45.18% | 93.55 | 25.65% |
| GRU | 3.93M | 24.45% | 24.67% | 37.70% | 43.40% | 112.34 | 24.66% |
| LSTM | 3.96M | 24.39% | 24.56% | 37.42% | 43.13% | 115.39 | 24.55% |

Target OOV rate: 1.45% for every row. "Without repeats" drops 370 test examples (0.12%). Combined weights: Transformer 0.4, Bi-LSTM 0.2, counts 0.4; the gated CNN was the best CNN but got weight 0 on validation. Hybrid is the Bi-LSTM plus counts (counts weight 0.5).

**Best model:** Combined in both languages, by validation Top-1. It is also best on test Top-1, Top-3, Top-5 and perplexity. The Transformer is the best single model in both languages.

**Settings.** LSTM, Bi-LSTM and GRU: embedding 64, hidden 128, Adam 0.001, batch 512, seed 42, 3 epochs (CPU). Gated CNN and gated word-character CNN: Telugu settings, 5 epochs (Apple GPU; a timing test gave the same validation result as CPU). Transformer: embedding 128, 4 layers, 4 heads, causal attention over the 8-word prefix, learned positions, pre-norm, dropout 0.1, AdamW 0.001, 5 epochs, prediction from the last position. Four 1-epoch pilots per language (2 or 4 layers, learning rate 0.001 or 0.0003, 1,000,000 training examples, full validation) chose 4 layers and 0.001 in both languages. Its 5.93M parameters are 0.90× the Bi-LSTM. Counts: contexts of one to four words seen at least 3 times. Combined: the best model of each family (recurrent, CNN, Transformer) by validation Top-1, plus counts, weights on a 0.1 grid screened on 60,000 validation examples, top 10 rechecked on full validation.

**Data.** Hindi: [iamshnoo/alpaca-cleaned-hindi](https://huggingface.co/datasets/iamshnoo/alpaca-cleaned-hindi), 51,760 rows machine-translated from yahma/alpaca-cleaned with NLLB-1.3B; no license on its card, so check upstream terms. 5,203,483 training, 293,336 validation examples. English: [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) (CC BY 4.0), the source the Hindi set was translated from, 51,760 rows; 5,165,991 training, 285,746 validation examples. Hindi has fewer rows than Telugu (72,524 records from two sources) but more examples, because its sentences are longer.

**Pretrained models (500-example test sample, seed 42).** Same method as Telugu: LoRA r=16, 2,500 steps on 20,000 training sentences, whole words ranked by greedy completion of the 30 most likely word-start tokens, out-of-vocabulary targets count as misses. The Hindi BLOOM fine-tune diverged at learning rate 5e-5 (stopped by the loss guard), so it was rerun at 2e-5; the others used 5e-5. GPT-2 was run for English only, since its tokenizer is English-centric.

| Model | Parameters | Hindi Top-1 | Hindi Top-5 | English Top-1 | English Top-5 |
|---|---:|---:|---:|---:|---:|
| Combined | 12.55M + counts | **36.0%** | **60.2%** | 31.6% | 49.4% |
| Transformer | 5.93M | 33.2% | 56.6% | 27.6% | 47.6% |
| bloom-560m, LoRA fine-tuned | 560M | 30.4% | 51.8% | **32.0%** | **52.0%** |
| bloom-560m, no fine-tuning | 560M | 25.4% | 45.4% | 29.2% | 49.4% |
| GPT-2, LoRA fine-tuned | 124M | – | – | 31.6% | 50.6% |
| GPT-2, no fine-tuning | 124M | – | – | 28.8% | 46.6% |

In English, fine-tuned bloom-560m edges the Combined model by 0.4 points on this sample (2 of 500 examples), which is within noise, at 45× the size. In Hindi the project models stay clearly ahead.

**Limits.** One seed per model; the same test set was reused for reporting. The recurrent models ran 3 epochs and the CNNs and Transformer 5 (the recurrent models were still improving), so gaps between families are not a fair measure of the architectures alone. Hindi is machine-translated. Data, vocabulary coverage (OOV 7.21% Telugu, 1.04% Hindi, 1.45% English) and sentence length all differ, so compare languages only as indicative, never as a ranking.

Reproduce (Hindi source: `data/train-00000-of-00001-9c3cdcdad5ba72d6.parquet` from that dataset, saved as `data_hindi/source/alpaca_hindi.parquet`; English source: `alpaca_data_cleaned.json` saved in `data_english/source/`):

```bash
python src/prepare_hindi_data.py --data-dir data_hindi
python src/prepare_english_data.py
python src/train_hindi_comparison.py --data-dir data_hindi --output-dir outputs/hindi_comparison --epochs 3
python src/evaluate_hindi_clean.py
./run_hindi_families.sh      # Hindi GRU, CNNs and Transformer pilots
./run_english_families.sh    # English LSTM, Bi-LSTM, GRU, CNNs and Transformer pilots
python src/train_families_hindi.py --family transformer --language hindi --layers 4 --lr 0.001 --epochs 5 --device mps
python src/train_families_hindi.py --family transformer --language english --layers 4 --lr 0.001 --epochs 5 --device mps
python src/train_ensemble_hindi.py --language hindi
python src/train_ensemble_hindi.py --language english
python src/summarize_families_hindi.py --language hindi
python src/summarize_families_hindi.py --language english
./run_pretrained.sh && ./run_pretrained_rest.sh   # bloom-560m and GPT-2 on the 500-example sample
./run_telugu_families.sh     # Telugu GRU, Transformer pilots and final, new Combined, summary
```

Records: `outputs/hindi_comparison/` (Hindi LSTM and Bi-LSTM), `outputs/hindi_families/` and `outputs/english_families/` (every other model, `ensemble_selection.json`, and `comparison.json` with all rows above).

## Run

macOS/Linux: `./run_app.sh` (creates `.venv` on first run). Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/server.py
```

The website has Write (suggestions as you type), Compare, Generate and Results tabs. A Telugu / Hindi / English switch sits at the top. Telugu opens by default, with the combined model (ensemble) selected; Hybrid v2, Bi-LSTM v2 and the LSTM baseline are selectable, followed by the new Combined (all families), Transformer and GRU. Hindi and English list only the models actually trained, best on validation first, with Combined as the default; their Results tab is read from `comparison.json`. Best checkpoints are included. Open http://127.0.0.1:8501. No training dataset is required for this demo.

## Data and reproduction

Native Telugu instruction/input/output fields come from:

- [Telugu-LLM-Labs Alpaca](https://huggingface.co/datasets/Telugu-LLM-Labs/telugu_alpaca_yahma_cleaned_filtered_romanized): 28,910 records; provenance records CC BY 4.0.
- [Telugu-LLM-Labs GPTeacher](https://huggingface.co/datasets/Telugu-LLM-Labs/telugu_teknium_GPTeacher_general_instruct_filtered_romanized): 43,614 records; provenance records MIT.

Training: 3,483,073 examples. Validation: 198,928. Test: 200,092. Vocabulary: 20,000 tokens, fitted on training only. Context: eight words. Splits group normalized instructions and remove exact sentence overlap. See `data_telugu_llm/provenance.json` for source hashes and attribution. Respect upstream terms.

Downloaded CSVs and generated sequence shards are excluded from Git. Obtain the source CSVs as `alpaca.csv` and `gpteacher.csv` in a fresh directory `data_rebuilt/source/`, matching the checksums in `src/prepare_telugu_llm_data.py`.

```powershell
python src/prepare_telugu_llm_data.py --data-dir data_rebuilt
python src/train_telugu_comparison.py --data-dir data_rebuilt --output-dir outputs/reproduced --epochs 5
```

Improvement scripts expect prepared data at `data_telugu_llm` and verify parent-checkpoint hashes. Rebuild in a separate checkout: the included provenance file intentionally prevents overwriting an existing prepared directory. Historical machine paths in experiment records describe the original run, not installation requirements.

## Repository guide

- [Full architecture and flowcharts](docs/ARCHITECTURE.md): all models, formulas, training and prediction flow.
- `src/`: preparation, training, evaluation, model and UI source.
- `screenshots/`: the website in Telugu, Hindi and English (Write tab), and the Hindi and English Results tabs.
- `outputs/`: best checkpoints, hybrid count models, histories, configurations and metrics. Resume-checkpoint duplicates and runtime logs are excluded.
- `presentations/`: slide versions with formulas and flowcharts. They predate the newest accuracy experiments; use the results table above for current figures.
- [Historical README](docs/README_HISTORICAL.md): archived earlier description, including legacy experiments.
- `FINAL_MODEL_RESULTS.md`: original four-model benchmark. Later results are in the accuracy/hybrid output folders.

Baseline training: Adam, learning rate 0.001, batch 512, seed 42, five epochs. Bi-LSTM v2 fine-tuned at 0.0003 through epoch 10; validation selected epoch 9. V3 used AdamW, learning rate 0.0001, weight decay 0.01 and training-only dropout 0.15 through epoch 14; validation selected epoch 13. Hybrid parameters are selected by validation top-1.

## Limits

The vocabulary is fixed; unseen target words count as misses. Several plausible next words may exist, while evaluation checks one recorded target. LSTM and Bi-LSTM were repeated across five seeds (Top-1 17.94 ± 0.07 and 18.72 ± 0.10; see `outputs/telugu_seeds/summary.json`); other results use one seed, and all use repeated reporting on the same test set; a fresh untouched evaluation set is needed for independent confirmation. Exact duplicate checks do not exclude semantic duplication. Novelty, gate-specific benefit, and 80–90% accuracy are not established. Training can take hours on CPU and cannot continue through shutdown.

