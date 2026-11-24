# Comparison with published next-word prediction work

No Telugu next-word prediction paper was found, so these are the closest published results. Accuracy from different papers is not directly comparable: language, data, vocabulary and test method all differ.

| Model | Language | Reported accuracy | Test method | Source checked |
|---|---|---:|---|---|
| Bi-LSTM on n-grams | Bangla | 99% (99.74% elsewhere in the paper) | No split stated | Full text |
| LSTM | Assamese | 88.20% | Unknown | Secondhand only |
| Bi-LSTM | Hindi | 81.07% | Described as validation | Abstract only |
| BLSTM-GRU | Amharic | 78.6% | Test set of Bible text; top-1 not stated | Full text |
| GRU | Dzongkha | 74.03% | Validation split | Full text |
| LSTM | Hindi | 59.46% | Described as validation | Abstract only |
| **Combined model (this project)** | Telugu | **21.15%** top-1, 32.97% top-3, 38.24% top-5 | 200,092 de-duplicated test examples | `outputs/` |
| Hybrid v2 (this project) | Telugu | 20.84% top-1, 37.46% top-5 | 200,092 de-duplicated test examples | `outputs/` |
| Bi-LSTM v2 (this project) | Telugu | 19.42% top-1, 30.47% top-3 | Same | `outputs/` |
| LSTM (this project) | Telugu | 18.03% top-1, 28.78% top-3 | Same | `outputs/` |
| Gboard server CIFG (LSTM variant) | English | 16.5% top-1, 27.1% top-3 | Separate test logs | Full text |
| Gboard n-gram | English | 13.0% top-1, 22.1% top-3 | Separate test logs | Full text |

Gboard is the closest benchmark because it reports the same top-1/top-3 metric on held-out data. It uses real user typing in English with a 10,000-word vocabulary, so it is evidence of a sensible result, not proof of a better model.

Marking the project's own models on training data shows how much the test method matters (100,000 examples per split, `outputs/telugu_causal_lm/train_vs_test_check.json`): Hybrid v2 scores 31.97% top-1 / 63.70% top-5 on training data versus 20.75% / 37.42% on unseen test data. This shows the direction of the effect; it does not by itself explain figures near 99%.

## Sources

- Gboard: [Hard et al., Federated Learning for Mobile Keyboard Prediction](https://arxiv.org/abs/1811.03604)
- Bangla: [Islam et al., 2024](https://arxiv.org/abs/2405.01873)
- Amharic: [Endalie et al., PLoS ONE 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9387859/)
- Dzongkha: [Chhetri et al.](https://arxiv.org/pdf/2607.11515)
- Hindi: [Sharma et al., ICDSE 2019](https://www.researchgate.net/publication/338947141_Next_Word_Prediction_in_Hindi_Using_Deep_Learning_Techniques)
- Assamese: Barman & Boruah, Procedia Computer Science 2018, as cited in the Amharic paper
