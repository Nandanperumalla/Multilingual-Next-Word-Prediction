# Presentation Speaker Notes

Short notes per slide, followed by likely professor questions.

## Slide 1: Multilingual Next-Word Prediction

Good morning. This project compares next-word prediction models for Telugu, Hindi and English under one evaluation method. Telugu and Hindi are complete apart from the Hindi pretrained test. English models are still training.

## Slide 2: Abstract

Abstract length is 117 words. Stress three things: one method for all languages, measured results on unseen data, and clear caveats. Abbreviations: Top-k means the correct word appears among the first k suggestions.

## Slide 3: Introduction

Next-word prediction ranks likely next words from typed context. Google Gboard [1] is the best-known deployed example. Our models output a fixed vocabulary of 20,000 words. The figure is the real project website.

## Slide 4: Problem Statement

The problem has two parts: predicting the next word, and measuring that ability fairly. The OOV rate and the seen versus unseen gap are measured in this project. OOV means out of vocabulary. Hybrid v2 blends a Bi-LSTM with word counts.

## Slide 5: Research Motivation

Four motivations. Bloom-560m is a pretrained 560-million-parameter multilingual language model. The last card is a real research question: does a large pretrained model beat a small model trained for this task?

## Slide 6: Research Objectives

Five measurable objectives. Telugu has met all of them. Hindi has met all except the pretrained comparison. English data is prepared and its training is in progress.

## Slide 7: Literature Review: Keyboards, Telugu and Hindi

Gboard is the closest benchmark because it reports Top-1 and Top-3 on held-out data. CIFG means coupled input and forget gate, an LSTM variant. Two Telugu studies were located; only the first abstract could be read, so no scores are quoted for them. Figures for Sharma et al. come from the abstract.

## Slide 8: Literature Review: Related Languages and Methods

High scores appear here, but test methods differ, so they cannot be ranked against ours. The Islam et al. paper gives 99% in its abstract and 99.74% for 5-grams in its body. The Assamese figure was seen only through citing papers.

## Slide 9: Research Gap

Important correction: the project report says no Telugu paper was found, but later searching located two. So we do not claim to be first. The gap is transparent, large-scale evaluation across model families. This reflects our search and is not proof that no such study exists.

## Slide 10: Existing System

The left column shows the existing approaches and the right column shows their limits. N-gram models count word sequences. The bloom limitations are hypotheses from our own test and are discussed later.

## Slide 11: Proposed System

One pipeline handles all three languages, so differences between models are not caused by different data handling. The combined model adds model probabilities using fixed weights chosen on validation data. The website has Write, Compare, Generate and Results pages.

## Slide 12: Proposed System Architecture

Read left to right for training and bottom row for use. Asterisk: GRU and Transformer were trained for Hindi and English, not for Telugu. Telugu added Bi-LSTM fine-tuning instead. All boxes match the code in src and the records in outputs.

## Slide 13: Complete System Workflow

This slide shows the order of events, not the components. The upper lane runs once, offline. The lower lane runs each time the user types. Inference uses saved weights only; typed text does not update the models. PAD and UNK are the padding and unknown-word tokens.

## Slide 14: Research Methodology

Six stages. Top-k counts how often the true word is among the k highest-ranked words. PAD and UNK are removed from ranking, so an OOV target can never score. In cross-entropy, OOV targets map to UNK. Perplexity is the exponential of cross-entropy; lower is better. Weights were picked on validation Top-1: 110 candidates for Telugu and 287 for Hindi, screened on 60,000 validation examples.

## Slide 15: Proposed Model Architecture

Upper part: the Telugu combined model. Weights are 0.3 LSTM, 0.3 Bi-LSTM v2, 0.1 Bi-LSTM v3 and 0.3 counts. Counts back off from long to short contexts, with confidence n/(n+5) for three- and four-word contexts. Lower part: the Hindi Transformer, the highest Hindi single model. It uses a causal mask and predicts from the last position.

## Slide 16: Dataset Description

All counts come from the provenance files. Hindi has fewer rows than Telugu but more examples, because its sentences are longer: 5.2 million versus 3.48 million. Hindi is machine-translated with NLLB-1.3B, so it is not naturally written text. Project files do not record how the Telugu sources were produced. About 3.83% of Hindi test examples sit in translation repetition loops.

## Slide 17: Experimental Setup

Only recorded settings are shown. Telugu baselines ran on CPU with 4 threads; the machine itself is not recorded. Hindi ran on the Apple M5 laptop. Recurrent models used 3 Hindi epochs and CNN and Transformer used 5, so Hindi epochs are unequal. Adam is from [20]. The Transformer pilot used 1 million training examples for one epoch and compared 2 and 4 layers at two learning rates on validation data.

## Slide 18: Experimental Results: Telugu

All rows use the same 200,092 test examples. Accuracy is in percent; PP is perplexity, lower is better. A dash means not measured. Combined is highest on all three accuracy measures. The word-character CNN has the lowest measured perplexity. Five seeds: LSTM Top-1 17.94 ± 0.07 and Bi-LSTM 18.72 ± 0.10.

## Slide 19: Experimental Results: Hindi

All rows use the same 296,193 test examples, one seed, machine-translated text. Combined is highest on every measure, with 34.75% validation Top-1, and also has the lowest perplexity. Hybrid leads Transformer on Top-1 and Top-3, but not on Top-5. Recurrent models stopped at epoch 3 while still improving, so they had fewer epochs than CNN and Transformer.

## Slide 20: Comparative Performance Analysis

The chart shows only models trained in both languages, using the baseline runs. It is not a language ranking: data sources differ, Hindi is machine-translated, OOV differs (7.21% vs 1.04%) and Telugu recurrent models had 5 epochs against 3. Gboard is English logs with a 10,000-word vocabulary, so it is a guide only. The 500-example sample is slightly easier than the full Telugu test set.

## Slide 21: Discussion and Limitations

The left column is measured. The middle column gives possible explanations that have not been tested, so please read them as hypotheses. The right column gives the limits on any claim. Hindi Combined weights were screened on 60,000 validation examples and selected on validation data only.

## Slide 22: Conclusion and Future Scope

Conclusions are limited to what was measured. The first future step finishes the running work. We do not claim novelty or superiority over published work, because languages, data and test methods differ.

## Slide 23: References (1 of 2)

References follow IEEE style and are numbered by first citation in this deck.

## Slide 24: References (2 of 2)

References follow IEEE style and are numbered by first citation in this deck.

## Likely Professor Questions

**Q: Why is your accuracy so far below the 59–99% in other papers?**

A: Those papers use different languages and data, and several state no separate unseen test set. Ours counts OOV targets as wrong and uses grouped unseen text. On seen data Hybrid v2 scores 31.97% Top-1, against 20.75% unseen.

**Q: Your report says no Telugu paper exists. Is that true?**

A: No. The report line is out of date. Two earlier Telugu studies were located [5], [6]. We could read only one abstract, so we quote no scores from them. We do not claim to be first.

**Q: Is Hindi easier than Telugu, since it scores higher?**

A: We cannot say. The data source differs, Hindi is machine-translated, its OOV rate is 1.04% against 7.21%, and it has more examples. The comparison is indicative only.

**Q: Why does a Bi-LSTM not cheat by seeing the future?**

A: It reads only the observed prefix in both directions. The target word is never in its input.

**Q: How do you know there is no data leakage?**

A: Splits are made by a hash of the normalized instruction, so all text from one instruction stays in one split, and exact repeated sentences are removed. Near-duplicates with different wording are not ruled out.

**Q: What does perplexity tell you that accuracy does not?**

A: It measures how much probability the model gives to the true word across the whole distribution. It is the exponential of cross-entropy, and lower is better. OOV targets map to UNK in this measure.

**Q: Is a 0.31 point gain from the combined model meaningful?**

A: It is small, and the combined model has not been repeated across seeds. Seed spread for LSTM and Bi-LSTM is about 0.1 points, which suggests a real but modest gain. The Hindi gain over the best single model is larger, at 1.75 points.

**Q: Why did the large pretrained model do worse?**

A: Measured: 8.6% against 22.6% on a 500-example sample. Likely reasons, not tested: its tokenizer splits Telugu words into pieces and the fine-tune covered under 1% of the examples. Its word ranking is also approximate.

**Q: Why were the Hindi models trained for different numbers of epochs?**

A: Recurrent models used 3 epochs to fit the available time, and were still improving. CNN and Transformer used 5. So Hindi family differences are not on equal training budgets.

**Q: Why is English not finished?**

A: Its data is prepared and training is running. No English score is shown because none exists yet.

**Q: Why did the word-character CNN get zero weight in Hindi Combined?**

A: The validation search gave it no weight, so it added nothing beyond the Transformer, Bi-LSTM and counts on that measure. This does not show that it has no value in other settings.

**Q: Is the Gboard comparison fair?**

A: Only as a guide. It uses the same type of measure, but real English typing with a 10,000-word vocabulary.
