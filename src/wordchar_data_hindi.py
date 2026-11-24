"""wordchar_data.prepare for any language. Spelling comes from each split; the alphabet from training only."""
import json
import numpy as np
import torch
from text_utils import iter_language_lines, tokenize


def prepare(base, vocab, language, raw_name, max_chars=32):
    alphabet = set()
    for line in iter_language_lines(base / raw_name, language=language):
        for word in tokenize(line, language=language):
            alphabet.update(word)
    char_to_idx = {c: i + 2 for i, c in enumerate(sorted(alphabet))}
    surfaces = {'<pad>': 0}
    surface_words = [0]
    surface_chars = [[0] * max_chars]
    splits = []
    for split in ['', 'validation', 'test']:
        folder = base / split
        manifest = json.loads((folder / 'sequences_manifest.json').read_text())
        length = vocab['seq_length']
        rows = np.empty((manifest['total_sequences'], length + 1), dtype=np.int64)
        offset = 0
        for line in iter_language_lines(folder / raw_name, language=language):
            words = tokenize(line, language=language)
            if len(words) < 2:
                continue
            ids = []
            for word in words:
                if word not in surfaces:
                    surfaces[word] = len(surface_words)
                    surface_words.append(vocab['word_to_idx'].get(word, 1))
                    chars = [char_to_idx.get(c, 1) for c in word[-max_chars:]]
                    surface_chars.append(chars + [0] * (max_chars - len(chars)))
                ids.append(surfaces[word])
            for i in range(1, len(ids)):
                prefix = ids[max(0, i - length):i]
                rows[offset, :-1] = [0] * (length - len(prefix)) + prefix
                rows[offset, -1] = vocab['word_to_idx'].get(words[i], 1)
                offset += 1
        if offset != len(rows):
            raise ValueError('Word-character sequence count differs from baseline')
        splits.append(torch.from_numpy(rows))
    metadata = dict(char_to_idx=char_to_idx, surface_to_idx=surfaces, max_chars=max_chars,
                    encoding='NFC Unicode codepoints; suffix truncation')
    return splits, surface_words, surface_chars, metadata
