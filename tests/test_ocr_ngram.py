import torch
from transformers import NoRepeatNGramLogitsProcessor

from app.stages.ocr import _NoRepeatNGram


def test_matches_transformers_no_repeat_ngram() -> None:
    gen = torch.Generator().manual_seed(0)
    vocab = 12  # 词表小，随机序列里才会经常出现重复的 n 元组
    for n in (2, 3, 4):
        ours, ref = _NoRepeatNGram(n), NoRepeatNGramLogitsProcessor(n)
        for length in range(1, 30):
            ids = torch.randint(0, vocab, (8, length), generator=gen)
            scores = torch.randn(8, vocab, generator=gen)
            assert torch.equal(ours(ids, scores.clone()), ref(ids, scores.clone())), (n, length)
