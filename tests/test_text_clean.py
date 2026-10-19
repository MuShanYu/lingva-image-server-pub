import pytest

from app.stages.text_clean import clean_text, is_noise


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("なんだって …！？", "なんだって…！？"),
        ("そう ... か", "そう…か"),
        ("えっ......", "えっ……"),
        ("ま・・・まさか", "ま…まさか"),
        ("2/2 ABC", "２／２ＡＢＣ"),
        ("  \n", ""),
    ],
)
def test_clean_text(raw: str, expected: str) -> None:
    assert clean_text(raw) == expected


@pytest.mark.parametrize(
    ("text", "confidence", "noise"),
    [
        ("", 1.0, True),
        ("！？", 1.0, True),
        ("…", 1.0, True),
        ("ー～", 1.0, True),
        ("ッ", 0.99, False),
        ("ッ", 0.5, True),
        ("は！", 0.9, False),
        ("すまんな千恵", 0.9, False),
        ("お客様のご要望においておりますので", 0.49, True),
    ],
)
def test_is_noise(text: str, confidence: float, noise: bool) -> None:
    assert is_noise(text, confidence) is noise
