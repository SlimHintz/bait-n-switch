from baitnswitch.text import normalize, word_count


def test_strips_tweet_furniture():
    assert normalize("RT @BuzzFeed: This quiz is wild http://t.co/x #quiz") == "this quiz is wild quiz"


def test_folds_accents_and_curly_quotes():
    assert normalize("Café owner’s “secret”") == "cafe owner's \"secret\""


def test_collapses_whitespace():
    assert normalize("  Breaking:\n\tnews  ") == "breaking: news"


def test_word_count_ignores_punctuation():
    assert word_count("Wait... what?!") == 2
