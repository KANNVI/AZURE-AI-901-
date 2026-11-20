import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from analyzer import FeedbackAnalyzer

a = FeedbackAnalyzer(backend="offline")

def test_positive():
    r = a.analyze(["The delivery was quick and the packaging was excellent."])[0]
    assert r["sentiment"] == "positive" and r["positive"] > r["negative"]

def test_negative():
    r = a.analyze(["The app is slow and keeps crashing. Terrible experience."])[0]
    assert r["sentiment"] == "negative"

def test_neutral_and_empty():
    assert a.analyze(["The product is okay."])[0]["sentiment"] == "neutral"
    assert a.analyze(["", "   "]) == []

def test_summary_mentions_counts():
    res = a.analyze(["Great and friendly", "Terrible and slow"])
    assert "2 feedback items" in a.summarize(res)
