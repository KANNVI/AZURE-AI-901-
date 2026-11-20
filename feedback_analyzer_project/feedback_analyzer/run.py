"""Command-line runner:  python run.py [data/sample_feedback.csv]"""
import sys, csv, json
from analyzer import FeedbackAnalyzer

path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_feedback.csv"
with open(path, newline="", encoding="utf-8") as f:
    texts = [row["feedback"] for row in csv.DictReader(f)]
a = FeedbackAnalyzer()
print(f"Backend: {a.backend}\n")
res = a.analyze(texts)
for i, r in enumerate(res, 1):
    print(f"{i:>2}. [{r['sentiment']:<8}] +{r['positive']:.2f} ={r['neutral']:.2f} -{r['negative']:.2f}  {r['key_phrases']}  | {r['text'][:60]}")
print("\nSUMMARY:", a.summarize(res))
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
