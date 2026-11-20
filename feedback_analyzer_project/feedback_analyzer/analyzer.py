"""Customer Feedback Insight Analyzer - core logic.

Pipeline:  (image) -> OCR -> language / sentiment / key phrases -> summary

Two back-ends:
  * "azure"   : Azure AI Vision (Read OCR), Azure AI Language, Azure OpenAI
  * "offline" : simple local rules so the project runs with no keys (demo/tests)
Credentials are read from environment variables, never hard-coded.
"""
import os
import re
from collections import Counter

STOP = set("""a an the and or but if so to of in on at for with from by is are was were be been it its this that these
those i me my we our you your they them their he she his her as not no very too just also than then there here
have has had do does did can could would should will am about after before again all any more most some such
only own same s t don didn really got get one two three""".split())
POS = set("""good great excellent amazing awesome love loved quick fast friendly helpful easy smooth perfect best nice
happy satisfied fantastic wonderful recommend reliable clean superb brilliant""".split())
NEG = set("""bad poor terrible awful slow late broken worst rude delay delayed disappointed disappointing useless dirty
expensive problem issue issues refund waste unhelpful horrible faulty damaged missing failed confusing crashing crash fail""".split())


def _words(text):
    return re.findall(r"[a-zA-Z']+", text.lower())


class FeedbackAnalyzer:
    def __init__(self, backend=None):
        if backend is None:
            backend = "azure" if os.getenv("AZURE_LANGUAGE_ENDPOINT") and os.getenv("AZURE_LANGUAGE_KEY") else "offline"
        self.backend = backend
        if backend == "azure":
            from azure.core.credentials import AzureKeyCredential
            from azure.ai.textanalytics import TextAnalyticsClient
            self._lang = TextAnalyticsClient(os.environ["AZURE_LANGUAGE_ENDPOINT"],
                                             AzureKeyCredential(os.environ["AZURE_LANGUAGE_KEY"]))

    # ---------- stage 1: OCR ----------
    def ocr(self, image_bytes):
        if self.backend != "azure" or not os.getenv("AZURE_VISION_ENDPOINT"):
            raise RuntimeError("OCR needs Azure AI Vision: set AZURE_VISION_ENDPOINT and AZURE_VISION_KEY.")
        from azure.ai.vision.imageanalysis import ImageAnalysisClient
        from azure.ai.vision.imageanalysis.models import VisualFeatures
        from azure.core.credentials import AzureKeyCredential
        client = ImageAnalysisClient(os.environ["AZURE_VISION_ENDPOINT"],
                                     AzureKeyCredential(os.environ["AZURE_VISION_KEY"]))
        result = client.analyze(image_data=image_bytes, visual_features=[VisualFeatures.READ])
        lines = []
        if result.read is not None:
            for block in result.read.blocks:
                lines += [ln.text for ln in block.lines]
        return "\n".join(lines)

    # ---------- stage 2: text analytics ----------
    def analyze(self, texts):
        texts = [t.strip() for t in texts if t and t.strip()]
        return self._analyze_azure(texts) if self.backend == "azure" else self._analyze_offline(texts)

    def _analyze_azure(self, texts):
        out = []
        for i in range(0, len(texts), 10):          # service limit: 10 docs per request
            chunk = texts[i:i + 10]
            langs = self._lang.detect_language(documents=chunk)
            sent = self._lang.analyze_sentiment(documents=chunk)
            keys = self._lang.extract_key_phrases(documents=chunk)
            for t, l, s, k in zip(chunk, langs, sent, keys):
                if l.is_error or s.is_error or k.is_error:
                    out.append(dict(text=t, language="unknown", sentiment="error", positive=0, neutral=0, negative=0, key_phrases=[]))
                    continue
                c = s.confidence_scores
                out.append(dict(text=t, language=l.primary_language.name, sentiment=s.sentiment,
                                positive=round(c.positive, 2), neutral=round(c.neutral, 2),
                                negative=round(c.negative, 2), key_phrases=list(k.key_phrases)))
        return out

    def _analyze_offline(self, texts):
        out = []
        for t in texts:
            w = _words(t)
            p = n = 0
            for j, x in enumerate(w):                # negation: "not good" counts as negative
                negated = j > 0 and w[j - 1] in ("not", "never", "no")
                if x in POS:
                    n, p = (n + 1, p) if negated else (n, p + 1)
                elif x in NEG:
                    p, n = (p + 1, n) if negated else (p, n + 1)
            total = max(p + n, 1)
            pos, neg = p / total, n / total
            label = "positive" if p > n else "negative" if n > p else "neutral"
            kp = [w_ for w_, _ in Counter(x for x in w if x not in STOP and len(x) > 3).most_common(3)]
            out.append(dict(text=t, language="English (assumed)", sentiment=label,
                            positive=round(pos, 2), neutral=round(1 - pos - neg, 2),
                            negative=round(neg, 2), key_phrases=kp))
        return out

    # ---------- stage 3: summary ----------
    def summarize(self, results):
        counts = Counter(r["sentiment"] for r in results)
        phrases = Counter(p.lower() for r in results for p in r["key_phrases"]).most_common(8)
        stats = (f"{len(results)} feedback items: " +
                 ", ".join(f"{counts.get(k, 0)} {k}" for k in ("positive", "neutral", "negative")) +
                 ". Top themes: " + ", ".join(p for p, _ in phrases) + ".")
        if self.backend == "azure" and os.getenv("AZURE_OPENAI_ENDPOINT"):
            from openai import AzureOpenAI
            client = AzureOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                                 api_key=os.environ["AZURE_OPENAI_KEY"],
                                 api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"))
            resp = client.chat.completions.create(
                model=os.environ["AZURE_OPENAI_DEPLOYMENT"],   # your deployment name
                temperature=0.2, max_tokens=250,
                messages=[
                    {"role": "system", "content": "You summarise customer feedback for a manager in 4-5 neutral "
                     "sentences. Use ONLY the data provided. Do not invent facts."},
                    {"role": "user", "content": stats}])
            return resp.choices[0].message.content.strip()
        # offline template summary
        tone = "mostly positive" if counts["positive"] > counts["negative"] else \
               "mostly negative" if counts["negative"] > counts["positive"] else "mixed"
        return f"Overall feedback is {tone}. " + stats
