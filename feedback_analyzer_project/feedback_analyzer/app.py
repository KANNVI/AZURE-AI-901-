"""Streamlit web UI:  streamlit run app.py"""
import io
import pandas as pd
import streamlit as st
from analyzer import FeedbackAnalyzer

st.set_page_config(page_title="Customer Feedback Insight Analyzer", layout="wide")
st.title("Customer Feedback Insight Analyzer")
a = FeedbackAnalyzer()
st.caption(f"Backend: **{a.backend}** (set Azure environment variables to use Azure AI services)")

texts = []
tab1, tab2, tab3 = st.tabs(["Paste text", "Upload CSV", "Scanned image (OCR)"])
with tab1:
    raw = st.text_area("One feedback per line", height=180)
    texts += [l for l in raw.splitlines() if l.strip()]
with tab2:
    f = st.file_uploader("CSV with a 'feedback' column", type="csv")
    if f is not None:
        texts += pd.read_csv(f)["feedback"].dropna().astype(str).tolist()
with tab3:
    img = st.file_uploader("Photo/scan of a feedback form", type=["png", "jpg", "jpeg"])
    if img is not None:
        try:
            text = a.ocr(img.getvalue())
            st.text_area("Extracted text", text, height=120)
            texts += [l for l in text.splitlines() if l.strip()]
        except Exception as e:
            st.warning(str(e))

if st.button("Analyze", type="primary") and texts:
    res = a.analyze(texts)
    df = pd.DataFrame(res)
    df["key_phrases"] = df["key_phrases"].apply(", ".join)
    c1, c2 = st.columns([1, 2])
    c1.subheader("Sentiment counts")
    c1.bar_chart(df["sentiment"].value_counts())
    c2.subheader("Summary")
    c2.write(a.summarize(res))
    st.subheader("Details")
    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False), "results.csv")
