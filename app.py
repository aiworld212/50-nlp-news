import streamlit as st
import pickle
import pandas as pd

PIPE_PATH = "tfidf_logreg_pipeline.pkl"
LE_PATH   = "label_encoder.pkl"


@st.cache_resource
def load_model():
    with open(PIPE_PATH, "rb") as f:
        pipe = pickle.load(f)
    with open(LE_PATH, "rb") as f:
        le = pickle.load(f)
    return pipe, le

pipe, le = load_model()

st.title("BBC News Topic Classifier")

input_text = st.text_area("Paste article text here", height=200)

if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Enter a news article text.")
    else:
        probs = pipe.predict_proba([input_text])[0]
        label = le.inverse_transform([probs.argmax()])[0]

        st.subheader(f"Predicted Topic: {label}")
        st.write(f"Confidence: {probs.max()*100:.2f}%")

        df = pd.DataFrame({
            "Topic": le.classes_,
            "Probability": probs
        }).sort_values("Probability", ascending=False)

        st.table(df)

uploaded = st.file_uploader("Upload CSV (must contain column: text)")

if uploaded:
    df = pd.read_csv(uploaded)
    if "text" not in df.columns:
        st.error("CSV must have a column named 'text'")
    else:
        preds = pipe.predict(df["text"].astype(str))
        df["predicted_label"] = le.inverse_transform(preds)
        st.dataframe(df.head())
        st.download_button("Download Results", df.to_csv(index=False), "results.csv")
