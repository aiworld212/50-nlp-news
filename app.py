
import streamlit as st
import pickle
import pandas as pd
import os

MODELS_DIR = 'bbc_news_project_output/models'
PIPE_PATH = os.path.join(MODELS_DIR, 'tfidf_logreg_pipeline.pkl')
LE_PATH = os.path.join(MODELS_DIR, 'label_encoder.pkl')

@st.cache_resource
def load_model():
    with open(PIPE_PATH, 'rb') as f:
        pipe = pickle.load(f)
    with open(LE_PATH, 'rb') as f:
        le = pickle.load(f)
    return pipe, le

pipe, le = load_model()

st.set_page_config(page_title='BBC News Topic Classifier', layout='centered')
st.title('BBC News Topic Classifier')
st.markdown('Enter a news article or headline and the model will predict the topic.')

input_text = st.text_area('Paste article text or headline here', height=200)
if st.button('Predict'):
    if not input_text.strip():
        st.warning('Please enter article text.')
    else:
        probs = pipe.predict_proba([input_text])[0]
        idx = probs.argmax()
        label = le.inverse_transform([idx])[0]
        st.subheader('Predicted Topic: ' + label)
        st.write('Confidence: {:.2f}%'.format(probs[idx]*100))
        prob_df = pd.DataFrame({'topic': le.classes_, 'probability': probs}).sort_values('probability', ascending=False)
        st.table(prob_df)

st.write('---')
st.header('Batch predict from CSV')
uploaded = st.file_uploader('Upload a CSV with a text column (column name: text)', type=['csv'])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    if 'text' not in df.columns:
        st.error("CSV must contain a 'text' column. Rename if necessary.")
    else:
        st.write('Running predictions...')
        preds = pipe.predict(df['text'].astype(str).tolist())
        pred_labels = le.inverse_transform(preds)
        df['pred_label'] = pred_labels
        st.dataframe(df.head())
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button('Download results CSV', csv, 'predictions.csv', 'text/csv')

st.write('\n---\n')
st.info('Model files expected at: ' + MODELS_DIR + '\nEnsure the pipeline and label encoder pickles are present.')
