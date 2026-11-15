from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
DATASET_OVERVIEW = ROOT / 'data' / 'figures' / 'dataset_overview.png'


st.set_page_config(page_title='CS747 AI Text Detection', layout='wide')
st.title('CS747 — Prediction of AI Generated Text')
st.write('Project repository for reproducible AI-generated text detection evaluation. Dataset preparation, model packaging, and benchmarking scaffolding are underway.')

if DATASET_OVERVIEW.exists():
    st.image(str(DATASET_OVERVIEW), caption='Dataset overview')
else:
    st.info('Dataset overview image has not been generated yet. Run the dataset preparation pipeline to create it.')

st.subheader('Navigation')
st.write('- Model Comparison')
st.write('- CPU Benchmarking')
st.write('- Generalization')
st.write('- Live Text Detection')
st.caption('Model integrations and benchmark results are not yet available in this placeholder dashboard.')
