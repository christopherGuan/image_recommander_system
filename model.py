from sentence_transformers import SentenceTransformer
import code.streamlit as st

@st.cache_data()
def get_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
