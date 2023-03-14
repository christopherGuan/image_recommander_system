import pandas as pd
import code.streamlit as st

@st.cache_data()
def get_embeddings():
    return pd.read_csv('/Users/chrisguan/Documents/senior_project/demo/ids_embeddings.csv')
