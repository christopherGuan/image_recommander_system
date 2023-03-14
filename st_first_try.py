import code.streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from PIL import Image
import time

start_time = time.time()
st.set_page_config(page_title="Text -> Images", page_icon="🐣", layout="wide")
st.markdown("<h1 style='text-align: center;'>Text-Image Recommending System</h1>", unsafe_allow_html=True)

@st.cache_data()
def get_embeddings():
    return pd.read_csv('/Users/chrisguan/Documents/senior_project/demo/ids_embeddings.csv')

@st.cache_data()
def get_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@st.cache_data()
def get_path():
    return '/Users/chrisguan/Documents/senior_project/dataset/new/images/'

def recommend_images(imageLike, emb_df, model, path):
    
    embeddings_input = model.encode(imageLike)

    cosine_sims = cosine_similarity(
        [embeddings_input],
        emb_df.iloc[:, 2:]
    )[0]
    
    top_indices = np.argsort(cosine_sims)[::-1][:100]
    top_table = emb_df.iloc[top_indices]

    # create an empty list to store the images and captions
    image_list = []

    # Loop through the 'num' and 'cap' columns in parallel
    for num, cap in zip(top_table['image_id'], top_table['caption']):
        # Construct the full path to the file
        file_path = os.path.join(path, str(num))
        # Open the image file and resize it to reduce the image size
        image = Image.open(file_path).resize((850, 850))
        # add the image and caption to the list
        image_list.append((image, cap))
    return image_list

imageLike = st.text_input("")
if imageLike:
    emb_df = get_embeddings()
    model = get_model()
    path = get_path()
    # Call the recommend_images function and cache the result
    image_list = recommend_images(imageLike, emb_df, model, path)
    
    # create a grid of images and captions
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    i = 0

    for col in cols:
        for j in range(20):
            if i < len(image_list):

                # display the image and caption
                col.image(image_list[i][0], caption=image_list[i][1])

                with col.container():
                    download_button_str = f"Download"
                    download_file_path = os.path.join(path, f"{i+1}.jpg")
                    col.download_button(download_button_str, download_file_path)
                i += 1
                
    end_time = time.time()
    total_time = end_time - start_time
    st.text(total_time)
else:
  print('describe the image you expect')