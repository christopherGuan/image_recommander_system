import code.streamlit as st
from recommend_images import recommend_images
from embeddings import get_embeddings
from model import get_model
from path import get_path
import time
import os

start_time = time.time()
st.set_page_config(page_title="Text -> Images", page_icon="🐣", layout="wide")
st.markdown("<h1 style='text-align: center;'>Text-Image Recommending System</h1>", unsafe_allow_html=True)

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
