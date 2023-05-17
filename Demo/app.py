import streamlit as st
from recommend_images import recommend_images
from embeddings import get_embeddings
from model import get_model
from path import get_path
from dalle_api import generate_dalle_image
import os


st.set_page_config(page_title="Text -> Images", page_icon="🐣", layout="wide")
st.markdown("<h1 style='text-align: center;'>Text-Image Recommending System</h1>", unsafe_allow_html=True)


#Two columns of inputs
colA, colB = st.columns([8, 2])
#Sentence Input
imageLike = colA.text_input("")
colA.write('Describe The Image You Expect')
#Similarity Input
similarity_threshold = colB.number_input('',min_value=0.0, max_value=1.0, step=0.01, value=0.1)
colB.write('Similarity Threshold')


if imageLike:
    
    emb_df = get_embeddings()
    model = get_model()
    path = get_path()
    image_list = recommend_images(imageLike, emb_df, model, path, similarity_threshold)

    # create a grid of images and captions
    i = 0
    while i < len(image_list):
        row = st.container()
        with row:
            col1, col2, col3 = st.columns(3)
            cols = [col1, col2, col3]
            for col in cols:
                if i < len(image_list):
                    # display the image and caption
                    col.image(image_list[i][0], caption=image_list[i][1])
                    with col.container():
                        download_button_str = f"Download" 
                        download_file_path = os.path.join(path, f"{i+1}.jpg")
                        col.download_button(download_button_str, download_file_path)
                    i += 1
                    
    if len(image_list) < 10:
        
        img = generate_dalle_image(prompt=f"{imageLike}")

        col_name = f"col{(len(image_list)%3)+1}"
        col_obj = eval(col_name)
        col_obj.image(img, use_column_width=True)
        col_obj.write("<div style='color:green; text-align:center'>DALL-E generated image based on your input</div>", unsafe_allow_html=True)
        with col_obj.container():
            download_button_str = f"Download" 
            download_file_path = os.path.join(path, f"{i+1}.jpg")
            col_obj.download_button(download_button_str, download_file_path)        