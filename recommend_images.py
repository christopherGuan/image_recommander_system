import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from PIL import Image
import code.streamlit as st

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
