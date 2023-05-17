import openai
import requests
from PIL import Image
from io import BytesIO
import streamlit as st

openai.api_key = input("Enter your openai api_key: ")

def generate_dalle_image(prompt):
    # Generate an image using DALL-E API
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )

    # Get the URL of the generated image
    image_url = response['data'][0]['url']

    # Download the image and return the image content
    image_content = requests.get(image_url).content
    image = Image.open(BytesIO(image_content)).resize((850, 850))
    
    # Create a download button for the image
    # file_name = prompt.replace(" ", " ") + ".jpg"
    # st.download_button(
    #     label="Download image",
    #     data=image_content,
    #     file_name=file_name,
    #     mime="image/jpeg"
    # )
    return image