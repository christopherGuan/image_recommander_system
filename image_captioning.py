import torch
import clip
from PIL import Image

# Load the CLIP model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

# Load the input image
image_path = "path/to/image.jpg"
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

# Encode the image using CLIP's text encoder
with torch.no_grad():
    image_features = model.encode_image(image)

# Generate a caption based on the image features
caption = clip.tokenize(["a photo of "])[0] + clip.generate_text(
    image_features,
    model,
    temperature=1.0,
    top_p=0.9,
    top_k=40,
    repetition_penalty=1.0,
    num_beams=5,
    num_return_sequences=1,
)[0].strip()

# Output the generated caption
print("Generated Caption: ", caption)

