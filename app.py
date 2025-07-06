import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model

import os
import gdown

model_path = "face_mask_detection_model.h5"


# The File ID is everything between '/d/' and '/view?'
new_file_id = "1ZXJUNy-dFF7VTm8AFls4XXe2808O469v"

if not os.path.exists(model_path):
    url = f"https://drive.google.com/uc?id={new_file_id}"
    print(f"Downloading model from: {url}") # Added for clarity in logs
    gdown.download(url, model_path, quiet=False)
else:
    print(f"Model already exists at: {model_path}, skipping download.") # Added for clarity in logs

# Load the trained model
model = load_model(model_path)


st.title("😷 Face Mask Detection App")
st.write("Upload an image to detect whether a face mask is worn or not.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)

    img_array = np.array(image)
    img_array = cv2.resize(img_array, (224, 224))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    confidence = prediction[0][0]

    label = "🚫 No Mask" if confidence > 0.5 else "😷 Mask"
    confidence_pct = confidence * 100 if confidence > 0.5 else (1 - confidence) * 100

    st.markdown(f"### Prediction: **{label}**")
    st.markdown(f"**Confidence:** {confidence_pct:.2f}%")
