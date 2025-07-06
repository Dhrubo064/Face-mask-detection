# Face Mask Detection

## Overview

The Face Mask Detection project uses TensorFlow and Keras to classify whether a person is wearing a face mask. It is built using a Convolutional Neural Network (CNN) and can be tested with images.

## Key Features

1. Detects if a person is wearing a face mask in images.

2. Built using TensorFlow 2.14.0 and Keras.

3. Provides a simple web interface using Streamlit.

## Installation

1. python -m venv venv

2. venv\Scripts\activate #For windows

3. source venv/bin/activate #For mac

4. pip install streamlit tensorflow opencv-python pillow

5. streamlit run app.py

## Model Usage

The app will display an interface where you can upload an image, and it will classify whether the person in the image is wearing a mask or not.
