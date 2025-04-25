from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from io import BytesIO

app = Flask(__name__)

# Load your VGG16 model
model = load_model('model/vgg16_model.h5')

# Define image size
IMG_SIZE = (224, 224)

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Convert the file to a BytesIO stream
    img_bytes = BytesIO(file.read())
    
    # Load the image
    img = load_img(img_bytes, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = int(np.round(prediction[0][0]))  # assuming binary output
    return jsonify({'prediction': result})


if __name__ == '__main__':
    app.run(debug=True)
