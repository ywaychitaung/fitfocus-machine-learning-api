from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import io

from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the disease prediction model
model = load_model('models/disease_model.h5')

# Load class names from the disease_labels.txt file
with open('models/disease_labels.txt', 'r') as file:
    class_names = [line.strip() for line in file.readlines()]

# Preprocess the image
def preprocess_image(file_content):
    img = image.load_img(file_content, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Get prediction
def get_prediction(img_array):
    predictions = model.predict(img_array)
    predicted_class = int(np.argmax(predictions))
    predicted_class_label = class_names[predicted_class]
    probabilities = np.exp(predictions) / np.sum(np.exp(predictions), axis=-1)
    return predicted_class_label, probabilities[0]

# Index route
@app.route('/')
def index():
    return 'Welcome to the Disease Prediction API!'

# Disease prediction route
@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No file part', 400

    uploaded_file = request.files['image']
    if uploaded_file.filename == '':
        return 'No selected file', 400

    file_content = io.BytesIO(uploaded_file.read())
    img_array = preprocess_image(file_content)
    predicted_class_label, probabilities = get_prediction(img_array)

    return jsonify({
        'prediction': predicted_class_label,
        'probabilities': {
            class_name: prob.item() for class_name, prob in zip(class_names, probabilities)
        }
    })

if __name__ == "__main__":
    app.run()