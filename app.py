from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import io

from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the models
models = {
    'foods': load_model('models/foods.h5'),
    'gym-equipments': load_model('models/gym-equipments.h5')
}

# Define class names and calorie estimates
class_info = {
    'foods': {
        'class_names': [
            'apple_pie', 'bibimbap', 'cheesecake', 'dumplings', 'fish_and_chips',
            'grilled_salmon', 'hot_dog', 'ice_cream', 'lobster_roll_sandwich',
            'onion_rings', 'pizza', 'ramen'
        ],
        'calorie_estimates': {
            'apple_pie': 500, 'bibimbap': 634, 'cheesecake': 321, 'dumplings': 215,
            'fish_and_chips': 861, 'grilled_salmon': 180, 'hot_dog': 150,
            'ice_cream': 280, 'lobster_roll_sandwich': 340, 'onion_rings': 335,
            'pizza': 300, 'ramen': 371
        }
    },
    'gym-equipments': {
        'class_names': [
            'barbell', 'dumbell', 'gym-ball', 'kettle-ball', 'leg-press',
            'punching-bag', 'roller-abs', 'states-bicycle', 'step', 'treadmill'
        ],
        'calorie_estimates': {
            'barbell': 38, 'dumbell': 36, 'gym-ball': 45, 'kettle-ball': 200,
            'leg-press': 80, 'punching-bag': 76, 'roller-abs': 48,
            'states-bicycle': 100, 'step': 95, 'treadmill': 113
        }
    }
}

# Preprocess the image
def preprocess_image(file_content):
    img = image.load_img(file_content, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Get prediction
def get_prediction(img_array, model_key):
    model = models[model_key]
    class_names = class_info[model_key]['class_names']
    predictions = model.predict(img_array)
    predicted_class = int(np.argmax(predictions))
    predicted_class_label = class_names[predicted_class]
    probabilities = np.exp(predictions) / np.sum(np.exp(predictions), axis=-1)
    return predicted_class_label, probabilities[0]

# Index route
@app.route('/')
def index():
    return 'Welcome to Fit Focus Machine Learning API!'

# Unified prediction route
@app.route('/api/<model_key>', methods=['POST'])
def predict(model_key):
    if model_key not in models:
        return 'Invalid model type', 400

    if 'image' not in request.files:
        return 'No file part', 400

    uploaded_file = request.files['image']
    if uploaded_file.filename == '':
        return 'No selected file', 400

    file_content = io.BytesIO(uploaded_file.read())
    img_array = preprocess_image(file_content)
    predicted_class_label, probabilities = get_prediction(img_array, model_key)
    
    calorie_estimates = class_info[model_key]['calorie_estimates']
    estimated_calories = calorie_estimates.get(predicted_class_label, 0)
    response_key = 'estimated_calories' if model_key == 'foods' else 'estimated_calories_burned'

    return jsonify({
        'prediction': predicted_class_label,
        'probabilities': {
            class_name: prob.item() for class_name, prob in zip(class_info[model_key]['class_names'], probabilities)
        },
        response_key: estimated_calories
    })

if __name__ == "__main__":
    app.run()
    # app.run(ssl_context=('cert.pem', 'key.pem'))

