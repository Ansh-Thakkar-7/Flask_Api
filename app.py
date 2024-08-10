from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

# Load your model
model = tf.keras.models.load_model('amanking.h5')

@app.route('/')
def home():
    return "Welcome to the Plant Disease Prediction API! Use the /predict endpoint to make predictions."

@app.route('/predict', methods=['POST'])

def predict():
    print("hello")
    file = request.files['image']
    # Read the file as an image
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (100, 100))
    img = img.astype("float32") / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction[0])

    class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight']
    
    return jsonify({'predicted_class': class_name[predicted_class_index]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
