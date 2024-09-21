from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from playsound import playsound
from twilio.rest import Client

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'

# Load pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# List of known wild animals
wild_animals = ['tiger', 'lion', 'elephant', 'bear', 'wolf', 'leopard', 'cheetah', 'gorilla', 'chimpanzee', 'fox', 'hyena']

from PIL import Image

# Function to classify an image
def classify_image(img_path):
    try:
        img = Image.open(img_path).resize((224, 224))  # Open image with PIL and resize
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict
        preds = model.predict(img_array)
        predictions = decode_predictions(preds, top=1)[0]

        return predictions[0][1].lower(), predictions[0][2]
    except Exception as e:
        print("Error:", e)
        return None, None


# Function to produce beep sound
def beep_sound(sound_file):
    try:
        playsound(sound_file)
        print("Sound played successfully.")
    except Exception as e:
        print("Error playing sound:", e)

# Function to send message using Twilio
def send_message(account_sid, auth_token, twilio_phone_number, your_phone_number, message):
    try:
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body=message,
            from_=twilio_phone_number,
            to=your_phone_number
        )
        print("Message sent successfully!")
    except Exception as e:
        print("Error sending message:", e)

# Function to check if image is blurry
def is_blurry(image_path, threshold=100):
    img = cv2.imread(image_path)
    if img is None:
        return True
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            animal, confidence = classify_image(filepath)
            if animal is None:
                result = "No animal detected in the image."
            else:
                result = f"Animal detected: {animal.capitalize()}, \nConfidence: {confidence}"
                if animal in wild_animals:
                    result += "\nWild animal detected! Message sent successfully to forest officers"
                    # Produce beep sound
                    beep_sound('C:/Users/ESWARI/Downloads/alarm-car-or-home-62554.mp3')
                    # Send message to officers
                    message = f"Wild animal detected: {animal.capitalize()}, please take necessary actions"
                    send_message('AC620328813375e92fdd6cde2124957362', '11b6e07378ca9db8a34f255a0d517464', '+16467625162', '+919059475441', message)
                    #print("Message sent successfully to forest officers")
                else:
                    result += "\nNo wild animal detected."
            
            return render_template('result.html', result=result, image_path=filepath)
    else:
        # Handle GET request (if needed)
        return render_template('result.html')  # Placeholder for GET request handling
 # Placeholder for GET request handling

if __name__ == '__main__':
    app.run(debug=True)