from flask import Flask, request, jsonify
from fer import FER
import cv2
import numpy as np

app = Flask(__name__)

# Route to display the web page
@app.route('/')
def index():
    return '''
    <h1>Emotion Detection</h1>
    <form method="POST" action="/detect" enctype="multipart/form-data">
        <input type="file" name="image">
        <button type="submit">Upload</button>
    </form>
    '''

# Route to handle emotion detection
@app.route('/detect', methods=['POST'])
def detect_emotion():
    image = request.files['image']  # Get the uploaded image
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    emotion_detector = FER()  # Initialize the emotion detector
    emotions = emotion_detector.detect_emotions(img)  # Detect emotions
    if emotions:
        return jsonify({'emotion': emotions[0]['emotions']})  # Return the detected emotions
    return jsonify({'error': 'No face detected'})

if __name__ == "__main__":
    app.run(debug=True)
