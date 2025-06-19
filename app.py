from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
import pickle

app = Flask(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, 
                      max_num_hands=1, 
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)

# Load the trained model
with open('hand_gesture_model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    scaler = data['scaler']

def get_relative_landmarks(hand_landmarks):
    landmarks = []
    base_x, base_y = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y
    
    for landmark in hand_landmarks.landmark:
        rel_x = landmark.x - base_x
        rel_y = landmark.y - base_y
        rel_z = landmark.z - hand_landmarks.landmark[0].z
        
        ref_distance = np.sqrt(
            (hand_landmarks.landmark[9].x - base_x)**2 + 
            (hand_landmarks.landmark[9].y - base_y)**2
        )
        
        if ref_distance > 0:
            rel_x /= ref_distance
            rel_y /= ref_distance
            rel_z /= ref_distance
        
        landmarks.extend([rel_x, rel_y, rel_z])
    
    return landmarks

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        prediction_text = ""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get landmarks and predict
                features = get_relative_landmarks(hand_landmarks)
                features_scaled = scaler.transform([features])
                prediction = model.predict(features_scaled)[0]
                prediction_text = f"Prediction: {prediction}"
                
                # Draw landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Add prediction text
        cv2.putText(frame, prediction_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                  mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(debug=True)

import webbrowser
import threading

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    threading.Timer(1.0, open_browser).start()
    app.run(debug=False)