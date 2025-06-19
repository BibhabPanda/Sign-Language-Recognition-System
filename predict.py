import cv2
import numpy as np
import mediapipe as mp
import pickle

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, 
                      max_num_hands=1, 
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)

# Load the trained model
with open('hand_gesture_model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    scaler = data['scaler']

# Function to calculate relative landmark positions (same as training)
def get_relative_landmarks(hand_landmarks, image_shape):
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

def predict_gesture(frame):
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get landmarks and predict
            features = get_relative_landmarks(hand_landmarks, frame.shape)
            features_scaled = scaler.transform([features])
            
            # Get prediction and probabilities
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)
            
            # Display prediction
            cv2.putText(frame, f"Gesture: {prediction}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2%}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

# Main function for real-time prediction
def main():
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Predict gesture
        output_frame = predict_gesture(frame)
        
        # Display the resulting frame
        cv2.imshow('Hand Gesture Recognition', output_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()