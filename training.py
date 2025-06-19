import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os
from glob import glob

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Function to calculate relative landmark positions
def get_relative_landmarks(hand_landmarks, image_shape):
    landmarks = []
    base_x, base_y = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y
    
    for landmark in hand_landmarks.landmark:
        # Calculate relative position to wrist (landmark 0)
        rel_x = landmark.x - base_x
        rel_y = landmark.y - base_y
        rel_z = landmark.z - hand_landmarks.landmark[0].z
        
        # Normalize by distance between wrist and middle finger MCP (landmark 9)
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

# Function to process images from folder
def process_images_from_folder(data_folder):
    dataset = {}
    
    # Get all subdirectories (each represents a label)
    labels = [d for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))]
    
    for label in labels:
        label_path = os.path.join(data_folder, label)
        print(f"Processing label: {label}")
        
        # Get all image files in the label directory
        image_files = glob(os.path.join(label_path, "*.jpg")) + \
                     glob(os.path.join(label_path, "*.png")) + \
                     glob(os.path.join(label_path, "*.jpeg"))
        
        label_data = []
        
        for image_file in image_files:
            # Read image
            image = cv2.imread(image_file)
            if image is None:
                continue
                
            # Convert to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get relative landmarks
                    relative_landmarks = get_relative_landmarks(hand_landmarks, image.shape)
                    label_data.append(relative_landmarks)
                    
                    # Optional: visualize processing
                    if False:  # Set to True to visualize processing
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        cv2.imshow(f"Processing {label}", image)
                        cv2.waitKey(100)
                        cv2.destroyAllWindows()
        
        if label_data:
            dataset[label] = label_data
            print(f"Processed {len(label_data)} samples for {label}")
        else:
            print(f"No valid hand landmarks found for {label}")
    
    return dataset

# Main function to train model
def main():
    # Path to your dataset folder (should contain subfolders for each gesture)
    data_folder = "hand_gesture_dataset"
    
    if not os.path.exists(data_folder):
        print(f"Error: Dataset folder '{data_folder}' not found!")
        print("Please create a folder with this structure:")
        print(f"{data_folder}/")
        print("├── gesture1/")
        print("│   ├── image1.jpg")
        print("│   ├── image2.jpg")
        print("│   └── ...")
        print("├── gesture2/")
        print("│   ├── image1.jpg")
        print("│   └── ...")
        print("└── ...")
        return
    
    # Process images from folder
    dataset = process_images_from_folder(data_folder)
    
    if not dataset:
        print("No valid training data found!")
        return
    
    # Prepare training data
    X = []
    y = []
    
    for label, samples in dataset.items():
        for sample in samples:
            X.append(sample)
            y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train Random Forest classifier
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    
    # Save model and scaler
    with open('hand_gesture_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
    print("Model saved as 'hand_gesture_model.pkl'")

if __name__ == "__main__":
    main()