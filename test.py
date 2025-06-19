import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
from glob import glob
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

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

def evaluate_test_folder(test_folder='test_data', model_path='hand_gesture_model.pkl'):
    # Load the trained model
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        scaler = data['scaler']
    
    # Get all subdirectories (each represents a label)
    labels = [d for d in os.listdir(test_folder) if os.path.isdir(os.path.join(test_folder, d))]
    
    true_labels = []
    pred_labels = []
    confidences = []
    image_paths = []
    
    for label in labels:
        label_path = os.path.join(test_folder, label)
        print(f"Processing test images for label: {label}")
        
        # Get all image files in the label directory
        image_files = glob(os.path.join(label_path, "*.jpg")) + \
                     glob(os.path.join(label_path, "*.png")) + \
                     glob(os.path.join(label_path, "*.jpeg"))
        
        for image_file in image_files:
            # Read and process the image
            image = cv2.imread(image_file)
            if image is None:
                continue
                
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                # Get landmarks for the first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]
                features = get_relative_landmarks(hand_landmarks, image.shape)
                
                # Scale features and predict
                features_scaled = scaler.transform([features])
                prediction = model.predict(features_scaled)[0]
                confidence = np.max(model.predict_proba(features_scaled))
                
                # Store results
                true_labels.append(label)
                pred_labels.append(prediction)
                confidences.append(confidence)
                image_paths.append(image_file)
                
                # Optional: visualize processing
                if False:  # Set to True to visualize processing
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    cv2.putText(image, f"True: {label}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(image, f"Pred: {prediction} ({confidence:.2f})", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Processing", image)
                    cv2.waitKey(500)
                    cv2.destroyAllWindows()
            else:
                print(f"No hand detected in {image_file}")
    
    # Generate evaluation metrics
    if true_labels:
        print("\nEvaluation Metrics:")
        print(classification_report(true_labels, pred_labels))
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, pred_labels, labels=labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'image_path': image_paths,
            'true_label': true_labels,
            'predicted_label': pred_labels,
            'confidence': confidences
        })
        
        # Save results to CSV
        results_df.to_csv('test_results.csv', index=False)
        print("\nDetailed results saved to 'test_results.csv'")
        
        # Calculate and print accuracy
        accuracy = np.mean(np.array(true_labels) == np.array(pred_labels))
        print(f"\nOverall Accuracy: {accuracy:.2%}")
        
        return results_df
    else:
        print("No valid test images processed")
        return None

if __name__ == "__main__":
    # Path to your test folder (should contain subfolders for each gesture)
    test_folder = "test_data"
    
    if not os.path.exists(test_folder):
        print(f"Error: Test folder '{test_folder}' not found!")
        print("Please create a folder with this structure:")
        print(f"{test_folder}/")
        print("├── gesture1/")
        print("│   ├── image1.jpg")
        print("│   ├── image2.jpg")
        print("│   └── ...")
        print("├── gesture2/")
        print("│   ├── image1.jpg")
        print("│   └── ...")
        print("└── ...")
    else:
        results = evaluate_test_folder(test_folder)