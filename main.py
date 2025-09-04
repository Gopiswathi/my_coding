# main.py
# PCB Fault Detection - Starter Code

# Libraries
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model('pcb_fault_model.h5')  # Make sure model file is in same folder

# Function to preprocess image
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))  # Resize to model input size
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)
    return img

# Function to predict PCB defect
def predict_defect(img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    if prediction[0][0] > 0.5:
        return "Defective PCB"
    else:
        return "Normal PCB"

# Test example
if __name__ == "__main__":
    test_image = 'test_pcb.jpg'  # Replace with your test image
    result = predict_defect(test_image)
    print(f"Prediction: {result}")
