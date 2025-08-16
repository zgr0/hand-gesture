import mediapipe as mp
import cv2
import numpy as np

print("MediaPipe version:", mp.__version__)

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Try to create a simple image for testing instead of camera
test_image = np.zeros((480, 640, 3), dtype=np.uint8)

try:
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands:
        print("MediaPipe Hands initialized successfully!")
        
        # Process the test image
        results = hands.process(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
        print("Image processing successful!")
        
except Exception as e:
    print(f"Error: {e}")
    print("MediaPipe initialization failed")
