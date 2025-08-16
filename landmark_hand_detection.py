import cv2
import numpy as np
import math

class HandLandmarkDetector:
    def __init__(self):
        # Define hand landmark connections (similar to MediaPipe)
        self.HAND_CONNECTIONS = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm connections
            (5, 9), (9, 13), (13, 17)
        ]
        
    def detect_hand_contour(self, frame):
        """Detect hand contour using background subtraction and skin detection"""
        height, width = frame.shape[:2]
        
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define skin color range
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour (assume it's the hand)
        if contours:
            hand_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(hand_contour) > 5000:
                return hand_contour
        
        return None
    
    def estimate_landmarks(self, contour, frame):
        """Estimate 21 hand landmarks from contour using improved finger detection"""
        if contour is None:
            return None
            
        # Get convex hull and defects
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate moments for center
        M = cv2.moments(contour)
        if M['m00'] == 0:
            return None
            
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        landmarks = []
        
        # Wrist (landmark 0) - bottom center of bounding box
        wrist = (cx, y + h - 5)
        landmarks.append(wrist)
        
        # Find fingertips using convexity defects
        fingertips = []
        finger_valleys = []
        
        if defects is not None and len(defects) > 0:
            # Analyze convexity defects to find fingertips and valleys
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                
                # Calculate distances
                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                
                # Calculate angle using cosine rule
                if b != 0 and c != 0:
                    angle = math.acos((b**2 + c**2 - a**2) / (2*b*c)) * 180 / math.pi
                    
                    # If angle is less than 90 degrees and depth is significant, it's likely a finger valley
                    if angle <= 90 and d > 20:
                        finger_valleys.append(far)
                        # The points around the valley are potential fingertips
                        if start[1] < wrist[1] - h//4:  # Above palm area
                            fingertips.append(start)
                        if end[1] < wrist[1] - h//4:
                            fingertips.append(end)
        
        # Remove duplicate fingertips that are too close
        filtered_tips = []
        for tip in fingertips:
            is_unique = True
            for existing_tip in filtered_tips:
                distance = math.sqrt((tip[0] - existing_tip[0])**2 + (tip[1] - existing_tip[1])**2)
                if distance < 30:  # Too close to existing tip
                    is_unique = False
                    break
            if is_unique:
                filtered_tips.append(tip)
        
        # Sort fingertips by x-coordinate (left to right)
        filtered_tips.sort(key=lambda p: p[0])
        
        # Ensure we have at least some fingertips, use hull points if defects failed
        if len(filtered_tips) < 3:
            hull_points = []
            for i in range(len(hull)):
                point = tuple(contour[hull[i][0]][0])
                if point[1] < wrist[1] - h//6:  # Above palm
                    hull_points.append(point)
            
            hull_points.sort(key=lambda p: p[1])  # Sort by y (topmost first)
            hull_points = hull_points[:5]  # Take top 5
            hull_points.sort(key=lambda p: p[0])  # Sort by x (left to right)
            filtered_tips = hull_points
        
        # Pad with default points if needed
        while len(filtered_tips) < 5:
            default_x = x + len(filtered_tips) * w // 5
            default_y = y + 10
            filtered_tips.append((default_x, default_y))
        
        # Take only 5 fingertips
        filtered_tips = filtered_tips[:5]
        
        # Create landmarks for each finger
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        
        for i, tip in enumerate(filtered_tips):
            if i == 0:  # Thumb - different structure
                # Thumb landmarks
                thumb_base = (wrist[0] - w//6, wrist[1] - h//8)
                thumb_mcp = (tip[0] + (thumb_base[0] - tip[0]) * 0.7, tip[1] + (thumb_base[1] - tip[1]) * 0.7)
                thumb_ip = (tip[0] + (thumb_base[0] - tip[0]) * 0.3, tip[1] + (thumb_base[1] - tip[1]) * 0.3)
                
                landmarks.extend([thumb_base, thumb_mcp, thumb_ip, tip])
            else:
                # Other fingers
                # Calculate finger base (MCP joint)
                finger_base_x = wrist[0] + (i - 2) * w // 6
                finger_base_y = wrist[1] - h // 5
                mcp = (finger_base_x, finger_base_y)
                
                # PIP joint (about 2/3 from tip to base)
                pip_x = tip[0] + (mcp[0] - tip[0]) * 0.65
                pip_y = tip[1] + (mcp[1] - tip[1]) * 0.65
                pip = (int(pip_x), int(pip_y))
                
                # DIP joint (about 1/3 from tip to base)
                dip_x = tip[0] + (mcp[0] - tip[0]) * 0.35
                dip_y = tip[1] + (mcp[1] - tip[1]) * 0.35
                dip = (int(dip_x), int(dip_y))
                
                landmarks.extend([mcp, pip, dip, tip])
        
        # Ensure we have exactly 21 landmarks
        while len(landmarks) < 21:
            landmarks.append((cx, cy))
            
        return landmarks[:21]
    
    def draw_landmarks(self, frame, landmarks):
        """Draw landmarks and connections like MediaPipe"""
        if landmarks is None:
            return frame
            
        # Draw connections
        for connection in self.HAND_CONNECTIONS:
            if connection[0] < len(landmarks) and connection[1] < len(landmarks):
                pt1 = tuple(map(int, landmarks[connection[0]]))
                pt2 = tuple(map(int, landmarks[connection[1]]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        
        # Draw landmarks
        for i, landmark in enumerate(landmarks):
            center = tuple(map(int, landmark))
            cv2.circle(frame, center, 5, (255, 0, 255), -1)  # Magenta circles
            # Add landmark numbers for debugging
            cv2.putText(frame, str(i), (center[0] + 8, center[1] + 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return frame

def main():
    print("Starting MediaPipe-style hand landmark detection")
    
    # Initialize detector
    detector = HandLandmarkDetector()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect hand contour
        contour = detector.detect_hand_contour(frame)
        
        if contour is not None:
            # Draw contour
            cv2.drawContours(frame, [contour], -1, (0, 255, 255), 2)
            
            # Estimate landmarks
            landmarks = detector.estimate_landmarks(contour, frame)
            
            # Draw landmarks and connections
            frame = detector.draw_landmarks(frame, landmarks)
        
        # Add instructions
        cv2.putText(frame, "Show your hand to the camera", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Hand Landmark Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
