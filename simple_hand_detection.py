import cv2
import numpy as np
import math

class SimpleHandLandmarkDetector:
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
        
        print("Simple hand detection initialized (no face detection)")
        
    def detect_hand_contour(self, frame):
        """Detect hand contour using skin detection without face exclusion"""
        height, width = frame.shape[:2]
        
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define multiple skin color ranges for better detection
        skin_ranges = [
            # Light skin
            ([0, 20, 70], [20, 255, 255]),
            # Medium skin  
            ([0, 35, 80], [25, 255, 255]),
            # Darker skin
            ([0, 50, 50], [30, 255, 200])
        ]
        
        # Combine all skin masks
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        for lower, upper in skin_ranges:
            lower_skin = np.array(lower, dtype=np.uint8)
            upper_skin = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the best hand contour
        best_contour = self.find_best_hand_contour(contours, frame)
        
        return best_contour, combined_mask
        
    def find_best_hand_contour(self, contours, frame):
        """Find the best hand contour based on area and position"""
        if not contours:
            return None
            
        # Filter by minimum area
        valid_contours = [c for c in contours if cv2.contourArea(c) > 3000]
        
        if not valid_contours:
            return None
            
        # Find contours in the center and right side of frame (typical hand positions)
        frame_height, frame_width = frame.shape[:2]
        center_x = frame_width // 2
        
        scored_contours = []
        for contour in valid_contours:
            # Calculate contour center
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
                
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Score based on position (prefer center-right area)
            position_score = 0
            if center_x * 0.3 < cx < frame_width:  # Right side or center
                position_score = 2
            elif center_x * 0.1 < cx < center_x * 1.5:  # Center area
                position_score = 1
                
            # Score based on shape
            shape_score = self.score_hand_shape(contour)
            
            total_score = position_score + shape_score
            scored_contours.append((contour, total_score))
        
        if not scored_contours:
            return None
            
        # Return the highest scoring contour
        best_contour = max(scored_contours, key=lambda x: x[1])[0]
        return best_contour
        
    def score_hand_shape(self, contour):
        """Score how hand-like a contour is"""
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        score = 0
        
        # Area score
        if 5000 < area < 40000:
            score += 2
        elif 3000 < area < 60000:
            score += 1
            
        # Aspect ratio score
        aspect_ratio = w / h if h > 0 else 0
        if 0.5 < aspect_ratio < 1.3:
            score += 2
        elif 0.4 < aspect_ratio < 1.6:
            score += 1
            
        # Convexity score
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if 0.6 < solidity < 0.85:  # Hands have moderate solidity
                score += 1
                
        return score
    
    def estimate_landmarks(self, contour, frame):
        """Estimate 21 hand landmarks from contour"""
        if contour is None:
            return None
            
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate center
        M = cv2.moments(contour)
        if M['m00'] == 0:
            return None
            
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        landmarks = []
        
        # Wrist (landmark 0)
        wrist = (cx, y + h - 10)
        landmarks.append(wrist)
        
        # Find fingertips using convex hull
        hull = cv2.convexHull(contour, returnPoints=True)
        hull_points = [tuple(point[0]) for point in hull]
        
        # Filter hull points that could be fingertips
        fingertip_candidates = []
        for pt in hull_points:
            # Must be above the palm area
            if pt[1] < wrist[1] - h//4:
                fingertip_candidates.append(pt)
        
        # Remove points too close to each other
        filtered_tips = []
        min_distance = 30
        for candidate in fingertip_candidates:
            if all(math.sqrt((candidate[0] - tip[0])**2 + (candidate[1] - tip[1])**2) > min_distance 
                   for tip in filtered_tips):
                filtered_tips.append(candidate)
        
        # Sort by x-coordinate and limit to 5
        filtered_tips.sort(key=lambda p: p[0])
        filtered_tips = filtered_tips[:5]
        
        # Ensure we have 5 fingertips
        while len(filtered_tips) < 5:
            default_x = x + len(filtered_tips) * w // 5
            default_y = y + 15
            filtered_tips.append((default_x, default_y))
        
        # Create landmarks for each finger
        for i, tip in enumerate(filtered_tips):
            if i == 0:  # Thumb
                thumb_base = (wrist[0] - w//6, wrist[1] - h//8)
                thumb_mcp = (int(tip[0] + (thumb_base[0] - tip[0]) * 0.7), 
                           int(tip[1] + (thumb_base[1] - tip[1]) * 0.7))
                thumb_ip = (int(tip[0] + (thumb_base[0] - tip[0]) * 0.3), 
                          int(tip[1] + (thumb_base[1] - tip[1]) * 0.3))
                landmarks.extend([thumb_base, thumb_mcp, thumb_ip, tip])
            else:  # Other fingers
                # MCP joint
                finger_base_x = wrist[0] + (i - 2) * w // 6
                finger_base_y = wrist[1] - h // 5
                mcp = (finger_base_x, finger_base_y)
                
                # PIP joint
                pip_x = int(tip[0] + (mcp[0] - tip[0]) * 0.65)
                pip_y = int(tip[1] + (mcp[1] - tip[1]) * 0.65)
                pip = (pip_x, pip_y)
                
                # DIP joint
                dip_x = int(tip[0] + (mcp[0] - tip[0]) * 0.35)
                dip_y = int(tip[1] + (mcp[1] - tip[1]) * 0.35)
                dip = (dip_x, dip_y)
                
                landmarks.extend([mcp, pip, dip, tip])
        
        # Ensure exactly 21 landmarks
        while len(landmarks) < 21:
            landmarks.append((cx, cy))
            
        return landmarks[:21]
    
    def draw_landmarks(self, frame, landmarks):
        """Draw landmarks and connections"""
        if landmarks is None:
            return frame
            
        # Draw connections
        for connection in self.HAND_CONNECTIONS:
            if connection[0] < len(landmarks) and connection[1] < len(landmarks):
                pt1 = tuple(map(int, landmarks[connection[0]]))
                pt2 = tuple(map(int, landmarks[connection[1]]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        
        # Draw landmarks with different colors
        colors = [
            (255, 0, 0),      # Wrist - red
            (255, 165, 0), (255, 165, 0), (255, 165, 0), (255, 165, 0),  # Thumb - orange
            (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),          # Index - green
            (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255),          # Middle - blue
            (255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255),  # Ring - magenta
            (255, 255, 0), (255, 255, 0), (255, 255, 0), (255, 255, 0)   # Pinky - cyan
        ]
        
        for i, landmark in enumerate(landmarks):
            center = tuple(map(int, landmark))
            color = colors[i] if i < len(colors) else (255, 255, 255)
            cv2.circle(frame, center, 4, color, -1)
            # Add landmark numbers
            cv2.putText(frame, str(i), (center[0] + 6, center[1] + 6), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return frame

def main():
    print("Starting Simple Hand Landmark Detection")
    print("No face detection - works with basic OpenCV installations")
    
    # Initialize detector
    detector = SimpleHandLandmarkDetector()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Press 'q' to quit, 'm' to show mask")
    show_mask = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect hand contour
        contour, mask = detector.detect_hand_contour(frame)
        
        # Show mask if requested
        if show_mask:
            cv2.imshow('Skin Detection Mask', mask)
        
        if contour is not None:
            # Draw contour
            cv2.drawContours(frame, [contour], -1, (0, 255, 255), 2)
            
            # Show contour info
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            score = detector.score_hand_shape(contour)
            
            cv2.putText(frame, f'Area: {int(area)}', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f'Score: {score}', (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f'Size: {w}x{h}', (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Estimate landmarks
            landmarks = detector.estimate_landmarks(contour, frame)
            
            # Draw landmarks and connections
            frame = detector.draw_landmarks(frame, landmarks)
            
            cv2.putText(frame, "Hand Detected!", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Add instructions
        cv2.putText(frame, "Show your hand to the camera", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if show_mask:
            cv2.putText(frame, "Mask Mode ON", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.imshow('Simple Hand Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow('Skin Detection Mask')
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
