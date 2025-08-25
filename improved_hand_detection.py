import cv2
import numpy as np
import math

class AdvancedHandLandmarkDetector:
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
        
        # Initialize face cascade for face detection with error handling
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                print("Warning: Could not load face cascade. Face detection disabled.")
                self.face_cascade = None
            else:
                print("Face detection initialized successfully.")
        except Exception as e:
            print(f"Warning: Face detection not available ({e}). Continuing without face exclusion.")
            self.face_cascade = None
        
        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # Tracking variables
        self.last_hand_position = None
        self.hand_confidence = 0
        self.max_confidence = 10
        
        # Multiple skin tone ranges for better detection
        self.skin_ranges = [
            # Light skin
            ([0, 20, 70], [20, 255, 255]),
            # Medium skin
            ([0, 35, 80], [25, 255, 255]),
            # Dark skin
            ([0, 50, 50], [30, 255, 200])
        ]
        
    def detect_motion_regions(self, frame):
        """Detect regions with motion to focus hand detection"""
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        return fg_mask
        
    def detect_hand_contour(self, frame):
        """Enhanced hand contour detection with multiple techniques"""
        height, width = frame.shape[:2]
        
        # 1. Face detection to exclude face regions
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []
        if self.face_cascade is not None:
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Create a mask for faces to exclude them
        face_mask = np.ones((height, width), dtype=np.uint8) * 255
        for (x, y, w, h) in faces:
            # Make face region black in the mask (with larger margin)
            margin = 30
            y_start = max(0, y - margin)
            y_end = min(height, y + h + margin)
            x_start = max(0, x - margin)
            x_end = min(width, x + w + margin)
            face_mask[y_start:y_end, x_start:x_end] = 0
            
        # 2. Motion detection
        motion_mask = self.detect_motion_regions(frame)
        
        # 3. Multi-range skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        combined_skin_mask = np.zeros((height, width), dtype=np.uint8)
        
        for lower, upper in self.skin_ranges:
            lower_skin = np.array(lower, dtype=np.uint8)
            upper_skin = np.array(upper, dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            combined_skin_mask = cv2.bitwise_or(combined_skin_mask, skin_mask)
        
        # 4. Combine all masks
        # Prioritize areas with motion and skin color, exclude faces
        final_mask = cv2.bitwise_and(combined_skin_mask, face_mask)
        
        # If we have motion, focus on moving skin areas
        if np.sum(motion_mask) > 1000:  # Threshold for significant motion
            motion_skin_mask = cv2.bitwise_and(final_mask, motion_mask)
            if np.sum(motion_skin_mask) > 500:
                final_mask = motion_skin_mask
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the best hand candidate
        best_contour = self.select_best_hand_contour(contours, frame)
        
        return best_contour, final_mask, faces
        
    def select_best_hand_contour(self, contours, frame):
        """Select the best hand contour using multiple criteria"""
        if not contours:
            self.hand_confidence = max(0, self.hand_confidence - 1)
            return None
            
        # Filter contours by area
        valid_contours = [c for c in contours if cv2.contourArea(c) > 3000]
        
        if not valid_contours:
            self.hand_confidence = max(0, self.hand_confidence - 1)
            return None
            
        best_contour = None
        best_score = 0
        
        for contour in valid_contours:
            score = self.score_hand_contour(contour)
            
            # Bonus for tracking continuity
            if self.last_hand_position is not None:
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    distance = math.sqrt((cx - self.last_hand_position[0])**2 + 
                                       (cy - self.last_hand_position[1])**2)
                    # Bonus for nearby positions (smooth tracking)
                    if distance < 100:
                        score += 2
                    elif distance < 200:
                        score += 1
            
            if score > best_score:
                best_score = score
                best_contour = contour
        
        # Update tracking
        if best_contour is not None and best_score > 3:
            M = cv2.moments(best_contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                self.last_hand_position = (cx, cy)
                self.hand_confidence = min(self.max_confidence, self.hand_confidence + 1)
                return best_contour
        
        self.hand_confidence = max(0, self.hand_confidence - 1)
        return None
        
    def score_hand_contour(self, contour):
        """Score a contour based on how hand-like it is"""
        if contour is None:
            return 0
            
        # Calculate properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Hull and defects
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        
        score = 0
        
        # Area score (prefer medium-sized objects)
        if 3000 < area < 50000:
            score += 2
        elif 1000 < area < 80000:
            score += 1
            
        # Aspect ratio score
        aspect_ratio = w / h if h > 0 else 0
        if 0.4 < aspect_ratio < 1.5:
            score += 2
        elif 0.3 < aspect_ratio < 2.0:
            score += 1
            
        # Solidity score (hands have lower solidity due to fingers)
        solidity = area / hull_area if hull_area > 0 else 0
        if 0.6 < solidity < 0.85:
            score += 2
        elif 0.5 < solidity < 0.9:
            score += 1
            
        # Convexity defects (fingers create defects)
        if len(hull_indices) > 3:
            defects = cv2.convexityDefects(contour, hull_indices)
            if defects is not None:
                significant_defects = 0
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    if d > 1000:  # Significant defect depth
                        significant_defects += 1
                        
                if 3 <= significant_defects <= 8:
                    score += 2
                elif 2 <= significant_defects <= 10:
                    score += 1
                    
        # Perimeter to area ratio
        if perimeter > 0:
            ratio = perimeter**2 / area
            if 15 < ratio < 40:  # Hands typically have this range
                score += 1
                
        return score
    
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
        
        # Improved fingertip detection using defects
        fingertips = self.find_fingertips(contour, defects, wrist)
        
        # Create landmarks for each finger
        for i, tip in enumerate(fingertips[:5]):  # Ensure max 5 fingers
            if i == 0:  # Thumb - different structure
                # Thumb landmarks
                thumb_base = (wrist[0] - w//6, wrist[1] - h//8)
                thumb_mcp = (int(tip[0] + (thumb_base[0] - tip[0]) * 0.7), 
                           int(tip[1] + (thumb_base[1] - tip[1]) * 0.7))
                thumb_ip = (int(tip[0] + (thumb_base[0] - tip[0]) * 0.3), 
                          int(tip[1] + (thumb_base[1] - tip[1]) * 0.3))
                
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
    
    def find_fingertips(self, contour, defects, wrist):
        """Find fingertips using convexity defects and geometry"""
        # Get convex hull points
        hull = cv2.convexHull(contour, returnPoints=True)
        hull_points = [tuple(point[0]) for point in hull]
        
        # Filter points that are above the wrist (potential fingertips)
        candidates = [pt for pt in hull_points if pt[1] < wrist[1] - 20]
        
        # Remove points that are too close to each other
        fingertips = []
        min_distance = 30
        
        # Sort by y-coordinate (topmost first)
        candidates.sort(key=lambda p: p[1])
        
        for candidate in candidates:
            if all(math.sqrt((candidate[0] - tip[0])**2 + (candidate[1] - tip[1])**2) > min_distance 
                   for tip in fingertips):
                fingertips.append(candidate)
                
        # Limit to 5 fingertips and sort by x-coordinate
        fingertips = fingertips[:5]
        fingertips.sort(key=lambda p: p[0])
        
        # Fill with default positions if needed
        x, y, w, h = cv2.boundingRect(contour)
        while len(fingertips) < 5:
            default_x = x + len(fingertips) * w // 5
            default_y = y + 10
            fingertips.append((default_x, default_y))
            
        return fingertips
    
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
        
        # Draw landmarks with different colors for different parts
        colors = {
            0: (255, 0, 0),      # Wrist - red
            1: (255, 165, 0),    # Thumb - orange
            2: (255, 165, 0),
            3: (255, 165, 0),
            4: (255, 165, 0),
            5: (0, 255, 0),      # Index - green
            6: (0, 255, 0),
            7: (0, 255, 0),
            8: (0, 255, 0),
            9: (0, 0, 255),      # Middle - blue
            10: (0, 0, 255),
            11: (0, 0, 255),
            12: (0, 0, 255),
            13: (255, 0, 255),   # Ring - magenta
            14: (255, 0, 255),
            15: (255, 0, 255),
            16: (255, 0, 255),
            17: (255, 255, 0),   # Pinky - cyan
            18: (255, 255, 0),
            19: (255, 255, 0),
            20: (255, 255, 0)
        }
        
        for i, landmark in enumerate(landmarks):
            center = tuple(map(int, landmark))
            color = colors.get(i, (255, 255, 255))
            cv2.circle(frame, center, 4, color, -1)
            # Add landmark numbers for debugging (smaller text)
            cv2.putText(frame, str(i), (center[0] + 6, center[1] + 6), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
        
        return frame

def main():
    print("Starting Advanced Hand Landmark Detection")
    print("Features: Motion detection, multi-range skin detection, hand tracking")
    
    # Initialize detector
    detector = AdvancedHandLandmarkDetector()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Press 'q' to quit, 'd' to toggle debug mode, 'r' to reset tracking")
    debug_mode = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect hand contour
        contour, mask, faces = detector.detect_hand_contour(frame)
        
        # Debug visualization
        if debug_mode:
            cv2.imshow('Detection Mask', mask)
            
            # Show face detection
            debug_frame = frame.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(debug_frame, 'Face (Excluded)', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.imshow('Face Detection', debug_frame)
        
        if contour is not None:
            # Draw contour
            cv2.drawContours(frame, [contour], -1, (0, 255, 255), 2)
            
            # Show contour info
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            score = detector.score_hand_contour(contour)
            
            cv2.putText(frame, f'Area: {int(area)}', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f'Score: {score}', (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f'Confidence: {detector.hand_confidence}', (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Estimate landmarks
            landmarks = detector.estimate_landmarks(contour, frame)
            
            # Draw landmarks and connections
            frame = detector.draw_landmarks(frame, landmarks)
        else:
            cv2.putText(frame, f'Confidence: {detector.hand_confidence}', (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Add instructions
        cv2.putText(frame, "Show your hand to the camera", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if debug_mode:
            cv2.putText(frame, "Debug Mode ON", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.imshow('Advanced Hand Landmark Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_mode = not debug_mode
            if not debug_mode:
                cv2.destroyWindow('Detection Mask')
                cv2.destroyWindow('Face Detection')
        elif key == ord('r'):
            # Reset tracking
            detector.last_hand_position = None
            detector.hand_confidence = 0
            print("Tracking reset")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
