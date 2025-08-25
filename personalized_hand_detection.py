import cv2
import numpy as np
import math

class PersonalizedHandDetector:
    def __init__(self):
        # Define hand landmark connections
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]
        
        # Your calibrated optimal skin detection values
        self.calibrated_hsv_lower = np.array([0, 132, 0], dtype=np.uint8)
        self.calibrated_hsv_upper = np.array([56, 255, 255], dtype=np.uint8)
        
        # Backup ranges for fallback
        self.backup_ranges = [
            ([0, 120, 0], [60, 255, 255]),    # Slightly wider than calibrated
            ([0, 100, 0], [70, 255, 255]),    # Even wider for difficult lighting
        ]
        
        print("Personalized hand detector initialized with your calibrated values")
        print(f"HSV Range: {self.calibrated_hsv_lower} to {self.calibrated_hsv_upper}")
        
    def detect_skin_calibrated(self, frame):
        """Primary skin detection using calibrated values"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.calibrated_hsv_lower, self.calibrated_hsv_upper)
        return mask
        
    def detect_skin_backup(self, frame):
        """Backup skin detection with wider ranges"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        for lower, upper in self.backup_ranges:
            lower_skin = np.array(lower, dtype=np.uint8)
            upper_skin = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
            
        return combined_mask
        
    def detect_hand_contour(self, frame):
        """Detect hand contour using calibrated values with fallback"""
        # Try calibrated values first
        skin_mask = self.detect_skin_calibrated(frame)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area
        valid_contours = [c for c in contours if cv2.contourArea(c) > 1500]
        
        # If no good contours found with calibrated values, try backup
        if not valid_contours:
            backup_mask = self.detect_skin_backup(frame)
            backup_mask = cv2.morphologyEx(backup_mask, cv2.MORPH_OPEN, kernel)
            backup_mask = cv2.morphologyEx(backup_mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(backup_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [c for c in contours if cv2.contourArea(c) > 1500]
            skin_mask = backup_mask  # Use backup mask for display
            
        if not valid_contours:
            return None, skin_mask, False
            
        # Find the best contour
        best_contour = self.select_best_contour(valid_contours, frame)
        used_calibrated = len([c for c in contours if cv2.contourArea(c) > 1500]) > 0
        
        return best_contour, skin_mask, used_calibrated
        
    def select_best_contour(self, contours, frame):
        """Select the most hand-like contour"""
        if not contours:
            return None
            
        frame_height, frame_width = frame.shape[:2]
        
        scored_contours = []
        for contour in contours:
            score = self.score_contour(contour, frame_width, frame_height)
            scored_contours.append((contour, score))
            
        if not scored_contours:
            return None
            
        # Return the highest scoring contour
        best_contour = max(scored_contours, key=lambda x: x[1])
        
        # Lower threshold since we have calibrated values
        if best_contour[1] > 1:
            return best_contour[0]
        else:
            return None
            
    def score_contour(self, contour, frame_width, frame_height):
        """Score how hand-like a contour is"""
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        score = 0
        
        # Area score (optimized for calibrated detection)
        if 2000 < area < 40000:
            score += 3
        elif 1500 < area < 60000:
            score += 2
        elif 1000 < area < 80000:
            score += 1
            
        # Aspect ratio
        aspect_ratio = w / h if h > 0 else 0
        if 0.5 < aspect_ratio < 1.3:
            score += 2
        elif 0.4 < aspect_ratio < 1.6:
            score += 1
            
        # Position preference (center-right area)
        center_x = x + w // 2
        if frame_width * 0.3 < center_x < frame_width * 0.8:
            score += 1
            
        # Convexity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if 0.6 < solidity < 0.9:
                score += 2
            elif 0.5 < solidity < 0.95:
                score += 1
                
        return score
        
    def estimate_landmarks(self, contour, frame):
        """Estimate hand landmarks"""
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
        
        # Wrist (landmark 0) - bottom center
        wrist = (cx, y + h - 15)
        landmarks.append(wrist)
        
        # Find fingertips using convex hull and defects
        fingertips = self.find_fingertips(contour, wrist)
        
        # Create landmarks for each finger
        for i, tip in enumerate(fingertips[:5]):
            if i == 0:  # Thumb
                thumb_base = (wrist[0] - w//5, wrist[1] - h//6)
                thumb_mcp = (int(tip[0] + (thumb_base[0] - tip[0]) * 0.7), 
                           int(tip[1] + (thumb_base[1] - tip[1]) * 0.7))
                thumb_ip = (int(tip[0] + (thumb_base[0] - tip[0]) * 0.3), 
                          int(tip[1] + (thumb_base[1] - tip[1]) * 0.3))
                landmarks.extend([thumb_base, thumb_mcp, thumb_ip, tip])
            else:  # Other fingers
                finger_base_x = wrist[0] + (i - 2.5) * w // 5
                finger_base_y = wrist[1] - h // 4
                mcp = (int(finger_base_x), int(finger_base_y))
                
                pip_x = int(tip[0] + (mcp[0] - tip[0]) * 0.7)
                pip_y = int(tip[1] + (mcp[1] - tip[1]) * 0.7)
                pip = (pip_x, pip_y)
                
                dip_x = int(tip[0] + (mcp[0] - tip[0]) * 0.35)
                dip_y = int(tip[1] + (mcp[1] - tip[1]) * 0.35)
                dip = (dip_x, dip_y)
                
                landmarks.extend([mcp, pip, dip, tip])
        
        # Ensure 21 landmarks
        while len(landmarks) < 21:
            landmarks.append((cx, cy))
            
        return landmarks[:21]
        
    def find_fingertips(self, contour, wrist):
        """Find fingertips using convex hull"""
        # Get convex hull
        hull = cv2.convexHull(contour, returnPoints=True)
        hull_points = [tuple(point[0]) for point in hull]
        
        # Filter hull points above wrist
        x, y, w, h = cv2.boundingRect(contour)
        candidates = []
        
        for pt in hull_points:
            if pt[1] < wrist[1] - h//8:  # Above palm area
                candidates.append(pt)
        
        # Remove points too close to each other
        filtered_tips = []
        min_distance = 25
        
        for candidate in candidates:
            if all(math.sqrt((candidate[0] - tip[0])**2 + (candidate[1] - tip[1])**2) > min_distance 
                   for tip in filtered_tips):
                filtered_tips.append(candidate)
        
        # Sort by x-coordinate
        filtered_tips.sort(key=lambda p: p[0])
        
        # Ensure 5 fingertips
        while len(filtered_tips) < 5:
            default_x = x + len(filtered_tips) * w // 5
            default_y = y + 20
            filtered_tips.append((default_x, default_y))
            
        return filtered_tips[:5]
        
    def draw_landmarks(self, frame, landmarks):
        """Draw landmarks and connections with enhanced visualization"""
        if landmarks is None:
            return frame
            
        # Enhanced color scheme
        colors = [
            (255, 0, 0),      # Wrist - red
            (255, 100, 0), (255, 140, 0), (255, 180, 0), (255, 220, 0),  # Thumb - orange gradient
            (0, 255, 0), (40, 255, 40), (80, 255, 80), (120, 255, 120),  # Index - green gradient
            (0, 0, 255), (40, 40, 255), (80, 80, 255), (120, 120, 255),  # Middle - blue gradient
            (255, 0, 255), (255, 40, 255), (255, 80, 255), (255, 120, 255),  # Ring - magenta gradient
            (0, 255, 255), (40, 255, 255), (80, 255, 255), (120, 255, 255)   # Pinky - cyan gradient
        ]
        
        # Draw connections with varying thickness
        for connection in self.HAND_CONNECTIONS:
            if connection[0] < len(landmarks) and connection[1] < len(landmarks):
                pt1 = tuple(map(int, landmarks[connection[0]]))
                pt2 = tuple(map(int, landmarks[connection[1]]))
                
                # Thicker lines for main structure
                thickness = 3 if connection[0] == 0 else 2
                cv2.line(frame, pt1, pt2, (0, 255, 0), thickness)
        
        # Draw landmarks with enhanced styling
        for i, landmark in enumerate(landmarks):
            center = tuple(map(int, landmark))
            color = colors[i] if i < len(colors) else (255, 255, 255)
            
            # Draw landmark with border
            cv2.circle(frame, center, 6, color, -1)
            cv2.circle(frame, center, 7, (0, 0, 0), 1)  # Black border
            
            # Highlight fingertips
            if i in [4, 8, 12, 16, 20]:  # Fingertip indices
                cv2.circle(frame, center, 8, (255, 255, 255), 2)
            
        return frame

def main():
    print("Personalized Hand Detection")
    print("Using your calibrated skin detection values")
    print("HSV Range: [0, 132, 0] to [56, 255, 255]")
    
    detector = PersonalizedHandDetector()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Optimize camera settings
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
    
    print("Controls:")
    print("'q' - quit")
    print("'m' - toggle mask view")
    print("'s' - show statistics")
    
    show_mask = False
    show_stats = False
    detection_count = 0
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        # Detect hand
        contour, mask, used_calibrated = detector.detect_hand_contour(frame)
        
        if show_mask:
            cv2.imshow('Skin Detection Mask', mask)
            
        if contour is not None:
            detection_count += 1
            
            # Draw contour with enhanced styling
            cv2.drawContours(frame, [contour], -1, (0, 255, 255), 3)
            
            # Show detailed contour info
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            score = detector.score_contour(contour, frame.shape[1], frame.shape[0])
            
            # Status text with better visibility
            status_color = (0, 255, 0) if used_calibrated else (0, 165, 255)
            status_text = "CALIBRATED" if used_calibrated else "BACKUP"
            
            cv2.putText(frame, f'Detection: {status_text}', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, f'Area: {int(area)}', (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f'Score: {score}', (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if show_stats:
                detection_rate = (detection_count / frame_count) * 100
                cv2.putText(frame, f'Detection Rate: {detection_rate:.1f}%', (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Estimate and draw landmarks
            landmarks = detector.estimate_landmarks(contour, frame)
            frame = detector.draw_landmarks(frame, landmarks)
            
            # Success indicator
            cv2.putText(frame, "HAND DETECTED!", (200, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "No hand detected", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if show_stats:
                detection_rate = (detection_count / frame_count) * 100
                cv2.putText(frame, f'Detection Rate: {detection_rate:.1f}%', (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Instructions
        cv2.putText(frame, "Personalized Hand Detection", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow('Personalized Hand Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow('Skin Detection Mask')
        elif key == ord('s'):
            show_stats = not show_stats
    
    # Final statistics
    final_detection_rate = (detection_count / frame_count) * 100
    print(f"\nFinal Statistics:")
    print(f"Total frames: {frame_count}")
    print(f"Successful detections: {detection_count}")
    print(f"Detection rate: {final_detection_rate:.1f}%")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
