import cv2
import numpy as np

class SkinCalibrator:
    def __init__(self):
        self.hsv_lower = [0, 20, 70]
        self.hsv_upper = [20, 255, 255]
        self.ycrcb_lower = [0, 133, 77]
        self.ycrcb_upper = [255, 173, 127]
        
        # Create trackbars for HSV
        cv2.namedWindow('HSV Controls')
        cv2.createTrackbar('H_min', 'HSV Controls', self.hsv_lower[0], 180, lambda x: None)
        cv2.createTrackbar('S_min', 'HSV Controls', self.hsv_lower[1], 255, lambda x: None)
        cv2.createTrackbar('V_min', 'HSV Controls', self.hsv_lower[2], 255, lambda x: None)
        cv2.createTrackbar('H_max', 'HSV Controls', self.hsv_upper[0], 180, lambda x: None)
        cv2.createTrackbar('S_max', 'HSV Controls', self.hsv_upper[1], 255, lambda x: None)
        cv2.createTrackbar('V_max', 'HSV Controls', self.hsv_upper[2], 255, lambda x: None)
        
        # Create trackbars for YCrCb
        cv2.namedWindow('YCrCb Controls')
        cv2.createTrackbar('Y_min', 'YCrCb Controls', self.ycrcb_lower[0], 255, lambda x: None)
        cv2.createTrackbar('Cr_min', 'YCrCb Controls', self.ycrcb_lower[1], 255, lambda x: None)
        cv2.createTrackbar('Cb_min', 'YCrCb Controls', self.ycrcb_lower[2], 255, lambda x: None)
        cv2.createTrackbar('Y_max', 'YCrCb Controls', self.ycrcb_upper[0], 255, lambda x: None)
        cv2.createTrackbar('Cr_max', 'YCrCb Controls', self.ycrcb_upper[1], 255, lambda x: None)
        cv2.createTrackbar('Cb_max', 'YCrCb Controls', self.ycrcb_upper[2], 255, lambda x: None)
        
    def get_hsv_values(self):
        """Get current HSV values from trackbars"""
        h_min = cv2.getTrackbarPos('H_min', 'HSV Controls')
        s_min = cv2.getTrackbarPos('S_min', 'HSV Controls')
        v_min = cv2.getTrackbarPos('V_min', 'HSV Controls')
        h_max = cv2.getTrackbarPos('H_max', 'HSV Controls')
        s_max = cv2.getTrackbarPos('S_max', 'HSV Controls')
        v_max = cv2.getTrackbarPos('V_max', 'HSV Controls')
        
        return ([h_min, s_min, v_min], [h_max, s_max, v_max])
        
    def get_ycrcb_values(self):
        """Get current YCrCb values from trackbars"""
        y_min = cv2.getTrackbarPos('Y_min', 'YCrCb Controls')
        cr_min = cv2.getTrackbarPos('Cr_min', 'YCrCb Controls')
        cb_min = cv2.getTrackbarPos('Cb_min', 'YCrCb Controls')
        y_max = cv2.getTrackbarPos('Y_max', 'YCrCb Controls')
        cr_max = cv2.getTrackbarPos('Cr_max', 'YCrCb Controls')
        cb_max = cv2.getTrackbarPos('Cb_max', 'YCrCb Controls')
        
        return ([y_min, cr_min, cb_min], [y_max, cr_max, cb_max])
        
    def detect_skin_hsv(self, frame, lower, upper):
        """Detect skin using HSV"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        return mask
        
    def detect_skin_ycrcb(self, frame, lower, upper):
        """Detect skin using YCrCb"""
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, np.array(lower), np.array(upper))
        return mask

def main():
    print("Skin Detection Calibrator")
    print("=========================")
    print("Instructions:")
    print("1. Position your hand clearly in the camera view")
    print("2. Adjust the trackbars to get the best skin detection")
    print("3. Press 'p' to print optimal values")
    print("4. Press 'r' to reset to defaults")
    print("5. Press 'q' to quit")
    print()
    
    calibrator = SkinCalibrator()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    mode = 'hsv'  # Start with HSV mode
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        
        if mode == 'hsv':
            lower, upper = calibrator.get_hsv_values()
            mask = calibrator.detect_skin_hsv(frame, lower, upper)
            cv2.putText(frame, "HSV Mode - Press 'c' to switch to YCrCb", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            lower, upper = calibrator.get_ycrcb_values()
            mask = calibrator.detect_skin_ycrcb(frame, lower, upper)
            cv2.putText(frame, "YCrCb Mode - Press 'c' to switch to HSV", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
        
        # Find contours for feedback
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 1000]
        
        # Draw contours on original frame
        cv2.drawContours(frame, valid_contours, -1, (0, 255, 0), 2)
        
        # Show statistics
        skin_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        skin_percentage = (skin_pixels / total_pixels) * 100
        
        cv2.putText(frame, f'Skin detection: {skin_percentage:.1f}%', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f'Contours found: {len(valid_contours)}', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if len(valid_contours) > 0:
            largest_area = max([cv2.contourArea(c) for c in valid_contours])
            cv2.putText(frame, f'Largest area: {int(largest_area)}', (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Show your hand clearly", (10, frame.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Adjust trackbars for best detection", (10, frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Camera Feed', frame)
        cv2.imshow('Skin Mask', mask)
        cv2.imshow('Cleaned Mask', mask_clean)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            mode = 'ycrcb' if mode == 'hsv' else 'hsv'
            print(f"Switched to {mode.upper()} mode")
        elif key == ord('p'):
            if mode == 'hsv':
                lower, upper = calibrator.get_hsv_values()
                print(f"\nOptimal HSV values:")
                print(f"Lower: {lower}")
                print(f"Upper: {upper}")
                print(f"Code: lower_skin = np.array({lower}, dtype=np.uint8)")
                print(f"Code: upper_skin = np.array({upper}, dtype=np.uint8)")
            else:
                lower, upper = calibrator.get_ycrcb_values()
                print(f"\nOptimal YCrCb values:")
                print(f"Lower: {lower}")
                print(f"Upper: {upper}")
                print(f"Code: lower_skin = np.array({lower}, dtype=np.uint8)")
                print(f"Code: upper_skin = np.array({upper}, dtype=np.uint8)")
        elif key == ord('r'):
            print("Reset to default values")
            if mode == 'hsv':
                cv2.setTrackbarPos('H_min', 'HSV Controls', 0)
                cv2.setTrackbarPos('S_min', 'HSV Controls', 20)
                cv2.setTrackbarPos('V_min', 'HSV Controls', 70)
                cv2.setTrackbarPos('H_max', 'HSV Controls', 20)
                cv2.setTrackbarPos('S_max', 'HSV Controls', 255)
                cv2.setTrackbarPos('V_max', 'HSV Controls', 255)
            else:
                cv2.setTrackbarPos('Y_min', 'YCrCb Controls', 0)
                cv2.setTrackbarPos('Cr_min', 'YCrCb Controls', 133)
                cv2.setTrackbarPos('Cb_min', 'YCrCb Controls', 77)
                cv2.setTrackbarPos('Y_max', 'YCrCb Controls', 255)
                cv2.setTrackbarPos('Cr_max', 'YCrCb Controls', 173)
                cv2.setTrackbarPos('Cb_max', 'YCrCb Controls', 127)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
