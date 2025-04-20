import os
import mss
import cv2
import numpy as np
import time
import glob
import json
from datetime import datetime
from ultralytics import YOLO
from openpyxl import Workbook

def main():
    # Get the user's home directory
    home_dir = os.path.expanduser("~")

    # Define dynamic paths
    save_path = os.path.join(home_dir, "yolo_detection")
    screenshots_path = os.path.join(save_path, "screenshots")
    
    # Ensure necessary directories exist
    os.makedirs(screenshots_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "runs", "detect"), exist_ok=True)

    # Define pattern classes
    classes = ['Head and shoulders bottom', 'Head and shoulders top', 'M_Head', 'StockLine', 'Triangle', 'W_Bottom']

    # Read config if available
    config = {"input_size": 640}
    if os.path.exists("config.json"):
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
            print(f"Loaded configuration: {config}")
        except Exception as e:
            print(f"Error loading config: {e}")

    # Load YOLOv8 model - set correct path
    model_path = "model.pt"
    
    print(f"Looking for model at: {os.path.abspath(model_path)}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    print("Model loaded successfully!")

    # Define screen capture region 
    monitor = {"top": 0, "left": 0, "width": 800, "height": 600}
    print(f"Initial screen capture area: {monitor}")
    
    # Allow user to adjust the capture area
    print("\nAdjust capture area? Current setting:")
    print(f"Top: {monitor['top']}, Left: {monitor['left']}, Width: {monitor['width']}, Height: {monitor['height']}")
    
    if input("Adjust capture area? (y/n): ").lower() == 'y':
        try:
            monitor["top"] = int(input(f"Top (current: {monitor['top']}): ") or monitor["top"])
            monitor["left"] = int(input(f"Left (current: {monitor['left']}): ") or monitor["left"])
            monitor["width"] = int(input(f"Width (current: {monitor['width']}): ") or monitor["width"])
            monitor["height"] = int(input(f"Height (current: {monitor['height']}): ") or monitor["height"])
        except ValueError:
            print("Invalid input, using default settings")
    
    print(f"Using screen capture area: {monitor}")

    # Create Excel file for results
    excel_file = os.path.join(save_path, "classification_results.xlsx")
    wb = Workbook()
    ws = wb.active
    ws.append(["Timestamp", "Screenshot Path", "Pattern", "Confidence"])

    # Initialize capture interval and total captures
    interval = 60  # seconds between captures
    total_captures = 30  # total number of captures
    
    # Allow adjustment of interval and total captures
    try:
        new_interval = input(f"Seconds between captures (current: {interval}): ")
        if new_interval:
            interval = int(new_interval)
            
        new_total = input(f"Total number of captures (current: {total_captures}): ")
        if new_total:
            total_captures = int(new_total)
    except ValueError:
        print("Invalid input, using default settings")
    
    print(f"Capture interval: {interval} seconds")
    print(f"Total captures: {total_captures}")
    
    # Create a named window for preview
    cv2.namedWindow("Screen Capture Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Screen Capture Preview", 800, 600)
    
    # Show initial capture to verify area
    with mss.mss() as sct:
        # Take initial screenshot for preview
        sct_img = sct.grab(monitor)
        preview_img = np.array(sct_img)
        preview_img = cv2.cvtColor(preview_img, cv2.COLOR_BGRA2BGR)
        
        # Display for verification
        cv2.imshow("Screen Capture Preview", preview_img)
        
        print("\nVerify that the capture area shows your TradingView chart correctly.")
        print("Press any key to continue, or Esc to exit...")
        
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # Esc key
            print("Setup cancelled by user")
            cv2.destroyAllWindows()
            return
        
        # Start capture and detection
        print("\nStarting pattern detection with preview...")
        print(f"Press 'q' at any time to stop")
        
        try:
            for capture_num in range(total_captures):
                # Capture screenshot
                start_time = time.time()
                
                # Take screenshot
                sct_img = sct.grab(monitor)
                img = np.array(sct_img)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                # Save screenshot
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                image_name = f"capture_{timestamp}.png"
                image_path = os.path.join(screenshots_path, image_name)
                cv2.imwrite(image_path, img)
                
                # Show preview
                display_img = img.copy()
                cv2.putText(
                    display_img,
                    f"Capture {capture_num+1}/{total_captures} - {timestamp}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Run model prediction
                print(f"\nProcessing capture {capture_num+1}/{total_captures}")
                try:
                    results = model.predict(
                        source=image_path,
                        imgsz=config["input_size"],
                        conf=0.25  # Confidence threshold
                    )
                    
                    result = results[0]
                    detected = False
                    
                    # Process detections
                    if len(result.boxes) > 0:
                        for i, box in enumerate(result.boxes):
                            # Get class and confidence
                            cls_idx = int(box.cls[0].item())
                            conf = round(box.conf[0].item(), 2)
                            label = classes[cls_idx]
                            
                            print(f"Detected: {label} (Confidence: {conf*100:.1f}%)")
                            
                            # Add to display image
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(
                                display_img,
                                f"{label}: {conf:.2f}",
                                (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2
                            )
                            
                            # Log to Excel
                            ws.append([timestamp, image_path, label, conf])
                            detected = True
                            
                    if not detected:
                        print("No patterns detected")
                        ws.append([timestamp, image_path, "No pattern detected", 0.0])
                
                except Exception as e:
                    print(f"Error during prediction: {e}")
                    ws.append([timestamp, image_path, "Error", 0.0])
                
                # Show the image with detections
                cv2.imshow("Screen Capture Preview", display_img)
                
                # Save Excel
                wb.save(excel_file)
                
                # Calculate time until next capture
                elapsed_time = time.time() - start_time
                wait_time = max(1, interval - int(elapsed_time))
                
                # Wait until next capture, checking for key press every second
                for i in range(wait_time):
                    # Update countdown display
                    countdown_img = display_img.copy()
                    cv2.putText(
                        countdown_img,
                        f"Next capture in {wait_time - i} seconds...",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )
                    cv2.imshow("Screen Capture Preview", countdown_img)
                    
                    # Check for key press
                    key = cv2.waitKey(1000) & 0xFF
                    if key == ord('q'):
                        print("\nStopped by user")
                        break
                
                # Check if we need to exit the loop
                if key == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("\nProcess interrupted by user")
        
        finally:
            # Clean up
            cv2.destroyAllWindows()
            
            # Save final results
            wb.save(excel_file)
            
            print("\nDetection complete.")
            print(f"Results saved to: {excel_file}")
            print(f"Screenshots saved to: {screenshots_path}")

if __name__ == "__main__":
    main()