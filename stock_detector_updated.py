import os
import mss
import cv2
import numpy as np
import time
import glob
import json
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

    # Load YOLOv8 model - Change path if needed
    model_path = "model.pt"
    
    print(f"Looking for model at: {os.path.abspath(model_path)}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    print("Model loaded successfully!")

    # Define screen capture region - Modify as needed for your screen
    monitor = {"top": 0, "left": 0, "width": 800, "height": 600}
    print(f"Screen capture area: {monitor}")

    # Create an Excel file for results
    excel_file = os.path.join(save_path, "classification_results.xlsx")
    wb = Workbook()
    ws = wb.active
    ws.append(["Timestamp", "Predicted Image Path", "Pattern", "Confidence"])

    # Initialize video writer
    video_path = os.path.join(save_path, "annotated_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 0.5
    video_writer = None

    print(f"Starting stock pattern detection...")
    print(f"Taking screenshots every 60 seconds.")
    print(f"Results will be saved to {excel_file}")
    print(f"Video will be saved to {video_path}")
    print(f"Press Ctrl+C to stop at any time.")

    # Set total captures (30 minutes)
    total_captures = 30

    with mss.mss() as sct:
        try:
            for capture_num in range(total_captures):
                # Capture screenshot
                print(f"\nCapture {capture_num+1}/{total_captures}")
                sct_img = sct.grab(monitor)
                img = np.array(sct_img)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                # Save screenshot
                timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                image_name = f"capture_{timestamp}.png"
                image_path = os.path.join(screenshots_path, image_name)
                cv2.imwrite(image_path, img)
                print(f"Screenshot saved: {image_path}")

                # Run inference with YOLOv8
                print("Running pattern detection...")
                results = model.predict(
                    source=image_path,
                    save=True,
                    imgsz=config["input_size"],
                    conf=0.25  # Confidence threshold
                )
                
                # Process results
                result = results[0]
                predict_dir = os.path.dirname(result.save_dir)
                
                # Find the processed image
                processed_images = glob.glob(os.path.join(predict_dir, "*.jpg"))
                if processed_images:
                    processed_image = max(processed_images, key=os.path.getmtime)
                else:
                    processed_image = image_path
                
                # Extract detected patterns
                if len(result.boxes) > 0:
                    for box in result.boxes:
                        # Get class and confidence
                        cls_idx = int(box.cls[0].item())
                        conf = round(box.conf[0].item(), 2)
                        label = classes[cls_idx]
                        
                        print(f"Detected: {label} (Confidence: {conf*100:.1f}%)")
                        
                        # Log to Excel
                        ws.append([timestamp, processed_image, label, conf])
                else:
                    print("No patterns detected")
                    ws.append([timestamp, processed_image, "No pattern detected", 0.0])
                
                # Save Excel
                wb.save(excel_file)
                
                # Create video frame
                annotated_img = cv2.imread(processed_image)
                if annotated_img is not None:
                    # Add timestamp 
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(
                        annotated_img, 
                        f"Time: {timestamp.replace('_', ' ')}", 
                        (10, 30), 
                        font, 0.7, (0, 255, 0), 2, cv2.LINE_AA
                    )
                    
                    # Initialize video writer if needed
                    if video_writer is None:
                        height, width, _ = annotated_img.shape
                        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                    
                    # Add to video
                    video_writer.write(annotated_img)
                
                # Wait for next capture (skip wait on last iteration)
                if capture_num < total_captures - 1:
                    print(f"Waiting 60 seconds for next capture...")
                    time.sleep(60)  # Wait 60 seconds
                
        except KeyboardInterrupt:
            print("\nProcess interrupted by user")
        
        finally:
            # Save final results
            wb.save(excel_file)
            if video_writer is not None:
                video_writer.release()
            
            print(f"\nDetection complete.")
            print(f"Results saved to: {excel_file}")
            print(f"Video saved to: {video_path}")
            print(f"Screenshots saved to: {screenshots_path}")

if __name__ == "__main__":
    main()