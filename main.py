import os
print("Current working directory:", os.getcwd())
import mss
import cv2
import numpy as np
import time
import glob
from ultralytics import YOLO
from openpyxl import Workbook
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stock Market Pattern Detection from Screen Capture')
    parser.add_argument('--model', type=str, default="model.pt", help='Path to YOLO model file')
    parser.add_argument('--interval', type=int, default=60, help='Capture interval in seconds')
    parser.add_argument('--top', type=int, default=0, help='Top coordinate of screen capture area')
    parser.add_argument('--left', type=int, default=683, help='Left coordinate of screen capture area')
    parser.add_argument('--width', type=int, default=683, help='Width of screen capture area')
    parser.add_argument('--height', type=int, default=768, help='Height of screen capture area')
    args = parser.parse_args()

    # Get the user's home directory
    home_dir = os.path.expanduser("~")

    # Define dynamic paths
    save_path = os.path.join(home_dir, "yolo_detection")
    screenshots_path = os.path.join(save_path, "screenshots")
    detect_path = os.path.join(save_path, "runs", "detect")

    # Ensure necessary directories exist
    os.makedirs(screenshots_path, exist_ok=True)
    os.makedirs(detect_path, exist_ok=True)

    # Define pattern classes
    classes = ['Head and shoulders bottom', 'Head and shoulders top', 'M_Head', 'StockLine', 'Triangle', 'W_Bottom']

    # Load YOLOv8 model
    model_path = args.model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    print("Model loaded successfully!")

    # Define screen capture region
    monitor = {
        "top": args.top, 
        "left": args.left, 
        "width": args.width, 
        "height": args.height
    }
    print(f"Screen capture area: {monitor}")

    # Create an Excel file
    excel_file = os.path.join(save_path, "classification_results.xlsx")
    wb = Workbook()
    ws = wb.active
    ws.append(["Timestamp", "Predicted Image Path", "Label"])  # Headers

    # Initialize video writer
    video_path = os.path.join(save_path, "annotated_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 0.5  # Adjust frames per second as needed
    video_writer = None

    print(f"Starting screen capture. Press 'q' to quit.")
    print(f"Taking screenshots every {args.interval} seconds.")
    print(f"Results will be saved to {excel_file}")
    print(f"Video will be saved to {video_path}")

    with mss.mss() as sct:
        start_time = time.time()
        last_capture_time = start_time  # Track the last capture time
        frame_count = 0
        
        try:
            while True:
                # Continuously capture the screen
                sct_img = sct.grab(monitor)
                img = np.array(sct_img)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                # Check if interval seconds have passed since last YOLO prediction
                current_time = time.time()
                time_since_last = current_time - last_capture_time
                
                # Display countdown
                countdown = max(0, args.interval - int(time_since_last))
                display_img = img.copy()
                cv2.putText(display_img, f"Next capture in: {countdown}s", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                
                if time_since_last >= args.interval:
                    # Take screenshot for YOLO prediction
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    image_name = f"predicted_images_{timestamp}_{frame_count}.png"
                    image_path = os.path.join(screenshots_path, image_name)
                    cv2.imwrite(image_path, img)
                    print(f"Screenshot taken: {image_path}")

                    # Run YOLO model and get save directory
                    results = model(image_path, save=True)
                    predict_path = results[0].save_dir if results else None

                    # Find the latest annotated image inside predict_path
                    if predict_path and os.path.exists(predict_path):
                        annotated_images = sorted(glob.glob(os.path.join(predict_path, "*.jpg")), 
                                                 key=os.path.getmtime, reverse=True)
                        final_image_path = annotated_images[0] if annotated_images else image_path
                    else:
                        final_image_path = image_path  # Fallback to original image

                    # Determine predicted label
                    if results and results[0].boxes:
                        class_indices = results[0].boxes.cls.tolist()
                        predicted_label = classes[int(class_indices[0])]
                    else:
                        predicted_label = "No pattern detected"

                    # Insert data into Excel (store path instead of image)
                    ws.append([timestamp, final_image_path, predicted_label])

                    # Read the image for video processing
                    annotated_img = cv2.imread(final_image_path)
                    if annotated_img is not None:
                        # Add timestamp and label text to the image
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(annotated_img, f"{timestamp}", (10, 30), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(annotated_img, f"{predicted_label}", (10, 60), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                        
                        # Initialize video writer if not already initialized
                        if video_writer is None:
                            height, width, layers = annotated_img.shape
                            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                        
                        video_writer.write(annotated_img)

                    print(f"Frame {frame_count}: {final_image_path} -> {predicted_label}")
                    frame_count += 1

                    # Update the last capture time
                    last_capture_time = current_time

                    # Save the Excel file after each capture
                    wb.save(excel_file)

                # Display the screen capture with countdown
                cv2.imshow("Stock Pattern Detection - Screen Capture", display_img)

                # Break if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quitting...")
                    break
                
                # Short sleep to reduce CPU usage
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            # Save final Excel file
            wb.save(excel_file)
            print(f"Results saved to {excel_file}")
            
            # Release video writer
            if video_writer is not None:
                video_writer.release()
                print(f"Video saved at {video_path}")

            # Clean up
            cv2.destroyAllWindows()
            print("Cleanup complete")

if __name__ == "__main__":
    main()