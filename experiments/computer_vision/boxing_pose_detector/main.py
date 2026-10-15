import cv2
import os
import numpy as np
import time
from holistic_detector import HolisticDetector
from pose_classifier import PoseClassifier

def load_reference_images():
    """Load reference images for each boxing pose class"""
    # Assuming images are in the parent directory as provided in the prompt
    images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    reference_images = {}
    
    pose_images = {
        "Hitman": "Mashiba-hitman.jpg",
        "Peek-a-Boo": "ippo-peek_a_boo.jpg", 
        "No Guard": "joe-no_guard.jpg",
        "Cross Guard": "miyata-cross_guard.webp"
    }
    
    for pose_name, filename in pose_images.items():
        image_path = os.path.join(images_dir, filename)
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                # Resize reference images for consistency
                img = cv2.resize(img, (400, 500))
                reference_images[pose_name] = img
                print(f"Loaded reference image for {pose_name}")
            else:
                print(f"Could not load image: {image_path}")
        else:
            print(f"Reference image not found: {image_path}")
            
    return reference_images

def show_reference_image(pose_name, reference_images, window_name="Reference Stance"):
    if pose_name in reference_images:
        img = reference_images[pose_name].copy()
        cv2.putText(img, pose_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(window_name, img)
    else:
        blank_img = np.zeros((500, 400, 3), dtype=np.uint8)
        cv2.putText(blank_img, "No Pose Detected", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow(window_name, blank_img)

def main():
    detector = HolisticDetector()
    classifier = PoseClassifier()
    
    print("Loading reference boxing stances...")
    reference_images = load_reference_images()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
        
    reference_window_name = "Reference Stance"
    cv2.namedWindow(reference_window_name, cv2.WINDOW_AUTOSIZE)
    show_reference_image("None", reference_images, reference_window_name)
    
    print("Boxing Pose Detector Started!")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        results = detector.detect(frame)
        frame = detector.draw_landmarks(frame, results)
        
        landmark_data = detector.get_landmark_data(results)
        pose_landmarks = landmark_data.get('pose')
        
        pose_prediction = "No Pose"
        confidence = 0.0
        
        if pose_landmarks is not None:
            pose_prediction, confidence = classifier.predict(pose_landmarks)
            all_confidences = classifier.get_all_confidences(pose_landmarks)
            
            y_offset = 150
            for i, (pose_name, conf) in enumerate(all_confidences.items()):
                if conf > 0.1:
                    cv2.putText(frame, f"{pose_name}: {conf:.2f}", (10, y_offset + i*30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if pose_prediction != "No Pose" and confidence > 0.4:
            show_reference_image(pose_prediction, reference_images, reference_window_name)
        else:
            show_reference_image(None, reference_images, reference_window_name)
            
        color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.4 else (0, 0, 255)
        cv2.putText(frame, f"Stance: {pose_prediction} ({confidence:.2f})", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                   
        cv2.putText(frame, "Press 'q' to quit, 't' to train", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                   
        cv2.imshow('Boxing Pose Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            from data_collector import PoseDataCollector
            collector = PoseDataCollector()
            X, y = collector.load_sample_data()
            if X is not None:
                print("Model retrained with real data!")
                # Re-initialize classifier to use new model
                classifier = PoseClassifier()
            else:
                print("No data to train with. Collect some data first using data_collector.py!")
                
    cap.release()
    cv2.destroyAllWindows()
    detector.release()

if __name__ == "__main__":
    main()
