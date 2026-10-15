import cv2
import numpy as np
import os
import json
import pickle
from datetime import datetime
from holistic_detector import HolisticDetector
from pose_classifier import PoseClassifier

class PoseDataCollector:
    def __init__(self, data_dir="pose_data"):
        self.detector = HolisticDetector()
        self.classifier = PoseClassifier()
        self.data_dir = data_dir
        self.current_pose = 0
        self.collected_samples = 0
        self.samples_per_pose = 100
        self.auto_collect = False
        self.collection_delay = 5
        self.frame_counter = 0
        self.awaiting_confirmation = False

        os.makedirs(data_dir, exist_ok=True)

        self.pose_labels = self.classifier.pose_labels

        print("Boxing Pose Data Collector")
        print("Controls:")
        for key, label in self.pose_labels.items():
            print(f"  '{key}' - Set pose to {label}")
        print("  'a' - Toggle auto collection (start/stop)")
        print("  's' - Save collected data to files")
        print("  't' - Train model with collected data")
        print("  'l' - Load previously saved data")
        print("  'q' - Quit")
    
    def collect_data(self):
        print("Initializing webcam...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Camera 0 not available, trying camera 1...")
            cap = cv2.VideoCapture(1)
            
        if not cap.isOpened():
            print("Error: Could not open any webcam")
            return
            
        collected_data = []
        collected_labels = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            self.frame_counter += 1
            
            results = self.detector.detect(frame)
            frame = self.detector.draw_landmarks(frame, results)
            
            landmark_data = self.detector.get_landmark_data(results)
            pose_landmarks = landmark_data.get('pose')
            
            if (self.auto_collect and pose_landmarks is not None and
                self.frame_counter % self.collection_delay == 0):

                features = self.classifier.extract_features(pose_landmarks)
                collected_data.append(features)
                collected_labels.append(self.current_pose)
                self.collected_samples += 1

                if self.collected_samples >= self.samples_per_pose:
                    self.auto_collect = False
                    self.awaiting_confirmation = True

            if self.awaiting_confirmation:
                cv2.putText(frame, "100 samples collected!", (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.putText(frame, "Press ENTER to continue to next pose", (150, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "Press BACKSPACE to re-collect this pose", (140, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "Press S to save and quit", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            auto_status = "ON" if self.auto_collect else "OFF"
            current_pose_name = self.pose_labels.get(self.current_pose, "Unknown")
            cv2.putText(frame, f"Pose: {current_pose_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {self.collected_samples}/{self.samples_per_pose}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Auto Collect: {auto_status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if self.auto_collect and pose_landmarks is not None:
                cv2.circle(frame, (50, 130), 20, (0, 255, 0), -1)
            elif self.auto_collect:
                cv2.circle(frame, (50, 130), 20, (0, 0, 255), -1)
            
            cv2.imshow('Boxing Pose Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF

            if self.awaiting_confirmation:
                if key == 13:
                    print(f"Completed collecting for {self.pose_labels[self.current_pose]}")
                    self.collected_samples = 0
                    self.current_pose = (self.current_pose + 1) % len(self.pose_labels)
                    self.awaiting_confirmation = False
                    if self.current_pose == 0:
                        print("All poses completed! You can train the model now.")
                elif key == 8:
                    print(f"Re-collecting: {self.pose_labels[self.current_pose]}")
                    self.collected_samples = 0
                    collected_data = [f for f, l in zip(collected_data, collected_labels) if l != self.current_pose]
                    collected_labels = [l for l in collected_labels if l != self.current_pose]
                    self.awaiting_confirmation = False
                elif key == ord('s'):
                    if len(collected_data) > 0:
                        self.save_data(collected_data, collected_labels)
                    break
            elif key == ord('q'):
                break
            elif key >= ord('0') and key <= ord(str(len(self.pose_labels)-1)):
                self.current_pose = int(chr(key))
                self.collected_samples = 0
                print(f"Switched to: {self.pose_labels[self.current_pose]}")
            elif key == ord('a'):
                self.auto_collect = not self.auto_collect
            elif key == ord('s'):
                if len(collected_data) > 0:
                    self.save_data(collected_data, collected_labels)
            elif key == ord('l'):
                loaded_data, loaded_labels = self.load_data()
                if loaded_data is not None:
                    collected_data = loaded_data
                    collected_labels = loaded_labels
            elif key == ord('t'):
                if len(collected_data) > 0:
                    self.train_model(collected_data, collected_labels)
        
        cap.release()
        cv2.destroyAllWindows()
        self.detector.release()
        
    def train_model(self, X, y):
        if len(X) < 10:
            print("Need at least 10 samples to train. Collect more data!")
            return
        print(f"Training model with {len(X)} samples...")
        self.classifier.compare_models(np.array(X), np.array(y))
        print("Model training completed!")

    def save_data(self, X, y):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        features_file = os.path.join(self.data_dir, f"pose_features_{timestamp}.npy")
        labels_file = os.path.join(self.data_dir, f"pose_labels_{timestamp}.npy")
        
        np.save(features_file, np.array(X))
        np.save(labels_file, np.array(y))
        
        metadata = {
            "timestamp": timestamp,
            "num_samples": len(X),
            "pose_labels": self.pose_labels,
            "collection_date": datetime.now().isoformat()
        }
        
        metadata_file = os.path.join(self.data_dir, f"pose_metadata_{timestamp}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        np.save(os.path.join(self.data_dir, "pose_features_latest.npy"), np.array(X))
        np.save(os.path.join(self.data_dir, "pose_labels_latest.npy"), np.array(y))
        print("Data saved.")

    def load_data(self, filename_prefix="latest"):
        features_file = os.path.join(self.data_dir, f"pose_features_{filename_prefix}.npy")
        labels_file = os.path.join(self.data_dir, f"pose_labels_{filename_prefix}.npy")
        
        if not os.path.exists(features_file) or not os.path.exists(labels_file):
            print("No saved data found.")
            return None, None
            
        X = np.load(features_file)
        y = np.load(labels_file)
        print(f"Loaded {len(X)} samples")
        return X.tolist(), y.tolist()

    def load_sample_data(self):
        X, y = self.load_data("latest")
        if X is not None:
            self.classifier.train_model(X, y)
        return X, y

if __name__ == "__main__":
    collector = PoseDataCollector()
    collector.collect_data()
