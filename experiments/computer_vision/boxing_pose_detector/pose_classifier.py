import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pickle
import os
import mlflow
import mlflow.sklearn
from datetime import datetime
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns

# Load environment variables
load_dotenv()

class PoseClassifier:
    def __init__(self, model_path=None):
        """
        Initialize pose classifier
        
        Args:
            model_path: Path to saved model file
        """
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.pose_labels = {
            0: "Hitman",
            1: "Peek-a-Boo", 
            2: "No Guard",
            3: "Cross Guard",
        }
        self.model_path = model_path or "pose_classifier_model.pkl"
        
        # Set MLflow experiment
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "boxing-pose-classification")
        mlflow.set_experiment(experiment_name)
        
        # Load existing model if available
        if os.path.exists(self.model_path):
            self.load_model()
    
    def extract_features(self, pose_landmarks):
        """
        Extract features from pose landmarks for classification
        
        Args:
            pose_landmarks: Array of pose landmarks (33, 3)
            
        Returns:
            features: Extracted feature vector
        """
        if pose_landmarks is None:
            return np.zeros(20)  # Return zeros if no pose detected
        
        # Convert to numpy array if needed
        landmarks = np.array(pose_landmarks)
        
        # Extract key features for pose classification
        features = []
        
        # 1. Shoulder width (distance between shoulders)
        left_shoulder = landmarks[11]  # Left shoulder
        right_shoulder = landmarks[12]  # Right shoulder
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        features.append(shoulder_width)
        
        # 2. Hip width (distance between hips)
        left_hip = landmarks[23]  # Left hip
        right_hip = landmarks[24]  # Right hip
        hip_width = np.linalg.norm(left_hip - right_hip)
        features.append(hip_width)
        
        # 3. Body height (shoulder to hip distance)
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        body_height = np.linalg.norm(shoulder_center - hip_center)
        features.append(body_height)
        
        # 4. Arm angles (elbow to shoulder to wrist angles)
        left_elbow = landmarks[13]
        left_wrist = landmarks[15]
        left_arm_angle = self._calculate_angle(left_elbow, left_shoulder, left_wrist)
        features.append(left_arm_angle)
        
        right_elbow = landmarks[14]
        right_wrist = landmarks[16]
        right_arm_angle = self._calculate_angle(right_elbow, right_shoulder, right_wrist)
        features.append(right_arm_angle)
        
        # 5. Hand heights relative to shoulders
        left_hand_height = left_shoulder[1] - left_wrist[1]  # Y coordinate difference
        right_hand_height = right_shoulder[1] - right_wrist[1]
        features.extend([left_hand_height, right_hand_height])
        
        # 6. Knee angles
        left_knee = landmarks[25]
        left_ankle = landmarks[27]
        right_knee = landmarks[26]
        right_ankle = landmarks[28]
        
        left_knee_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
        features.extend([left_knee_angle, right_knee_angle])
        
        # 7. Torso orientation (shoulder to hip angle)
        torso_angle = self._calculate_angle(left_shoulder, hip_center, right_hip)
        features.append(torso_angle)
        
        # 8. Arm extension (distance from shoulder to wrist)
        left_arm_extension = np.linalg.norm(left_shoulder - left_wrist)
        right_arm_extension = np.linalg.norm(right_shoulder - right_wrist)
        features.extend([left_arm_extension, right_arm_extension])
        
        # 9. Leg positions relative to hips
        left_leg_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
        right_leg_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
        features.extend([left_leg_angle, right_leg_angle])
        
        # 10. Overall body symmetry
        left_side_avg = np.mean([left_shoulder, left_elbow, left_wrist, left_hip, left_knee], axis=0)
        right_side_avg = np.mean([right_shoulder, right_elbow, right_wrist, right_hip, right_knee], axis=0)
        body_symmetry = np.linalg.norm(left_side_avg - right_side_avg)
        features.append(body_symmetry)
        
        # 11. Hand movement indicators (distance from center)
        center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        left_hand_offset = abs(left_wrist[0] - center_x)
        right_hand_offset = abs(right_wrist[0] - center_x)
        features.extend([left_hand_offset, right_hand_offset])
        
        # 12. Vertical body alignment
        nose = landmarks[0]
        vertical_alignment = abs(nose[0] - (left_shoulder[0] + right_shoulder[0]) / 2)
        features.append(vertical_alignment)
        
        # Specifically for boxing: distance between hands (helps differentiate Peek-a-boo and others)
        hands_distance = np.linalg.norm(left_wrist - right_wrist)
        features.append(hands_distance)
        
        return np.array(features)
    
    def _calculate_angle(self, point1, point2, point3):
        """
        Calculate angle between three points
        """
        vector1 = point1 - point2
        vector2 = point3 - point2
        
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        cos_angle = dot_product / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def train_model(self, X, y):
        """
        Train the pose classification model using MLOps principles
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                "n_estimators": self.model.n_estimators,
                "max_depth": self.model.max_depth,
                "random_state": self.model.random_state,
                "model_type": "RandomForestClassifier",
                "features_count": len(X[0]) if len(X) > 0 else 0
            })
            
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            
            print(f"Model trained with accuracy: {accuracy:.2f}")
            print("\nClassification Report:")
            
            unique_labels = np.unique(y)
            target_names = [self.pose_labels.get(label, f"Class_{label}") for label in unique_labels]
            
            report = classification_report(y_test, y_pred, 
                                      labels=unique_labels,
                                      target_names=target_names)
            print(report)
            
            # Log classification report as artifact
            report_path = "classification_report.txt"
            with open(report_path, "w") as f:
                f.write(report)
            mlflow.log_artifact(report_path)
            
            # Log model
            mlflow.sklearn.log_model(self.model, "boxing_pose_classifier_model")
            
            # Log confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=target_names, yticklabels=target_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            cm_path = "confusion_matrix.png"
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)
            plt.close()
            
            # Save model locally
            self.save_model()
            
            # Remove temp files
            for path in [report_path, cm_path]:
                if os.path.exists(path):
                    os.remove(path)
    
    def predict(self, pose_landmarks):
        if not self._is_model_trained():
            return "No Model", 0.0
            
        features = self.extract_features(pose_landmarks)
        features = features.reshape(1, -1)

        try:
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            confidence = np.max(probabilities)
        except Exception:
            return "No Model", 0.0
        
        pose_name = self.pose_labels.get(prediction, "Unknown")
        return pose_name, confidence
    
    def get_all_confidences(self, pose_landmarks):
        if not self._is_model_trained():
            return {}
        
        features = self.extract_features(pose_landmarks)
        features = features.reshape(1, -1)

        try:
            probabilities = self.model.predict_proba(features)[0]
        except Exception:
            return {}
        
        confidences = {}
        for i, prob in enumerate(probabilities):
            if i in self.pose_labels:
                pose_name = self.pose_labels[i]
                confidences[pose_name] = prob
        
        return confidences

    def _is_model_trained(self):
        if self.model is None:
            return False

        try:
            check_is_fitted(self.model)
            return True
        except (NotFittedError, TypeError, ValueError, AttributeError):
            return False
    
    def save_model(self):
        if self.model is not None:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

    def compare_models(self, X, y):
        """
        Compare RandomForest, XGBoost and LightGBM models
        Returns the best model based on accuracy
        """
        unique_labels = np.unique(y)
        target_names = [self.pose_labels.get(label, f"Class_{label}") for label in unique_labels]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            "XGBoost": XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                reg_lambda=1.0,
                reg_alpha=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss',
                verbosity=0
            ),
            "LightGBM": LGBMClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                reg_lambda=1.0,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )
        }

        results = {}
        best_accuracy = 0
        best_model_name = None
        best_model = None

        print("=" * 60)
        print("MODEL COMPARISON: RandomForest vs XGBoost vs LightGBM")
        print("=" * 60)

        for name, model in models.items():
            print(f"\n{'=' * 20} {name} {'=' * 20}")

            with mlflow.start_run(run_name=name):
                mlflow.log_params({
                    "model_type": name,
                    "n_estimators": model.n_estimators if hasattr(model, 'n_estimators') else 'N/A',
                    "max_depth": model.max_depth if hasattr(model, 'max_depth') else 'N/A',
                    "random_state": model.random_state if hasattr(model, 'random_state') else 'N/A'
                })

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                mlflow.log_metric("accuracy", accuracy)

                print(f"Accuracy: {accuracy:.4f}")
                print("\nClassification Report:")
                report = classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names)
                print(report)

                cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=target_names, yticklabels=target_names)
                plt.title(f'Confusion Matrix - {name}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                cm_path = f"confusion_matrix_{name.replace(' ', '_')}.png"
                plt.savefig(cm_path)
                mlflow.log_artifact(cm_path)
                plt.close()

                results[name] = {
                    "model": model,
                    "accuracy": accuracy,
                    "report": report
                }

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_name = name
                    best_model = model

            os.remove(f"confusion_matrix_{name.replace(' ', '_')}.png") if os.path.exists(f"confusion_matrix_{name.replace(' ', '_')}.png") else None

        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        for name, result in results.items():
            marker = " <-- BEST" if name == best_model_name else ""
            print(f"{name}: {result['accuracy']:.4f}{marker}")

        print(f"\nBest Model: {best_model_name} with accuracy {best_accuracy:.4f}")
        self.model = best_model
        self.save_model()
        mlflow.sklearn.log_model(best_model, f"best_model_{best_model_name.replace(' ', '_')}")

        return best_model_name, best_accuracy, results
