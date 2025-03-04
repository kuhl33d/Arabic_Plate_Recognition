import os
import sys
import cv2
import numpy as np
from sklearn.base import BaseEstimator
from sklearn import model_selection as ms
from sklearn.metrics import precision_score
import logging
import opencv

class FaceRecognizer(BaseEstimator):
    def __init__(self):
        self.model = cv2.face.LBPHFaceRecognizer_create()

    def fit(self, X, y):
        self.model.train(np.array(X), np.array(y))
        return self

    def predict(self, T):
        if T.ndim == 3:  # If T is a list of images
            return [self.model.predict(T[i])[0] for i in range(T.shape[0])]
        else:  # If T is a single image
            return self.model.predict(T)[0]

def validate_model():
    try:
        # Load images and labels
        X, y = opencv.load_images_from_db()
        if len(X) == 0:
            logging.error("No images available for validation")
            return
            
        y = np.asarray(y, dtype=np.int32)
        
        # Create cross-validation splits
        cv = ms.StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        # Initialize model
        estimator = FaceRecognizer()
        
        # Perform cross-validation
        precision_scores = ms.cross_val_score(
            estimator, X, y, 
            scoring='precision_weighted',
            cv=cv
        )
        
        # Print results
        logging.info("Individual precision scores: %s", precision_scores)
        logging.info("Average precision score: %f", np.mean(precision_scores))
        
        return np.mean(precision_scores)
        
    except Exception as e:
        logging.error("Validation error: %s", str(e))
        return None

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    validate_model()