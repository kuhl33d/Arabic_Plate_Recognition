import os
import sys
import cv2
import numpy as np
import logging
import shutil
from peewee import *
import datetime

# Constants
MODEL_FILE = "data/models/model.mdl"
IMAGE_DIR = "data/images"
DATABASE_FILE = "data/images.db"
FACE_CASCADE_FILE = "data/haarcascade_frontalface_alt.xml"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Database setup
db = SqliteDatabase(DATABASE_FILE, pragmas={
    'journal_mode': 'wal',
    'cache_size': -1024 * 64
})

class DatabaseConnection:
    def __enter__(self):
        if db.is_closed():
            db.connect()
        return db

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not db.is_closed():
            db.close()

class BaseModel(Model):
    class Meta:
        database = db

class Label(BaseModel):
    name = CharField(unique=True)
    created_at = DateTimeField(default=datetime.datetime.now)

    def persist(self):
        try:
            path = os.path.join(IMAGE_DIR, self.name)
            
            # Clear existing directory if it has 10 or more images
            if os.path.exists(path) and len(os.listdir(path)) >= 10:
                shutil.rmtree(path)

            # Create directory if it doesn't exist
            if not os.path.exists(path):
                os.makedirs(path)
                logging.info(f"Created directory for user: {self.name}")

            return self
        except Exception as e:
            logging.error(f"Error persisting label {self.name}: {str(e)}")
            raise

class Image(BaseModel):
    path = CharField(unique=True)
    label = ForeignKeyField(Label, backref='images')
    created_at = DateTimeField(default=datetime.datetime.now)

    def persist(self, cv_image):
        try:
            path = os.path.join(IMAGE_DIR, self.label.name)
            nr_of_images = len(os.listdir(path))
            
            if nr_of_images >= 10:
                return 'Done'

            # Detect and process face
            faces = detect_faces(cv_image)
            if not faces:
                return 'No face detected'

            # Process the first detected face
            face = faces[0]
            x, y, w, h = map(int, face)
            
            # Crop and preprocess face
            face_img = cv_image[y:y+h, x:x+w]
            gray_face = to_grayscale(face_img)
            resized_face = cv2.resize(gray_face, (100, 100))

            # Save the processed face image
            filename = f"{nr_of_images}.jpg"
            image_path = os.path.abspath(os.path.join(path, filename))
            cv2.imwrite(image_path, resized_face)

            # Save to database
            self.path = image_path
            self.save()

            logging.info(f"Saved face image {filename} for user {self.label.name}")
            return 'Success'

        except Exception as e:
            logging.error(f"Error persisting image: {str(e)}")
            return None

def initialize_database():
    """Initialize database and create tables"""
    try:
        db.connect()
        db.create_tables([Label, Image], safe=True)
        logging.info("Database initialized successfully")
    except Exception as e:
        logging.error(f"Database initialization error: {str(e)}")
        raise

def cleanup_database():
    """Clean up database connection"""
    if not db.is_closed():
        db.close()
        logging.info("Database connection closed")

def detect_faces(img):
    """Detect faces in image using Haar Cascade"""
    try:
        # Check if cascade file exists
        if not os.path.exists(FACE_CASCADE_FILE):
            raise FileNotFoundError(f"Cascade file not found: {FACE_CASCADE_FILE}")

        # Load cascade classifier
        cascade = cv2.CascadeClassifier(FACE_CASCADE_FILE)
        
        # Convert to grayscale
        gray = to_grayscale(img)
        
        # Detect faces
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces.tolist() if len(faces) > 0 else []

    except Exception as e:
        logging.error(f"Face detection error: {str(e)}")
        return []

def to_grayscale(img):
    """Convert image to grayscale if needed"""
    try:
        if img is None:
            raise ValueError("Input image is None")
            
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return cv2.equalizeHist(gray)  # Improve contrast
        return img

    except Exception as e:
        logging.error(f"Grayscale conversion error: {str(e)}")
        return img

def load_images_to_db(path):
    """Load images from directory into database"""
    if not os.path.exists(path):
        logging.warning(f"Directory does not exist: {path}")
        return

    try:
        with db.atomic():
            for dirname, dirnames, filenames in os.walk(path):
                for subdirname in dirnames:
                    subject_path = os.path.join(dirname, subdirname)
                    label, created = Label.get_or_create(name=subdirname)
                    
                    if created:
                        logging.info(f"Created new label: {subdirname}")
                    
                    for filename in os.listdir(subject_path):
                        file_path = os.path.abspath(os.path.join(subject_path, filename))
                        Image.get_or_create(path=file_path, label=label)

    except Exception as e:
        logging.error(f"Error loading images to database: {str(e)}")
        raise

def load_images_from_db():
    """Load all images from database"""
    images, labels = [], []
    try:
        with DatabaseConnection():
            for label in Label.select():
                for image in label.images:
                    try:
                        # Load and preprocess image
                        cv_image = cv2.imread(image.path, cv2.IMREAD_GRAYSCALE)
                        if cv_image is None:
                            continue
                            
                        cv_image = cv2.resize(cv_image, (100, 100))
                        images.append(np.asarray(cv_image, dtype=np.uint8))
                        labels.append(label.id)
                    except Exception as e:
                        logging.error(f"Error loading image {image.path}: {str(e)}")

        return images, np.array(labels)
    except Exception as e:
        logging.error(f"Error loading images from database: {str(e)}")
        return [], np.array([])

def train():
    """Train the face recognition model"""
    try:
        # Load images and labels
        images, labels = load_images_from_db()
        if len(images) == 0:
            logging.warning("No images available for training")
            return False

        # Convert to numpy arrays
        images_np = np.array(images, dtype=np.uint8)
        labels_np = np.array(labels, dtype=np.int32)

        # Create and train model
        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(images_np, labels_np)
        
        # Save model
        model.write(MODEL_FILE)
        logging.info("Model trained and saved successfully")
        return True

    except Exception as e:
        logging.error(f"Training error: {str(e)}")
        return False

def predict(cv_image):
    """Predict face in image"""
    try:
        # Detect faces
        faces = detect_faces(cv_image)
        if not faces:
            return None

        # Process first face
        face = faces[0]
        x, y, w, h = map(int, face)
        
        # Crop and preprocess face
        face_img = cv_image[y:y+h, x:x+w]
        gray_face = to_grayscale(face_img)
        resized_face = cv2.resize(gray_face, (100, 100))

        # Load model and predict
        model = cv2.face.LBPHFaceRecognizer_create()
        model.read(MODEL_FILE)
        
        label_id, confidence = model.predict(resized_face)
        
        # Get label from database
        with DatabaseConnection():
            label = Label.get(Label.id == label_id)

        return {
            'face': {
                'name': label.name,
                'confidence': float(confidence),
                'coords': {
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h
                }
            }
        }

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return None

def setup_directories():
    """Create necessary directories if they don't exist"""
    try:
        os.makedirs(IMAGE_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
        logging.info("Directories created successfully")
    except Exception as e:
        logging.error(f"Error creating directories: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Initialize everything
        setup_directories()
        initialize_database()
        load_images_to_db("data/images")
        train()
        logging.info("Initialization completed successfully")
    except Exception as e:
        logging.error(f"Initialization error: {str(e)}")
        sys.exit(1)