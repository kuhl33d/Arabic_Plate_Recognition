import logging
import tornado.escape
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.websocket
import os.path
import uuid
from PIL import Image
import time
from io import BytesIO
import numpy
import json
from tornado.options import define, options
import opencv

define("port", default=8888, help="run on the given port", type=int)

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/", MainHandler),
            (r"/harvesting", HarvestHandler),
            (r"/predict", PredictHandler),
            (r"/train", TrainHandler)
        ]

        settings = dict(
            cookie_secret="your_secret_here",
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            xsrf_cookies=False,
            debug=True
        )
        super().__init__(handlers, **settings)

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

    def post(self):
        try:
            name = self.get_argument("label", None)
            if not name or not name.strip():
                self.set_status(400)
                self.write({"error": "Label cannot be empty"})
                return

            logging.info(f"Processing label: {name}")
            label = opencv.Label.get_or_create(name=name)[0]
            label.persist()
            
            self.set_secure_cookie('label', name)
            self.write({"status": "success"})
        except Exception as e:
            logging.error(f"Error processing label: {str(e)}")
            self.set_status(500)
            self.write({"error": str(e)})

class SocketHandler(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True  # Allow cross-origin requests

    def open(self):
        logging.info('New WebSocket connection established')

    def on_message(self, message):
        try:
            image = Image.open(BytesIO(message))
            cv_image = numpy.array(image)
            self.process(cv_image)
        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")
            self.write_message(json.dumps({
                "status": "error",
                "message": "Failed to process image"
            }))

    def on_close(self):
        logging.info('WebSocket connection closed')

    def process(self, cv_image):
        pass

class HarvestHandler(SocketHandler):
    def process(self, cv_image):
        try:
            label_name = self.get_secure_cookie('label')
            if not label_name:
                self.write_message(json.dumps({
                    'status': 'error',
                    'message': 'No label set'
                }))
                return

            label = opencv.Label.get(opencv.Label.name == label_name.decode())
            faces = opencv.detect_faces(cv_image)
            
            if not faces:
                self.write_message(json.dumps({
                    'status': 'error',
                    'message': 'No face detected'
                }))
                return

            result = opencv.Image(label=label).persist(cv_image)
            
            if result == 'Done':
                self.write_message(json.dumps({
                    'status': 'complete',
                    'message': 'Training data collection complete'
                }))
            elif result == 'Success':
                self.write_message(json.dumps({
                    'status': 'success',
                    'message': 'Image captured'
                }))
            else:
                self.write_message(json.dumps({
                    'status': 'error',
                    'message': 'Failed to save image'
                }))

        except Exception as e:
            logging.error(f"Error in harvest handler: {str(e)}")
            self.write_message(json.dumps({
                'status': 'error',
                'message': str(e)
            }))

class TrainHandler(tornado.web.RequestHandler):
    def post(self):
        try:
            success = opencv.train()
            if success and opencv.verify_model():
                self.write(json.dumps({
                    "status": "success",
                    "message": "Model trained successfully"
                }))
            else:
                self.set_status(500)
                self.write(json.dumps({
                    "status": "error",
                    "message": "Training or verification failed"
                }))
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            self.set_status(500)
            self.write(json.dumps({
                "status": "error",
                "message": str(e)
            }))

class PredictHandler(SocketHandler):
    def process(self, cv_image):
        try:
            result = opencv.predict(cv_image)
            if result is not None:
                self.write_message(json.dumps({
                    "status": "success",
                    "result": result
                }))
            else:
                self.write_message(json.dumps({
                    "status": "error",
                    "message": "No face detected or prediction failed"
                }))
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            self.write_message(json.dumps({
                "status": "error",
                "message": str(e)
            }))

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('face_recognition.log')
        ]
    )

def initialize_application():
    """Initialize application requirements"""
    try:
        # Create necessary directories
        os.makedirs("data/images", exist_ok=True)
        os.makedirs("data/models", exist_ok=True)
        
        # Initialize database
        opencv.initialize_database()
        
        # Check for cascade file
        cascade_file = "data/haarcascade_frontalface_alt.xml"
        if not os.path.exists(cascade_file):
            raise FileNotFoundError(
                f"Cascade file not found: {cascade_file}. "
                "Please download it from OpenCV's GitHub repository."
            )
        
        # Clean up invalid images
        opencv.cleanup_invalid_images()
        
        # Load initial data
        opencv.load_images_to_db("data/images")
        
        # Train and verify model if there's data
        if opencv.Image.select().count() > 0:
            if not opencv.train() or not opencv.verify_model():
                raise Exception("Model training or verification failed")
            
        logging.info("Application initialized successfully")
        
    except Exception as e:
        logging.error(f"Initialization error: {str(e)}")
        raise

def main():
    try:
        # Parse command line arguments
        tornado.options.parse_command_line()
        
        # Setup logging
        setup_logging()
        
        # Initialize application
        initialize_application()
        
        # Create and start server
        app = Application()
        app.listen(options.port)
        
        logging.info(f"Server started on port {options.port}")
        logging.info("Press Ctrl+C to stop the server")
        
        # Start IOLoop
        tornado.ioloop.IOLoop.instance().start()
        
    except KeyboardInterrupt:
        logging.info("Server shutdown initiated")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise
    finally:
        opencv.cleanup_database()
        logging.info("Server shutdown complete")

if __name__ == "__main__":
    main()