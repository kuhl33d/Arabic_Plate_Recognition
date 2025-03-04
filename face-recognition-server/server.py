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
            (r"/", SetupHarvestHandler),
            (r"/harvesting", HarvestHandler),
            (r"/predict", PredictHandler),
            (r"/train", TrainHandler)
        ]

        # settings = dict(
        #     cookie_secret=os.environ.get('COOKIE_SECRET', "default_secret_key"),
        #     template_path=os.path.join(os.path.dirname(__file__), "templates"),
        #     static_path=os.path.join(os.path.dirname(__file__), "static"),
        #     xsrf_cookies=False,
        #     autoescape=None,
        #     debug=True
        # )
        settings = dict(
            cookie_secret="your_secret_here",
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            xsrf_cookies=False,
            debug=True
        )
        tornado.web.Application.__init__(self, handlers, **settings)

class SocketHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        logging.info('New WebSocket connection established')

    def on_message(self, message):
        try:
            image = Image.open(BytesIO(message))
            cv_image = numpy.array(image)
            self.process(cv_image)
        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")
            self.write_message(json.dumps({"error": "Failed to process image"}))

    def on_close(self):
        logging.info('WebSocket connection closed')

    def process(self, cv_image):
        pass

class SetupHarvestHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("harvest.html")

    def post(self):
        try:
            name = self.get_argument("label", None)
            if not name or not name.strip():
                self.write_error(400, message="Label cannot be empty")
                return

            logging.info(f"Processing label: {name}")
            label = opencv.Label.get_or_create(name=name)[0]
            label.persist()
            
            self.set_secure_cookie('label', name)
            self.redirect("/")
        except Exception as e:
            logging.error(f"Error in setup harvest: {str(e)}")
            self.write_error(500)

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
            
            self.write_message(json.dumps({
                'status': 'success',
                'message': 'Image captured'
            }))

            if result == 'Done':
                self.write_message(json.dumps({
                    'status': 'complete',
                    'message': 'Training data collection complete'
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
            opencv.train()
            self.write(json.dumps({"status": "success"}))
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            self.write_error(500)

class PredictHandler(SocketHandler):
    def process(self, cv_image):
        try:
            result = opencv.predict(cv_image)
            if result is not None:
                self.write_message(json.dumps(result))
            else:
                self.write_message(json.dumps({
                    "error": "No face detected or prediction failed"
                }))
        except Exception as e:
            logging.error(f"Prediction processing error: {str(e)}")
            self.write_message(json.dumps({
                "error": "Failed to process image",
                "details": str(e)
            }))

def initialize_application():
    # Create necessary directories
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("data/models", exist_ok=True)
    
    # Initialize database
    opencv.initialize_database()
    
    # Clear existing data
    with opencv.db.atomic():
        opencv.Image.delete().execute()
        opencv.Label.delete().execute()
    
    # Load initial data
    opencv.load_images_to_db("data/images")
    
    # Train model if there's data
    if opencv.Image.select().count() > 0:
        opencv.train()

def main():
    try:
        tornado.options.parse_command_line()
        
        # Initialize everything
        initialize_application()
        
        # Start server
        app = Application()
        app.listen(options.port)
        logging.info(f"Server started on port {options.port}")
        tornado.ioloop.IOLoop.instance().start()
    
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise
    finally:
        opencv.cleanup_database()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()