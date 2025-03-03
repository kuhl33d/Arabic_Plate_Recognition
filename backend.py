from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from Extract_Character import *
from Character_Recognizer import *
from digit_recognizer_ import *
from Car_Plate_Detection import *

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize models
cr = Character_Recognizer()
nr = Number_Recognizer()
Ec = Extract_Characters()
cp = Car_Plate_Detection()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image):
    try:
        PlateImg = cp.Detect_Plate(image)
        
        if PlateImg is None or isinstance(PlateImg, bool):
            return {"success": False, "message": "No plate found in image"}

        numbers, characters = Ec.extract(PlateImg)
        word = []
        
        for i in range(len(numbers)):
            word.append(nr.ocr(numbers[i]))

        for i in range(len(characters)):
            word.append(cr.ocr(characters[i]))

        return {"success": True, "plate_number": ','.join(word)}
    
    except Exception as e:
        return {"success": False, "message": f"Error processing image: {str(e)}"}

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/recognize_plate', methods=['POST'])
def recognize_plate():
    try:
        if 'image' not in request.files:
            return jsonify({"success": False, "message": "No image file provided"})

        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"success": False, "message": "No selected file"})

        if file and allowed_file(file.filename):
            # Save the file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Read and process the image
            image = cv2.imread(filepath)
            
            # Remove the temporary file
            os.remove(filepath)

            if image is None:
                return jsonify({"success": False, "message": "Failed to read image"})

            result = process_image(image)
            return jsonify(result)

        return jsonify({"success": False, "message": "Invalid file type"})

    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {str(e)}"})

@app.route('/recognize_plate_stream', methods=['POST'])
def recognize_plate_stream():
    try:
        # Get raw image data from request
        nparr = np.frombuffer(request.data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"success": False, "message": "Failed to decode image stream"})

        result = process_image(image)
        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {str(e)}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)