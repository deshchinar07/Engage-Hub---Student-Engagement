# import ssl
import numpy as np
import tensorflow as tf
from keras.utils import img_to_array
import cv2
from flask import Flask, request, jsonify
from pymongo import MongoClient
from datetime import datetime
import certifi

app = Flask(__name__)

client = MongoClient('mongodb+srv://yuj-poc:p0qhaR2JpsI9VQMF@poc-yuj.759wdjf.mongodb.net/?retryWrites=true&w=majority&tls=true')
db = client['FaceRecognition']
collection = db['FRUpdated']

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
classifier = tf.keras.models.load_model('model_78.h5')
classifier.load_weights('model_weights_78.h5')

try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
    print("Error loading cascade classifiers")

def generate_metadata(uploaded_file, output, eLevel, is_valid_format):
    filename_parts = uploaded_file.filename.rsplit('.', 1)
    filename_without_extension = filename_parts[0]
    filename_parts = filename_without_extension.split('_')

    metadata = None  # Initialize metadata here

    if len(filename_parts) != 5:
        is_valid_format = False
        warning = "Warning: Please give Filename in 'SchoolName_Date_Grade_Division_Subject.' format"
    else:
        warning = None

        date_index = 1
        if len(filename_parts) > date_index and "NA" in filename_parts[date_index]:
            date_string = str(datetime.now())
            filename_parts[date_index] = date_string
    
        metadata = {
            "filename": uploaded_file.filename,
            "emotion": output,
            "engagement_level": str(eLevel),
            "SchoolName": filename_parts[0],
            "Date": filename_parts[date_index] if len(filename_parts) > date_index else None,
            "Grade": filename_parts[2] if len(filename_parts) > 2 else None,
            "Division": filename_parts[3] if len(filename_parts) > 3 else None,
            "Subject": filename_parts[4] if len(filename_parts) > 4 else None
        }

        collection.insert_one(metadata)

        print("Emotion: " + output + "; Engagement Level = " + str(eLevel))
        print("Metadata:", metadata)
    
    return metadata, is_valid_format, warning if not is_valid_format else None

@app.route('/predict', methods=['POST'])
def predict_emotion():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file'})

    is_valid_format = True
    warning = None

    image_data = uploaded_file.read()
    print("Image Data",image_data)
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), -1)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)
    output = "No face detected"
    eLevel = "No emotion detected"

    for (x, y, w, h) in faces:
        cv2.rectangle(img=img_gray, pt1=(x, y), pt2=(x+w, y + h), color=(0, 255, 255), thickness=2)
        roi_gray = img_gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            maxindex = int(np.argmax(classifier.predict(roi)[0]))
            finalout = emotion_labels[maxindex]
            output = str(finalout)
            eLevel = 1

            if output == "Surprise":
                eLevel = 5
            elif output == "Fear" or output == "Angry":
                eLevel = 4
            elif output == "Happy" or output == "Sad":
                eLevel = 3
            elif output == "Neutral":
                eLevel = 2

            metadata, is_valid_format, warning = generate_metadata(uploaded_file, output, eLevel, is_valid_format)

    response_data = {
        "Filename": uploaded_file.filename,
        "Emotion": output,
        "Engagement Level": str(eLevel),
    }

    if warning:
        response_data["Warning"] = warning

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
