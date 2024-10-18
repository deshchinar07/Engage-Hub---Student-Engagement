import numpy as np
import tensorflow as tf
from keras.utils import img_to_array
import cv2
from flask import Flask, request, jsonify

app = Flask(__name__)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
#Load model.
classifier = tf.keras.models.load_model('model_78.h5')
# load weights into new model
classifier.load_weights('model_weights_78.h5')
# Load face using OpenCV
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
    print("Error loading cascade classifiers")

@app.route('/predict', methods=['POST'])
def predict_emotion():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file'})

    image_data = uploaded_file.read()
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), -1)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5) 
    output = "No face detected"
    eLevel = "No emition detected"
    for (x, y, w, h) in faces:
        cv2.rectangle(img= img_gray, pt1=(x, y), pt2=(x+w, y + h), color=(0, 255, 255), thickness=2) 
        roi_gray = img_gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims (roi, axis=0)
            maxindex = int (np.argmax(classifier.predict(roi)[0]))
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
            print("Emotion: " + output + "; Engagement Level = " + str(eLevel) + "\n")
            
    return jsonify({"Filename": uploaded_file.filename, "Emotion": output, "Engagement Level": str(eLevel)})
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)


#Fix putText() and see if last 3 lines are inside or outside for loop

# import numpy as np
# import tensorflow as tf
# from keras.utils import img_to_array
# import cv2
# from flask import Flask, request, jsonify

# app = Flask(__name__)

# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# classifier = tf.keras.models.load_model('model_78.h5')
# classifier.load_weights("model_weights_78.h5")

# try:
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# except Exception:
#     print("Error loading cascade classifiers")

# @app.route('/predict', methods=['POST'])
# def predict_emotion():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})

#     uploaded_file = request.files['file']
#     if uploaded_file.filename == '':
#         return jsonify({'error': 'No selected file'})

#     image_data = uploaded_file.read()
#     image = cv2.imdecode(np.frombuffer(image_data, np.uint8), -1)

#     img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)

#     results = []

#     for (x, y, w, h) in faces:
#         roi_gray = img_gray[y:y + h, x:x + w]
#         roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

#         if np.sum([roi_gray]) != 0:
#             roi = roi_gray.astype('float') / 255.0
#             roi = img_to_array(roi)
#             roi = np.expand_dims(roi, axis=0)
#             prediction = classifier.predict(roi)[0]
#             maxindex = int(np.argmax(prediction))
#             finalout = emotion_labels[maxindex]
#             results.append(finalout)

#     return jsonify({'emotions': results})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=80)

