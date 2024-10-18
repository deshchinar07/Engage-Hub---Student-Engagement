import numpy as np
import tensorflow as tf
from keras.utils import img_to_array
import cv2
from flask import Flask, request, jsonify
from pymongo import MongoClient
from datetime import datetime
import os
import gdown
import matplotlib.pyplot as plt


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

def generate_metadata(filename, output, eLevel):
    filename_parts = filename.rsplit('.', 1)
    filename_without_extension = filename_parts[0]
    filename_parts = filename_without_extension.split('_')

    metadata = None

    if len(filename_parts) != 5:
        warning = "Warning: Please give Filename in 'SchoolName_Date_Grade_Division_Subject.' format"
        return None, warning
    else:
        warning = None

    date_index = 1
    if len(filename_parts) > date_index and "NA" in filename_parts[date_index]:
        date_string = str(datetime.now())
        filename_parts[date_index] = date_string

    metadata = {
        "filename": filename,
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

    return metadata, None

def download_video_from_google_drive(url, output_path):
    gdown.download(url, output_path, quiet=False)

def extract_frames_at_intervals(video_path, interval_seconds=60):
    vidcap = cv2.VideoCapture(video_path)

    if not vidcap.isOpened():
        print("Error opening video file:", video_path)
        return None

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    interval_frames = int(interval_seconds * fps)

    print(f"Frames per second (FPS): {fps}")
    print(f"Interval frames to skip: {interval_frames}")

    success, image = vidcap.read()
    count = 0
    frame_number = 1
    output_data = []

    while success:
        if count % interval_frames == 0:
            print("Image",image)
            output, eLevel = predict(image)
            print("output",output)
            print("elevel",eLevel)
            output_data.append({
                "Frame": frame_number,
                "Emotion": output,
                "Engagement Level": str(eLevel)
            })
            frame_number += 1
        
        success, image = vidcap.read()
        count += 1
    
    vidcap.release()
    print(f"Extracted {frame_number} frames at {interval_seconds} seconds intervals.")
    
    return output_data

def predict(frame):
    # image = cv2.imdecode(np.frombuffer(frame, np.uint8), -1)
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

    return output, eLevel


def save_plot(output_data, drive_url):    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    plot_filename = f"{timestamp}_plot.png"
    
    frames = [data['Frame'] for data in output_data]
    eLevels = [data['Engagement Level'] for data in output_data]

    plt.figure(figsize=(10, 6))
    plt.plot(frames, eLevels, marker='o', linestyle='-', color='b')
    plt.title('Time vs Engagement Level')
    plt.xlabel('Lecture')
    plt.ylabel('Engagement Level')
    plt.grid(True)

    plt.savefig(plot_filename)

    gdown.upload(plot_filename, drive_url, quiet=False)

    os.remove(plot_filename)

    plot_drive_link = f"https://drive.google.com/file/d/{drive_url.split('/')[-2]}/view"
    return plot_drive_link


@app.route('/process_video', methods=['POST'])
def process_video_from_drive():
    drive_url = request.json.get('drive_url')

    if not drive_url:
        return jsonify({'error': 'No Google Drive URL provided'})

    temp_video_path = "temp_video.mp4"
    download_video_from_google_drive(drive_url, temp_video_path)

    output_data = extract_frames_at_intervals(temp_video_path)

    if output_data is None:
        return jsonify({'error': 'Error extracting frames from video'})
    
    plot_drive_link = save_plot(output_data, drive_url)


    os.remove(temp_video_path)

    return jsonify({'plot_drive_link': plot_drive_link})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
