import numpy as np
import tensorflow as tf
from keras.utils import img_to_array
import cv2
from flask import Flask, request, jsonify
from pymongo import MongoClient
from datetime import datetime
import os
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from oauth2client.service_account import ServiceAccountCredentials

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

def download_video_from_drive(file_id, output_path):
    try:
        drive = initialize_drive()
        file_drive = drive.CreateFile({'id': file_id})
        file_drive.GetContentFile(output_path)
    except Exception as e:
        raise e

def get_file_parent_folder_id(drive, file_id):
    try:
        file = drive.CreateFile({'id': file_id})
        file.FetchMetadata(fields='parents')
        parents = file['parents']
        if parents:
            return parents[0]['id'] 
        else:
            raise Exception("No parent folder found for the file.")
    except Exception as e:
        raise e

def create_folder_if_not_exists(drive, parent_folder_id, folder_name):
    try:
        query = f"title='{folder_name}' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
        folder_list = drive.ListFile({'q': query}).GetList()
        if folder_list:
            return folder_list[0]['id']
        else:
            folder_metadata = {
                'title': folder_name,
                'parents': [{'id': parent_folder_id}],
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = drive.CreateFile(folder_metadata)
            folder.Upload()
            return folder['id']
    except Exception as e:
        raise e

def check_if_plot_exists(drive, folder_id, plot_filename):
    try:
        query = f"title='{plot_filename}' and '{folder_id}' in parents and mimeType='image/png'"
        file_list = drive.ListFile({'q': query}).GetList()
        if file_list:
            return f"https://drive.google.com/file/d/{file_list[0]['id']}/view"
        return None
    except Exception as e:
        raise e

def extract_frames_at_intervals(video_path, interval_seconds=20):
    try:
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
                output, eLevel = predict(image)
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
    
    except Exception as e:
        raise e

def predict(frame):
    try:
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
    
    except Exception as e:
        raise e

def save_plot(output_data, video_file_id, plot_filename, interval_seconds=20):  
    try:
        drive = initialize_drive()
        drive_folder_id = get_file_parent_folder_id(drive, video_file_id)
        result_folder_id = create_folder_if_not_exists(drive, drive_folder_id, "AI Result")

        frames = [data['Frame'] for data in output_data]
        eLevels = [data['Engagement Level'] for data in output_data]

        engagement_level_labels = {
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "No emotion detected": 1
        }

        eLevels_mapped = [engagement_level_labels.get(e, e) for e in eLevels]

        time_in_minutes = [(frame * interval_seconds) / 60.0 for frame in frames]

        fig, ax = plt.subplots(figsize=(15, 9))
        ax.plot(time_in_minutes, eLevels_mapped, marker='o', linestyle='-', color='b', label='Engagement Level')
        ax.set_title('Time vs Engagement Level')
        ax.set_xlabel('Lecture Minutes')
        ax.set_ylabel('Engagement Level')
        ax.grid(True)
        ax.set_yticks(list(engagement_level_labels.values()))
        ax.legend() 

        fig.savefig(plot_filename)
        plt.close('all')  
        
        file_drive = drive.CreateFile({'title': plot_filename, 'parents': [{'id': result_folder_id}]})
        file_drive.SetContentFile(plot_filename)
        file_drive.Upload()

        plot_drive_link = f"https://drive.google.com/file/d/{file_drive['id']}/view"
        return plot_drive_link
    
    except Exception as e:
        raise e

def initialize_drive():
    scopes = ["https://www.googleapis.com/auth/drive"]
    credentials = ServiceAccountCredentials.from_json_keyfile_name('service_account.json', scopes)
    gauth = GoogleAuth()
    gauth.credentials = credentials
    drive = GoogleDrive(gauth)
    return drive

def list_folders(drive, parent_folder_id=None):
    try:
        query = "mimeType='application/vnd.google-apps.folder'"
        if parent_folder_id:
            query += f" and '{parent_folder_id}' in parents"
        folder_list = drive.ListFile({'q': query}).GetList()
        folders = [{'title': folder['title'], 'id': folder['id']} for folder in folder_list]
        return folders
    except Exception as e:
        raise e
    
def list_files(drive, folder_id):
    try:
        query = f"'{folder_id}' in parents"
        file_list = drive.ListFile({'q': query}).GetList()
        files = [{'title': file['title'], 'id': file['id'], 'mimeType': file['mimeType']} for file in file_list]
        return files
    except Exception as e:
        raise e

@app.route('/list_folders', methods=['GET'])
def list_first_level_folders():
    try:
        drive = initialize_drive()
        parent_folder_id = "1DlA_7qYRLsJb64b09RG8yRuuO6awLhK3"
        folders = list_folders(drive, parent_folder_id)
        return jsonify({'folders': folders})
    except Exception as e:
        return jsonify({'error': str(e)})
    

@app.route('/list_subfolders', methods=['GET'])
def list_subfolders():
    parent_folder_id = request.args.get('parent_folder_id')
    if not parent_folder_id:
        return jsonify({'error': 'No parent folder ID provided'})
    
    try:
        drive = initialize_drive()
        subfolders = list_folders(drive, parent_folder_id)
        return jsonify({'folders': subfolders})
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/list_files', methods=['GET'])
def list_files_in_folder():
    folder_id = request.args.get('folder_id')
    if not folder_id:
        return jsonify({'error': 'No folder ID provided'})
    
    try:
        drive = initialize_drive()
        files = list_files(drive, folder_id)
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/process_video', methods=['POST'])
def process_video_from_drive():
    file_id = request.json.get('file_id')
    
    if not file_id:
        return jsonify({'error': 'No file ID provided'})

    try:
        drive = initialize_drive()
        file_drive = drive.CreateFile({'id': file_id})
        file_drive.FetchMetadata(fields='title')
        original_filename = file_drive['title']
    except Exception as e:
        return jsonify({'error': str(e)})

    temp_video_path = original_filename
    plot_filename = original_filename.rsplit('.', 1)[0] + "_plot.png"

    try:
        drive_folder_id = get_file_parent_folder_id(drive, file_id)
        result_folder_id = create_folder_if_not_exists(drive, drive_folder_id, "AI Result")
        existing_plot_link = check_if_plot_exists(drive, result_folder_id, plot_filename)
        if existing_plot_link:
            return jsonify({'plot_drive_link': existing_plot_link})
    except Exception as e:
        raise e

    try:
        download_video_from_drive(file_id, temp_video_path)
    except Exception as e:
        os.remove(temp_video_path)
        raise e

    try:
        output_data = extract_frames_at_intervals(temp_video_path)
    except Exception as e:
        os.remove(temp_video_path)
        raise e

    if output_data is None:
        return jsonify({'error': 'Error extracting frames from video'})

    try:
        plot_drive_link = save_plot(output_data, file_id, plot_filename)
    except Exception as e:
        os.remove(temp_video_path)
        os.remove(plot_filename)
        raise e
    
    os.remove(plot_filename)
    os.remove(temp_video_path)

    return jsonify({'plot_drive_link': plot_drive_link})

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'})

    image_file = request.files['image']
    image_np = np.fromstring(image_file.read(), np.uint8)
    frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    try:
        output, eLevel = predict(frame)
    except Exception as e:
        return jsonify({'error': str(e)})

    return jsonify({
        'Emotion': output,
        'Engagement Level': str(eLevel),
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
