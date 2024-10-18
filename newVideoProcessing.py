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
from scipy.spatial import distance as dist
from scipy.stats import mode

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

# Track the face IDs and their centroids
faceTrackers = {}
nextFaceID = 0
face_images = {}

def track_faces(faces):
    try:
        global faceTrackers, nextFaceID
        
        currentCentroids = []
        print("I am in track_faces")
        for (x, y, w, h) in faces:
            cx = x + w // 2
            cy = y + h // 2
            currentCentroids.append((cx, cy))
            
        if not currentCentroids:
            print("No faces detected.")
            return faceTrackers

        if len(faceTrackers) == 0:
            for i in range(len(currentCentroids)):
                faceTrackers[nextFaceID] = currentCentroids[i]
                print(f"New face detected: Assigning FaceID {nextFaceID} to centroid {currentCentroids[i]}")
                nextFaceID += 1
        else:
            objectIDs = list(faceTrackers.keys())
            objectCentroids = list(faceTrackers.values())
            print("currentCentroids",currentCentroids)
            print("objectIDs",objectIDs)
            print("objectCentroids",objectCentroids)
            
            D = dist.cdist(np.array(objectCentroids), np.array(currentCentroids))
            
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            usedRows = set()
            usedCols = set()
            
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                faceID = objectIDs[row]
                faceTrackers[faceID] = currentCentroids[col]
                print(f"Updating FaceID {faceID} with new centroid {currentCentroids[col]}")
                usedRows.add(row)
                usedCols.add(col)
            
            for i in range(len(currentCentroids)):
                if i not in usedCols:
                    faceTrackers[nextFaceID] = currentCentroids[i]
                    print(f"New face detected: Assigning FaceID {nextFaceID} to centroid {currentCentroids[i]}")
                    nextFaceID += 1

        return faceTrackers
    except Exception as e:
        raise e

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
        print("I am in download_video_from_drive")
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

def check_if_folder_exists(drive, parent_folder_id, folder_name):
    try:
        query = f"title='{folder_name}' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
        folder_list = drive.ListFile({'q': query}).GetList()
        if folder_list:
            return folder_list[0]['id']  
        return None  
    except Exception as e:
        raise e
    
def check_if_folder_exists_and_get_plots(drive, parent_folder_id, folder_name):
    try:
        query = f"title='{folder_name}' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
        folder_list = drive.ListFile({'q': query}).GetList()
        
        if not folder_list:
            return None, None  
        
        folder_id = folder_list[0]['id']
        
        query = f"'{folder_id}' in parents and mimeType='image/png'"
        file_list = drive.ListFile({'q': query}).GetList()
        
        plot_links = []
        for file in file_list:
            plot_links.append(f"https://drive.google.com/file/d/{file['id']}/view")
        
        return folder_id, plot_links  
    
    except Exception as e:
        raise e

    
def extract_frames_at_intervals(video_path, interval_seconds=20):
    try:
        print(" I am in extract_frames_at_intervals")
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
        overall_pred = []

        while success:
            if count % interval_frames == 0:
                overall_results = predict_overall(image)
                if overall_results:
                    overall_pred.append({
                        "Frame": frame_number,
                        "Engagement Level": overall_results
                        })
                results = predict(image)
                if results:
                    for result in results:
                        output_data.append({
                            "Frame": frame_number,
                            "FaceID": result["FaceID"],
                            "Emotion": result["Emotion"],
                            "Engagement Level": str(result["Engagement Level"])
                        })
                    frame_number += 1
            
            success, image = vidcap.read()
            count += 1
        
        vidcap.release()
        print(f"Extracted {frame_number} frames at {interval_seconds} seconds intervals.")
        
        return output_data, overall_pred
    
    except Exception as e:
        raise e
    
def save_plot_overall(output_data, file_specific_folder_id, plot_filename_overall, interval_seconds=20):  
    try:
        drive = initialize_drive()
        # drive_folder_id = get_file_parent_folder_id(drive, video_file_id)
        # result_folder_id = create_folder_if_not_exists(drive, drive_folder_id, "AI Result")
        print("Overall_Plot---------------------------",output_data)
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

        fig.savefig(plot_filename_overall)
        plt.close('all')  
        
        file_drive = drive.CreateFile({'title': plot_filename_overall, 'parents': [{'id': file_specific_folder_id}]})
        file_drive.SetContentFile(plot_filename_overall)
        file_drive.Upload()

        plot_drive_link = f"https://drive.google.com/file/d/{file_drive['id']}/view"
        return [plot_drive_link]
    
    except Exception as e:
        raise e  
     
def predict_overall(frame):
    try:
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)
        emotions = []
        engagement_levels = []
        
        if len(faces) == 0:
            return "No emotion detected"

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
                emotions.append(output)

                if output == "Surprise":
                    engagement_levels.append(5)
                elif output == "Fear" or output == "Angry":
                    engagement_levels.append(4)
                elif output == "Happy" or output == "Sad":
                    engagement_levels.append(3)
                elif output == "Neutral":
                    engagement_levels.append(2)
                else:
                    engagement_levels.append(1)

        if engagement_levels:
            print("Engagement Levels:", engagement_levels)
            overall_engagement_level = mode(engagement_levels).mode[0]
            print("Overall Engagement Level:", overall_engagement_level)
        else:
            overall_engagement_level = "1"

        return overall_engagement_level
    
    except Exception as e:
        raise e
    
def predict(frame):
    try:
        print("I am in predict")
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)
        tracked_faces = track_faces(faces)
        print("TrackedFaces:", tracked_faces)
        results = []
        
        detected_face_ids = set()

        for faceID, (cx, cy) in tracked_faces.items():
            output = "No Emotion Detected"
            eLevel = 1
            face_detected = False

            for (x, y, w, h) in faces:
                if x <= cx <= x + w and y <= cy <= y + h:
                    face_detected = True
                    detected_face_ids.add(faceID)
                    cv2.rectangle(img=img_gray, pt1=(x, y), pt2=(x+w, y + h), color=(0, 255, 255), thickness=2)
                    face_image = frame[y:y+h, x:x+w]
                    face_image_path = f"face_{faceID}.png"
                    face_images[faceID] = face_image_path
                    print("face_images",face_images)
                    cv2.imwrite(face_image_path, face_image)
                    roi_gray = img_gray[y:y+h, x:x+w]
                    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                    if np.sum([roi_gray]) != 0:
                        roi = roi_gray.astype('float') / 255.0
                        roi = img_to_array(roi)
                        roi = np.expand_dims(roi, axis=0)
                        prediction = classifier.predict(roi)
                        maxindex = int(np.argmax(prediction[0]))
                        finalout = emotion_labels[maxindex]
                        output = str(finalout)

                        engagement_levels = {
                            "Surprise": 5,
                            "Fear": 4,
                            "Angry": 4,
                            "Happy": 3,
                            "Sad": 3,
                            "Neutral": 2,
                        }
                        eLevel = engagement_levels.get(output, 1)  

            results.append({
                "FaceID": faceID,
                "Emotion": output,
                "Engagement Level": eLevel
            })
            
        for faceID in faceTrackers.keys():
            if faceID not in detected_face_ids:
                results.append({
                    "FaceID": faceID,
                    "Emotion": "No Emotion Detected",
                    "Engagement Level": 1
                })


        print("Results:", results)
        return results  

    except Exception as e:
        raise e


def save_plot(output_data, file_specific_folder_id, interval_seconds=20):
    try:
        plot_filenames = []
        drive = initialize_drive()
        # drive_folder_id = get_file_parent_folder_id(drive, video_file_id)
        # result_folder_id = create_folder_if_not_exists(drive, drive_folder_id, "AI Result")
        
        face_data = {}
        for data in output_data:
            face_id = data['FaceID']
            # print("Face_ID",face_id)
            if face_id not in face_data:
                face_data[face_id] = {'frames': [], 'eLevels': []}
                
            face_data[face_id]['frames'].append(data['Frame'])
            face_data[face_id]['eLevels'].append(data['Engagement Level'])

        print("Face_Data",face_data)
        plot_links = []
        for face_id, data in face_data.items():
            frames = data['frames']
            print("frames",frames)
            eLevels = data['eLevels']

            engagement_level_labels = {
                "2": 2,
                "3": 3,
                "4": 4,
                "5": 5,
                "1": 1         #1 is no emotion detected
            }

            eLevels_mapped = [engagement_level_labels.get(e, e) for e in eLevels]
            time_in_minutes = [(frame * interval_seconds) / 60.0 for frame in frames]

            plot_filename = f"{face_id}_plot.png"
            plot_filenames.append(plot_filename)
            fig, ax = plt.subplots(figsize=(15, 9))
            ax.plot(time_in_minutes, eLevels_mapped, marker='o', linestyle='-', color='b', label='Engagement Level')
            ax.set_title(f'Time vs Engagement Level for FaceID {face_id}')
            ax.set_xlabel('Lecture Minutes')
            ax.set_ylabel('Engagement Level')
            ax.grid(True)
            ax.set_yticks(list(engagement_level_labels.values()))
            ax.legend()
            if face_id in face_images:
                # print('Face_id',face_id)
                # print('Face_id',face_images)
                face_image_filename = face_images[face_id]
                print("face_image_filename",face_image_filename)
                if os.path.exists(face_image_filename):
                    print("I am in face_image_filename")
                    face_img = plt.imread(face_image_filename)
                    ax_inset = fig.add_axes([0.88, 0.70, 0.15, 0.15])
                    ax_inset.imshow(face_img)
                    ax_inset.axis('off')
                    os.remove(face_image_filename)

            fig.savefig(plot_filename)
            plt.close('all')

            file_drive = drive.CreateFile({'title': plot_filename, 'parents': [{'id': file_specific_folder_id}]})
            file_drive.SetContentFile(plot_filename)
            file_drive.Upload()

            plot_drive_link = f"https://drive.google.com/file/d/{file_drive['id']}/view"
            plot_links.append(plot_drive_link)
            

        return plot_links, plot_filenames
    
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
    
def predict_image(frame):
    try:
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)
        print("Faces",faces)
        results = []
        
        if len(faces) == 0:
            return ["No face detected"], ["No emotion detected"]

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
                # eLevel = 1

                if output == "Surprise":
                    engagement_level = 5 
                elif output == "Fear" or output == "Angry":
                    engagement_level = 4
                elif output == "Happy" or output == "Sad":
                    engagement_level = 3
                elif output == "Neutral":
                    engagement_level = 2
                else:
                    engagement_level = 1
                    
                results.append({
                    "Emotion": output,
                    "Engagement Level": engagement_level
                })

        return results
    
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
    plot_filename_overall = original_filename.rsplit('.', 1)[0] + "_plot.png"
    file_specific_folder_name = original_filename.rsplit('.', 1)[0]

    try:
        drive_folder_id = get_file_parent_folder_id(drive, file_id)
        result_folder_id = create_folder_if_not_exists(drive, drive_folder_id, "AI Result")
        file_specific_folder_id, plot_links = check_if_folder_exists_and_get_plots(drive, result_folder_id, file_specific_folder_name)
        if plot_links:
            return jsonify({'plot_drive_link': plot_links})
        if not file_specific_folder_id:
            file_specific_folder_id = create_folder_if_not_exists(drive, result_folder_id, file_specific_folder_name)
    except Exception as e:
        raise e

    try:
        download_video_from_drive(file_id, temp_video_path)
    except Exception as e:
        os.remove(temp_video_path)
        raise e

    try:
        output_data, overall_pred = extract_frames_at_intervals(temp_video_path)
        print("Output_Data",output_data)
        if not output_data:
            return jsonify({'error': 'No faces detected in the video.'})
    except Exception as e:
        os.remove(temp_video_path)
        for face_id in face_images:
            if os.path.exists(face_images[face_id]):
                os.remove(face_images[face_id])
        raise e

    # if output_data is None:
    #     return jsonify({'error': 'Error extracting frames from video'})

    try:
        plot_drive_link, plot_filenames = save_plot(output_data, file_specific_folder_id)
        overall_plot_drive_link = save_plot_overall(overall_pred, file_specific_folder_id, plot_filename_overall)

    except Exception as e:
        if os.path.exists(plot_filename_overall):
            os.remove(plot_filename_overall)
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if plot_filenames:
            for plot_filename in plot_filenames:
                if os.path.exists(plot_filename):
                    os.remove(plot_filename)
        if face_images:
            for face_id in face_images:
                if os.path.exists(face_images[face_id]):
                    os.remove(face_images[face_id])
        raise e
    
    finally:
        if os.path.exists(plot_filename_overall):
            os.remove(plot_filename_overall)
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if plot_filenames:
            for plot_filename in plot_filenames:
                if os.path.exists(plot_filename):
                    os.remove(plot_filename)
        if face_images:
            for face_id in face_images:
                if os.path.exists(face_images[face_id]):
                    os.remove(face_images[face_id])
        
    return jsonify({'global_plot': overall_plot_drive_link,
                    'plot_drive_link': plot_drive_link})

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'})

    image_file = request.files['image']
    image_np = np.fromstring(image_file.read(), np.uint8)
    frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    try:
        output = predict_image(frame)
        print("Output",output)
    except Exception as e:
        return jsonify({'error': str(e)})

    return output

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
