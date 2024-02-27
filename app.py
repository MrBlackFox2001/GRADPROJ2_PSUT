from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
from sklearn.preprocessing import RobustScaler
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import subprocess
import xml_to_df
app = Flask(__name__)
model = None

def load_model_on_startup():
    # Use a raw string for the Windows path
    model_path = r"C:\Users\USER\Desktop\h5\ecggraddd.h5"
    return load_model(model_path)

@app.before_first_request
def initialize_model():
    global model
    model = load_model_on_startup()

@app.route('/')
def index():
    return render_template('index.html')  # Make sure 'index.html' exists in the 'templates' directory

@app.route('/upload', methods=['POST'])
def upload_file():
    # if 'img' not in request.files:
    #     return jsonify({'error': 'No file part'}), 400
    file = request.files["file"]
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        file.save(file_path)
        try:
            prediction = predict(file_path)
            print(prediction, "PredictionFlag")
            os.remove(file_path)  # Remove the file after prediction
            return jsonify({'prediction': prediction})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

def predict(file_path):
    fileExt = file_path.rsplit('.', 1)[1].lower()
    print(fileExt)
    if fileExt in {'jpg', 'jpeg', 'png', 'bmp'}:
        print("Predicting from an image")
        return predict_image(file_path)
    if fileExt == 'xml':
        print("Predicting from an XML")
        return predict_xml(file_path)
    if fileExt == 'csv':
        print("Predicting from a CSV")
        return predict_raw(file_path)
        

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'bmp', "xml", "csv"}
           
def predict_image(image_path):
    command = f'plotdigitizer "{image_path}" -p 0,0 -p 2,0 -p 0,1 -l 2,29 -l 4,5 -l 22,5'
    output = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    print("STDOUT:", output.stdout)
    print("STDERR:", output.stderr)

    if output.returncode != 0:
        raise RuntimeError("plotdigitizer failed to run correctly")

    image_path2 = os.path.splitext(image_path)[0] + ".jpg.traj.csv"
    if not os.path.exists(image_path2):
        raise FileNotFoundError(f"Expected file not found: {image_path2}")

    df = pd.read_csv(image_path2, sep=' ', header=None)
    scaler = RobustScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    df = df.head(32).T.reset_index(drop=True)

    df.columns = ['0_pre-RR', '0_post-RR', '0_pPeak', '0_tPeak', '0_rPeak', '0_sPeak',
       '0_qPeak', '0_qrs_interval', '0_pq_interval', '0_qt_interval',
       '0_st_interval', '0_qrs_morph0', '0_qrs_morph1', '0_qrs_morph2',
       '0_qrs_morph3', '0_qrs_morph4', '1_pre-RR', '1_post-RR', '1_pPeak',
       '1_tPeak', '1_rPeak', '1_sPeak', '1_qPeak', '1_qrs_interval',
       '1_pq_interval', '1_qt_interval', '1_st_interval', '1_qrs_morph0',
       '1_qrs_morph1', '1_qrs_morph2', '1_qrs_morph3', '1_qrs_morph4']

    df = df.apply(pd.to_numeric, errors='coerce')

    predictions = model.predict(df.values) 

    predicted_label_index = np.argmax(predictions)
    class_labels = ["N", "The test shows a Ventricular heartbeat.` These heartbeats often occur when the normal electrical conduction system of the heart is disrupted or when the SA and AV nodes fail to initiate and conduct impulses properly.", "The image shows a Ventricular heartbeat.` These heartbeats often occur when the normal electrical conduction system of the heart is disrupted or when the SA and AV nodes fail to initiate and conduct impulses properly.", "Fusion"]
    predicted_label = class_labels[predicted_label_index]
    print(f"The predicted label is: {predicted_label}")
    return predicted_label

def predict_xml(xml_file):
    # Load the mode
    # Convert XML to DataFrame
    df = xml_to_df.convert_xml_to_df(xml_file)
    df.columns = df.columns.str.replace('patient_record__', '')
    # Define selected feature names
    selected_features =['0_pre-RR', '0_post-RR', '0_pPeak', '0_tPeak', '0_rPeak', '0_sPeak',
       '0_qPeak', '0_qrs_interval', '0_pq_interval', '0_qt_interval',
       '0_st_interval', '0_qrs_morph0', '0_qrs_morph1', '0_qrs_morph2',
       '0_qrs_morph3', '0_qrs_morph4', '1_pre-RR', '1_post-RR', '1_pPeak',
       '1_tPeak', '1_rPeak', '1_sPeak', '1_qPeak', '1_qrs_interval',
       '1_pq_interval', '1_qt_interval', '1_st_interval', '1_qrs_morph0',
       '1_qrs_morph1', '1_qrs_morph2', '1_qrs_morph3', '1_qrs_morph4']
    df = df[selected_features]
    # Filter selected features
    scaler = RobustScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    predictions = model.predict(df.values) 

    predicted_label_index = np.argmax(predictions)
    class_labels = ["N", "VEB", "The test shows a Normal heartbeat.", "Fusion"]
    predicted_label = class_labels[predicted_label_index]
    print(f"The predicted label is: {predicted_label}")
    return predicted_label


def predict_raw(csv_file_path):
    # Read the CSV file into a Pandas DataFrame
    raw_data = pd.read_csv(csv_file_path)

    # Assuming model is defined somewhere in your code
    predictions = model.predict(raw_data.values)

    class_labels = ["N", "The test shows a Ventricular heartbeat", "SVEB", "Fusion"]

    predicted_label_index = np.argmax(predictions)
    predicted_label = class_labels[predicted_label_index]
    print(f"The predicted label is: {predicted_label}")
    return predicted_label

    

if __name__ == '__main__':
        app.run(host='127.0.0.1', port=5000, debug=True)