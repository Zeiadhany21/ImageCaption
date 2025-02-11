from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import pickle
import numpy as np
from model.image_caption_model import predict_caption, load_a_model, extract_features

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded_images'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}


model = load_a_model()



tokenizer_path = r"D:\zewail.city\pythonProject1\model\tokenizer.pkl"
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print("[INFO] POST request received.")

        if 'file' not in request.files:
            print("[ERROR] No file part in request.")
            return 'No file part'

        file = request.files['file']
        if file.filename == '':
            print("[ERROR] No selected file.")
            return 'No selected file'

        if file and allowed_file(file.filename):
            print("[INFO] File is allowed.")


            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            print(f"[DEBUG] File saved at: {file_path}, Exists: {os.path.exists(file_path)}")


            try:
                print(f"[DEBUG] Extracting features from: {file_path}")
                image_features = extract_features(file_path)


                print("[DEBUG] Extracted image features:", type(image_features))
                print("[DEBUG] Feature shape:", image_features.shape)
                print("[INFO] Image features extracted successfully.")
            except Exception as e:
                print(f"[ERROR] Feature extraction failed: {e}")
                return 'Error extracting features'

            try:
                caption = predict_caption( model, file_path , image_features=image_features, tokenizer = tokenizer )
                print(f"[INFO] Caption generated: {caption}")
            except Exception as e:
                print(f"[ERROR] Caption generation failed: {e}")
                return 'Error generating caption'

            return render_template('index.html', uploaded_image=file_path, caption=caption)

    return render_template('index.html', uploaded_image=None, caption=None)

if __name__ == '__main__':
    app.run(debug=True)
