from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

# -------------------------------
# Flask App Initialization
# -------------------------------
app = Flask(__name__)

UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create static folder if not exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# -------------------------------
# Load Trained Model
# -------------------------------
model = load_model("pneumonia_model.h5", compile=False)

# -------------------------------
# Prediction Function (UPDATED)
# -------------------------------
def predict_pneumonia(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prob = model.predict(img)[0][0]        # value between 0 and 1
    percentage = round(prob * 100, 2)      # convert to percentage

    if prob > 0.5:
        result = "PNEUMONIA"
    else:
        result = "NORMAL"

    return result, percentage

# -------------------------------
# Route (UPDATED)
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    percentage = None

    if request.method == "POST":
        file = request.files["file"]

        if file and file.filename != "":
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            result, percentage = predict_pneumonia(filepath)

    return render_template("index.html", result=result, percentage=percentage)

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
