
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='./templates')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the custom optimizer class if necessary
class CustomAdam(Adam):
    pass

model1 = load_model("densenet_model_epoch_v902.h5", custom_objects={'Adam': CustomAdam})
model2 = load_model("inception_model_epoch_06.h5", custom_objects={'Adam': CustomAdam})
model3 = load_model("xception_model_epoch_v102.h5", custom_objects={'Adam': CustomAdam})

class_names_path = 'classnames.txt'

def read_class_names(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

classes = read_class_names(class_names_path)

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('predict.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('predict.html', error='No selected file')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            img_array = preprocess_image(file_path)
            
            pred1 = model1.predict(img_array)
            pred2 = model2.predict(img_array)
            pred3 = model3.predict(img_array)
            print("Individual Predictions:")
            print("Model 1:", pred1)
            print("Model 2:", pred2)
            print("Model 3:", pred3)
            
            ensemble_pred = (pred1 + pred2 + pred3) / 3
            print("Ensemble Prediction:", ensemble_pred)

            predicted_class_index = np.argmax(np.mean(ensemble_pred, axis=0))
            predicted_class_name = classes[predicted_class_index]
            
            print("Predicted Class Index:", predicted_class_index)
            print("Predicted Class Name:", predicted_class_name)

            return render_template('result.html', predicted_breed=predicted_class_name)
    return render_template('predict.html', error=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
