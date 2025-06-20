import os
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# import gdown

<<<<<<< HEAD
# # URL from Google Drive (must be shareable)
# gdown.download("https://drive.google.com/uc?id=1jyxcjacRq-gAAxX-G55_3trQycDeVDHA", "dahlia_vgg_final.h5", quiet=False)


# output = "dahlia_vgg_final.h5"

# if not os.path.exists(output):
#     import gdown
#     gdown.download("https://drive.google.com/uc?id=1jyxcjacRq-gAAxX-G55_3trQycDeVDHA", output, quiet=False)

model = load_model('dahlia_vgg_final.h5')

# try:
#     model = load_model(output)
#     print("✅ Model loaded successfully!")
# except Exception as e:
#     print(f"❌ Error loading model: {e}")
=======
# URL from Google Drive (must be shareable)
gdown.download("https://drive.google.com/uc?id=1jyxcjacRq-gAAxX-G55_3trQycDeVDHA", "dahlia_vgg_final.h5", quiet=False)


output = "dahlia_vgg_final.h5"

if not os.path.exists(output):
    import gdown
    gdown.download("https://drive.google.com/uc?id=1jyxcjacRq-gAAxX-G55_3trQycDeVDHA", output, quiet=False)

try:
    model = load_model(output)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
>>>>>>> 5d74e14d764f6bb773d1b9a0ee2940bf102001bc


# Dahlia class names (update these based on your model training)
class_names = ['Anemone Dahlia', 'Ball Dahlia', 'Cactus Dahlia', 'Collarate Dahlia', 'Waterlily Dahlia', 'Non Dahlia','Object']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return "No file part in the request"

        file = request.files['file']

        if file.filename == '':
            return "No file selected"

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Load and preprocess image
            img = image.load_img(filepath, target_size=(224, 224))  
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Prediction
            prediction = model.predict(img_array)
            confidence = np.max(prediction)
            predicted_class = class_names[np.argmax(prediction)]
            
            # Check if the confidence is high enough to be a dahlia
            if confidence < 0.8:  
                return render_template('result.html', 
                                    image_file=filepath, 
                                    prediction="Not Recognized", 
                                    confidence=f"{confidence*100:.2f}%")
            
            return render_template('result.html', 
                                image_file=filepath, 
                                prediction=predicted_class,
                                confidence=f"{confidence*100:.2f}%")

        return "Invalid file type"

    except Exception as e:
        return f"Error occurred: {e}"


if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True)
