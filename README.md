# Dahlia Flower Classification App

A web application that classifies Dahlia flower images using a pre-trained CNN model.

## Setup

1. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Make sure the model file (`dahlia_vgg_final.h5`) is in the root directory of the project.

3. Run the Flask application:

   ```
   python app.py
   ```

4. Open your browser and navigate to `http://127.0.0.1:5000` to use the application.

## Usage

1. Click on the "Choose File" button to select a Dahlia flower image (supported formats: JPG, JPEG, PNG).
2. Click "Classify" to upload the image and get the classification result.
3. The application will display the predicted Dahlia type and the confidence level.

## Notes

- The model was trained on various Dahlia flower types.
- For best results, use clear images of Dahlia flowers with minimal background distractions.
- The input images are resized to 224x224 pixels for classification.
