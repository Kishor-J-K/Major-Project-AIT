# app.py
from flask import Flask, request, render_template
import os
import torch
from torchvision import models
import torch.nn as nn
from src.inference import predict_class

app = Flask(__name__)

# --- Model Setup ---
# Define the number of classes (should match your labels.json)
NUM_CLASSES = 23

# Build the model architecture
model = models.resnet50(pretrained=False) # The warning about 'pretrained' is expected and can be ignored.
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) #
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES) #

# Load the trained model weights
# Use a relative path for better portability
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'sound_model.pth')
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# --- App Routes ---

@app.route('/')
def index():
    # Renders the main page with an empty result
    return render_template('index.html', result="")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', result="No file part in request.")
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', result="No file selected.")
    
    if file:
        # Ensure the uploads directory exists
        uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Save the uploaded file temporarily
        file_path = os.path.join(uploads_dir, file.filename)
        file.save(file_path)
        
        # Predict the class of the uploaded audio file
        predicted_class = predict_class(file_path, model)
        
        # Clean up the class name (replace underscores with spaces)
        display_result = f"Predicted Species: {predicted_class.replace('_', ' ')}"
        
        # Render the page again with the prediction result
        return render_template('index.html', result=display_result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
