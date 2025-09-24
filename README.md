# Audio Classification Web Application

This project is a web application for classifying audio files using a trained deep learning model. The application allows users to upload audio files, which are then processed and classified by the model, returning the predicted class name.

## Project Structure

```
audio-classification-webapp
├── app.py                # Main entry point of the web application
├── requirements.txt      # List of dependencies
├── model
│   └── sound_model.pth   # Trained model weights
├── src
│   ├── inference.py      # Functions for model inference
│   └── utils.py          # Utility functions for audio processing
├── templates
│   └── index.html        # HTML template for the main page
├── static
│   └── style.css         # CSS styles for the web application
└── README.md             # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd audio-classification-webapp
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```
   python app.py
   ```

5. **Access the web application:**
   Open your web browser and go to `http://127.0.0.1:5000`.

## Usage Guidelines

- Use the provided form on the main page to upload an audio file in `.wav` format.
- After uploading, the application will process the audio and display the predicted class name.

## Additional Information

- Ensure that the model weights (`sound_model.pth`) are located in the `model` directory.
- Modify the `requirements.txt` file to add any additional libraries as needed for your specific use case.