from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# Load your trained model
model = tf.keras.models.load_model('fine_tuned_flood_detection_model.h5')

# Initialize FastAPI app
app = FastAPI()

# Define your class names
class_names = ['Flooding', 'No Flooding'] # Replace with your actual class names

# Function to preprocess the image
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")  # Ensure the image is in RGB format
    image = image.resize((224, 224))  # Resize to the expected size
    image = np.array(image) / 255.0   # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to decode predictions
def decode_predictions(predictions, class_names):
    decoded = []
    for pred in predictions:
        class_idx = np.argmax(pred)
        class_label = class_names[class_idx]
        decoded.append((class_label, float(pred[class_idx])))
    return decoded


@app.get("/home")
def home():
    return {"message": "Hello World", "health_check" : "OK"}


# Define a prediction endpoint
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image file
        image = Image.open(io.BytesIO(await file.read()))
        # Preprocess the image
        input_data = preprocess_image(image)
        # Make prediction
        prediction = model.predict(input_data)
        # Decode the prediction
        decoded_prediction = decode_predictions(prediction, class_names)
        return JSONResponse(content={'prediction': decoded_prediction})
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=400)

# To run the app: `uvicorn app:app --reload`

