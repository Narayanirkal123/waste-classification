from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Labels for the 9 waste categories
labels = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Paper', 'Plastic', 'Miscellaneous Trash', 'Textile Trash', 'Vegetation']

# Load TFLite models
densenet_interpreter = tf.lite.Interpreter(model_path='models/densenet169.tflite')
inception_interpreter = tf.lite.Interpreter(model_path='models/inceptionv4.tflite')
efficientnet_interpreter = tf.lite.Interpreter(model_path='models/efficientnetb2.tflite')

# Allocate tensors for each interpreter
densenet_interpreter.allocate_tensors()
inception_interpreter.allocate_tensors()
efficientnet_interpreter.allocate_tensors()

# Get input and output tensor details
densenet_input_details = densenet_interpreter.get_input_details()
densenet_output_details = densenet_interpreter.get_output_details()
inception_input_details = inception_interpreter.get_input_details()
inception_output_details = inception_interpreter.get_output_details()
efficientnet_input_details = efficientnet_interpreter.get_input_details()
efficientnet_output_details = efficientnet_interpreter.get_output_details()

# Image preprocessing (assumes 224x224 input size, adjust if needed)
def preprocess_image(image: Image.Image, target_size=(224, 224)):
    image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess image
    image = Image.open(io.BytesIO(await file.read()))
    img_array = preprocess_image(image)

    # Run predictions for each model
    # DenseNet169
    densenet_interpreter.set_tensor(densenet_input_details[0]['index'], img_array)
    densenet_interpreter.invoke()
    densenet_pred = densenet_interpreter.get_tensor(densenet_output_details[0]['index'])

    # InceptionV4
    inception_interpreter.set_tensor(inception_input_details[0]['index'], img_array)
    inception_interpreter.invoke()
    inception_pred = inception_interpreter.get_tensor(inception_output_details[0]['index'])

    # EfficientNetB2
    efficientnet_interpreter.set_tensor(efficientnet_input_details[0]['index'], img_array)
    efficientnet_interpreter.invoke()
    efficientnet_pred = efficientnet_interpreter.get_tensor(efficientnet_output_details[0]['index'])

    # Get predicted classes
    densenet_class = labels[np.argmax(densenet_pred[0])]
    inception_class = labels[np.argmax(inception_pred[0])]
    efficientnet_class = labels[np.argmax(efficientnet_pred[0])]

    # Ensemble: majority voting
    predictions = [densenet_class, inception_class, efficientnet_class]
    ensemble_class = max(set(predictions), key=predictions.count)

    return {
        "densenet169": densenet_class,
        "inceptionv4": inception_class,
        "efficientnetb2": efficientnet_class,
        "ensemble": ensemble_class
    }

@app.get("/")
async def root():
    return {"message": "Waste Classification API"}