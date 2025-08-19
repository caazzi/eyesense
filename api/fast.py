from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from . import utils
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import operator

# 1. Receive the image
# 2. Prepare it (resize, normalize, tensor)
# 3. Load model
# 4. Predict

app = FastAPI()
app.state.model = utils.load_model_from_gcp()
# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/predict")
async def predict(img: UploadFile = File(...)):
    try:
        # Read and convert the image to a usable format
        contents = await img.read()
        image = Image.open(io.BytesIO(contents))

        X = utils.prepare_image(image, 299, 299)

        CLASS_NAMES = ['cataract', 'degeneration', 'diabetes', 'glaucoma', 'hypertension', 'myopia', 'normal']
        predictions = app.state.model.predict(X)

        dict_pred = {cls: round(float(pred), 4) for cls, pred in zip(CLASS_NAMES, predictions[0])}
        pred_list = sorted(dict_pred.items(), key=operator.itemgetter(1), reverse=True)

        return JSONResponse(content={"result":pred_list}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {'Is it working?': True}
