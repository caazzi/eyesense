from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from utils import prepare_image, load_model_from_gcp
from PIL import Image
import io
import numpy as np

# Receberia a imagem
# Prepararia ela (size, normalization, tensor)
# Load model
# Predict

app = FastAPI()
app.state.model = load_model_from_gcp()
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
        # Ler e converter a imagem para um formato utilizável
        contents = await img.read()
        image = Image.open(io.BytesIO(contents))

        X = prepare_image(image, 299, 299)
         
        CLASS_NAMES = ['cataract', 'degeneration', 'diabets', 'glaucoma', 'hypertension', 'myopia', 'normal', 'others']
        predictions = app.state.model.predict(X)
        predicted_class = CLASS_NAMES[np.argmax(predictions, axis=1)[0]]

        return JSONResponse(content={"result":predicted_class}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {'Is it working?': True}
