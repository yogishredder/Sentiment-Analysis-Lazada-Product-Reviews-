from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
import numpy as np
import util as utils
import data_pipeline as data_pipeline
import preprocessing as preprocessing
import function as fc

config = utils.load_config()
model_data = utils.pickle_load(config["production_model_path"])

class api_data(BaseModel):
    cleaned_data : str
    review_len : float
    punct : float

app = FastAPI()

@app.get("/")
def home():
    return "Hello, FastAPI up!"

@app.post("/predict/")
def predict(data: api_data):    
    # Convert data api to dataframe
    data = pd.DataFrame(data).set_index(0).T.reset_index(drop = True)  # type: ignore
    data.columns = config["predictors"]