import pickle
import uuid
import datetime
import numpy as np

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import mlflow.pyfunc
model_name= "heart-attack-model-v2"
version = 1
model_uri = "models:/{model_name}/{version}".format(model_name=model_name,version=version)

model = mlflow.pyfunc.load_model(model_uri=model_uri)
#model = pickle.load(open('model.pkl', 'rb'))

# from domino_prediction_logging.prediction_client import PredictionClient
from domino_data_capture.data_capture_client import DataCaptureClient

features = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh',
       'exng', 'oldpeak', 'slp', 'caa', 'thall']

target = ["target"]

# pred_client = PredictionClient(features, target)
data_capture_client = DataCaptureClient(features, target)

def predict(age, sex, cp, trtbps, chol, fbs, restecg, thalachh,
       exng, oldpeak, slp, caa, thall, _id=None):
    
    data = [ {
        "age" : age, 
        "sex" : sex, 
        "cp" : cp, 
        "trtbps" : trtbps, 
        "chol" : chol, 
        "fbs" : fbs, 
        "restecg" : restecg, 
        "thalachh" : thalachh,
        "exng" : exng, 
        "oldpeak" : oldpeak, 
        "slp" : slp, 
        "caa" :caa, 
        "thall" :thall
    }]
    df = pd.DataFrame(data)
    print(df)

    prediction = model.predict(df).tolist()


    # Record eventID and current time
    if _id is None:
        print("No ID found! Creating a new one.")
        _id = str(datetime.datetime.now())
        # custid = uuid.uuid4()
    print('ID is: {}'.format(_id))

    feature_values=[age, sex, cp, trtbps, chol, fbs, restecg, thalachh,
       exng, oldpeak, slp, caa, thall]
    # pred_client.record(feature_values, prediction, event_id=custid)
    data_capture_client.capturePrediction(feature_values, prediction,event_id=_id)

    return dict(prediction=prediction[0])

#result = predict(1,1,1,1,1,1,1)