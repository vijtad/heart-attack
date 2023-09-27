import pickle
import uuid
import datetime
import numpy as np


import mlflow.pyfunc
model_name= "heart-attack-LR-model"
model = "models:/{model_name}/1".format(model_name=model_name)

# from domino_prediction_logging.prediction_client import PredictionClient
from domino_data_capture.data_capture_client import DataCaptureClient

features = ['thal','exang','cp','ca','sex','oldpeak','slope']

target = ["target"]

# pred_client = PredictionClient(features, target)
data_capture_client = DataCaptureClient(features, target)

def predict(thal,exang,cp,ca,sex,oldpeak,slope, _id=None):
    feature_values = [density, volatile_acidity, chlorides, is_red, alcohol]
    prediction = model.predict([feature_values]).tolist()


    # Record eventID and current time
    if _id is None:
        print("No ID found! Creating a new one.")
        wine_id = str(datetime.datetime.now())
        # custid = uuid.uuid4()
    print('ID is: {}'.format(_id))

    # pred_client.record(feature_values, prediction, event_id=custid)
    data_capture_client.capturePrediction(feature_values, prediction,event_id=_id)

    return dict(prediction=prediction[0])