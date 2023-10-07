import mlflow


# define a custom model
class heart_attack_model(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import pickle
        with open(context.artifacts["heart_attack_model"], "rb") as f:
            self.model = pickle.load(f)
            # from domino_prediction_logging.prediction_client import PredictionClient
            from domino_data_capture.data_capture_client import DataCaptureClient

            features = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh',
                   'exng', 'oldpeak', 'slp', 'caa', 'thall']

            target = ["target"]

            # pred_client = PredictionClient(features, target)
            self.data_capture_client = DataCaptureClient(features, target)

#    def predict(age, sex, cp, trtbps, chol, fbs, restecg, thalachh,
#           exng, oldpeak, slp, caa, thall, _id=None):
    def predict(self, context, model_input):

        '''
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
        '''
        print(type(model_input))
        print(model_input)
        #print('age', model_input[0].age)
        #data = [model_input]
        #df = pd.DataFrame(data)
        #print(df)

        prediction = self.model.predict(model_input).tolist()


        # Record eventID and current time
        _id = str(datetime.datetime.now())
        # custid = uuid.uuid4()
        print('ID is: {}'.format(_id))

        feature_values=[model_input.age, model_input.sex, model_input.cp, model_input.trtbps, model_input.chol, 
                        model_input.fbs, model_input.restecg, model_input.thalachh,
           model_input.exng, model_input.oldpeak, model_input.slp, model_input.caa, model_input.thall]
        # pred_client.record(feature_values, prediction, event_id=custid)
        self.data_capture_client.capturePrediction(model_input.values.tolist()[0], prediction,event_id=_id)

        return dict(prediction=prediction[0])


