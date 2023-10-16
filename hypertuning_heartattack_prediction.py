#!/usr/bin/env python
# coding: utf-8

# # Model Experimentation Tracking (MLFow) - Hyperparamter Optimization

# Record and query experiments: Code, data, config, results, parameters, metrics
# 
# ![Data](images/MLflow_Model_experimentation.png)

# ## Import Packages

# In[3]:


# Data analysis library
import numpy as np
import pandas as pd
import joblib

# Machine Learning library
import sklearn
from sklearn.metrics import roc_curve, auc, accuracy_score, plot_confusion_matrix, plot_roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import LGBMClassifier
from lightgbm import plot_importance, plot_metric

# Model experimentation library
import mlflow
import mlflow.lightgbm
from mlflow.tracking import MlflowClient

# Hyperparameter tunning library
import optuna

# Plotting library
import matplotlib.pyplot as plt
# Prevent figures from displaying by turning interactive mode off using the function
plt.ioff()
import warnings
warnings.filterwarnings("ignore")


# In[4]:


print(f'Numpy version is {np.__version__}')
print(f'Pandas version is {pd.__version__}')
print(f'sklearn version is {sklearn.__version__}')
print(f'mlflow version is {mlflow.__version__}')
print(f'joblib version is {joblib.__version__}')
print(f'optuna version is {optuna.__version__}')


# ## Load data

# In[5]:


## Files
data_file='/mnt/data/Heart-Attack-prediction/heart.csv'

# Load train loan dataset 
try:
    data = pd.read_csv(data_file)
    print("The dataset has {} samples with {} features.".format(*data.shape))
except:
    print("The dataset could not be loaded. Is the dataset missing?")
    


# ## Introduction To The Data

# In[6]:


data.head()


# In[7]:


data['output'].value_counts()


# ## Initialize MLflow
# 
# **Experiments** : You can organize runs into experiments, which group together runs for a specific task. 
# 
# **Tracking URI**: MLflow runs can be recorded to local files, to a database, or remotely to a tracking server. By default, the MLflow Python API logs runs locally to files in an mlruns directory wherever you ran your program
# 
# #### MLflow Tracking Servers 
# MLflow tracking server has two components for storage: a **backend store** and an **artifact store**
# 
# The **backend store** is where MLflow Tracking Server stores experiment and run metadata as well as params, metrics, and tags for runs. MLflow supports two types of backend stores: **file store and database-backed store**.
# 
# The **artifact store** is a location suitable for large data (such as an S3 bucket or shared NFS file system) and is where clients log their artifact output (for example, models).
# 
#     Amazon S3 and S3-compatible storage
#     Azure Blob Storage
#     Google Cloud Storage
#     FTP server
#     SFTP Server
#     NFS
#     HDFS

# In[8]:


experiment_name = "heart_attack_predictions_v2"

# Initialize client
client = MlflowClient()

# If experiment doesn't exist then it will create new
# else it will take the experiment id and will use to to run the experiments
try:
    # Create experiment 
    experiment_id = client.create_experiment(experiment_name)
except:
    # Get the experiment id if it already exists
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id


# ## Prepare data for model training

# In[10]:


exclude_feature = []
# Define Target columns
target = data['output']

# Define numeric and categorical features
numeric_features = [ 'age', 'trtbps', 'chol', 'thalachh', 'oldpeak' ]
categorical_features = [ 'sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall' ]


# Define final feature list for training and validation
features = numeric_features + categorical_features
# Final data for training and validation
data = data[features]
data = data.fillna(0)

# Split data in train and vlaidation
X_train, X_valid, y_train, y_valid = train_test_split(data, target, test_size=0.15, random_state=10)

# Perform label encoding for categorical variable
for feature in categorical_features:
    le = LabelEncoder()
    le.fit(X_train.loc[:, feature])
    X_train.loc[:, feature] = le.transform(X_train.loc[:, feature])
    X_valid.loc[:, feature] = le.transform(X_valid.loc[:, feature])


# ## Lightgbm Hyperparameter tunning + MLFlow for model tracking

# ### Define model training function to train and track model results

# In[11]:


def model_training_tracking(params):
    # Launching Multiple Runs in One Program.This is easy to do because the ActiveRun object returned by mlflow.start_run() is a 
    # Python context manager. You can “scope” each run to just one block of code as follows:
    with mlflow.start_run(experiment_id=experiment_id, run_name='Lightgbm_model') as run:
        # Get run id 
        run_id = run.info.run_uuid
        
        # Set the notes for the run
        MlflowClient().set_tag(run_id,
                               "mlflow.note.content",
                               "This is experiment for hyperparameter optimzation for lightgbm models for the Campus Recruitment Dataset")
        
        # Define customer tag
        tags = {"Application": "Payment Monitoring Platform",
                "release.candidate": "PMP",
                "release.version": "2.2.0"}

        # Set Tag
        mlflow.set_tags(tags)
                        
        # Log python environment details
        mlflow.log_artifact('requirements.txt')
        
        # logging params
        mlflow.log_params(params)

        # Perform model training
        lgb_clf = LGBMClassifier(**params)
        lgb_clf.fit(X_train, y_train, 
                    eval_set = [(X_train, y_train), (X_valid, y_valid)], 
                    early_stopping_rounds=50,
                    verbose=20)

        # Log model artifacts
        mlflow.sklearn.log_model(lgb_clf, "model")

        # Perform model evaluation 
        lgb_valid_prediction = lgb_clf.predict_proba(X_valid)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_valid, lgb_valid_prediction)
        roc_auc = auc(fpr, tpr) # compute area under the curve
        print("=====================================")
        print("Validation AUC:{}".format(roc_auc))
        print("=====================================")   

        # log metrics
        mlflow.log_metrics({"Validation_AUC": roc_auc})

        # Plot and save feature importance details
        ax = plot_importance(lgb_clf, height=0.4)
        filename = './images/lgb_validation_feature_importance.png'
        plt.savefig(filename)
        # log model artifacts
        mlflow.log_artifact(filename)

        ax = plot_metric(lgb_clf.evals_result_)
        filename = './images/lgb_validation_metrics_comparision.png'
        plt.savefig(filename)
        # log model artifacts
        mlflow.log_artifact(filename)

        # Plot and save metrics details    
        plot_confusion_matrix(lgb_clf, X_valid, y_valid, 
                              display_labels=['Placed', 'Not Placed'],
                              cmap='magma')
        plt.title('Confusion Matrix')
        filename = './images/lgb_validation_confusion_matrix.png'
        plt.savefig(filename)
        # log model artifacts
        mlflow.log_artifact(filename)

        # Plot and save AUC details  
        plot_roc_curve(lgb_clf, X_valid, y_valid, name='Validation')
        plt.title('ROC AUC Curve')
        filename = './images/lgb_validation_roc_curve.png'
        plt.savefig(filename)
        # log model artifacts
        mlflow.log_artifact(filename)
        
        return roc_auc


# ### Define an objective function to be maximized

# In[12]:


def objective(trial):

    param = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 1e-1, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "random_state": 42,
    }
    
    auc = model_training_tracking(param)
    return auc


# ### Create a study object and optimize the objective function

# In[15]:


# Create a study object and optimize the objective function.
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=40)
trial = study.best_trial
print('AUC: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))


# ## Load best lightgbm model
# 
# Check Mlflow UI and pick the best model for model deployment

# In[78]:


import mlflow 
client = mlflow.tracking.MlflowClient()
runs = client.search_runs(experiment_id, "", order_by=["metrics.auc DESC"], max_results=1)

#Fetching Run ID for
run_id = runs[0]._info.run_id

#best_run_id = runs[0].params #['mlflow.domino.run_id']
lgb_best_model = mlflow.pyfunc.load_model("runs:/" + run_id + "/model")


# In[79]:


# Make prediction aganist Validation data
lgb_best_val_prediction = lgb_best_model.predict(X_valid)
lgb_best_val_prediction


# In[80]:


import mlflow
import pickle
import datetime


# define a custom model
class heart_attack_model(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
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

        predictions = self.model.predict(model_input).tolist()

        print('prediction        : ', type(predictions))
        print('prediction Values : ', predictions)

        # Record eventID and current time
        _id = str(datetime.datetime.now())
        # custid = uuid.uuid4()
        print('ID is: {}'.format(_id))

        feature_values=[model_input.age, model_input.sex, model_input.cp, model_input.trtbps, model_input.chol, 
                        model_input.fbs, model_input.restecg, model_input.thalachh,
           model_input.exng, model_input.oldpeak, model_input.slp, model_input.caa, model_input.thall]
        feature_values = model_input.values.tolist()
        print('feature_values', feature_values)
        # pred_client.record(feature_values, prediction, event_id=custid)
        for i in range(len(feature_values)) :
            print('feature_values[i] : ', feature_values[i])
            print('predictions[i] : ', predictions[i])
            self.data_capture_client.capturePrediction(feature_values[i], [predictions[i]],event_id=_id)

        return dict(prediction=predictions)




# In[91]:


import os 
import pickle
#mlflow.end_run()
from mlflow.models.signature import infer_signature
model_tmp = './'
registered_model = "lgb_heart_attack_model"
heart_attack_model_dir_path = os.path.join(model_tmp, "model/model.pkl")
run = mlflow.get_run(run_id)
with mlflow.start_run(run_id=run_id) :

    with open(heart_attack_model_dir_path, "wb") as f:
        pickle.dump(lgb_best_model, f)


    # Create a dictionary to tell MLflow where the necessary artifacts are
    artifacts = {
                "heart_attack_model": heart_attack_model_dir_path,
            }
    model_signature = infer_signature(X_train, 
                  lgb_best_model.predict(X_train))

    mlflow.pyfunc.log_model(
                    artifact_path='heart_attack_model_path',
                    python_model=heart_attack_model(),
                    code_path=["./model_src"],
                    artifacts=artifacts,
                    registered_model_name=registered_model,
                    signature=model_signature
                )



# ## Reference
# 
# ### Model experimentation
# https://www.mlflow.org/docs/latest/tracking.html#
# 
# ### Hyperparameter Optimization
# https://github.com/optuna/optuna
