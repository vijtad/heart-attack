import subprocess

# This is a sample Python/Flask app showing Domino's App publishing functionality.
# You can publish an app by clicking on "Publish" and selecting "App" in your
# quick-start project.

import json
import flask
from flask import request, redirect, url_for
import numpy as np
import logging
import requests
import sys
import time

class ReverseProxied(object):
  def __init__(self, app):
      self.app = app
  def __call__(self, environ, start_response):
      script_name = environ.get('HTTP_X_SCRIPT_NAME', '')
      if script_name:
          environ['SCRIPT_NAME'] = script_name
          path_info = environ['PATH_INFO']
          if path_info.startswith(script_name):
              environ['PATH_INFO'] = path_info[len(script_name):]
      # Setting wsgi.url_scheme from Headers set by proxy before app
      scheme = environ.get('HTTP_X_SCHEME', 'https')
      if scheme:
        environ['wsgi.url_scheme'] = scheme
      # Setting HTTP_HOST from Headers set by proxy before app
      remote_host = environ.get('HTTP_X_FORWARDED_HOST', '')
      remote_port = environ.get('HTTP_X_FORWARDED_PORT', '')
      if remote_host and remote_port:
          environ['HTTP_HOST'] = f'{remote_host}:{remote_port}'
      return self.app(environ, start_response)

app = flask.Flask(__name__)
#app.wsgi_app = ReverseProxied(app.wsgi_app)

# Homepage which uses a template file
@app.route('/')
def index_page():
  return flask.render_template("index.html")

@app.route('/', methods=['POST'])
def my_form_post():
    age = request.form['age']
    print('age', age, ' : ', type(age))
    sex = request.form['sex']
    print('sex', sex, ' : ', type(sex))
    cp = request.form['cp']
    trtbps = request.form['trtbps']
    chol = request.form['chol']
    fbs = request.form['fbs']
    restecg = request.form['restecg']
    thalachh = request.form['thalachh']

    exng = request.form['exng']
    oldpeak = request.form['oldpeak']
    slp = request.form['slp']

    caa = request.form['caa']
    thall = request.form['thall']

    data = {
        "age": float(age),
        "sex": float(sex),
        "cp": float(cp),
        "trtbps": float(trtbps),
        "chol": float(chol),
        "fbs": float(fbs),
        "restecg": float(restecg),
        "thalachh": float(thalachh),
        "exng": float(exng),
        "oldpeak": oldpeak,
        "slp": float(slp),
        "caa": float(caa),
        "thall": float(thall)
    }


    # change logging setup as required
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)  

    # TO EDIT: update the example request parameters for your model
    REQUEST_PARAMETERS = data
    
    # TO EDIT: copy these values from "Calling your Model" on the Model API overview page
    DOMINO_URL = "https://cdc-sandbox.domino-eval.com:443"
    MODEL_ID = "6536d231f66ed97e65981017"
    MODEL_ACCESS_TOKEN = "fc0xyHJJhEjWgUUouPSiMOo91Si3dSNQ4HXvnvzEep5WgxjXeajzjU1EzCcililo"
 
    # DO NOT EDIT these values
    MODEL_BASE_URL = f"{DOMINO_URL}/api/modelApis/async/v1/{MODEL_ID}"
    SUCCEEDED_STATUS = "succeeded"
    FAILED_STATUS = "failed"
    QUEUED_STATUS = "queued"
    TERMINAL_STATUSES = [SUCCEEDED_STATUS, FAILED_STATUS]
    PENDING_STATUSES = [QUEUED_STATUS]
    MAX_RETRY_DELAY_SEC = 60


    ### CREATE REQUEST ###

    create_response = None
    retry_delay_sec = 0
    while (
        create_response is None
        or (500 <= create_response.status_code < 600)  # retry for transient 5xx errors
    ):
        # status polling with a time interval that backs off up to MAX_RETRY_DELAY_SEC
        if retry_delay_sec > 0:
            time.sleep(retry_delay_sec)
        retry_delay_sec = min(max(retry_delay_sec * 2, 1), MAX_RETRY_DELAY_SEC)
        
        create_response = requests.post(
            MODEL_BASE_URL,
            headers={"Authorization": f"Bearer {MODEL_ACCESS_TOKEN}"},
            json={"parameters": REQUEST_PARAMETERS}
        )

    if create_response.status_code != 200:
        raise Exception(f"create prediction request failed, response: {create_response}")

    prediction_id = create_response.json()["asyncPredictionId"]
    logging.info(f"prediction id: {prediction_id}")


    ### POLL STATUS AND RETRIEVE RESULT ###

    status_response = None
    retry_delay_sec = 0
    while (
            status_response is None
            or (500 <= status_response.status_code < 600)  # retry for transient 5xx errors
            or (status_response.status_code == 200 and status_response.json()["status"] in PENDING_STATUSES)
    ):
        # status polling with a time interval that backs off up to MAX_RETRY_DELAY_SEC
        if retry_delay_sec > 0:
            time.sleep(retry_delay_sec)
        retry_delay_sec = min(max(retry_delay_sec * 2, 1), MAX_RETRY_DELAY_SEC)

        status_response = requests.get(
            f"{MODEL_BASE_URL}/{prediction_id}",
            headers={"Authorization": f"Bearer {MODEL_ACCESS_TOKEN}"},
        )

    if status_response.status_code != 200:
        raise Exception(f"prediction status request failed, response: {create_response}")

    prediction_status = status_response.json()["status"]
    if prediction_status == SUCCEEDED_STATUS:  # succeeded response includes the prediction result in "result"
        result = status_response.json()["result"]
        logging.info(f"prediction succeeded, result:\n{json.dumps(result, indent = 2)}")
    elif prediction_status == FAILED_STATUS:  # failed response includes the error messages in "errors"
        errors = status_response.json()["errors"]
        logging.error(f"prediction failed, errors:\n{json.dumps(errors, indent = 2)}")
    else:
        raise Exception(f"unexpected terminal prediction response status: {prediction_status}") 
    print(response.status_code)
    print(response.headers)
    print(response.json())

    return response.json()


