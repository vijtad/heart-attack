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
        "age": number(age),
        "sex": number(sex),
        "cp": number(cp),
        "trtbps": number(trtbps),
        "chol": number(chol),
        "fbs": number(fbs),
        "restecg": number(restecg),
        "thalachh": number(thalachh),
        "exng": number(exng),
        "oldpeak": oldpeak,
        "slp": number(slp),
        "caa": number(caa),
        "thall": number(thall)
    }


    # change logging setup as required
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)  

    DOMINO_URL = 'https://cdc-sandbox.domino-eval.com'
    MODEL_ID = '6536d231f66ed97e65981017'
    MODEL_ACCESS_TOKEN = 'CueoJUbSdWoGigkCCQNNfoa89eC8f0xZKuS0LUNyL3ijayyhSIj3HL4Qulvwmbtx'
    AUTH=(f"{MODEL_ACCESS_TOKEN}", f"{MODEL_ACCESS_TOKEN}")

    response = requests.post(f"{DOMINO_URL}/models/{MODEL_ID}/latest/model",
       auth=AUTH,
       json={"data": data}
    )

    print(response.status_code)
    print(response.headers)
    logging.info(response.json())

    return response.json()


