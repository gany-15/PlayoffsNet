from flask import Flask
from os import path

def nba():
    nba = Flask(__name__)
    from .models.auth import auth
    nba.register_blueprint(auth, url_prefix = "/")
    return nba