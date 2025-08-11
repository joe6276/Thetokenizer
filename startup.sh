#!/bin/bash
set -e

# Azure sets $PORT automatically
export FLASK_APP=app.py
export FLASK_ENV=production
export PYTHONPATH=/home/site/wwwroot

cd /home/site/wwwroot

# Start the app with Gunicorn (faster, production-ready)
gunicorn --bind=0.0.0.0:$PORT app:app
