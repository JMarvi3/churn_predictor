@ECHO OFF
if not exist venv (
    python -m venv venv
    call venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate
)
set FLASK_DEBUG=True
set FLASK_APP=app.py
flask run