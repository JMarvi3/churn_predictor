if [ ! -d venv ]; then
	python3 -m venv venv
	source venv/bin/activate
	echo 'Installing'
	pip install -r requirements.txt
else
	source venv/bin/activate
fi
FLASK_APP=app.py flask run
