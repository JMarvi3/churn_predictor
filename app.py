import os
import pickle
from datetime import datetime, timezone
from functools import wraps
from urllib.parse import urlparse, urljoin

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from flask import Flask, render_template, redirect, url_for, request, \
    jsonify, abort, flash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from werkzeug.security import check_password_hash
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired

import database as db
from models.role import Role
from models.user import User, getUser
from models.login import Login, addLogin, getLastLoginDate


class Model:
    def make_images(self):
        self.make_heatmap()
        self.make_confusion()
        self.make_internet_churn()
        self.make_churn_by_phone()
        self.make_churn_by_internet_options()

    def make_heatmap(self):
        df2 = self.df.drop('Churn', axis=1)
        for arg in df2.columns:
            if df2[arg].dtype != 'int64' and df2[arg].dtype != 'float64':
                df2[arg] = pd.Categorical(df2[arg]).codes
        fig = plt.figure(figsize=(15, 15))
        df2.columns = ['Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Tenure', 'Phone Service', 'Multiple Lines',
                       'Internet Service', 'Online Security', 'Online Backup', 'Device Protection', 'Tech Support',
                       'StreamingTV', 'StreamingMovies', 'Contract', 'Paperless Billing', 'Payment Method',
                       'Monthly Charges', 'Total Charges']
        sns.set(font_scale=1)
        sns.heatmap(df2.corr().round(2), cbar=False, cmap='Blues', annot=True)
        plt.tight_layout()
        plt.grid(False)
        fig.savefig('static/img/heatmap.png', transparent=True)

    def make_confusion(self):
        fig = plt.figure()
        sns.set(font_scale=1.5)
        sns.heatmap(self.confusion_matrix.round(2), annot=True, cbar=False, cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.grid(False)
        fig.savefig('static/img/confusion.png', transparent=True)

    def make_internet_churn(self):
        df = self.df
        values = ['No', 'DSL', 'Fiber optic']
        data = [[df.loc[(df['InternetService'] == val) & (df['Churn'] == 'Yes')].size /
                 df[df['InternetService'] == val].size] for val in values]
        ax = pd.DataFrame(data, columns=['Churn'], index=['None' if val == 'No' else val for val in values]). \
            plot(kind='bar', legend=False, in_layout=True)
        plt.title('Churn by Internet Service')
        ax.set_ylim(0, np.amax(data) + .2)
        for spine in ax.spines.values():
            spine.set_color('black')
        for p in ax.patches:
            width = p.get_width()
            height = p.get_height()
            ax.annotate('{:.1%}'.format(height), (p.get_x() + .5 * width, p.get_y() + height + 0.01), ha='center')
        plt.yticks([])
        plt.tight_layout()
        plt.grid(False)
        plt.savefig('static/img/internet_churn.png', transparent=True)

    def make_churn_by_internet_options(self):
        df = self.df
        values = ['No', 'Yes']
        columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                   'StreamingMovies']
        column_labels = ['Online Security', 'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV',
                         'Streaming Movies']
        data = [[df.loc[(df[col] == val) & (df['Churn'] == 'Yes')].size / df[df[col] == val].size for val in values] for
                col in columns]
        ax = pd.DataFrame(data, columns=values, index=column_labels).plot(kind='bar',
                                                                          title='Churn by Optional Internet Services',
                                                                          figsize=(10, 8))
        ax.set_ylim(0, np.amax(data) + .2)
        for spine in ax.spines.values():
            spine.set_color('black')
        for p in ax.patches:
            width = p.get_width()
            height = p.get_height()
            ax.annotate('{:.1%}'.format(height), (p.get_x() + width, p.get_y() + height + 0.01), ha='center',
                        rotation=45)
        plt.yticks([])
        plt.tight_layout()
        plt.grid(False)
        plt.savefig('static/img/internet_option_churn.png', transparent=True)

    def make_churn_by_phone(self):
        df = self.df
        values = ['None', 'Single Line', 'Multiple Lines']
        data = [
            [df.loc[(df['PhoneService'] == 'No') & (df['Churn'] == 'Yes')].size /
             df[df['PhoneService'] == 'No'].size],
            [df.loc[(df['MultipleLines'] == 'No') & (df['Churn'] == 'Yes')].size /
             df[df['MultipleLines'] == 'No'].size],
            [df.loc[(df['MultipleLines'] == 'Yes') & (df['Churn'] == 'Yes')].size /
             df[df['MultipleLines'] == 'Yes'].size]]
        ax = pd.DataFrame(data, columns=['Churn'], index=values).plot(kind='bar', legend=False,
                                                                      title='Churn by Phone Service')
        plt.yticks([])
        for spine in ax.spines.values():
            spine.set_color('black')
        ax.set_ylim(0, np.amax(data) + .2)
        for p in ax.patches:
            width = p.get_width()
            height = p.get_height()
            ax.annotate('{:.1%}'.format(height), (p.get_x() + 0.5 * width, p.get_y() + height + 0.01), ha='center')
        plt.tight_layout()
        plt.grid(False)
        plt.savefig('static/img/phone_churn.png', transparent=True)

    def __init__(self):
        df = pd.read_csv('data/Telco-Customer-Churn.csv').drop('customerID', axis=1)
        # https://github.com/IBM/telco-customer-churn-on-icp4d
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').replace(np.nan, 0)
        x = df.drop('Churn', axis=1)
        y = df['Churn']
        cat_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
        transformer = ColumnTransformer([('enc', OneHotEncoder(), cat_features)], remainder='passthrough')
        x_transformed = transformer.fit_transform(x)
        clf = RandomForestClassifier(n_estimators=600, min_samples_split=4, min_samples_leaf=1, max_depth=None,
                                     class_weight={'Yes': 20})
        x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.8)
        clf.fit(x_train, y_train)
        y_preds = clf.predict(x_test)
        self.df = df
        self.x_test = x_test
        self.y_test = y_test
        self.clf = clf
        self.transformer = transformer
        self.seed = datetime.now(timezone.utc)
        self.score = round(100*clf.score(x_test, y_test), 2)
        self.confusion_matrix = confusion_matrix(y_test, y_preds) / len(y_test)
        self.classification_report = pd.DataFrame(classification_report(y_test, y_preds, output_dict=True)).T.round(3)
        os.makedirs('static/img', exist_ok=True)
        self.make_images()
        pickle.dump(self, open('model.pickle', 'wb'))


dotenv.load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
db.db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)

if not os.path.isfile('model.pickle'):
    model = Model()
else:
    model = pickle.load(open('model.pickle', 'rb'))


@login_manager.unauthorized_handler
def unauthorized():
    return redirect(url_for('login'))


@login_manager.user_loader
def load_user(user_id):
    # return db.getUser(user_id)
    return getUser(user_id)


@app.before_request
def ensure_secure():
    if not app.debug and request.scheme == 'http':
        return abort(403, description='You must use SSL/TLS to access this webpage.')


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # ensure_secure() will always be called before this
        if not any(role.name == 'admin' for role in current_user.roles):
            flash('You need to be an administrator to do that.')
            return redirect(request.referrer or url_for('index'))
        else:
            return f(*args, **kwargs)

    return decorated_function


@app.route('/login', methods=['GET', 'POST'])
def login():

    class LoginForm(FlaskForm):
        username = StringField('Username', [InputRequired()])
        password = PasswordField('Password', [InputRequired()])
        submit = SubmitField('Submit')

    def is_safe_url(target):
        ref_url = urlparse(request.host_url)
        test_url = urlparse(urljoin(request.host_url, target))
        return test_url.scheme in ('http', 'https') and ref_url.netloc == test_url.netloc

    error = None
    if current_user.is_authenticated:
        next_url = request.args.get('next')
        if not is_safe_url(next_url):
            abort(400)
        return redirect(next_url or url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = load_user(form.username.data)
        if user and check_password_hash(user.password, form.password.data):
            login_user(user)
            message = 'Logged in successfully.'
            last_login_date = getLastLoginDate(user.name)
            if last_login_date:
                message += f' Last logged in at {last_login_date}.'
            flash(message)
            addLogin(name=user.name, request=request)
            next_url = request.args.get('next')
            if not is_safe_url(next_url):
                abort(400)
            return redirect(next_url or url_for('index'))
        else:
            addLogin(name=form.username.data, request=request, success=False)
            error = 'Incorrect username or password.'
    return render_template('login.html', form=form, error=error)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/')
@login_required
def index():
    return render_template('index.html', model=model, now=datetime.now(timezone.utc), debug=app.debug)


@app.route('/rebuild')
@login_required
@admin_required
def rebuild():
    global model
    model = Model()
    return redirect(url_for('index'))


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict_form.html', debug=app.debug)
    else:
        columns = model.df.columns[:-1]
        values = [request.values[col] for col in columns]
        customer = pd.DataFrame(np.array(values, ndmin=2), columns=columns)
        customer['SeniorCitizen'] = customer['SeniorCitizen'].astype(np.int64)
        customer['tenure'] = customer['tenure'].astype(np.int64)
        customer['MonthlyCharges'] = customer['MonthlyCharges'].astype(np.float64)
        customer['TotalCharges'] = customer['TotalCharges'].astype(np.float64)
        customer_transformed = model.transformer.transform(customer)
        prediction = model.clf.predict(customer_transformed)[0]
        probabilities = model.clf.predict_proba(customer_transformed)[0]
        if prediction == 'Yes':
            probability = probabilities[1]
        else:
            probability = probabilities[0]
        return jsonify(probabilities=list(probabilities * 100),
                       prediction=f"Predict customer will {'not ' if prediction == 'No' else ''}"
                                  f"leave with probability {round(probability * 100)}%")


@app.route('/favicon.ico')
def favicon():
    return redirect(url_for('static', filename='favicon.ico'))


if app.debug:
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True

with app.app_context():
    db.db.create_all()

if __name__ == '__main__':
    app.run()
