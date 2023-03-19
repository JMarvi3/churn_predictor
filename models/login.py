from datetime import timezone
from database import db


class Login(db.Model):
    __tablename__ = "logins"

    id = db.Column(db.Integer, primary_key=True)
    host = db.Column(db.String)
    name = db.Column(db.String)
    ip = db.Column(db.String)
    success = db.Column(db.Boolean)
    date = db.Column(db.DateTime(timezone=True), server_default=db.sql.func.now())


def addLogin(name, request, success=True):
    db.session.add(Login(host=request.host.split(':')[0], name=name, ip=':'.join(request.access_route),
                         success=success))
    db.session.commit()


def format_date(date):
    utc_date = date.astimezone(timezone.utc)
    return utc_date.strftime('%B {}, %Y %r %Z').format(utc_date.day)


def getLastLoginDate(name):
    login = Login.query.filter_by(name=name, success=True).order_by(Login.date.desc()).limit(1).one_or_none()

    # login = db.query(Login).filter(Login.name == name, Login.success). \
    #     order_by(Login.date.desc()).limit(1).one_or_none()
    return format_date(login.date) if login else None
