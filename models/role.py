from database import db


class Role(db.Model):
    __tablename__ = "roles"

    name = db.Column(db.String, primary_key=True, index=True)