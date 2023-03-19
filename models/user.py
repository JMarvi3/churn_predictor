from werkzeug.security import generate_password_hash
from database import db
from models.role import Role


roles = db.Table('user_role',
                 db.Column('user_name', db.String, db.ForeignKey('users.name'), primary_key=True),
                 db.Column('role_name', db.String, db.ForeignKey('roles.name'), primary_key=True)
                 )


class User(db.Model):
    __tablename__ = "users"

    name = db.Column(db.String, primary_key=True, index=True)
    password = db.Column(db.String, nullable=False)
    roles = db.relationship('Role', secondary=roles, lazy='subquery',
                            backref=db.backref('users', lazy=True))

    def is_authenticated(self):
        return True

    def is_active(self):
        return True

    def is_anonymous(self):
        return False

    def get_id(self):
        return self.name


def getUser(name):
    return User.query.filter_by(name=name).first()


def changePassword(user, password):
    user.password = generate_password_hash(password)
    db.session.commit()


def addUser(name, password, roles=None):
    user = User.query.filter_by(name=name).first() or User(name=name)
    user.password = generate_password_hash(password)
    if roles:
        for role_name in roles:
            role = Role.query.filter_by(name=role_name).first() or Role(name=role_name)
            user.roles.append(role)
    db.session.add(user)
    db.session.commit()
    return user


def removeUser(user: User):
    # user.query.filter_by(user=user).delete()
    # user.delete()
    db.session.delete(user)
    db.session.commit()

def addRole(user, role_name):
    role = Role.query.filter_by(name=role_name).first() or Role(name=role_name)
    if role not in user.roles:
        user.roles.append(role)
        db.session.commit()


def removeRole(user, role_name):
    role = Role.query.filter_by(name=role_name).first()
    if role:
        user.roles.remove(role)
    db.session.commit()
