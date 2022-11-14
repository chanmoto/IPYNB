import bcrypt

from db.connection import session
from db.models import Users
from models import models



def getAll():
    users = session.query(Users.fullname, Users.email, Users.password, Users.isOlder).order_by(Users.id)
    return [user for user in users]


def getById(id):
    user = session.query(Users.fullname, Users.email, Users.password, Users.isOlder).filter(
        Users.id == id
        ).first()

    return user


def addOne(user:models.Users):
    hashed_pwd = bcrypt.hashpw(bytes(user.password, encoding="UTF-8"), bcrypt.gensalt(12))
    user_data = Users(fullname=user.fullname, email=user.email, password=hashed_pwd.decode("UTF-8"), isOlder=user.isOlder)

    session.add(user_data)
    session.commit()

    return


def modifyById(user:models.UsersOptional, id):
    if user.fullname:
        session.query(Users).filter(Users.id == id).update(
            {
                Users.fullname: user.fullname
            }
        )
        session.commit()

        return

    elif user.email:
        session.query(Users).filter(Users.id == id).update(
            {
                Users.email: user.email
            }
        )
        session.commit()

        return

    elif user.password:
        hashed_pwd = bcrypt.hashpw(bytes(user.password, encoding="UTF-8"), bcrypt.gensalt(12))
        session.query(Users).filter(Users.id == id).update(
            {
                Users.password: hashed_pwd.decode("UTF-8")
            }
        )
        session.commit()
        
        return

    elif str(user.isOlder):
        session.query(Users).filter(Users.id == id).update(
            {
                Users.isOlder: user.isOlder
            }
        )
        session.commit()

        return
    else:
        return True


def updateById(user:models.Users, id):
    hashed_pwd = bcrypt.hashpw(bytes(user.password, encoding="UTF-8"), bcrypt.gensalt(12))
    session.query(Users).filter(Users.id == id).update(
        {
            Users.fullname: user.fullname,
            Users.email: user.email,
            Users.password: hashed_pwd.decode("UTF-8"),
            Users.isOlder: user.isOlder
        }
    )
    session.commit()
    
    return


def removeById(id:int):
    session.query(Users).filter(Users.id == id).delete()
    session.commit()
    
    return
