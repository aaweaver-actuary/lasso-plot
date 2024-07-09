"""Database models and Pydantic models for data validation."""

from __future__ import annotations
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    ForeignKey,
    Float,
    DateTime,
    Text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from pydantic import BaseModel, Field, ValidationError
from typing import Optional
from datetime import datetime
from app.config.security import hasher

# SQLAlchemy setup
Base = declarative_base()
engine = create_engine("sqlite:///app.db")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class User(Base):
    """Represents a user of the app.

    Each user has a unique name and email, and a hashed password for security.

    Attributes
    ----------
    id : int, primary key, indexed
        The unique identifier for the user.
    name : str, indexed
        The user's name.
    email : str, unique, indexed
        The user's email address.
    hashed_password : str
        The hashed password for the user. Uses a secure hashing algorithm, and
        there is no way to retrieve the original password from the hash.
    user_settings : List[UserSettings]
        The settings unique to this user.
    user_sessions : List[Session]
        The list of sessions.id this user has been a part of.
    """

    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    user_settings = relationship("UserSettings", back_populates="user")
    user_sessions = relationship("Session", back_populates="user")
    incorrect_password_attempts = Column(Integer, default=0)

    def __repr__(self) -> str:
        """Return a string representation of the user."""
        return f"<User {self.name}>"

    def __str__(self) -> str:
        """Return the name of the user when converted to a string."""
        return self.name

    def password_attempt(self, attempt: str) -> tuple[bool, int]:
        """Check if a password attempt is correct."""
        if self.incorrect_password_attempts >= 3:
            return False, self.incorrect_password_attempts
        if hasher.verify(attempt, self.hashed_password):
            self.incorrect_password_attempts = 0
            return True, 0
        self.incorrect_password_attempts += 1
        return False, self.incorrect_password_attempts


class UserSettings(Base):
    """Represents the settings unique to a user.

    Each user can have multiple settings, each with a unique key and value.

    Attributes
    ----------
    id : int, primary key, indexed
        The unique identifier for the setting.
    user_id : int, foreign key
        The user this setting belongs to.
    setting_key : str, indexed
        The name of the setting.
    setting_value : str
        The value of the setting.
    """

    __tablename__ = "user_settings"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    setting_key = Column(String, index=True)
    setting_value = Column(Text)


class Session(Base):
    """Represents a session of the app.

    A new session occurs each time a user logs in, and expires either when
    the user logs out or after a period of inactivity.

    Attributes
    ----------
    id : int, primary key, indexed
        The unique identifier for the session.
    user_id : int, foreign key
        The user this session belongs to. Each session is associated with a
        single user.
    session_start : datetime, default=datetime.utcnow
        The UTC time when the session started. Defaults to the current time
        when a new session is created.
    session_end : datetime
        The UTC time when the session ended. If the session is still active,
        this field will be None.
    session_settings : List[SessionSettings]
        The settings unique to this session.
    transactions : List[Transaction]
        The list of transactions this session has been a part of.
    user : User
        The user this session belongs to.
    """

    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    session_start = Column(DateTime, default=datetime.utcnow)
    session_end = Column(DateTime)
    session_settings = relationship("SessionSettings", back_populates="session")
    transactions = relationship("Transaction", back_populates="session")
    user = relationship("User", back_populates="sessions")


class SessionSettings(Base):
    """Represents the settings unique to a session.

    Each session can have multiple settings, each with a unique key and value.

    Attributes
    ----------
    id : int, primary key, indexed
        The unique identifier for the setting.
    session_id : int, foreign key
        The session this setting belongs to.
    setting_key : str, indexed
        The name of the setting.
    setting_value : str
        The value of the setting.
    session : Session
        The session this setting belongs to.
    """

    __tablename__ = "session_settings"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))
    setting_key = Column(String, index=True)
    setting_value = Column(Text)
    session = relationship("Session", back_populates="session_settings")


class TransactionType(Base):
    """Represents a transaction type in the app.

    Each transaction type has a unique name and a description.

    Attributes
    ----------
    id : int, primary key, indexed
        The unique identifier for the transaction type.
    name : str, indexed
        The name of the transaction type.
    description : str
        The description of the transaction type.
    """

    __tablename__ = "transaction_types"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text)


class Transaction(Base):
    """Represents a transaction in the app.

    Each transaction is associated with a session and has a timestamp and an associated action.

    Attributes
    ----------
    id : int, primary key, indexed
        The unique identifier for the transaction.
    session_id : int, foreign key
        The session this transaction belongs to.
    user_id : int, foreign key
        The user this transaction belongs to.
    transaction_type_id : int, foreign key
        The type of the transaction.
    timestamp : datetime, default=datetime.utcnow
        The UTC time when the transaction occurred. Defaults to the current time
        when a new transaction is created.
    amount : float
        The amount of the transaction. Defaults to 0 and only changes when a
        transaction specifically does something numeric.
    """

    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    transaction_type_id = Column(Integer, ForeignKey("transaction_types.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    amount = Column(Float)


class GlobalSettings(Base):
    """Represents the global settings of the app.

    Each global setting has a unique key and value.

    Attributes
    ----------
    id : int, primary key, indexed
        The unique identifier for the setting.
    setting_key : str, indexed
        The name of the setting.
    setting_value : str
        The value of the setting.
    """

    __tablename__ = "global_settings"
    id = Column(Integer, primary_key=True, index=True)
    setting_key = Column(String, index=True)
    setting_value = Column(Text)


# Pydantic models for data validation
class UserModel(BaseModel):
    """Pydantic model for user data validation."""

    name: str
    email: str
    password: str


class UserSettingsModel(BaseModel):
    """Pydantic model for user settings data validation."""

    setting_key: str
    setting_value: str


class SessionModel(BaseModel):
    """Pydantic model for session data validation."""

    user_id: int
    session_start: Optional[datetime] = None
    session_end: Optional[datetime] = None


class SessionSettingsModel(BaseModel):
    """Pydantic model for session settings data validation."""

    setting_key: str
    setting_value: str


class TransactionTypeModel(BaseModel):
    """Pydantic model for transaction type data validation."""

    name: str
    description: str


class TransactionModel(BaseModel):
    """Pydantic model for transaction data validation."""

    session_id: int
    user_id: int
    transaction_type_id: int
    timestamp: datetime | None = None
    amount: float = 0.0


class GlobalSettingsModel(BaseModel):
    """Pydantic model for global settings data validation."""

    setting_key: str
    setting_value: str


# Create the tables
Base.metadata.create_all(bind=engine)


# Example usage
def create_user(db_session: Session, user: UserModel) -> User:
    """Create a new user in the database.

    Parameters
    ----------
    db_session : Session
        The database session to use.
    user : UserModel
        The user data to create. Uses the Pydantic model for validation.

    Returns
    -------
    User
        The user object created in the database.
    """
    # Create a new user
    db_user = User(name=user.name, email=user.email, password=user.password)

    # Add the user to the session and commit the changes
    db_session.add(db_user)
    db_session.commit()

    # Refresh the user object to get the updated values
    db_session.refresh(db_user)
    return db_user


def create_global_setting(
    db_session: Session, setting: GlobalSettingsModel
) -> GlobalSettings:
    """Create a new global setting in the database.

    Parameters
    ----------
    db_session : Session
        The database session to use.
    setting : GlobalSettingsModel
        The global setting data to create. Uses the Pydantic model for validation.

    Returns
    -------
    GlobalSettings
        The global setting object created in the database.
    """
    db_setting = GlobalSettings(
        setting_key=setting.setting_key, setting_value=setting.setting_value
    )
    db_session.add(db_setting)
    db_session.commit()
    db_session.refresh(db_setting)
    return db_setting


if __name__ == "__main__":
    db_session = SessionLocal()

    # Create a new user
    user = UserModel(
        name="John Doe", email="john.doe@example.com", password="securepassword"
    )
    try:
        db_user = create_user(db_session, user)
        print(f"Created user: {db_user}")
    except ValidationError as e:
        print(f"Error creating user: {e}")

    # Create a global setting
    global_setting = GlobalSettingsModel(
        setting_key="site_name", setting_value="Example Site"
    )
    try:
        db_setting = create_global_setting(db_session, global_setting)
        print(f"Created global setting: {db_setting}")
    except ValidationError as e:
        print(f"Error creating global setting: {e}")

    db_session.close()
