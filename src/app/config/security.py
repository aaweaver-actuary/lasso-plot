"""Define the hash configuration for hashing passwords."""

from passlib.context import CryptContext

# Define the hash configuration for hashing passwords.
hasher = CryptContext(schemes=["bcrypt"], deprecated="auto")

__all__ = ["hasher"]
