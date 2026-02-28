"""Password and token utilities."""

from datetime import UTC, datetime, timedelta

from jose import jwt
from passlib.context import CryptContext

from rivinityllm_tune.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bool(pwd_context.verify(plain_password, hashed_password))


def create_access_token(subject: str) -> str:
    expires = datetime.now(UTC) + timedelta(minutes=settings.access_token_ttl_minutes)
    payload = {"sub": subject, "exp": expires}
    return str(jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm))
