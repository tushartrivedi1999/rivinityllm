from datetime import datetime, timedelta
import hashlib
import secrets

from jose import JWTError, jwt
from passlib.context import CryptContext

from rivinity_nexus.config.settings import get_settings

settings = get_settings()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    return pwd_context.verify(password, hashed_password)


def hash_token(raw_value: str) -> str:
    return hashlib.sha256(raw_value.encode("utf-8")).hexdigest()


def create_access_token(subject: str, role: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=settings.jwt_access_token_expire_minutes)
    payload = {"sub": subject, "role": role, "type": "access", "exp": expire}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_refresh_token(subject: str) -> tuple[str, datetime]:
    expire = datetime.utcnow() + timedelta(days=settings.jwt_refresh_token_expire_days)
    raw = secrets.token_urlsafe(48)
    payload = {"sub": subject, "jti": raw, "type": "refresh", "exp": expire}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm), expire


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except JWTError:
        return None


def create_api_key() -> tuple[str, str, str]:
    raw = f"rnx_{secrets.token_urlsafe(32)}"
    prefix = raw[:12]
    return raw, prefix, hash_token(raw)
