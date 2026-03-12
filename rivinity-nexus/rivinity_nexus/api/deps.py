from collections.abc import Generator

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from sqlalchemy.orm import Session

from rivinity_nexus.auth.security import decode_token, hash_token
from rivinity_nexus.config.settings import get_settings
from rivinity_nexus.core.database import get_db_session
from rivinity_nexus.models.entities import ApiKey, User, UserRole

settings = get_settings()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)
api_key_scheme = APIKeyHeader(name=settings.api_key_header_name, auto_error=False)


def get_db() -> Generator[Session, None, None]:
    yield from get_db_session()


def _auth_from_bearer(token: str, db: Session) -> User | None:
    payload = decode_token(token)
    if not payload or payload.get("type") != "access":
        return None
    subject = payload.get("sub")
    if not subject:
        return None
    return db.query(User).filter(User.email == subject, User.is_active.is_(True)).first()


def _auth_from_api_key(raw_api_key: str, db: Session) -> User | None:
    hashed = hash_token(raw_api_key)
    key = db.query(ApiKey).filter(ApiKey.hashed_key == hashed, ApiKey.is_active.is_(True)).first()
    if not key:
        return None
    return db.query(User).filter(User.id == key.user_id, User.is_active.is_(True)).first()


def get_current_user(
    db: Session = Depends(get_db),
    bearer_token: str | None = Depends(oauth2_scheme),
    api_key: str | None = Depends(api_key_scheme),
) -> User:
    user = None
    if bearer_token:
        user = _auth_from_bearer(bearer_token, db)
    if not user and api_key:
        user = _auth_from_api_key(api_key, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
    return user


def require_role(required_role: UserRole):
    def _require_role(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role != required_role:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role permissions")
        return current_user

    return _require_role


def get_request_context(request: Request) -> tuple[str | None, str | None]:
    return request.headers.get("user-agent"), request.client.host if request.client else None
