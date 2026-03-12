from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from rivinity_nexus.api.deps import get_current_user, get_db, get_request_context, require_role
from rivinity_nexus.auth.security import (
    create_access_token,
    create_api_key,
    create_refresh_token,
    decode_token,
    hash_password,
    hash_token,
    verify_password,
)
from rivinity_nexus.data.schemas import (
    ApiKeyCreateRequest,
    ApiKeyCreateResponse,
    ApiKeyResponse,
    LoginRequest,
    RefreshTokenRequest,
    RefreshTokenResponse,
    SignupRequest,
    TokenResponse,
    UserResponse,
)
from rivinity_nexus.models.entities import ApiKey, Session as UserSession, User, UserRole

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def signup(payload: SignupRequest, db: Session = Depends(get_db)) -> User:
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    user = User(
        email=payload.email,
        full_name=payload.full_name,
        hashed_password=hash_password(payload.password),
        role=UserRole.user,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, request: Request, db: Session = Depends(get_db)) -> TokenResponse:
    user = db.query(User).filter(User.email == payload.email, User.is_active.is_(True)).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    access_token = create_access_token(user.email, user.role.value)
    refresh_token, expires_at = create_refresh_token(user.email)
    user_agent, ip_address = get_request_context(request)

    db.add(
        UserSession(
            user_id=user.id,
            refresh_token_hash=hash_token(refresh_token),
            user_agent=user_agent,
            ip_address=ip_address,
            expires_at=expires_at,
        )
    )
    db.commit()

    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=RefreshTokenResponse)
def refresh(payload: RefreshTokenRequest, db: Session = Depends(get_db)) -> RefreshTokenResponse:
    decoded = decode_token(payload.refresh_token)
    if not decoded or decoded.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

    refresh_hash = hash_token(payload.refresh_token)
    session = db.query(UserSession).filter(UserSession.refresh_token_hash == refresh_hash).first()
    if not session or session.is_revoked or session.expires_at < datetime.utcnow():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token expired or revoked")

    user = db.query(User).filter(User.id == session.user_id, User.is_active.is_(True)).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return RefreshTokenResponse(access_token=create_access_token(user.email, user.role.value))


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(payload: RefreshTokenRequest, db: Session = Depends(get_db)) -> None:
    refresh_hash = hash_token(payload.refresh_token)
    session = db.query(UserSession).filter(UserSession.refresh_token_hash == refresh_hash).first()
    if session:
        session.is_revoked = True
        db.commit()


@router.post("/api-keys", response_model=ApiKeyCreateResponse, status_code=status.HTTP_201_CREATED)
def create_user_api_key(
    payload: ApiKeyCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ApiKeyCreateResponse:
    raw_key, key_prefix, hashed_key = create_api_key()
    api_key = ApiKey(user_id=current_user.id, key_prefix=key_prefix, hashed_key=hashed_key, name=payload.name)
    db.add(api_key)
    db.commit()
    db.refresh(api_key)
    return ApiKeyCreateResponse(id=api_key.id, name=api_key.name, key=raw_key, created_at=api_key.created_at)


@router.get("/api-keys", response_model=list[ApiKeyResponse])
def list_user_api_keys(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[ApiKey]:
    return db.query(ApiKey).filter(ApiKey.user_id == current_user.id).order_by(ApiKey.created_at.desc()).all()


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_user_api_key(key_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> None:
    key = db.query(ApiKey).filter(ApiKey.id == key_id, ApiKey.user_id == current_user.id).first()
    if not key:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found")
    key.is_active = False
    db.commit()


@router.get("/users", response_model=list[UserResponse])
def list_users_for_admin(_: User = Depends(require_role(UserRole.admin)), db: Session = Depends(get_db)) -> list[User]:
    return db.query(User).order_by(User.created_at.desc()).all()
