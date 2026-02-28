"""In-memory user store for local development and testing."""

from dataclasses import dataclass

from rivinityllm_tune.auth.security import hash_password, verify_password


@dataclass
class StoredUser:
    email: str
    password_hash: str


class UserStore:
    def __init__(self) -> None:
        self._users: dict[str, StoredUser] = {}

    def create_user(self, email: str, password: str) -> StoredUser:
        if email in self._users:
            raise ValueError("User already exists")
        user = StoredUser(email=email, password_hash=hash_password(password))
        self._users[email] = user
        return user

    def authenticate(self, email: str, password: str) -> bool:
        user = self._users.get(email)
        if user is None:
            return False
        return verify_password(password, user.password_hash)


user_store = UserStore()
