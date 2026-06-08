"""
auth.py — Authentication logic: JWT tokens, password hashing, route protection.
"""
import bcrypt
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from api.database import get_db

# ── Config ────────────────────────────────────────────────────────────────────
SECRET_KEY = "cgpa-predict-secret-key-2026-do-not-share"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

security = HTTPBearer()


# ── Password Hashing ─────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


# ── JWT Tokens ────────────────────────────────────────────────────────────────
def create_token(user_id: int, email: str, role: str, name: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    payload = {
        "sub": str(user_id),
        "email": email,
        "role": role,
        "name": name,
        "exp": expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )


# ── Dependency: Get Current User ─────────────────────────────────────────────
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    return decode_token(credentials.credentials)


def require_role(role: str):
    """Dependency factory that checks user role."""
    def checker(user: dict = Depends(get_current_user)):
        if user.get("role") != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Requires '{role}' role.",
            )
        return user
    return checker


# ── User CRUD ─────────────────────────────────────────────────────────────────
def signup_user(name: str, email: str, password: str, role: str) -> dict:
    conn = get_db()
    # Check if email exists
    existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
    if existing:
        conn.close()
        raise HTTPException(status_code=400, detail="Email already registered")

    pw_hash = hash_password(password)
    cursor = conn.execute(
        "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)",
        (name, email, pw_hash, role),
    )
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()

    token = create_token(user_id, email, role, name)
    return {"token": token, "user": {"id": user_id, "name": name, "email": email, "role": role}}


def login_user(email: str, password: str) -> dict:
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not verify_password(password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_token(row["id"], row["email"], row["role"], row["name"])
    return {
        "token": token,
        "user": {"id": row["id"], "name": row["name"], "email": row["email"], "role": row["role"]},
    }
