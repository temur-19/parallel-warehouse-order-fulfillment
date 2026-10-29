import base64
import hashlib
import hmac
import json
import os
import secrets
import time

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession
from dotenv import load_dotenv

from app.database import get_db
from app.models import User


load_dotenv()

PASSWORD_ITERATIONS = 600_000
ACCESS_TOKEN_TTL_SECONDS = 30 * 60
bearer_scheme = HTTPBearer(auto_error=False)


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS
    )
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${_b64url_encode(salt)}${_b64url_encode(digest)}"


def is_password_hash(value: str) -> bool:
    return value.startswith("pbkdf2_sha256$")


def verify_password(password: str, stored_value: str) -> bool:
    if not is_password_hash(stored_value):
        return hmac.compare_digest(password, stored_value)

    try:
        algorithm, iterations_text, salt_text, digest_text = stored_value.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_text)
        if not 100_000 <= iterations <= 1_000_000:
            return False
        salt = _b64url_decode(salt_text)
        expected_digest = _b64url_decode(digest_text)
    except (ValueError, TypeError):
        return False

    actual_digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations
    )
    return hmac.compare_digest(actual_digest, expected_digest)


def _signing_key() -> bytes:
    secret = os.getenv("JWT_SECRET_KEY")
    if not secret or len(secret.encode("utf-8")) < 32:
        raise HTTPException(
            status_code=503,
            detail="JWT_SECRET_KEY sozlanmagan yoki 32 baytdan qisqa",
        )
    return secret.encode("utf-8")


def create_access_token(user: User) -> str:
    now = int(time.time())
    header = _b64url_encode(
        json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode()
    )
    payload = _b64url_encode(
        json.dumps(
            {
                "sub": str(user.id),
                "phone": user.phone,
                "iat": now,
                "exp": now + ACCESS_TOKEN_TTL_SECONDS,
            },
            separators=(",", ":"),
        ).encode()
    )
    signing_input = f"{header}.{payload}".encode("ascii")
    signature = hmac.new(_signing_key(), signing_input, hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64url_encode(signature)}"


def decode_access_token(token: str) -> dict[str, object]:
    try:
        header_text, payload_text, signature_text = token.split(".")
        signing_input = f"{header_text}.{payload_text}".encode("ascii")
        expected_signature = hmac.new(
            _signing_key(), signing_input, hashlib.sha256
        ).digest()
        if not hmac.compare_digest(
            expected_signature, _b64url_decode(signature_text)
        ):
            raise ValueError("Invalid token signature")

        header = json.loads(_b64url_decode(header_text))
        payload = json.loads(_b64url_decode(payload_text))
        if not isinstance(header, dict) or not isinstance(payload, dict):
            raise ValueError("Invalid token content")
        if header.get("alg") != "HS256":
            raise ValueError("Unsupported token algorithm")
        if not isinstance(payload.get("sub"), str) or not payload["sub"].isdigit():
            raise ValueError("Invalid token subject")
        if not isinstance(payload.get("exp"), int) or payload["exp"] <= int(time.time()):
            raise ValueError("Expired token")
        if not isinstance(payload.get("phone"), str):
            raise ValueError("Invalid token phone")
        return payload
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=401,
            detail="Token noto‘g‘ri yoki muddati tugagan",
            headers={"WWW-Authenticate": "Bearer"},
        ) from error


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Avval tizimga kiring",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(credentials.credentials)
    user_id = int(payload["sub"])
    user = await db.get(User, user_id)
    if user is None or user.phone != payload["phone"]:
        raise HTTPException(
            status_code=401,
            detail="Foydalanuvchi topilmadi yoki token bekor qilingan",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
