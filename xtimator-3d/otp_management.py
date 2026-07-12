import pyotp
import jwt
from jwt import (
    ExpiredSignatureError,
    InvalidSignatureError,
    InvalidTokenError,
    DecodeError,
    InvalidAlgorithmError,
    MissingRequiredClaimError,
)
import json
import re
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel
from functools import partial
from fastapi.concurrency import run_in_threadpool

from google.cloud import secretmanager

from email_notification import send_email, load_access_token
from helper import is_firebase_authenticated, pg_run


class PayloadRequestExternalOtp(BaseModel):
    user_id: str

class PayloadVerifyExternalOtp(BaseModel):
    user_id: str
    otp: str

OTP_EXPIRATION_MINUTES = 5
MAX_OTP_ATTEMPTS = 5
SECRET_SEED = "IFMK2JJHQAAA7KCJ26K3XEVXLCJNQCMM"

def load_secret_json(secret_path, version_id="latest"):
    client = secretmanager.SecretManagerServiceClient()

    secret_name = f"{secret_path}/versions/{version_id}"

    response = client.access_secret_version(request={"name": secret_name})
    secret_payload = response.payload.data.decode("UTF-8")

    return json.loads(secret_payload)

def normalize_email(email):
    return str(email or "").strip().lower()

def is_valid_email(email):
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None

def generate_otp():
    totp = pyotp.TOTP(SECRET_SEED)
    return totp.now()

def create_external_login_jwt(credentials, user_id):
    jwt_config = load_secret_json(credentials["JWT"]["secret_path"])

    now = datetime.now(timezone.utc)

    payload = {
        "user_id": user_id,
        "email": user_id,
        "login_method": "external_otp",
        "iat": now,
        "exp": now + timedelta(hours=jwt_config.get("expiration_hours", 24)),
        "type": "access",
    }

    return jwt.encode(
        payload,
        jwt_config["secret_key"],
        algorithm=jwt_config.get("algorithm", "HS256")
    )

def get_display_name(email):
    username = email.split("@")[0]

    parts = username.replace(".", " ").replace("_", " ").split()

    return " ".join(word.capitalize() for word in parts)

def trigger_otp_email(credentials, sender, recipient, otp_code):
    display_name = get_display_name(recipient)
    subject = "Your OTP for Xtimator Login"

    body_content = f"""
    <p>Hi {display_name},</p>
    <p>Your one-time password for signing in to Xtimator is:</p>
    <h1 style="letter-spacing: 5px;">{otp_code}</h1>
    <p>This OTP is valid for <b>{OTP_EXPIRATION_MINUTES} minutes</b>.</p>
    <p>Do not share this code with anyone.</p>
    <p>Regards,<br>FBM Xtimator Team</p>
    """

    access_token = load_access_token(
        credentials["Email"]["tenant_id"],
        credentials["Email"]["client_id"],
        credentials["Email"]["client_secret"]
    )

    send_email(access_token, sender, recipient, subject, body_content)

def is_jwt_authenticated(credentials, request, user_id):
    jwt_config = load_secret_json(credentials["JWT"]["secret_path"])
    authorization_header = request.headers.get("Authorization", None)
    if not authorization_header:
        return False, datetime.now(timezone.utc)
    id_token = authorization_header.split()[1]

    try:
        payload = jwt.decode(
            id_token,
            jwt_config["secret_key"],
            algorithms=[jwt_config.get("algorithm", "HS256")]
        )
        expiry = datetime.fromtimestamp(
            payload["exp"],
            tz=timezone.utc
        )
        authenticated = payload["email"].strip().lower() == user_id.strip().lower()
        return authenticated, expiry

    except ExpiredSignatureError:
        return True, datetime.now(timezone.utc)

    except InvalidSignatureError:
        return False, datetime.now(timezone.utc)

    except MissingRequiredClaimError:
        return False, datetime.now(timezone.utc)

    except (DecodeError, InvalidAlgorithmError, InvalidTokenError):
        return False, datetime.now(timezone.utc)

async def is_authenticated(credentials, pg_pool, request, user_id=None):
    query = f"""
        SELECT
            is_external
        FROM {credentials["CloudSQL"]["table_name_users"]}
        WHERE LOWER(user_id) = LOWER(%s)
        LIMIT 1;
    """
    is_external = await run_in_threadpool(
        partial(pg_run, credentials, pg_pool, query, params=(user_id,), fetch=True)
    )

    if not is_external:
        return dict(user_type="EXTERNAL", email=user_id, token="INVALID")

    if is_external[0]["is_external"]:
        is_user_authenticated, expiry = is_jwt_authenticated(credentials, request, user_id)
        if not is_user_authenticated:
            return dict(user_type="EXTERNAL", email=user_id, token="INVALID")
        if expiry <= datetime.now(timezone.utc):
            return dict(user_type="EXTERNAL", email=user_id, token="EXPIRED")
    #else:
    #    is_user_authenticated, user_id_authenticated = is_firebase_authenticated(credentials, request, user_id=user_id)
    #    if not is_user_authenticated:
    #        return dict(user_type="EXTERNAL", email=user_id, token="INVALID")
    #    if user_id_authenticated.strip().lower() != user_id.strip().lower():
    #        return dict(user_type="INTERNAL", email=user_id, token="INVALID")
