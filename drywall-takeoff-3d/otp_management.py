import random
import string
import jwt
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

def generate_otp_code():
    return "".join(random.choices(string.digits, k=6))

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
    subject = "Your OTP for Drywall Takeoff Login"

    body_content = f"""
    <p>Hi {display_name},</p>
    <p>Your one-time password is:</p>
    <h1 style="letter-spacing: 5px;">{otp_code}</h1>
    <p>This OTP is valid for <b>{OTP_EXPIRATION_MINUTES} minutes</b>.</p>
    <p>Do not share this code with anyone.</p>
    <p>Regards,<br>FBM Team</p>
    """

    access_token = load_access_token(
        credentials["Email"]["tenant_id"],
        credentials["Email"]["client_id"],
        credentials["Email"]["client_secret"]
    )

    send_email(access_token, sender, recipient, subject, body_content)

async def is_authenticated(credentials, pg_pool, request, user_id=None):
    authenticated_with_firebase, user_id_decoded = is_firebase_authenticated(credentials, request, user_id=user_id)
    if not authenticated_with_firebase:
        return dict(user_type="EXTERNAL", email=user_id_decoded, token="INVALID")

    user_email = normalize_email(user_id_decoded)
    if not is_valid_email(user_email):
        return dict(user_type="EXTERNAL", email=user_id_decoded, token="INVALID")

    query = f"""
        SELECT
            is_external
        FROM {credentials["CloudSQL"]["table_name_users"]}
        WHERE LOWER(user_id) = LOWER(%s)
        LIMIT 1;
    """
    is_external = await run_in_threadpool(
        partial(pg_run, pg_pool, query, params=(user_email,), fetch=True)
    )

    if not is_external:
        return dict(user_type="EXTERNAL", email=user_id_decoded, token="INVALID")

    if is_external[0]["is_external"]:
        query = f"""
            SELECT
                expires_at,
                is_verified
            FROM {credentials["CloudSQL"]["table_name_otp"]}
            WHERE LOWER(email) = LOWER(%s);
        """
        otp_status = await run_in_threadpool(
            partial(pg_run, pg_pool, query, params=(user_email,), fetch=True)
        )
        if not otp_status:
            return dict(user_type="EXTERNAL", email=user_id_decoded, token="EXPIRED")
        if otp_status[0]["expires_at"] < datetime.now(timezone.utc) or not otp_status[0]["is_verified"]:
            return dict(user_type="EXTERNAL", email=user_id_decoded, token="EXPIRED")
