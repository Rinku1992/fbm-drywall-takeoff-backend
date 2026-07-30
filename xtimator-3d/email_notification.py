import logging
import requests


def load_access_token(tenant_id, client_id, client_secret):
    TOKEN_URL = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    response = requests.post(
        TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "client_id": client_id,
            "scope": "https://graph.microsoft.com/.default",
            "client_secret": client_secret,
            "grant_type": "client_credentials",
        },
    )
    response.raise_for_status()
    return response.json()["access_token"]

def send_email(access_token, sender_email, recipient_email, subject, body_content):
    GRAPH_SENDMAIL_URL = f"https://graph.microsoft.com/v1.0/users/{sender_email}/sendMail"
    response = requests.post(
        GRAPH_SENDMAIL_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json={
            "message": {
                "subject": subject,
                "body": {"contentType": "HTML", "content": body_content},
                "toRecipients": [{"emailAddress": {"address": recipient_email}}],
            }
        },
    )

    if response.status_code != 202:
        raise Exception(f"Failed to send email: {response.status_code} - {response.text}")

def trigger(
    credentials,
    sender,
    recipient,
    user_name,
    plan_name,
    project_name,
    page_number,
    team_name,
    message="Floor Plan extraction completed",
):
    try:
        failed = any(keyword in message.lower() for keyword in ["failed", "failure", "floor plan extraction failed"])
        partial = any(keyword in message.lower() for keyword in ["partial", "incomplete", "scale not detected"])

        if failed:
            subject = f"Floor Plan Processing Failed - {plan_name} | Page Number - {page_number}"
            status_paragraph = (
                f"Your floor plan <b>{plan_name}</b> has failed to process successfully."
            )
            details_paragraph = (
                "The Xtimator output is not available because the extraction workflow failed."
            )
        elif partial:
            subject = f"Floor Plan Processing is Incomplete - {plan_name} | Page Number - {page_number}"
            status_paragraph = (
                f"Your floor plan <b>{plan_name}</b> could not be processed successfully due to missing architectural scale."
            )
            details_paragraph = (
                "Floor plan is waiting for the architectural scale input from the user to extract the quantity estimates."
            )
        else:
            subject = f"Floor Plan Processing Completed - {plan_name} | Page Number - {page_number}"
            status_paragraph = (
                f"Your floor plan <b>{plan_name}</b> has been processed successfully."
            )
            details_paragraph = (
                "The Xtimator output is now ready for review, including extracted rooms, "
                "wall segments, measurements, and quantity estimates."
            )

        body_content = f"""
        <p>Hi {user_name.split('@')[0]},</p>
        <p>{status_paragraph}</p>
        <p>{details_paragraph}</p>
        <p><b>Status:</b> {message}</p>
        <p><b>Project:</b> {project_name}</p>
        <p><b>Plan:</b> {plan_name}</p>

        <p>Regards,<br>{team_name}</p>
        """

        access_token = load_access_token(
            credentials["Email"]["tenant_id"],
            credentials["Email"]["client_id"],
            credentials["Email"]["client_secret"]
        )
        send_email(access_token, sender, recipient, subject, body_content)
        logging.info("SYSTEM: Email triggered successfully.")
    except Exception as e:
        logging.error(f"SYSTEM: An error occurred while sending status email: {e}")

def trigger_support(
    credentials,
    sender,
    recipient,
    subject,
    message,
):
    try:
        body_content = f"""
        <p>Hi XtimatorSupport,</p>
        <p><b>Status:</b> {message}</p>

        <p>Regards,<br>{user_name.split('@')[0]}</p>
        """

        access_token = load_access_token(
            credentials["Email"]["tenant_id"],
            credentials["Email"]["client_id"],
            credentials["Email"]["client_secret"]
        )
        send_email(access_token, sender, recipient, subject, body_content)
        logging.info("SYSTEM: Email triggered successfully.")
    except Exception as e:
        logging.error(f"SYSTEM: An error occurred while sending support email: {e}")
