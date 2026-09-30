"""
Nyika-Grid Predict AI — Alerts Module
------------------------------------------
Checks predicted load against a per-region threshold and dispatches an
alert. Actual SMS/email delivery is simulated (logged to alerts.log) so
the feature is fully testable without external credentials — the real
Twilio/SMTP integration points are included as commented-out examples.
"""

import datetime

ALERT_LOG_PATH = "alerts.log"


def check_load_alert(predicted_load_mw: float, threshold_mw: float) -> dict:
    """Return whether a load forecast crosses the alert threshold."""
    if predicted_load_mw >= threshold_mw:
        return {
            "triggered": True,
            "level": "HIGH",
            "message": (
                f"Predicted load {predicted_load_mw:.1f} MW exceeds threshold "
                f"{threshold_mw:.1f} MW — elevated load-shedding risk."
            ),
        }
    return {
        "triggered": False,
        "level": "NORMAL",
        "message": f"Predicted load {predicted_load_mw:.1f} MW is within normal range.",
    }


def send_alert(message: str, method: str = "log", recipient: str = "operations-team") -> str:
    """
    Dispatch an alert. Currently simulated: writes to alerts.log so the
    behaviour can be demoed and tested without real SMS/email credentials.

    To go live:
      - Email: use smtplib or an API like SendGrid
      - SMS: use the Twilio API

    Example (commented, not executed):

        from twilio.rest import Client
        client = Client(account_sid, auth_token)
        client.messages.create(body=message, from_=TWILIO_NUMBER, to=recipient_number)

        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(message)
        msg["Subject"] = "Nyika-Grid Alert"
        msg["From"] = sender_email
        msg["To"] = recipient_email
        with smtplib.SMTP("smtp.example.com", 587) as server:
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
    """
    timestamp = datetime.datetime.utcnow().isoformat()
    log_line = f"[{timestamp}] ({method} -> {recipient}) {message}\n"
    with open(ALERT_LOG_PATH, "a") as f:
        f.write(log_line)
    return log_line
