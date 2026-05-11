"""
notifications.py
─────────────────
Optional email notification system.

Sends alerts for:
  - Low attendance warnings
  - Daily attendance summary
  - Review queue items needing attention

Graceful no-op if SMTP is not configured in config.yaml.

Usage:
    from src.utils.notifications import send_daily_summary, send_low_attendance_alert
    send_daily_summary()  # only sends if SMTP is configured
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import date

from src.utils.config_loader import cfg
from src.utils.logger import log

# Load SMTP config (optional)
_SMTP_CFG = cfg.get("smtp", {})
_ENABLED = bool(_SMTP_CFG.get("host"))
_HOST = _SMTP_CFG.get("host", "")
_PORT = _SMTP_CFG.get("port", 587)
_USER = _SMTP_CFG.get("username", "")
_PASS = _SMTP_CFG.get("password", "")
_FROM = _SMTP_CFG.get("from_email", _USER)
_TO = _SMTP_CFG.get("admin_email", "")


def _send_email(subject: str, body_html: str) -> bool:
    """Send an email. Returns True on success, False on failure or if not configured."""
    if not _ENABLED:
        log.debug("Email notifications disabled (no SMTP config)")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = _FROM
        msg["To"] = _TO
        msg.attach(MIMEText(body_html, "html"))

        with smtplib.SMTP(_HOST, _PORT) as server:
            server.starttls()
            server.login(_USER, _PASS)
            server.sendmail(_FROM, [_TO], msg.as_string())

        log.info(f"Email sent: {subject}")
        return True
    except Exception as e:
        log.error(f"Failed to send email: {e}")
        return False


def send_low_attendance_alert(student_name: str, percentage: float) -> bool:
    """Alert admin when a student's attendance drops below threshold."""
    subject = f"⚠️ Low Attendance Alert — {student_name}"
    body = f"""
    <div style="font-family: Arial, sans-serif; padding: 20px;">
        <h2 style="color: #ef4444;">Low Attendance Alert</h2>
        <p><strong>{student_name}</strong> has an attendance rate of
           <strong style="color: #ef4444;">{percentage}%</strong></p>
        <p>Please follow up with the student or review their records in the
           Face Guard AI dashboard.</p>
        <hr>
        <p style="color: #888; font-size: 12px;">
            This is an automated alert from Face Guard AI Attendance System.
        </p>
    </div>
    """
    return _send_email(subject, body)


def send_daily_summary(present_count: int, total_enrolled: int, date_str: str = None) -> bool:
    """Send daily attendance summary to admin."""
    if date_str is None:
        date_str = date.today().isoformat()

    rate = round(present_count / total_enrolled * 100, 1) if total_enrolled else 0
    subject = f"📊 Daily Attendance Summary — {date_str}"
    body = f"""
    <div style="font-family: Arial, sans-serif; padding: 20px;">
        <h2>Daily Attendance Summary</h2>
        <p>Date: <strong>{date_str}</strong></p>
        <table style="border-collapse: collapse; margin-top: 15px;">
            <tr>
                <td style="padding: 8px 16px; border: 1px solid #ddd;">Present</td>
                <td style="padding: 8px 16px; border: 1px solid #ddd; font-weight: bold;
                    color: #10b981;">{present_count}</td>
            </tr>
            <tr>
                <td style="padding: 8px 16px; border: 1px solid #ddd;">Enrolled</td>
                <td style="padding: 8px 16px; border: 1px solid #ddd;">{total_enrolled}</td>
            </tr>
            <tr>
                <td style="padding: 8px 16px; border: 1px solid #ddd;">Attendance Rate</td>
                <td style="padding: 8px 16px; border: 1px solid #ddd; font-weight: bold;">{rate}%</td>
            </tr>
        </table>
        <hr>
        <p style="color: #888; font-size: 12px;">Face Guard AI Attendance System</p>
    </div>
    """
    return _send_email(subject, body)


def send_review_needed(pending_count: int) -> bool:
    """Alert admin about pending review items."""
    if pending_count == 0:
        return False

    subject = f"👁️ {pending_count} Face Crops Need Review"
    body = f"""
    <div style="font-family: Arial, sans-serif; padding: 20px;">
        <h2>Review Queue Alert</h2>
        <p>There are <strong>{pending_count}</strong> low-confidence face detections
           waiting for manual review.</p>
        <p>Please visit the <strong>Review Queue</strong> in the Face Guard AI dashboard
           to verify or reject these detections.</p>
        <hr>
        <p style="color: #888; font-size: 12px;">Face Guard AI Attendance System</p>
    </div>
    """
    return _send_email(subject, body)


def is_email_configured() -> bool:
    """Check if email notifications are configured."""
    return _ENABLED
