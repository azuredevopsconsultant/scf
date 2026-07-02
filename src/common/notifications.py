"""
src/common/notifications.py

Custom email helper for CONDITIONAL, content-rich mails (drift alert body,
confirmation-mail run summary) that native job email_notifications can't
express, since those only fire on fixed run-state events with a fixed body.

Credentials are pulled from a Databricks secret scope - never hardcode SMTP
creds in source. Create the scope once per workspace:

    databricks secrets create-scope scf-cohort-notifications
    databricks secrets put-secret scf-cohort-notifications smtp-user
    databricks secrets put-secret scf-cohort-notifications smtp-password

For most banks, swap this for your internal mail relay / Databricks SQL
Alerts / a Slack or Teams webhook - the send_email() signature is kept small
on purpose so it's a one-function change.
"""
import smtplib
import json
import urllib.request
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from databricks.sdk.runtime import dbutils

SECRET_SCOPE = "scf-cohort-notifications"
SMTP_HOST = "smtp.office365.com"
SMTP_PORT = 587


def send_email(to_addresses: list[str], subject: str, html_body: str, from_address: str = None):
    smtp_user = dbutils.secrets.get(SECRET_SCOPE, "smtp-user")
    smtp_password = dbutils.secrets.get(SECRET_SCOPE, "smtp-password")
    from_address = from_address or smtp_user

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_address
    msg["To"] = ", ".join(to_addresses)
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(from_address, to_addresses, msg.as_string())


def render_summary_table(rows: dict) -> str:
    """rows: {label: value} -> simple HTML table, used by every mail helper below."""
    tr = "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in rows.items())
    return f'<table border="1" cellpadding="6" cellspacing="0">{tr}</table>'


def send_drift_alert(to_addresses: list[str], environment: str, drift_summary: dict, run_url: str):
    subject = f"[{environment.upper()}] ⚠ Data Drift Detected - SCF Cohort Inference"
    body = f"""
    <p>Data drift was detected on the latest SCF Cohort inference run.</p>
    {render_summary_table(drift_summary)}
    <p>Job run: <a href="{run_url}">{run_url}</a></p>
    <p>Recommended action: review drifted features below the alert threshold
    before trusting downstream projections; consider triggering a retrain.</p>
    """
    send_email(to_addresses, subject, body)


def send_confirmation_mail(
    to_addresses: list[str], environment: str, pipeline_name: str, summary: dict, run_url: str
):
    subject = f"[{environment.upper()}] ✅ {pipeline_name.title()} Pipeline Run Complete - SCF Cohort"
    body = f"""
    <p>The SCF Cohort {pipeline_name} pipeline finished.</p>
    {render_summary_table(summary)}
    <p>Job run: <a href="{run_url}">{run_url}</a></p>
    """
    send_email(to_addresses, subject, body)


def send_webhook_notification(webhook_url: str, title: str, facts: dict[str, str], run_url: str):
    if not webhook_url:
        return

    payload = {
        "title": title,
        "run_url": run_url,
        "facts": facts,
    }
    req = urllib.request.Request(
        webhook_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        if resp.status >= 300:
            raise RuntimeError(f"Webhook call failed with status code {resp.status}")
