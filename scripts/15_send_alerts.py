#!/usr/bin/env python3
"""
Check for alerts and send notifications.

Supports:
- Email alerts (SMTP)
- File-based alerts (fallback)
"""

import json
import os
import smtplib
from email.mime.text import MIMEText
from pathlib import Path


def send_email_alert(subject: str, body: str, recipients: list, smtp_config: dict = None):
    """Envía alerta por email."""
    if smtp_config is None:
        # Default: use environment variables
        smtp_config = {
            "host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
            "port": int(os.getenv("SMTP_PORT", "587")),
            "user": os.getenv("SMTP_USER", ""),
            "password": os.getenv("SMTP_PASSWORD", ""),
            "from": os.getenv("SMTP_FROM", "xppm-dss@example.com"),
        }

    if not smtp_config.get("user") or not smtp_config.get("password"):
        print("⚠️ SMTP credentials not configured, skipping email")
        return False

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_config["from"]
    msg["To"] = ", ".join(recipients)

    try:
        with smtplib.SMTP(smtp_config["host"], smtp_config["port"]) as server:
            server.starttls()
            server.login(smtp_config["user"], smtp_config["password"])
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False


def check_alerts(metrics_dir: Path):
    """
    Check para alertas y envía notificaciones.
    """
    # Load drift report
    drift_path = metrics_dir / "drift_report.json"
    if not drift_path.exists():
        print("✅ No drift report found, no alerts")
        return

    with open(drift_path) as f:
        drift = json.load(f)

    if not drift["drift_detected"]:
        print("✅ No alerts")
        return

    # Build alert message
    alerts = []
    for signal_name, signal in drift["signals"].items():
        if signal["alert"]:
            alerts.append(f"- {signal_name}: {signal}")

    if alerts:
        subject = "⚠️ XPPM DSS: Drift Detected"
        body = f"""
Drift detected in XPPM Decision Support System.

Timestamp: {drift['timestamp']}
Lookback: {drift['lookback_days']} days

Alerts:
{chr(10).join(alerts)}

Action required: Review monitoring dashboard and consider retraining.
"""

        # Send (configurar recipients desde env o default)
        recipients = os.getenv("ALERT_RECIPIENTS", "data-team@example.com").split(",")

        # Try email first
        email_sent = send_email_alert(subject, body, recipients)

        if not email_sent:
            # Fallback: write to file
            alert_path = metrics_dir / f"alert_{drift['timestamp'].replace(':', '-')}.txt"
            with open(alert_path, "w") as f:
                f.write(body)
            print(f"💾 Alert saved to {alert_path}")
        else:
            print(f"📧 Alert sent to {recipients}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", default="artifacts/monitoring")
    args = parser.parse_args()

    check_alerts(Path(args.metrics_dir))
