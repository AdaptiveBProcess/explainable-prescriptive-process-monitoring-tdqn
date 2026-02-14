# XPPM DSS Deployment Guide

## Systemd Service Setup

### 1. Install service file

```bash
# Copy service file to systemd directory
sudo cp deploy/xppm-dss.service /etc/systemd/system/

# Edit paths in service file to match your environment
sudo nano /etc/systemd/system/xppm-dss.service
```

**Update these paths:**
- `User=ubuntu` → your user
- `WorkingDirectory=/home/ubuntu/xppm-project` → your project path
- `ExecStart=/home/ubuntu/xppm-project/.venv/bin/python` → your venv path
- `--bundle /home/ubuntu/xppm-project/artifacts/deploy/v1` → your bundle path

### 2. Activate service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable auto-start on boot
sudo systemctl enable xppm-dss

# Start service
sudo systemctl start xppm-dss

# Check status
sudo systemctl status xppm-dss

# View logs
sudo journalctl -u xppm-dss -f
```

### 3. Health check script

```bash
# Create health check script
cat > /usr/local/bin/xppm-health-check.sh << 'EOF'
#!/bin/bash
HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
if [ "$HEALTH" != "200" ]; then
    echo "Health check failed: HTTP $HEALTH"
    sudo systemctl restart xppm-dss
    exit 1
fi
exit 0
EOF

chmod +x /usr/local/bin/xppm-health-check.sh

# Add to crontab (every 5 minutes)
(crontab -l 2>/dev/null; echo "*/5 * * * * /usr/local/bin/xppm-health-check.sh") | crontab -
```

## Monitoring Setup

### 1. Configure cron jobs

```bash
# Add monitoring cron jobs
(crontab -l 2>/dev/null; echo "# XPPM DSS Monitoring
5 0 * * * cd /path/to/project && .venv/bin/python scripts/13_compute_monitoring_metrics.py
0 1 * * * cd /path/to/project && .venv/bin/python scripts/14_detect_drift.py
30 1 * * * cd /path/to/project && .venv/bin/python scripts/15_send_alerts.py
0 2 * * * cd /path/to/project && .venv/bin/python scripts/17_check_retraining_triggers.py
0 3 * * * cd /path/to/project && .venv/bin/python scripts/18_generate_dashboard.py") | crontab -
```

### 2. Configure alerting (optional)

Set environment variables for email alerts:

```bash
export SMTP_HOST=smtp.gmail.com
export SMTP_PORT=587
export SMTP_USER=your-email@gmail.com
export SMTP_PASSWORD=your-app-password
export SMTP_FROM=xppm-dss@example.com
export ALERT_RECIPIENTS=team@example.com
```

## Monitoring Outputs

- `artifacts/monitoring/daily_metrics.csv` - Daily metrics (CSV)
- `artifacts/monitoring/metrics_YYYY-MM-DD.json` - Daily metrics (JSON)
- `artifacts/monitoring/drift_report.json` - Drift detection report
- `artifacts/monitoring/retraining_ticket.json` - Retraining trigger (if activated)
- `artifacts/monitoring/dashboard.html` - Interactive dashboard

## Service Management

```bash
# Start
sudo systemctl start xppm-dss

# Stop
sudo systemctl stop xppm-dss

# Restart
sudo systemctl restart xppm-dss

# Status
sudo systemctl status xppm-dss

# Logs
sudo journalctl -u xppm-dss -f
sudo journalctl -u xppm-dss --since "1 hour ago"
```
