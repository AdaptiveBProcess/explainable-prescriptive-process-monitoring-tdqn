# XPPM DSS Production Deployment Guide

Complete step-by-step guide for deploying the XPPM Decision Support System to production.

---

## Prerequisites

- ✅ Bundle built: `artifacts/deploy/v1/` exists
- ✅ Virtual environment: `.venv` installed
- ✅ Python dependencies: All packages installed
- ✅ Smoke test passed: Server responds correctly

---

## Step 1: Deploy as Systemd Service

### 1.1 Edit service file

```bash
# Copy template
sudo cp deploy/xppm-dss.service /etc/systemd/system/

# Edit paths (REQUIRED)
sudo nano /etc/systemd/system/xppm-dss.service
```

**Update these paths:**
- `User=ubuntu` → your user
- `WorkingDirectory=/home/ubuntu/xppm-project` → your project path
- `ExecStart=/home/ubuntu/xppm-project/.venv/bin/python` → your venv path
- `--bundle /home/ubuntu/xppm-project/artifacts/deploy/v1` → your bundle path

### 1.2 Install and start service

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

### 1.3 Verify endpoints

```bash
# Health check
curl http://localhost:8000/health

# Version
curl http://localhost:8000/version

# Schema
curl http://localhost:8000/schema
```

---

## Step 2: Setup Log Rotation

### 2.1 Install logrotate config

```bash
# Edit paths in logrotate config
nano deploy/logrotate.conf

# Update paths:
# /path/to/artifacts/deploy/v1/ → your actual path
# /path/to/logs/ → your actual log path

# Install
sudo cp deploy/logrotate.conf /etc/logrotate.d/xppm-dss

# Test configuration
sudo logrotate -d /etc/logrotate.d/xppm-dss

# Force rotation (test)
sudo logrotate -f /etc/logrotate.d/xppm-dss
```

**Log rotation settings:**
- `decisions.jsonl`: Daily, keep 14 days
- `feedback.jsonl`: Daily, keep 30 days
- `monitoring_cron.log`: Weekly, keep 8 weeks
- `feedback_consolidation.log`: Weekly, keep 12 weeks

---

## Step 3: Setup Monitoring Cron Jobs

### 3.1 Run setup script

```bash
# Make executable
chmod +x deploy/setup_cron.sh

# Run (adjust paths if needed)
./deploy/setup_cron.sh /opt/xppm /opt/xppm/.venv/bin/python

# Or manually specify paths
./deploy/setup_cron.sh /home/ubuntu/xppm-project /home/ubuntu/xppm-project/.venv/bin/python
```

### 3.2 Verify cron jobs

```bash
# List installed jobs
crontab -l

# Should see:
# - Daily metrics: 00:05
# - Drift detection: 01:00
# - Retraining triggers: 01:10
# - Alerts: 01:30
# - Dashboard: 03:00
# - Feedback consolidation: Sunday 02:00
```

### 3.3 Manual testing

```bash
# Test each script manually
cd /path/to/project
source .venv/bin/activate

# Metrics
python scripts/13_compute_monitoring_metrics.py

# Drift
python scripts/14_detect_drift.py

# Triggers
python scripts/17_check_retraining_triggers.py

# Alerts
python scripts/15_send_alerts.py

# Dashboard
python scripts/18_generate_dashboard.py
```

---

## Step 4: Configure Alerting (Optional)

### 4.1 Email alerts (SMTP)

Set environment variables in systemd service:

```bash
# Edit service file
sudo nano /etc/systemd/system/xppm-dss.service

# Add to [Service] section:
Environment="SMTP_HOST=smtp.gmail.com"
Environment="SMTP_PORT=587"
Environment="SMTP_USER=your-email@gmail.com"
Environment="SMTP_PASSWORD=your-app-password"
Environment="SMTP_FROM=xppm-dss@example.com"
Environment="ALERT_RECIPIENTS=team@example.com,ops@example.com"

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart xppm-dss
```

**Note:** For Gmail, use an [App Password](https://support.google.com/accounts/answer/185833) instead of your regular password.

### 4.2 File-based alerts (fallback)

If SMTP is not configured, alerts will be written to:
- `artifacts/monitoring/alert_TIMESTAMP.txt`

---

## Step 5: Configure Monitoring Thresholds

### 5.1 Review configuration

```bash
# Edit monitoring config
nano configs/monitoring.yaml
```

**Key thresholds:**
- Coverage: Target >90%, warn <85%, critical <80%
- OOD rate: Target <5%, warn >10%, critical >20%
- Latency: p95 target <10ms, warn >20ms, critical >50ms
- Drift: Alert if >200% increase vs 7-day baseline
- Retraining: Trigger after 3 consecutive days of drift

### 5.2 Customize if needed

Adjust thresholds based on your operational requirements.

---

## Step 6: Health Check Automation

### 6.1 Create health check script

```bash
# Create script
sudo nano /usr/local/bin/xppm-health-check.sh
```

**Content:**
```bash
#!/bin/bash
HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
if [ "$HEALTH" != "200" ]; then
    echo "Health check failed: HTTP $HEALTH"
    sudo systemctl restart xppm-dss
    exit 1
fi
exit 0
```

### 6.2 Make executable and add to cron

```bash
# Make executable
sudo chmod +x /usr/local/bin/xppm-health-check.sh

# Add to crontab (every 5 minutes)
(crontab -l 2>/dev/null; echo "*/5 * * * * /usr/local/bin/xppm-health-check.sh") | crontab -
```

---

## Step 7: Verification End-to-End

### 7.1 Generate test decisions

```bash
# Start server (if not running)
sudo systemctl start xppm-dss

# Generate 20-50 test decisions
for i in {1..50}; do
    curl -X POST http://localhost:8000/v1/decision \
        -H "Content-Type: application/json" \
        -d @example_request.json
    sleep 0.1
done
```

### 7.2 Run monitoring pipeline manually

```bash
cd /path/to/project
source .venv/bin/activate

# Compute metrics
python scripts/13_compute_monitoring_metrics.py

# Check output
cat artifacts/monitoring/daily_metrics.csv

# Detect drift
python scripts/14_detect_drift.py

# Check drift report
cat artifacts/monitoring/drift_report.json

# Generate dashboard
python scripts/18_generate_dashboard.py

# Open dashboard
xdg-open artifacts/monitoring/dashboard.html
# or
open artifacts/monitoring/dashboard.html  # macOS
```

### 7.3 Verify outputs

**Expected outputs:**
- ✅ `artifacts/monitoring/daily_metrics.csv` (metrics)
- ✅ `artifacts/monitoring/metrics_YYYY-MM-DD.json` (daily JSON)
- ✅ `artifacts/monitoring/drift_report.json` (drift analysis)
- ✅ `artifacts/monitoring/dashboard.html` (interactive dashboard)

---

## Step 8: Daily Operations

### 8.1 Morning checklist

1. **Check service status:**
   ```bash
   sudo systemctl status xppm-dss
   ```

2. **Review daily metrics:**
   ```bash
   cat artifacts/monitoring/daily_metrics.csv | tail -1
   ```

3. **Check drift report:**
   ```bash
   cat artifacts/monitoring/drift_report.json | jq .
   ```

4. **Review alerts:**
   ```bash
   ls -lt artifacts/monitoring/alert_*.txt | head -5
   ```

5. **Check retraining tickets:**
   ```bash
   cat artifacts/monitoring/retraining_ticket.json 2>/dev/null || echo "No retraining triggered"
   ```

### 8.2 Weekly tasks

1. **Review dashboard:**
   ```bash
   open artifacts/monitoring/dashboard.html
   ```

2. **Check feedback consolidation:**
   ```bash
   ls -lh data/processed/D_offline_incremental.npz
   ```

3. **Review logs:**
   ```bash
   tail -100 logs/monitoring_cron.log
   ```

---

## Troubleshooting

### Service won't start

```bash
# Check logs
sudo journalctl -u xppm-dss -n 50

# Common issues:
# - Paths incorrect in service file
# - Permissions (user can't access bundle)
# - Port already in use (change --port)
```

### Cron jobs not running

```bash
# Check cron logs
grep CRON /var/log/syslog | tail -20

# Verify crontab
crontab -l

# Test script manually
cd /path/to/project && .venv/bin/python scripts/13_compute_monitoring_metrics.py
```

### Metrics not generated

```bash
# Check if decisions.jsonl exists
ls -lh artifacts/deploy/v1/decisions.jsonl

# Check if it has data
wc -l artifacts/deploy/v1/decisions.jsonl

# Run manually with verbose output
python scripts/13_compute_monitoring_metrics.py --date $(date +%Y-%m-%d)
```

### Dashboard not updating

```bash
# Check if metrics CSV exists
ls -lh artifacts/monitoring/daily_metrics.csv

# Regenerate manually
python scripts/18_generate_dashboard.py

# Check for errors
python scripts/18_generate_dashboard.py 2>&1
```

---

## Maintenance

### Update bundle

```bash
# Build new bundle
python scripts/10_build_deploy_bundle.py

# Update service to point to new bundle
sudo nano /etc/systemd/system/xppm-dss.service
# Update --bundle path

# Restart service
sudo systemctl restart xppm-dss
```

### Backup monitoring data

```bash
# Backup monitoring artifacts
tar -czf monitoring_backup_$(date +%Y%m%d).tar.gz \
    artifacts/monitoring/ \
    artifacts/deploy/v1/decisions.jsonl \
    artifacts/deploy/v1/feedback.jsonl
```

---

## Production Checklist

- [ ] Systemd service installed and running
- [ ] Health check script configured
- [ ] Log rotation configured
- [ ] Cron jobs installed
- [ ] Monitoring thresholds reviewed
- [ ] Alerting configured (email or file)
- [ ] End-to-end test completed
- [ ] Dashboard accessible
- [ ] Daily operations documented
- [ ] Backup strategy in place

---

## Support

For issues or questions:
1. Check logs: `sudo journalctl -u xppm-dss -f`
2. Review monitoring outputs: `artifacts/monitoring/`
3. Check cron logs: `logs/monitoring_cron.log`
4. Verify configuration: `configs/monitoring.yaml`
