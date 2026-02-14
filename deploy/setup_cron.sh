#!/bin/bash
# Setup cron jobs for XPPM DSS monitoring
# Usage: ./setup_cron.sh [PROJECT_PATH] [VENV_PATH]

set -e

# Default paths (adjust to your environment)
PROJECT_PATH="${1:-/opt/xppm}"
VENV_PATH="${2:-${PROJECT_PATH}/.venv/bin/python}"
LOG_DIR="${PROJECT_PATH}/logs"

# Create log directory
mkdir -p "$LOG_DIR"

# Backup existing crontab
echo "📋 Backing up existing crontab..."
crontab -l > "${LOG_DIR}/crontab_backup_$(date +%Y%m%d_%H%M%S).txt" 2>/dev/null || true

# Add monitoring cron jobs
echo "🔧 Setting up monitoring cron jobs..."

# Get current crontab (if exists)
CRON_TEMP=$(mktemp)
crontab -l > "$CRON_TEMP" 2>/dev/null || touch "$CRON_TEMP"

# Remove existing XPPM jobs (if any)
grep -v "# XPPM DSS Monitoring" "$CRON_TEMP" > "${CRON_TEMP}.new" || cp "$CRON_TEMP" "${CRON_TEMP}.new"
mv "${CRON_TEMP}.new" "$CRON_TEMP"

# Add new XPPM monitoring jobs
cat >> "$CRON_TEMP" << EOF

# XPPM DSS Monitoring
# Daily metrics computation (00:05)
5 0 * * * cd ${PROJECT_PATH} && ${VENV_PATH} scripts/13_compute_monitoring_metrics.py >> ${LOG_DIR}/monitoring_cron.log 2>&1

# Drift detection (01:00)
0 1 * * * cd ${PROJECT_PATH} && ${VENV_PATH} scripts/14_detect_drift.py >> ${LOG_DIR}/monitoring_cron.log 2>&1

# Retraining triggers check (01:10)
10 1 * * * cd ${PROJECT_PATH} && ${VENV_PATH} scripts/17_check_retraining_triggers.py >> ${LOG_DIR}/monitoring_cron.log 2>&1

# Send alerts (01:30)
30 1 * * * cd ${PROJECT_PATH} && ${VENV_PATH} scripts/15_send_alerts.py >> ${LOG_DIR}/monitoring_cron.log 2>&1

# Generate dashboard (03:00)
0 3 * * * cd ${PROJECT_PATH} && ${VENV_PATH} scripts/18_generate_dashboard.py >> ${LOG_DIR}/monitoring_cron.log 2>&1

# Weekly feedback consolidation (Sunday 02:00)
0 2 * * 0 cd ${PROJECT_PATH} && ${VENV_PATH} scripts/16_consolidate_feedback_to_offline_dataset.py --min-cases 500 >> ${LOG_DIR}/feedback_consolidation.log 2>&1
EOF

# Install new crontab
crontab "$CRON_TEMP"
rm "$CRON_TEMP"

echo "✅ Cron jobs installed successfully!"
echo ""
echo "📋 Installed jobs:"
echo "   - Daily metrics: 00:05"
echo "   - Drift detection: 01:00"
echo "   - Retraining triggers: 01:10"
echo "   - Alerts: 01:30"
echo "   - Dashboard: 03:00"
echo "   - Feedback consolidation: Sunday 02:00"
echo ""
echo "📝 Logs will be written to: ${LOG_DIR}/"
echo ""
echo "To verify: crontab -l"
echo "To edit: crontab -e"
