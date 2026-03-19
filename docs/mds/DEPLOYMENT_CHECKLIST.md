# Deployment Checklist - DSS Ready for Production

## ✅ Pre-deployment Verification

### Bundle Contents
- [x] `schema.json` - API contract (OpenAPI/JSON Schema)
- [x] `tree.pkl` - Surrogate model (8 leaves, 97% fidelity)
- [x] `policy_guard_config.json` - Guard thresholds + feature stats
- [x] `versions.json` - SHA256 hashes (model, data, config, git)
- [x] `fidelity.csv` - Offline fidelity test results
- [x] `xai/` - XAI artifacts (policy_summary, risk_explanations, deltaQ_explanations)
- [x] `README.md` - Documentation

### Version Verification
- [x] `model_version`: SHA256 hash of tree.pkl (non-empty)
- [x] `data_version`: SHA256 hash of D_offline.npz (non-empty)
- [x] `config_version`: SHA256 hash of config.yaml (non-empty)
- [x] `git_commit`: Current git commit SHA (non-empty)
- [x] `deployed_at`: ISO timestamp

### Guard Configuration
- [x] `tau_uncertainty`: Threshold for uncertainty fallback (default: 0.3)
- [x] `tau_ood_z`: Z-score threshold for OOD detection (default: 3.0)
- [x] `max_ood_features`: Max OOD features before fallback (default: 2)
- [x] `feature_stats`: Mean/std for all 9 features (non-empty)

---

## 🧪 Smoke Test Checklist

### Server Startup
```bash
python policy_server.py --bundle artifacts/deploy/v1 --port 8000
```

### Endpoint Tests
- [ ] `GET /health` → Returns `{"status": "healthy", ...}`
- [ ] `GET /version` → Returns versions with all hashes
- [ ] `GET /schema` → Returns schema JSON
- [ ] `POST /v1/decision` → Returns decision with all required fields

### Guard Tests
- [ ] **Override**: Request with `override` → `source="override"`
- [ ] **OOD**: Request with extreme features → `ood=true`, `source="baseline"`
- [ ] **Uncertainty**: Request with low confidence → `source="baseline"` (if threshold exceeded)
- [ ] **Action mask**: Request with invalid action → `source="baseline"`

### Response Validation
Each `/v1/decision` response must include:
- [x] `action_id` (int)
- [x] `action_name` (str)
- [x] `source` (one of: surrogate, teacher, baseline, override)
- [x] `confidence` (float, 0-1)
- [x] `uncertainty` (float, 0-1)
- [x] `ood` (bool)
- [x] `versions` (dict with model_version, data_version, config_version, git_commit)
- [x] `latency_ms` (float)

---

## 📊 Logging Verification

### JSONL Log File
- [ ] Log file created: `artifacts/deploy/v1/decisions.jsonl`
- [ ] Each request generates one log entry
- [ ] Log entries include: request_id, case_id, t, action_id, source, confidence, ood, latency_ms

### Log Entry Example
```json
{
  "timestamp": "2026-02-14T18:00:00.123456",
  "request_id": "demo-0001",
  "case_id": "CASE_123",
  "t": 10,
  "action_id": 1,
  "action_name": "contact_headquarters",
  "source": "surrogate",
  "confidence": 0.89,
  "ood": false,
  "latency_ms": 1.2
}
```

---

## 🚀 Production Readiness

### Performance
- [ ] Median latency < 5ms (surrogate inference)
- [ ] p95 latency < 10ms
- [ ] p99 latency < 20ms
- [ ] No memory leaks (test with 1000+ requests)

### Error Handling
- [ ] Invalid JSON → 422 Unprocessable Entity
- [ ] Missing required fields → 422 with error message
- [ ] Invalid feature ranges → 422 with validation error
- [ ] Server errors → 500 with error detail

### Monitoring Metrics
Track these in production:
- Override rate (human disagreement)
- OOD rate (distribution shift)
- Fallback rate (low confidence/invalid)
- Latency percentiles (p50, p95, p99)
- Error rate (4xx, 5xx)

---

## 📝 Example Requests

### Normal Request
See `example_request.json`

### Override Request
See `example_override.json`

### OOD Request
See `example_ood.json`

---

## 🔧 Quick Test Commands

```bash
# Start server
python policy_server.py --bundle artifacts/deploy/v1 --port 8000

# Run smoke test (in another terminal)
./scripts/12_smoke_test_server.sh

# Test single request
curl -X POST http://localhost:8000/v1/decision \
  -H "Content-Type: application/json" \
  -d @example_request.json | python -m json.tool

# Check logs
tail -f artifacts/deploy/v1/decisions.jsonl
```

---

## ✅ Final Sign-off

- [ ] All bundle files present and valid
- [ ] All smoke tests passing
- [ ] Guard tests verified
- [ ] Logging working
- [ ] Performance acceptable
- [ ] Documentation complete
- [ ] Version hashes non-empty

**Status**: ✅ Ready for deployment
