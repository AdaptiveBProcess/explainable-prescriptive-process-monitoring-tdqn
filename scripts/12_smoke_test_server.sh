#!/bin/bash
# Smoke test script for DSS server
# Usage: ./scripts/12_smoke_test_server.sh [base_url]

BASE_URL="${1:-http://localhost:8000}"

echo "=== DSS Server Smoke Test ==="
echo "Base URL: $BASE_URL"
echo ""

# Test 1: Health check
echo "1. Testing /health..."
curl -s "$BASE_URL/health" | python -m json.tool || echo "❌ Health check failed"
echo ""

# Test 2: Version
echo "2. Testing /version..."
curl -s "$BASE_URL/version" | python -m json.tool || echo "❌ Version endpoint failed"
echo ""

# Test 3: Schema
echo "3. Testing /schema..."
curl -s "$BASE_URL/schema" | python -m json.tool | head -20 || echo "❌ Schema endpoint failed"
echo ""

# Test 4: Decision (normal)
echo "4. Testing /v1/decision (normal request)..."
if [ -f "example_request.json" ]; then
    curl -s "$BASE_URL/v1/decision" \
        -X POST \
        -H "Content-Type: application/json" \
        -d @example_request.json | python -m json.tool || echo "❌ Decision endpoint failed"
else
    echo "⚠️  example_request.json not found, skipping"
fi
echo ""

# Test 5: Decision (override)
echo "5. Testing /v1/decision (override)..."
if [ -f "example_override.json" ]; then
    curl -s "$BASE_URL/v1/decision" \
        -X POST \
        -H "Content-Type: application/json" \
        -d @example_override.json | python -m json.tool || echo "❌ Override test failed"
else
    echo "⚠️  example_override.json not found, skipping"
fi
echo ""

# Test 6: Decision (OOD)
echo "6. Testing /v1/decision (OOD detection)..."
if [ -f "example_ood.json" ]; then
    curl -s "$BASE_URL/v1/decision" \
        -X POST \
        -H "Content-Type: application/json" \
        -d @example_ood.json | python -m json.tool || echo "❌ OOD test failed"
else
    echo "⚠️  example_ood.json not found, skipping"
fi
echo ""

echo "=== Smoke test complete ==="
