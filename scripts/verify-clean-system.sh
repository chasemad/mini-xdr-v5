#!/bin/bash
# Verify Mini-XDR System is Clean of Test Data
# This script checks for any remaining test/mock data

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DB_PATH="$PROJECT_ROOT/backend/xdr.db"

echo "=================================="
echo "Mini-XDR Clean System Verification"
echo "=================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if database exists
if [ ! -f "$DB_PATH" ]; then
    echo -e "${RED}✗ Database not found at $DB_PATH${NC}"
    exit 1
fi

echo "Checking database at: $DB_PATH"
echo ""

# Function to check count
check_count() {
    local table=$1
    local count=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM $table;" 2>/dev/null || echo "ERROR")

    if [ "$count" == "ERROR" ]; then
        echo -e "${RED}✗ Error checking $table${NC}"
        return 1
    elif [ "$count" -eq 0 ]; then
        echo -e "${GREEN}✓ $table: $count (CLEAN)${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ $table: $count${NC}"
        return 0
    fi
}

# Check main tables
echo "Database Tables:"
check_count "incidents"
check_count "events"
check_count "actions"
check_count "training_samples"
check_count "action_logs"
echo ""

# Check for test IP ranges
echo "Test IP Range Check:"
test_ip_count=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM events WHERE src_ip LIKE '203.0.113.%' OR src_ip LIKE '192.0.2.%' OR src_ip LIKE '198.51.100.%';" 2>/dev/null || echo "ERROR")

if [ "$test_ip_count" == "ERROR" ]; then
    echo -e "${RED}✗ Error checking test IPs${NC}"
elif [ "$test_ip_count" -eq 0 ]; then
    echo -e "${GREEN}✓ No test IP events found (RFC 5737 ranges)${NC}"
else
    echo -e "${RED}✗ Found $test_ip_count events from test IP ranges${NC}"
    echo "  Test IPs found:"
    sqlite3 "$DB_PATH" "SELECT DISTINCT src_ip FROM events WHERE src_ip LIKE '203.0.113.%' OR src_ip LIKE '192.0.2.%' OR src_ip LIKE '198.51.100.%';"
fi
echo ""

# Check evidence directory
echo "Evidence Directory:"
EVIDENCE_DIR="$PROJECT_ROOT/backend/evidence"
if [ -d "$EVIDENCE_DIR" ]; then
    evidence_count=$(find "$EVIDENCE_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')
    if [ "$evidence_count" -eq 0 ]; then
        echo -e "${GREEN}✓ Evidence directory is clean${NC}"
    else
        echo -e "${YELLOW}⚠ Found $evidence_count evidence files${NC}"
        find "$EVIDENCE_DIR" -type f
    fi
else
    echo -e "${YELLOW}⚠ Evidence directory not found${NC}"
fi
echo ""

# Check Qdrant storage
echo "Vector Storage:"
QDRANT_DIR="$PROJECT_ROOT/backend/qdrant_storage"
if [ -d "$QDRANT_DIR" ]; then
    qdrant_count=$(find "$QDRANT_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')
    if [ "$qdrant_count" -eq 0 ]; then
        echo -e "${GREEN}✓ Qdrant storage is clean (will rebuild from real data)${NC}"
    else
        echo -e "${YELLOW}⚠ Found $qdrant_count Qdrant files (may contain embeddings)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Qdrant storage directory not found${NC}"
fi
echo ""

# Database size
echo "Database Size:"
db_size=$(du -h "$DB_PATH" | cut -f1)
echo "  Size: $db_size"
echo ""

# Summary
echo "=================================="
echo "Summary"
echo "=================================="
if [ "$test_ip_count" -eq 0 ]; then
    echo -e "${GREEN}✓ System is clean and ready for production data${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Start T-Pot honeypot"
    echo "  2. Run simulated attacks"
    echo "  3. Monitor http://localhost:3000 for real incidents"
else
    echo -e "${RED}✗ System still contains test data${NC}"
    echo ""
    echo "To clean:"
    echo "  cd $PROJECT_ROOT/backend"
    echo "  sqlite3 xdr.db 'DELETE FROM incidents; DELETE FROM events; DELETE FROM actions;'"
    echo "  sqlite3 xdr.db 'VACUUM; ANALYZE;'"
fi

echo ""
echo "To manually verify:"
echo "  cd $PROJECT_ROOT/backend"
echo "  sqlite3 xdr.db 'SELECT * FROM incidents;'"
echo "  sqlite3 xdr.db 'SELECT * FROM events LIMIT 10;'"
echo ""
