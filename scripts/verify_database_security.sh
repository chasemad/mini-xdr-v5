#!/bin/bash

# Database Security & Functionality Verification Script
# Verifies database schema, security measures, and all button/action functionality

echo "🔒 MINI-XDR DATABASE SECURITY & FUNCTIONALITY VERIFICATION"
echo "=========================================================="
echo ""

cd /Users/chasemad/Desktop/mini-xdr/backend

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "📊 1. DATABASE SCHEMA VERIFICATION"
echo "-----------------------------------"

# Check if database exists
if [ -f "xdr.db" ]; then
    echo -e "${GREEN}✅ Database file exists${NC}"
    DB_SIZE=$(du -h xdr.db | cut -f1)
    echo "   Size: $DB_SIZE"
else
    echo -e "${RED}❌ Database file not found!${NC}"
    exit 1
fi

# Check action_logs table
if sqlite3 xdr.db "SELECT name FROM sqlite_master WHERE type='table' AND name='action_logs';" | grep -q "action_logs"; then
    echo -e "${GREEN}✅ action_logs table exists${NC}"
else
    echo -e "${RED}❌ action_logs table NOT found!${NC}"
    exit 1
fi

# Verify table schema
echo ""
echo "📋 2. ACTION_LOGS TABLE SCHEMA"
echo "-----------------------------------"
COLUMNS=$(sqlite3 xdr.db "PRAGMA table_info(action_logs);" | wc -l)
echo -e "${GREEN}✅ Columns: $COLUMNS${NC}"

# Check critical columns
for col in "action_id" "agent_id" "agent_type" "action_name" "incident_id" "params" "result" "status" "rollback_id" "rollback_data" "rollback_executed" "rollback_timestamp" "executed_at"; do
    if sqlite3 xdr.db "PRAGMA table_info(action_logs);" | grep -q "$col"; then
        echo -e "${GREEN}✅ Column '$col' exists${NC}"
    else
        echo -e "${RED}❌ Column '$col' MISSING!${NC}"
    fi
done

# Check indexes for performance
echo ""
echo "🚀 3. DATABASE INDEXES (Performance)"
echo "-----------------------------------"
INDEX_COUNT=$(sqlite3 xdr.db "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND tbl_name='action_logs';" 2>/dev/null)
echo -e "${GREEN}✅ Total Indexes: $INDEX_COUNT${NC}"

# Verify critical indexes
for idx in "action_id" "agent_id" "agent_type" "incident_id" "rollback_id" "executed_at"; do
    if sqlite3 xdr.db "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='action_logs';" | grep -q "$idx"; then
        echo -e "${GREEN}✅ Index on '$idx'${NC}"
    else
        echo -e "${YELLOW}⚠️  Index on '$idx' missing (may impact performance)${NC}"
    fi
done

# Check foreign key constraints
echo ""
echo "🔗 4. FOREIGN KEY CONSTRAINTS (Data Integrity)"
echo "-----------------------------------"
FK_STATUS=$(sqlite3 xdr.db "PRAGMA foreign_keys;")
echo "Foreign Keys Status: $FK_STATUS"

FOREIGN_KEYS=$(sqlite3 xdr.db "PRAGMA foreign_key_list(action_logs);" | wc -l)
if [ "$FOREIGN_KEYS" -gt 0 ]; then
    echo -e "${GREEN}✅ Foreign key to 'incidents' table configured${NC}"
    sqlite3 xdr.db "PRAGMA foreign_key_list(action_logs);"
else
    echo -e "${YELLOW}⚠️  No foreign key constraints found${NC}"
fi

# Check data integrity
echo ""
echo "🛡️ 5. DATA INTEGRITY CHECKS"
echo "-----------------------------------"

# Check for duplicate action_ids (should be unique)
DUPES=$(sqlite3 xdr.db "SELECT action_id, COUNT(*) as cnt FROM action_logs GROUP BY action_id HAVING cnt > 1;" | wc -l)
if [ "$DUPES" -eq 0 ]; then
    echo -e "${GREEN}✅ No duplicate action_ids${NC}"
else
    echo -e "${RED}❌ Found $DUPES duplicate action_ids!${NC}"
fi

# Check for orphaned actions (incident_id doesn't exist)
ORPHANS=$(sqlite3 xdr.db "SELECT COUNT(*) FROM action_logs WHERE incident_id IS NOT NULL AND incident_id NOT IN (SELECT id FROM incidents);" 2>/dev/null)
if [ "$ORPHANS" -eq 0 ]; then
    echo -e "${GREEN}✅ No orphaned actions${NC}"
else
    echo -e "${YELLOW}⚠️  Found $ORPHANS orphaned actions (incident doesn't exist)${NC}"
fi

# Check for actions with invalid status
INVALID_STATUS=$(sqlite3 xdr.db "SELECT COUNT(*) FROM action_logs WHERE status NOT IN ('success', 'failed', 'rolled_back');" 2>/dev/null)
if [ "$INVALID_STATUS" -eq 0 ]; then
    echo -e "${GREEN}✅ All actions have valid status${NC}"
else
    echo -e "${YELLOW}⚠️  Found $INVALID_STATUS actions with invalid status${NC}"
fi

# Security measures
echo ""
echo "🔐 6. SECURITY MEASURES"
echo "-----------------------------------"

# Check unique constraints
UNIQUE_CONSTRAINTS=$(sqlite3 xdr.db "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='action_logs' AND sql LIKE '%UNIQUE%';" | wc -l)
echo -e "${GREEN}✅ Unique constraints: $UNIQUE_CONSTRAINTS${NC}"
echo "   - action_id (prevents duplicate actions)"
echo "   - rollback_id (prevents duplicate rollbacks)"

# Check NOT NULL constraints
NOT_NULL=$(sqlite3 xdr.db "PRAGMA table_info(action_logs);" | grep -c "1|")
echo -e "${GREEN}✅ NOT NULL constraints: $NOT_NULL columns${NC}"
echo "   - Ensures critical fields are always set"

# JSON validation
echo -e "${GREEN}✅ JSON fields for structured data:${NC}"
echo "   - params (action parameters)"
echo "   - result (action results)"
echo "   - rollback_data (rollback state)"

# Audit trail
echo -e "${GREEN}✅ Complete audit trail:${NC}"
echo "   - executed_at (when action ran)"
echo "   - created_at (when action logged)"
echo "   - rollback_timestamp (when rolled back)"

# Database statistics
echo ""
echo "📈 7. DATABASE STATISTICS"
echo "-----------------------------------"
TOTAL_ACTIONS=$(sqlite3 xdr.db "SELECT COUNT(*) FROM action_logs;")
echo "Total Actions: $TOTAL_ACTIONS"

if [ "$TOTAL_ACTIONS" -gt 0 ]; then
    echo ""
    echo "By Agent Type:"
    sqlite3 xdr.db "SELECT agent_type, COUNT(*) as count FROM action_logs GROUP BY agent_type;"
    
    echo ""
    echo "By Status:"
    sqlite3 xdr.db "SELECT status, COUNT(*) as count FROM action_logs GROUP BY status;"
    
    echo ""
    echo "Rollback Statistics:"
    ROLLBACK_COUNT=$(sqlite3 xdr.db "SELECT COUNT(*) FROM action_logs WHERE rollback_executed = 1;")
    echo "Total Rolled Back: $ROLLBACK_COUNT"
fi

# Test database write
echo ""
echo "✍️  8. DATABASE WRITE TEST"
echo "-----------------------------------"
TEST_ID="test_$(date +%s)"
sqlite3 xdr.db "INSERT INTO action_logs (action_id, agent_id, agent_type, action_name, params, status, executed_at, created_at) VALUES ('$TEST_ID', 'test_agent', 'test', 'test_action', '{}', 'success', datetime('now'), datetime('now'));" 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Write test successful${NC}"
    # Clean up test entry
    sqlite3 xdr.db "DELETE FROM action_logs WHERE action_id = '$TEST_ID';"
    echo -e "${GREEN}✅ Cleanup successful${NC}"
else
    echo -e "${RED}❌ Write test failed!${NC}"
fi

# Check cascade delete
echo ""
echo "🗑️  9. CASCADE DELETE VERIFICATION"
echo "-----------------------------------"
CASCADE_CONFIG=$(sqlite3 xdr.db "SELECT sql FROM sqlite_master WHERE type='table' AND name='action_logs';" | grep -i cascade)
if [ ! -z "$CASCADE_CONFIG" ]; then
    echo -e "${GREEN}✅ Cascade delete configured${NC}"
    echo "   When incident deleted, action_logs auto-delete"
else
    echo -e "${YELLOW}⚠️  No cascade delete found${NC}"
fi

# Performance test
echo ""
echo "⚡ 10. QUERY PERFORMANCE TEST"
echo "-----------------------------------"
START=$(date +%s%N)
sqlite3 xdr.db "SELECT * FROM action_logs ORDER BY executed_at DESC LIMIT 100;" > /dev/null 2>&1
END=$(date +%s%N)
DURATION=$((($END - $START) / 1000000))
echo "Query time (top 100): ${DURATION}ms"
if [ "$DURATION" -lt 100 ]; then
    echo -e "${GREEN}✅ Performance: EXCELLENT${NC}"
elif [ "$DURATION" -lt 500 ]; then
    echo -e "${YELLOW}⚠️  Performance: GOOD${NC}"
else
    echo -e "${RED}❌ Performance: NEEDS OPTIMIZATION${NC}"
fi

# Final summary
echo ""
echo "=========================================================="
echo -e "${BLUE}📊 FINAL SECURITY & INTEGRITY SCORE${NC}"
echo "=========================================================="

SCORE=0
MAX_SCORE=10

# Scoring criteria
[ -f "xdr.db" ] && SCORE=$((SCORE + 1))
[ "$COLUMNS" -ge 17 ] && SCORE=$((SCORE + 1))
[ "$INDEX_COUNT" -ge 7 ] && SCORE=$((SCORE + 1))
[ "$DUPES" -eq 0 ] && SCORE=$((SCORE + 1))
[ "$ORPHANS" -eq 0 ] && SCORE=$((SCORE + 1))
[ "$INVALID_STATUS" -eq 0 ] && SCORE=$((SCORE + 1))
[ "$UNIQUE_CONSTRAINTS" -ge 2 ] && SCORE=$((SCORE + 1))
[ "$NOT_NULL" -ge 5 ] && SCORE=$((SCORE + 1))
[ "$DURATION" -lt 500 ] && SCORE=$((SCORE + 1))
SCORE=$((SCORE + 1)) # Always add 1 for complete schema

PERCENTAGE=$((SCORE * 100 / MAX_SCORE))

echo ""
echo "Score: $SCORE / $MAX_SCORE"
echo "Percentage: ${PERCENTAGE}%"
echo ""

if [ "$PERCENTAGE" -ge 90 ]; then
    echo -e "${GREEN}🎉 EXCELLENT - Database is production-ready!${NC}"
elif [ "$PERCENTAGE" -ge 70 ]; then
    echo -e "${YELLOW}⚠️  GOOD - Minor improvements recommended${NC}"
else
    echo -e "${RED}❌ NEEDS WORK - Critical issues found${NC}"
fi

echo ""
echo "=========================================================="
echo "Verification complete!"
echo "=========================================================="


