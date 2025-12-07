#!/bin/bash

# Clear All Incidents - Safe Deletion Script
# This script safely deletes all incidents and related actions

echo "🗑️  Mini-XDR Database Cleanup"
echo "=============================="
echo ""
echo "⚠️  WARNING: This will delete ALL incidents and related data!"
echo ""
echo "This includes:"
echo "  - All incidents"
echo "  - All manual actions"
echo "  - All workflow actions"
echo "  - All agent actions"
echo "  - All response impact metrics"
echo ""
read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ Cancelled. No data was deleted."
    exit 0
fi

echo ""
echo "📊 Counting current data..."

# Get current counts
INCIDENT_COUNT=$(curl -s http://localhost:8000/incidents \
  -H "x-api-key: demo-minixdr-api-key" \
  | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")

echo "Found: $INCIDENT_COUNT incidents"
echo ""

if [ "$INCIDENT_COUNT" = "0" ]; then
    echo "✅ No incidents to delete!"
    exit 0
fi

echo "🔄 Deleting all incidents..."

# Get all incident IDs
INCIDENT_IDS=$(curl -s http://localhost:8000/incidents \
  -H "x-api-key: demo-minixdr-api-key" \
  | python3 -c "import sys, json; data = json.load(sys.stdin); print(' '.join(str(inc['id']) for inc in data))" 2>/dev/null)

if [ -z "$INCIDENT_IDS" ]; then
    echo "❌ Failed to get incident IDs"
    exit 1
fi

# Delete each incident
DELETED=0
FAILED=0

for ID in $INCIDENT_IDS; do
    echo "  Deleting incident #$ID..."

    RESPONSE=$(curl -s -X DELETE http://localhost:8000/incidents/$ID \
      -H "x-api-key: demo-minixdr-api-key" \
      -w "\n%{http_code}")

    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)

    if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "204" ]; then
        DELETED=$((DELETED + 1))
        echo "    ✅ Deleted"
    else
        FAILED=$((FAILED + 1))
        echo "    ❌ Failed (HTTP $HTTP_CODE)"
    fi
done

echo ""
echo "📊 Cleanup Summary:"
echo "  ✅ Deleted: $DELETED incidents"
echo "  ❌ Failed: $FAILED incidents"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "🎉 All incidents deleted successfully!"
    echo ""
    echo "📝 Next steps:"
    echo "  1. Verify incidents are gone: http://localhost:3000/incidents"
    echo "  2. Run a fresh attack against T-Pot"
    echo "  3. Watch the enterprise UI for real-time updates"
else
    echo "⚠️  Some incidents failed to delete."
    echo "Check backend logs for details."
fi
