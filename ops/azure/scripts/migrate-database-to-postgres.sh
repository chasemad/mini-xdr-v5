#!/bin/bash
# ============================================================================
# Migrate SQLite Database to Azure PostgreSQL
# ============================================================================
# Exports data from SQLite and imports to PostgreSQL
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TERRAFORM_DIR="$PROJECT_ROOT/ops/azure/terraform"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       SQLite to PostgreSQL Migration                           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Get PostgreSQL connection string from Key Vault
if [ ! -f "$TERRAFORM_DIR/terraform.tfstate" ]; then
    echo -e "${RED}❌ Terraform state not found. Deploy infrastructure first.${NC}"
    exit 1
fi

KEY_VAULT_NAME=$(terraform -chdir="$TERRAFORM_DIR" output -raw key_vault_name)
POSTGRES_URL=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "postgres-connection-string" --query value -o tsv)

if [ -z "$POSTGRES_URL" ]; then
    echo -e "${RED}❌ PostgreSQL connection string not found in Key Vault${NC}"
    exit 1
fi

echo "Configuration:"
echo "  • SQLite: $PROJECT_ROOT/backend/xdr.db"
echo "  • PostgreSQL: $(echo $POSTGRES_URL | sed 's/:.*@/@/g')"  # Hide password
echo ""

# Check if SQLite database exists
if [ ! -f "$PROJECT_ROOT/backend/xdr.db" ]; then
    echo -e "${YELLOW}⚠️  SQLite database not found. Skipping data export.${NC}"
    echo "Creating fresh PostgreSQL database..."
else
    echo -e "${YELLOW}Step 1/3: Exporting data from SQLite...${NC}"
    
    # Export to SQL dump
    sqlite3 "$PROJECT_ROOT/backend/xdr.db" .dump > /tmp/sqlite_dump.sql
    echo -e "${GREEN}✓ SQLite data exported${NC}"
fi

# Step 2: Apply Alembic migrations to PostgreSQL
echo -e "${YELLOW}Step 2/3: Applying migrations to PostgreSQL...${NC}"

cd "$PROJECT_ROOT/backend"

# Update Alembic config temporarily
cp alembic.ini alembic.ini.backup
sed -i.bak "s|sqlalchemy.url.*|sqlalchemy.url = ${POSTGRES_URL}|" alembic.ini

# Activate virtual environment if it exists
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

# Run migrations
alembic upgrade head

# Restore original config
mv alembic.ini.backup alembic.ini
rm -f alembic.ini.bak

echo -e "${GREEN}✓ Database schema created${NC}"

# Step 3: Import data if we have it
if [ -f /tmp/sqlite_dump.sql ]; then
    echo -e "${YELLOW}Step 3/3: Importing data to PostgreSQL...${NC}"
    
    # Create Python script to migrate data
    python3 << PYEOF
import os
from sqlalchemy import create_engine, text
from app.models import Base, Incident, Event, AgentCredential, ActionLog

# Connect to both databases
sqlite_url = "sqlite:///$PROJECT_ROOT/backend/xdr.db"
postgres_url = "$POSTGRES_URL"

sqlite_engine = create_engine(sqlite_url)
postgres_engine = create_engine(postgres_url)

from sqlalchemy.orm import sessionmaker

SQLiteSession = sessionmaker(bind=sqlite_engine)
PostgresSession = sessionmaker(bind=postgres_engine)

sqlite_session = SQLiteSession()
postgres_session = PostgresSession()

print("Migrating incidents...")
incidents = sqlite_session.query(Incident).all()
for incident in incidents:
    postgres_session.merge(incident)
postgres_session.commit()
print(f"  ✓ Migrated {len(incidents)} incidents")

print("Migrating events...")
events = sqlite_session.query(Event).all()
for event in events:
    postgres_session.merge(event)
postgres_session.commit()
print(f"  ✓ Migrated {len(events)} events")

print("Migrating agent credentials...")
credentials = sqlite_session.query(AgentCredential).all()
for cred in credentials:
    postgres_session.merge(cred)
postgres_session.commit()
print(f"  ✓ Migrated {len(credentials)} credentials")

print("Migrating action logs...")
try:
    actions = sqlite_session.query(ActionLog).all()
    for action in actions:
        postgres_session.merge(action)
    postgres_session.commit()
    print(f"  ✓ Migrated {len(actions)} actions")
except Exception as e:
    print(f"  ⚠ No action logs to migrate: {e}")

sqlite_session.close()
postgres_session.close()

print("Migration complete!")
PYEOF
    
    echo -e "${GREEN}✓ Data imported successfully${NC}"
    rm /tmp/sqlite_dump.sql
else
    echo -e "${YELLOW}Step 3/3: No data to import (fresh database)${NC}"
fi

# Step 4: Verify migration
echo ""
echo -e "${YELLOW}Verification:${NC}"

python3 << PYEOF
from sqlalchemy import create_engine, text

postgres_url = "$POSTGRES_URL"
engine = create_engine(postgres_url)

with engine.connect() as conn:
    # Check tables exist
    result = conn.execute(text("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
    """))
    tables = [row[0] for row in result]
    print(f"  • Tables: {', '.join(tables)}")
    
    # Count records
    for table in ['incidents', 'events', 'agent_credentials']:
        if table in tables:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = result.scalar()
            print(f"  • {table}: {count} records")
PYEOF

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║        Database Migration Complete!                            ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "PostgreSQL is now the primary database."
echo ""
echo "Update backend configuration:"
echo "  1. Edit backend/.env"
echo "  2. Set DATABASE_URL=$POSTGRES_URL"
echo "  3. Restart backend: kubectl rollout restart deployment/mini-xdr-backend -n mini-xdr"
echo ""

