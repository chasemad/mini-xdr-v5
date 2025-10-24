from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
try:
    from sqlalchemy.ext.asyncio import async_sessionmaker
except ImportError:
    # SQLAlchemy 1.4 compatibility
    from sqlalchemy.orm import sessionmaker
    async_sessionmaker = sessionmaker
from sqlalchemy.orm import declarative_base
from .config import settings

Base = declarative_base()

# Ensure we use async SQLite URL
database_url = settings.database_url
if database_url.startswith('sqlite://'):
    database_url = database_url.replace('sqlite://', 'sqlite+aiosqlite://')

connect_args = {}

# Driver-specific connection arguments
if database_url.startswith("sqlite+aiosqlite://"):
    # aiosqlite supports the standard sqlite3 'timeout' parameter only
    connect_args = {
        "timeout": 60,
    }
elif database_url.startswith("postgresql+asyncpg://"):
    # asyncpg supports server_settings; omit non-standard args here for compatibility
    connect_args = {
        "server_settings": {
            "application_name": "mini-xdr-backend"
        }
    }

engine = create_async_engine(
    database_url,
    echo=False,
    future=True,
    pool_pre_ping=True,  # Verify connections before use
    connect_args=connect_args
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def get_db():
    """Dependency to get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Initialize database tables"""
    # Skip automatic table creation - tables should be created via Alembic migrations
    # async with engine.begin() as conn:
    #     await conn.run_sync(Base.metadata.create_all)
    pass
