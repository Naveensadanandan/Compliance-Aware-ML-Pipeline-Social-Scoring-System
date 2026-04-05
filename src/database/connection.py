"""
Database connection and session management.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from config import settings
from src.database.models import Base
import logging
logger = logging.getLogger(__name__)


# Create engine
engine = create_engine(
    settings.database_url,
    echo=False,  # Set to True for SQL query logging
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_database():
    """
    Initialize database by creating all tables.
    Should be run once during setup.
    """
    try:
        logger.info("Initializing database...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def drop_all_tables():
    """
    Drop all tables. Use with caution!
    Only for development/testing.
    """
    logger.warning("Dropping all tables...")
    Base.metadata.drop_all(bind=engine)
    logger.info("All tables dropped.")


@contextmanager
def get_db_session() -> Session:
    """
    Context manager for database sessions.
    Ensures proper cleanup and error handling.
    
    Usage:
        with get_db_session() as session:
            session.query(...)
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()


def get_db():
    """
    Dependency injection for FastAPI.
    
    Usage:
        @app.get("/")
        def read_root(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
