"""Database session management and helpers."""

from __future__ import annotations

from pathlib import Path

import structlog
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from src.storage.models import Base

log = structlog.get_logger()


class Database:
    """SQLite database manager."""

    def __init__(self, db_path: str = "data/trading.db"):
        self._db_path = db_path
        self._engine = None
        self._session_factory = None

    def init(self) -> None:
        """Initialize the database — create tables if they don't exist."""
        # Ensure parent directory exists
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._engine = create_engine(
            f"sqlite:///{self._db_path}",
            echo=False,
            connect_args={"check_same_thread": False},
        )
        self._session_factory = sessionmaker(bind=self._engine)

        # Create all tables
        Base.metadata.create_all(self._engine)
        log.info("database.initialized", path=self._db_path)

    def get_session(self) -> Session:
        """Get a new database session."""
        assert self._session_factory is not None, "Database not initialized. Call init() first."
        return self._session_factory()

    def save(self, obj) -> None:
        """Quick helper — save a single object."""
        with self.get_session() as session:
            session.add(obj)
            session.commit()

    def save_all(self, objects: list) -> None:
        """Quick helper — save multiple objects."""
        with self.get_session() as session:
            session.add_all(objects)
            session.commit()
