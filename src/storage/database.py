"""Database session management and helpers."""

from __future__ import annotations

from pathlib import Path

import structlog
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from src.storage.models import Base, NewsArticle

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
        """Quick helper — save multiple objects.

        NewsArticle rows are deduplicated by fingerprint so a single already-seen
        article cannot roll back the whole batch and create noisy IntegrityErrors.
        """
        if not objects:
            return

        if all(isinstance(obj, NewsArticle) for obj in objects):
            with self.get_session() as session:
                fingerprints = [obj.fingerprint for obj in objects]
                existing = {
                    fp
                    for (fp,) in session.query(NewsArticle.fingerprint)
                    .filter(NewsArticle.fingerprint.in_(fingerprints))
                    .all()
                }
                new_objects = [obj for obj in objects if obj.fingerprint not in existing]
                if not new_objects:
                    return
                session.add_all(new_objects)
                try:
                    session.commit()
                except IntegrityError:
                    # Race-safe fallback: persist individually and skip duplicates.
                    session.rollback()
                    for obj in new_objects:
                        session.add(obj)
                        try:
                            session.commit()
                        except IntegrityError:
                            session.rollback()
                return

        with self.get_session() as session:
            session.add_all(objects)
            session.commit()
