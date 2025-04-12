import sqlite3
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class MessageStore:
    def __init__(self):
        self.db_dir = "c:/Documents/projects/AI/data"
        self.db_path = os.path.join(self.db_dir, "conversations.db")
        self.conn = None

    def initialize(self):
        """Initialize the database"""
        try:
            # Create data directory if it doesn't exist
            os.makedirs(self.db_dir, exist_ok=True)
            
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._create_tables()
            logger.info(f"Message store initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def _create_tables(self):
        """Create necessary database tables"""
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_input TEXT NOT NULL,
                    ai_response TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    source VARCHAR(50),
                    session_id VARCHAR(100)
                )
            """)

    def store_conversation(self, user_input: str, ai_response: str, source: str = "web", session_id: str = None):
        """Store a conversation pair"""
        try:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO conversations (user_input, ai_response, source, session_id) VALUES (?, ?, ?, ?)",
                    (user_input, ai_response, source, session_id)
                )
            logger.info(f"Stored conversation from {source}")
        except Exception as e:
            logger.error(f"Failed to store conversation: {e}")