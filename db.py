"""
SQLite persistence for Live Mind Map sessions.
Uses aiosqlite for async access with WAL mode for concurrent reads.
"""

import json, time
import aiosqlite

DB_PATH = "livemind.db"
_db: aiosqlite.Connection | None = None


async def init_db(path: str = DB_PATH):
    """Initialize database connection and create schema."""
    global _db
    _db = await aiosqlite.connect(path)
    _db.row_factory = aiosqlite.Row
    await _db.execute("PRAGMA journal_mode=WAL")
    await _db.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            topic TEXT,
            created_at REAL NOT NULL,
            ended_at REAL,
            summary TEXT
        );
        CREATE TABLE IF NOT EXISTS segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(id),
            seq INTEGER NOT NULL,
            text TEXT NOT NULL,
            is_partial INTEGER NOT NULL DEFAULT 0,
            timestamp REAL NOT NULL,
            UNIQUE(session_id, seq)
        );
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(id),
            seq_at INTEGER NOT NULL,
            graph_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            trigger TEXT
        );
        CREATE TABLE IF NOT EXISTS actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(id),
            action_type TEXT NOT NULL,
            payload TEXT,
            created_at REAL NOT NULL
        );
    """)
    await _db.commit()
    return _db


async def close_db():
    """Close the database connection."""
    global _db
    if _db:
        await _db.close()
        _db = None


async def create_session(session_id: str, topic: str = "") -> dict:
    """Create a new session. Returns session dict."""
    now = time.time()
    await _db.execute(
        "INSERT INTO sessions (id, topic, created_at) VALUES (?, ?, ?)",
        (session_id, topic, now),
    )
    await _db.commit()
    return {"id": session_id, "topic": topic, "created_at": now}


async def end_session(session_id: str, summary: str = ""):
    """Mark a session as ended."""
    await _db.execute(
        "UPDATE sessions SET ended_at = ?, summary = ? WHERE id = ?",
        (time.time(), summary, session_id),
    )
    await _db.commit()


async def store_segment(session_id: str, seq: int, text: str, is_partial: bool, timestamp: float):
    """Store a transcript segment."""
    await _db.execute(
        "INSERT OR REPLACE INTO segments (session_id, seq, text, is_partial, timestamp) VALUES (?, ?, ?, ?, ?)",
        (session_id, seq, text, int(is_partial), timestamp),
    )
    await _db.commit()


async def get_segments_since(session_id: str, from_seq: int) -> list[dict]:
    """Get all segments with seq > from_seq for a session."""
    cursor = await _db.execute(
        "SELECT seq, text, is_partial, timestamp FROM segments WHERE session_id = ? AND seq > ? ORDER BY seq",
        (session_id, from_seq),
    )
    rows = await cursor.fetchall()
    return [{"seq": r["seq"], "text": r["text"], "is_partial": bool(r["is_partial"]), "timestamp": r["timestamp"]} for r in rows]


async def store_snapshot(session_id: str, seq_at: int, graph: dict, trigger: str = "periodic"):
    """Store a graph snapshot."""
    await _db.execute(
        "INSERT INTO snapshots (session_id, seq_at, graph_json, created_at, trigger) VALUES (?, ?, ?, ?, ?)",
        (session_id, seq_at, json.dumps(graph), time.time(), trigger),
    )
    await _db.commit()


async def get_latest_snapshot(session_id: str) -> dict | None:
    """Get the most recent snapshot for a session."""
    cursor = await _db.execute(
        "SELECT seq_at, graph_json, created_at, trigger FROM snapshots WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
        (session_id,),
    )
    row = await cursor.fetchone()
    if not row:
        return None
    return {
        "seq_at": row["seq_at"],
        "graph": json.loads(row["graph_json"]),
        "created_at": row["created_at"],
        "trigger": row["trigger"],
    }


async def store_action(session_id: str, action_type: str, payload: dict):
    """Store a user action (pin, hide, rename, merge, promote)."""
    await _db.execute(
        "INSERT INTO actions (session_id, action_type, payload, created_at) VALUES (?, ?, ?, ?)",
        (session_id, action_type, json.dumps(payload), time.time()),
    )
    await _db.commit()
