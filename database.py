#!/usr/bin/env python3
"""
db.py — MongoDB Integration Layer for Dr. Mind / Zenark
Database: zenark
All collections are accessed through typed async wrappers.
Motor (async pymongo) is used so FastAPI stays fully async.

Usage:
    from db import db
    await db.chats.save_session(...)
    await db.router_memory.upsert(...)
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bson import ObjectId
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

load_dotenv()

# ── BSON helpers ──────────────────────────────────────────────────────────────

def _serial(doc: dict) -> dict:
    """Convert ObjectId to str so Pydantic / JSON can handle it."""
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── Connection singleton ───────────────────────────────────────────────────────

class _MongoConn:
    _client: Optional[AsyncIOMotorClient] = None
    _db:     Optional[AsyncIOMotorDatabase] = None

    @classmethod
    def get(cls) -> AsyncIOMotorDatabase:
        if cls._db is None:
            uri    = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
            dbname = os.getenv("MONGODB_DB",  "zenark")
            cls._client = AsyncIOMotorClient(uri)
            cls._db     = cls._client[dbname]
        return cls._db

    @classmethod
    async def ping(cls) -> bool:
        try:
            await cls.get().command("ping")
            return True
        except Exception:
            return False


# ── Collection wrappers ────────────────────────────────────────────────────────

class ChatsCollection:
    """
    chats: { _id, session_id, messages, timestamp, userId, tool_history }
    Primary store for full conversation objects.
    """
    def _col(self): return _MongoConn.get()["chats"]

    async def save_session(
        self,
        session_id: str,
        user_id:    str,
        messages:   List[Dict],
        tool_history: List[Dict] | None = None,
    ) -> str:
        doc = {
            "session_id":   session_id,
            "userId":       user_id,
            "messages":     messages,
            "tool_history": tool_history or [],
            "timestamp":    _now(),
        }
        r = await self._col().insert_one(doc)
        return str(r.inserted_id)

    async def update_session(
        self,
        session_id: str,
        messages:   List[Dict],
        tool_history: List[Dict] | None = None,
    ) -> bool:
        update: Dict[str, Any] = {
            "$set": {
                "messages":  messages,
                "timestamp": _now(),
            }
        }
        if tool_history is not None:
            update["$set"]["tool_history"] = tool_history
        r = await self._col().update_one({"session_id": session_id}, update)
        return r.modified_count > 0

    async def upsert_session(
        self,
        session_id:   str,
        user_id:      str,
        messages:     List[Dict],
        tool_history: List[Dict] | None = None,
    ) -> str:
        """Insert or overwrite session — idempotent."""
        doc = {
            "session_id":   session_id,
            "userId":       user_id,
            "messages":     messages,
            "tool_history": tool_history or [],
            "timestamp":    _now(),
        }
        r = await self._col().update_one(
            {"session_id": session_id},
            {"$set": doc},
            upsert=True,
        )
        return session_id

    async def get_session(self, session_id: str) -> Optional[Dict]:
        doc = await self._col().find_one({"session_id": session_id})
        return _serial(doc) if doc else None

    async def get_user_sessions(self, user_id: str, limit: int = 20) -> List[Dict]:
        cursor = self._col().find(
            {"userId": user_id},
            sort=[("timestamp", -1)],
            limit=limit,
        )
        return [_serial(d) async for d in cursor]

    async def delete_session(self, session_id: str) -> bool:
        r = await self._col().delete_one({"session_id": session_id})
        return r.deleted_count > 0


class ChatSessionsNormalizedCollection:
    """
    chat_sessions_normalized:
      { _id, student_id, subject, speaker, message_text, timestamp, labels }
    Flat per-message store — one document per turn. Good for analytics.
    """
    def _col(self): return _MongoConn.get()["chat_sessions_normalized"]

    async def append_message(
        self,
        student_id:   str,
        speaker:      str,        # "student" | "dr_mind"
        message_text: str,
        labels:       List[str] | None = None,   # e.g. ["EXAM_STRESS", "NEGATIVE"]
        subject:      str = "clinical_interview",
    ) -> str:
        doc = {
            "student_id":   student_id,
            "subject":      subject,
            "speaker":      speaker,
            "message_text": message_text,
            "timestamp":    _now(),
            "labels":       labels or [],
        }
        r = await self._col().insert_one(doc)
        return str(r.inserted_id)

    async def get_student_history(
        self,
        student_id: str,
        limit:      int = 50,
    ) -> List[Dict]:
        cursor = self._col().find(
            {"student_id": student_id},
            sort=[("timestamp", -1)],
            limit=limit,
        )
        docs = [_serial(d) async for d in cursor]
        return list(reversed(docs))   # chronological order

    async def get_session_messages(
        self,
        student_id: str,
        since:      datetime | None = None,
    ) -> List[Dict]:
        query: Dict[str, Any] = {"student_id": student_id}
        if since:
            query["timestamp"] = {"$gte": since}
        cursor = self._col().find(query, sort=[("timestamp", 1)])
        return [_serial(d) async for d in cursor]


class RouterMemoryCollection:
    """
    router_memory:
      { _id, student_id, session_id, conversation_count, conversation_flow,
        dominant_emotions, last_emotion, last_tool, recurring_topics,
        tool_preferences, updated_at, preferred_language }
    One document per student — upserted after each turn.
    """
    def _col(self): return _MongoConn.get()["router_memory"]

    async def upsert(
        self,
        student_id:         str,
        session_id:         str,
        last_tool:          str,
        last_emotion:       str,
        preferred_language: str = "ENGLISH",
        dominant_emotions:  List[str] | None = None,
        recurring_topics:   List[str] | None = None,
        tool_preferences:   Dict[str, int] | None = None,
        conversation_flow:  List[str] | None = None,
    ) -> None:
        update: Dict[str, Any] = {
            "$set": {
                "session_id":         session_id,
                "last_tool":          last_tool,
                "last_emotion":       last_emotion,
                "preferred_language": preferred_language,
                "updated_at":         _now(),
            },
            "$inc": {"conversation_count": 1},
        }
        if dominant_emotions  is not None: update["$set"]["dominant_emotions"]  = dominant_emotions
        if recurring_topics   is not None: update["$set"]["recurring_topics"]   = recurring_topics
        if tool_preferences   is not None: update["$set"]["tool_preferences"]   = tool_preferences
        if conversation_flow  is not None: update["$set"]["conversation_flow"]  = conversation_flow
        await self._col().update_one(
            {"student_id": student_id},
            update,
            upsert=True,
        )

    async def get(self, student_id: str) -> Optional[Dict]:
        doc = await self._col().find_one({"student_id": student_id})
        return _serial(doc) if doc else None


class MeditationProgressCollection:
    """
    meditation_progress:
      { _id, session_id, user_id, completed, completed_at, created_at,
        time_spent, updated_at }
    """
    def _col(self): return _MongoConn.get()["meditation_progress"]

    async def start(self, user_id: str, session_id: str) -> str:
        doc = {
            "user_id":      user_id,
            "session_id":   session_id,
            "completed":    False,
            "completed_at": None,
            "time_spent":   0,
            "created_at":   _now(),
            "updated_at":   _now(),
        }
        r = await self._col().insert_one(doc)
        return str(r.inserted_id)

    async def complete(
        self, user_id: str, session_id: str, time_spent: int
    ) -> bool:
        r = await self._col().update_one(
            {"user_id": user_id, "session_id": session_id},
            {"$set": {
                "completed":    True,
                "completed_at": _now(),
                "time_spent":   time_spent,
                "updated_at":   _now(),
            }},
        )
        return r.modified_count > 0

    async def get_user_history(
        self, user_id: str, limit: int = 20
    ) -> List[Dict]:
        cursor = self._col().find(
            {"user_id": user_id},
            sort=[("created_at", -1)],
            limit=limit,
        )
        return [_serial(d) async for d in cursor]


class MeditationStreaksCollection:
    """
    meditation_streaks:
      { _id, user_id, current_streak, longest_streak, last_meditation_date,
        created_at, updated_at }
    """
    def _col(self): return _MongoConn.get()["meditation_streaks"]

    async def get(self, user_id: str) -> Optional[Dict]:
        doc = await self._col().find_one({"user_id": user_id})
        return _serial(doc) if doc else None

    async def update_after_session(self, user_id: str) -> Dict:
        today = datetime.now(timezone.utc).date().isoformat()
        existing = await self.get(user_id)
        if not existing:
            doc = {
                "user_id":              user_id,
                "current_streak":       1,
                "longest_streak":       1,
                "last_meditation_date": today,
                "created_at":           _now(),
                "updated_at":           _now(),
            }
            r = await self._col().insert_one(doc)
            doc["_id"] = str(r.inserted_id)
            return doc

        last = existing.get("last_meditation_date", "")
        current = existing.get("current_streak", 0)
        longest = existing.get("longest_streak", 0)

        from datetime import date
        try:
            last_date = date.fromisoformat(last)
            delta = (date.today() - last_date).days
        except Exception:
            delta = 2   # force reset on parse error

        if delta == 0:     # already meditated today
            new_streak = current
        elif delta == 1:   # consecutive day
            new_streak = current + 1
        else:              # streak broken
            new_streak = 1

        new_longest = max(longest, new_streak)
        await self._col().update_one(
            {"user_id": user_id},
            {"$set": {
                "current_streak":       new_streak,
                "longest_streak":       new_longest,
                "last_meditation_date": today,
                "updated_at":           _now(),
            }},
        )
        return {**existing, "current_streak": new_streak, "longest_streak": new_longest}


class JournalEntriesCollection:
    """
    journal_entries:
      { _id, user_id, mood, title, content, tags, time_spent,
        is_favorite, timestamp, created_at, updated_at }
    Read-only from Dr. Mind's perspective — journals are written by the student app.
    """
    def _col(self): return _MongoConn.get()["journal_entries"]

    async def recent_moods(self, user_id: str, limit: int = 5) -> List[str]:
        cursor = self._col().find(
            {"user_id": user_id},
            {"mood": 1},
            sort=[("created_at", -1)],
            limit=limit,
        )
        docs = [d async for d in cursor]
        return [d["mood"] for d in docs if "mood" in d]


class UsersCollection:
    """
    users: { _id, email, name, password, class, school, isActive, roles, __v, changedPass }
    """
    def _col(self): return _MongoConn.get()["users"]

    async def get_by_id(self, user_id: str) -> Optional[Dict]:
        try:
            doc = await self._col().find_one(
                {"_id": ObjectId(user_id)},
                {"password": 0},   # never return password
            )
        except Exception:
            doc = await self._col().find_one(
                {"_id": user_id},
                {"password": 0},
            )
        return _serial(doc) if doc else None

    async def get_by_email(self, email: str) -> Optional[Dict]:
        doc = await self._col().find_one(
            {"email": email},
            {"password": 0},
        )
        return _serial(doc) if doc else None

    async def is_active(self, user_id: str) -> bool:
        user = await self.get_by_id(user_id)
        return bool(user and user.get("isActive", False))


class AudiosCollection:
    """
    audios: { _id, title, description, genre, fileUrl, duration, isActive, createdAt }
    Used to fetch meditation audio tracks by genre/title.
    """
    def _col(self): return _MongoConn.get()["audios"]

    async def get_active(self, limit: int = 50) -> List[Dict]:
        cursor = self._col().find(
            {"isActive": True},
            sort=[("createdAt", -1)],
            limit=limit,
        )
        return [_serial(d) async for d in cursor]

    async def get_by_genre(self, genre: str, limit: int = 10) -> List[Dict]:
        cursor = self._col().find(
            {"isActive": True, "genre": genre},
            sort=[("createdAt", -1)],
            limit=limit,
        )
        return [_serial(d) async for d in cursor]

    async def get_by_title(self, title: str) -> Optional[Dict]:
        doc = await self._col().find_one({"title": title, "isActive": True})
        return _serial(doc) if doc else None


# ── Façade object — import this everywhere ────────────────────────────────────

class _DB:
    chats                    = ChatsCollection()
    normalized               = ChatSessionsNormalizedCollection()
    router_memory            = RouterMemoryCollection()
    meditation_progress      = MeditationProgressCollection()
    meditation_streaks       = MeditationStreaksCollection()
    journal_entries          = JournalEntriesCollection()
    users                    = UsersCollection()
    audios                   = AudiosCollection()

    def raw(self, collection_name: str):
        """Escape hatch — direct access to any collection."""
        return _MongoConn.get()[collection_name]

    async def ping(self) -> bool:
        return await _MongoConn.ping()


db = _DB()