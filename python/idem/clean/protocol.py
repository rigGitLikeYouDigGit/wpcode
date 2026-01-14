from __future__ import annotations
import typing as T
from dataclasses import dataclass, field, asdict
from datetime import datetime
import socket
import uuid
import os

"""
IDEM Discovery Protocol

For local same-machine connections, we use a dual approach:
1. Filesystem-based registration: Each session writes a JSON file to a shared temp directory
2. UDP multicast announcements: Sessions periodically broadcast their presence for quick discovery

Session Lifecycle:
- On startup: Generate unique session ID, write registration file, start broadcasting
- During operation: Periodically refresh registration timestamp
- On shutdown: Clean up registration file, send disconnect announcement

Message Types:
- ANNOUNCE: New session joining or periodic heartbeat
- UPDATE: Session name or status changed
- DISCONNECT: Session shutting down
- QUERY: Request all active sessions to respond
- RESPONSE: Reply to a query with session info
"""

@dataclass
class SessionInfo:
    """Information about an IDEM DCC session"""
    session_id: str  # Unique identifier (UUID)
    dcc_type: str  # e.g., "houdini", "maya", "blender"
    session_name: str  # User-editable name, default "{dcc_type}{n}"
    pid: int  # Process ID
    host: str  # Hostname
    port: int  # TCP port for direct communication (0 if not yet bound)
    groups: list[str] = field(default_factory=list)  # Connected group names
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> SessionInfo:
        return cls(**data)


@dataclass
class DiscoveryMessage:
    """Message sent over UDP multicast for discovery"""
    msg_type: str  # ANNOUNCE, UPDATE, DISCONNECT, QUERY, RESPONSE
    session_info: SessionInfo | None = None
    sender_id: str = ""  # Session ID of sender

    def to_dict(self) -> dict:
        result = {"msg_type": self.msg_type, "sender_id": self.sender_id}
        if self.session_info:
            result["session_info"] = self.session_info.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict) -> DiscoveryMessage:
        session_info = None
        if "session_info" in data and data["session_info"]:
            session_info = SessionInfo.from_dict(data["session_info"])
        return cls(
            msg_type=data["msg_type"],
            session_info=session_info,
            sender_id=data.get("sender_id", "")
        )


# Protocol constants
MULTICAST_GROUP = "239.255.43.21"  # Local multicast address
MULTICAST_PORT = 54321
HEARTBEAT_INTERVAL = 5.0  # Seconds between heartbeat announcements
SESSION_TIMEOUT = 15.0  # Seconds before considering a session dead


def generate_session_id() -> str:
    """Generate a unique session identifier"""
    return str(uuid.uuid4())


def get_default_session_name(dcc_type: str, existing_sessions: list[SessionInfo]) -> str:
    """Generate default session name like 'houdini1', 'maya2', etc."""
    # Find highest number for this DCC type
    max_num = 0
    for session in existing_sessions:
        if session.dcc_type == dcc_type and session.session_name.startswith(dcc_type):
            try:
                num = int(session.session_name[len(dcc_type):])
                max_num = max(max_num, num)
            except ValueError:
                pass
    return f"{dcc_type}{max_num + 1}"


def get_registry_dir() -> str:
    """Get the directory for session registration files"""
    import tempfile
    from pathlib import Path

    registry_dir = Path(tempfile.gettempdir()) / "idem_sessions"
    registry_dir.mkdir(exist_ok=True)
    return str(registry_dir)


def get_session_file_path(session_id: str) -> str:
    """Get the file path for a session's registration file"""
    from pathlib import Path
    return str(Path(get_registry_dir()) / f"{session_id}.json")
