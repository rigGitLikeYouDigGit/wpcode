from __future__ import annotations
import typing as T
import socket
import threading
import time
import orjson
from pathlib import Path
from datetime import datetime
from wplib import log

from .protocol import (
    SessionInfo, DiscoveryMessage,
    MULTICAST_GROUP, MULTICAST_PORT,
    HEARTBEAT_INTERVAL, SESSION_TIMEOUT,
    generate_session_id, get_default_session_name,
    get_registry_dir, get_session_file_path
)


class DiscoveryService:
    """
    Handles discovery and registration of IDEM sessions on the local machine.

    Uses a dual approach:
    - Filesystem registry for persistence and crash recovery
    - UDP multicast for fast, real-time discovery
    """

    def __init__(self, dcc_type: str, session_name: str | None = None, port: int = 0):
        self.dcc_type = dcc_type
        self.session_id = generate_session_id()
        self.port = port
        self._running = False
        self._heartbeat_thread: threading.Thread | None = None
        self._listener_thread: threading.Thread | None = None
        self._socket: socket.socket | None = None
        self._known_sessions: dict[str, SessionInfo] = {}
        self._lock = threading.Lock()
        self._callbacks: list[T.Callable[[str, SessionInfo], None]] = []

        # Create initial session info
        self.session_info = SessionInfo(
            session_id=self.session_id,
            dcc_type=dcc_type,
            session_name=session_name or dcc_type + "1",
            pid=__import__("os").getpid(),
            host=socket.gethostname(),
            port=port,
            groups=[],
            timestamp=datetime.now().timestamp()
        )

    def start(self):
        """Start the discovery service"""
        if self._running:
            return

        # Scan existing sessions and generate appropriate name if needed
        self._scan_filesystem_registry()
        if self.session_info.session_name == self.dcc_type + "1":
            # Update to proper numbered name
            self.session_info.session_name = get_default_session_name(
                self.dcc_type, list(self._known_sessions.values())
            )

        self._running = True

        # Write registration file
        self._write_registration()

        # Set up UDP multicast socket
        self._setup_socket()

        # Start background threads
        self._listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listener_thread.start()

        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

        # Send initial announcement
        self._send_message(DiscoveryMessage("ANNOUNCE", self.session_info, self.session_id))

        # Query for other sessions
        self._send_message(DiscoveryMessage("QUERY", sender_id=self.session_id))

        log.info(f"IDEM Discovery started: {self.session_info.session_name} ({self.session_id})")

    def stop(self):
        """Stop the discovery service and clean up"""
        if not self._running:
            return

        self._running = False

        # Send disconnect message
        self._send_message(DiscoveryMessage("DISCONNECT", self.session_info, self.session_id))

        # Clean up registration file
        try:
            Path(get_session_file_path(self.session_id)).unlink(missing_ok=True)
        except Exception as e:
            log.warning(f"Failed to remove registration file: {e}")

        # Close socket
        if self._socket:
            self._socket.close()

        # Wait for threads
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=1.0)
        if self._listener_thread:
            self._listener_thread.join(timeout=1.0)

        log.info(f"IDEM Discovery stopped: {self.session_info.session_name}")

    def get_sessions(self) -> list[SessionInfo]:
        """Get list of all known active sessions"""
        with self._lock:
            # Clean up stale sessions
            now = datetime.now().timestamp()
            stale = [
                sid for sid, info in self._known_sessions.items()
                if now - info.timestamp > SESSION_TIMEOUT
            ]
            for sid in stale:
                del self._known_sessions[sid]
                log.debug(f"Removed stale session: {sid}")

            return list(self._known_sessions.values())

    def update_session_name(self, new_name: str):
        """Update the session name"""
        self.session_info.session_name = new_name
        self.session_info.timestamp = datetime.now().timestamp()
        self._write_registration()
        self._send_message(DiscoveryMessage("UPDATE", self.session_info, self.session_id))

    def add_group(self, group_name: str):
        """Add this session to a group"""
        if group_name not in self.session_info.groups:
            self.session_info.groups.append(group_name)
            self.session_info.timestamp = datetime.now().timestamp()
            self._write_registration()
            self._send_message(DiscoveryMessage("UPDATE", self.session_info, self.session_id))

    def remove_group(self, group_name: str):
        """Remove this session from a group"""
        if group_name in self.session_info.groups:
            self.session_info.groups.remove(group_name)
            self.session_info.timestamp = datetime.now().timestamp()
            self._write_registration()
            self._send_message(DiscoveryMessage("UPDATE", self.session_info, self.session_id))

    def register_on_session_changed(self, callback: T.Callable[[str, SessionInfo], None]):
        """
        Register a callback for when sessions are discovered or updated.
        Callback signature: (event_type: str, session_info: SessionInfo)
        event_type is one of: "added", "updated", "removed"
        """
        self._callbacks.append(callback)

    def _setup_socket(self):
        """Set up UDP multicast socket"""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Bind to the multicast port
        self._socket.bind(("", MULTICAST_PORT))

        # Join multicast group
        mreq = socket.inet_aton(MULTICAST_GROUP) + socket.inet_aton("0.0.0.0")
        self._socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

        # Set socket to non-blocking for clean shutdown
        self._socket.settimeout(1.0)

    def _send_message(self, msg: DiscoveryMessage):
        """Send a discovery message via multicast"""
        if not self._socket:
            return

        try:
            data = orjson.dumps(msg.to_dict())
            self._socket.sendto(data, (MULTICAST_GROUP, MULTICAST_PORT))
        except Exception as e:
            log.warning(f"Failed to send discovery message: {e}")

    def _listen_loop(self):
        """Background thread to listen for discovery messages"""
        while self._running:
            try:
                data, addr = self._socket.recvfrom(4096)
                msg_dict = orjson.loads(data)
                msg = DiscoveryMessage.from_dict(msg_dict)

                # Ignore our own messages
                if msg.sender_id == self.session_id:
                    continue

                self._handle_message(msg)

            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    log.warning(f"Error in discovery listener: {e}")

    def _handle_message(self, msg: DiscoveryMessage):
        """Handle an incoming discovery message"""
        if msg.msg_type == "QUERY":
            # Respond with our session info
            self._send_message(DiscoveryMessage("RESPONSE", self.session_info, self.session_id))

        elif msg.msg_type in ("ANNOUNCE", "UPDATE", "RESPONSE"):
            if msg.session_info:
                with self._lock:
                    is_new = msg.session_info.session_id not in self._known_sessions
                    self._known_sessions[msg.session_info.session_id] = msg.session_info

                event_type = "added" if is_new else "updated"
                for callback in self._callbacks:
                    try:
                        callback(event_type, msg.session_info)
                    except Exception as e:
                        log.warning(f"Error in session callback: {e}")

        elif msg.msg_type == "DISCONNECT":
            if msg.session_info:
                with self._lock:
                    if msg.session_info.session_id in self._known_sessions:
                        del self._known_sessions[msg.session_info.session_id]

                for callback in self._callbacks:
                    try:
                        callback("removed", msg.session_info)
                    except Exception as e:
                        log.warning(f"Error in session callback: {e}")

    def _heartbeat_loop(self):
        """Background thread to send periodic heartbeat announcements"""
        while self._running:
            time.sleep(HEARTBEAT_INTERVAL)
            if self._running:
                self.session_info.timestamp = datetime.now().timestamp()
                self._write_registration()
                self._send_message(DiscoveryMessage("ANNOUNCE", self.session_info, self.session_id))

    def _write_registration(self):
        """Write session info to filesystem registry"""
        try:
            file_path = Path(get_session_file_path(self.session_id))
            file_path.write_bytes(orjson.dumps(self.session_info.to_dict()))
        except Exception as e:
            log.warning(f"Failed to write registration file: {e}")

    def _scan_filesystem_registry(self):
        """Scan filesystem registry for existing sessions"""
        registry_dir = Path(get_registry_dir())
        now = datetime.now().timestamp()

        for file_path in registry_dir.glob("*.json"):
            try:
                data = orjson.loads(file_path.read_bytes())
                session_info = SessionInfo.from_dict(data)

                # Check if session is still alive
                if now - session_info.timestamp > SESSION_TIMEOUT:
                    # Clean up stale file
                    file_path.unlink(missing_ok=True)
                else:
                    with self._lock:
                        self._known_sessions[session_info.session_id] = session_info

            except Exception as e:
                log.warning(f"Failed to read session file {file_path}: {e}")
