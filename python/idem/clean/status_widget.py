from __future__ import annotations
import typing as T
from PySide6 import QtWidgets, QtCore, QtGui
from datetime import datetime

from .protocol import SessionInfo
from .discovery import DiscoveryService


class SessionListModel(QtCore.QAbstractTableModel):
    """Model for displaying active IDEM sessions in a table"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sessions: list[SessionInfo] = []
        self._headers = ["Name", "Type", "Groups", "PID", "Status"]

    def set_sessions(self, sessions: list[SessionInfo]):
        """Update the list of sessions"""
        self.beginResetModel()
        self._sessions = sorted(sessions, key=lambda s: (s.dcc_type, s.session_name))
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._sessions)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self._headers)

    def data(self, index: QtCore.QModelIndex, role=QtCore.Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._sessions)):
            return None

        session = self._sessions[index.row()]
        col = index.column()

        if role == QtCore.Qt.DisplayRole:
            if col == 0:  # Name
                return session.session_name
            elif col == 1:  # Type
                return session.dcc_type
            elif col == 2:  # Groups
                return ", ".join(session.groups) if session.groups else "—"
            elif col == 3:  # PID
                return str(session.pid)
            elif col == 4:  # Status
                age = datetime.now().timestamp() - session.timestamp
                if age < 10:
                    return "Active"
                else:
                    return f"Idle ({int(age)}s)"

        elif role == QtCore.Qt.ForegroundRole:
            if col == 4:  # Status color
                age = datetime.now().timestamp() - session.timestamp
                if age < 10:
                    return QtGui.QColor(0, 180, 0)  # Green
                else:
                    return QtGui.QColor(200, 150, 0)  # Yellow

        elif role == QtCore.Qt.UserRole:
            # Store full session info
            return session

        return None

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._headers[section]
        return None

    def get_session(self, row: int) -> SessionInfo | None:
        """Get session info for a given row"""
        if 0 <= row < len(self._sessions):
            return self._sessions[row]
        return None


class IdemStatusWidget(QtWidgets.QWidget):
    """
    Widget to display the status of IDEM discovery system.

    Shows:
    - Current session info (this DCC instance)
    - List of all discovered sessions on the network
    - Ability to rename current session
    - Real-time updates as sessions join/leave
    """

    def __init__(self, discovery_service: DiscoveryService, parent=None):
        super().__init__(parent)
        self.discovery = discovery_service
        self._init_ui()
        self._setup_connections()

        # Register callback for session changes
        self.discovery.register_on_session_changed(self._on_session_changed)

        # Start update timer
        self._update_timer = QtCore.QTimer()
        self._update_timer.timeout.connect(self._refresh_sessions)
        self._update_timer.start(1000)  # Update every second

        # Initial refresh
        self._refresh_sessions()

    def _init_ui(self):
        """Initialize the UI"""
        layout = QtWidgets.QVBoxLayout(self)

        # === Current Session Section ===
        current_group = QtWidgets.QGroupBox("Current Session")
        current_layout = QtWidgets.QFormLayout(current_group)

        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setText(self.discovery.session_info.session_name)
        self.name_edit.setPlaceholderText("Session name")

        self.rename_btn = QtWidgets.QPushButton("Rename")
        self.rename_btn.setMaximumWidth(80)

        name_layout = QtWidgets.QHBoxLayout()
        name_layout.addWidget(self.name_edit)
        name_layout.addWidget(self.rename_btn)

        self.type_label = QtWidgets.QLabel(self.discovery.session_info.dcc_type)
        self.id_label = QtWidgets.QLabel(self.discovery.session_id[:8] + "...")
        self.id_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.pid_label = QtWidgets.QLabel(str(self.discovery.session_info.pid))
        self.groups_label = QtWidgets.QLabel("—")

        current_layout.addRow("Name:", name_layout)
        current_layout.addRow("Type:", self.type_label)
        current_layout.addRow("Session ID:", self.id_label)
        current_layout.addRow("PID:", self.pid_label)
        current_layout.addRow("Groups:", self.groups_label)

        layout.addWidget(current_group)

        # === Discovered Sessions Section ===
        sessions_group = QtWidgets.QGroupBox("Discovered Sessions")
        sessions_layout = QtWidgets.QVBoxLayout(sessions_group)

        # Session count label
        self.session_count_label = QtWidgets.QLabel("0 sessions found")
        sessions_layout.addWidget(self.session_count_label)

        # Session table
        self.session_table = QtWidgets.QTableView()
        self.session_model = SessionListModel()
        self.session_table.setModel(self.session_model)
        self.session_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.session_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.session_table.horizontalHeader().setStretchLastSection(True)
        self.session_table.verticalHeader().setVisible(False)
        self.session_table.setAlternatingRowColors(True)

        # Size columns appropriately
        header = self.session_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)  # Name
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)  # Type
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)  # Groups
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)  # PID
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)  # Status

        sessions_layout.addWidget(self.session_table)

        # Refresh button
        button_layout = QtWidgets.QHBoxLayout()
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.query_btn = QtWidgets.QPushButton("Query Network")
        button_layout.addWidget(self.refresh_btn)
        button_layout.addWidget(self.query_btn)
        button_layout.addStretch()

        sessions_layout.addLayout(button_layout)

        layout.addWidget(sessions_group)

        # Set window properties
        self.setWindowTitle("IDEM Discovery Status")
        self.resize(700, 500)

    def _setup_connections(self):
        """Connect signals and slots"""
        self.rename_btn.clicked.connect(self._on_rename_clicked)
        self.name_edit.returnPressed.connect(self._on_rename_clicked)
        self.refresh_btn.clicked.connect(self._refresh_sessions)
        self.query_btn.clicked.connect(self._on_query_clicked)

    def _on_rename_clicked(self):
        """Handle rename button click"""
        new_name = self.name_edit.text().strip()
        if new_name and new_name != self.discovery.session_info.session_name:
            self.discovery.update_session_name(new_name)
            self._update_current_session_ui()

    def _on_query_clicked(self):
        """Send a query to the network to discover all sessions"""
        from .protocol import DiscoveryMessage
        self.discovery._send_message(DiscoveryMessage("QUERY", sender_id=self.discovery.session_id))

    def _refresh_sessions(self):
        """Refresh the session list from the discovery service"""
        sessions = self.discovery.get_sessions()

        # Update model
        self.session_model.set_sessions(sessions)

        # Update count label
        count = len(sessions)
        self.session_count_label.setText(
            f"{count} session{'s' if count != 1 else ''} found"
        )

        # Update current session UI
        self._update_current_session_ui()

    def _update_current_session_ui(self):
        """Update the current session info display"""
        groups_text = ", ".join(self.discovery.session_info.groups) if self.discovery.session_info.groups else "—"
        self.groups_label.setText(groups_text)

        if self.name_edit.text() != self.discovery.session_info.session_name:
            self.name_edit.setText(self.discovery.session_info.session_name)

    def _on_session_changed(self, event_type: str, session_info: SessionInfo):
        """Handle session change notifications from discovery service"""
        # Schedule UI update on main thread
        QtCore.QMetaObject.invokeMethod(
            self, "_refresh_sessions", QtCore.Qt.QueuedConnection
        )

    def closeEvent(self, event):
        """Handle widget close"""
        self._update_timer.stop()
        super().closeEvent(event)


def create_status_window(discovery_service: DiscoveryService) -> IdemStatusWidget:
    """
    Convenience function to create a standalone status window.

    Args:
        discovery_service: The DiscoveryService instance to display status for

    Returns:
        IdemStatusWidget instance
    """
    widget = IdemStatusWidget(discovery_service)
    widget.show()
    return widget
