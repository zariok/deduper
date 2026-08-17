"""WebSocket event handlers for real-time progress updates.

Clients can:
- Subscribe to scan progress for a specific session (``subscribe_progress``)
- Subscribe to background scanner status updates (``subscribe_scanner``)
- Request a folder scan cancellation (``cancel_scan``)

The server pushes:
- ``scan_progress``   — per-file hashing / grouping progress
- ``scanner_init``    — full scanner + folder state on first subscribe
- ``scanner_status``  — overall background scanner state (incremental)
- ``folder_update``   — single-folder status change (incremental dropdown update)
- ``folder_status``   — per-folder scan state changes (to folder room subscribers)
- ``folder_refresh``  — notification that a folder was rescanned in the background
"""

from flask_socketio import emit, join_room, leave_room
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


def _serialize_folder_state(state) -> dict:
    """Convert a FolderState to a JSON-serializable dict.

    Reusable by both the WebSocket init payload and HTTP status endpoint.
    """
    from ..services.background_scanner import ScanStatus
    return {
        'status': state.status.value,
        'last_scan_time': state.last_scan_time,
        'last_modified_time': state.last_modified_time,
        'file_count': state.file_count,
        'error_message': state.error_message,
        'ready': state.status in (ScanStatus.COMPLETE, ScanStatus.STALE),
        'scan_progress': state.scan_progress,
        'scan_total': state.scan_total,
        'scan_message': state.scan_message,
        'duplicate_count': state.duplicate_count,
        # True when the count was computed at or after the newest change we know
        # about, so a stale folder can still show a number the UI just verified
        # instead of falling back to "(rescan needed)".
        'duplicate_count_fresh': (
            state.duplicate_count >= 0
            and state.duplicate_count_time >= state.last_modified_time
        ),
    }


def register_socketio_events(socketio):
    """Register all Socket.IO event handlers on the given *socketio* instance."""

    @socketio.on("connect")
    def handle_connect():
        logger.debug("WebSocket client connected")
        emit("connected", {"status": "ok"})

    @socketio.on("disconnect")
    def handle_disconnect():
        logger.debug("WebSocket client disconnected")

    # ------------------------------------------------------------------
    # Scan progress subscription
    # ------------------------------------------------------------------

    @socketio.on("subscribe_progress")
    def handle_subscribe_progress(data):
        """Client subscribes to progress updates for a scan session.

        Expects ``{"session_id": "..."}``
        The client is placed in a Socket.IO room named ``progress:<session_id>``
        so that ``emit_scan_progress`` can target only interested clients.
        """
        session_id = data.get("session_id")
        if session_id:
            room = f"progress:{session_id}"
            join_room(room)
            logger.debug(f"Client joined progress room: {room}")
            emit("subscribed", {"room": room})

    @socketio.on("unsubscribe_progress")
    def handle_unsubscribe_progress(data):
        session_id = data.get("session_id")
        if session_id:
            room = f"progress:{session_id}"
            leave_room(room)
            logger.debug(f"Client left progress room: {room}")

    # ------------------------------------------------------------------
    # Scanner status subscription
    # ------------------------------------------------------------------

    @socketio.on("subscribe_scanner")
    def handle_subscribe_scanner():
        """Client subscribes to background scanner status broadcasts.

        On subscribe, emits a ``scanner_init`` event with the full scanner
        status and all folder states so the client can populate the dropdown
        without an HTTP poll.
        """
        join_room("scanner")
        logger.debug("Client joined scanner room")
        emit("subscribed", {"room": "scanner"})

        # Send full initial state so the client can populate the dropdown
        try:
            from ..services.background_scanner import get_background_scanner
            scanner = get_background_scanner()
            if scanner:
                scanner_info = scanner.get_scanner_status()
                folder_states = scanner.get_all_folder_states()
                folders_info = {
                    name: _serialize_folder_state(state)
                    for name, state in folder_states.items()
                }
                emit("scanner_init", {
                    'scanner_state': scanner_info['state'],
                    'scanner_message': scanner_info['message'],
                    'active_scans': scanner_info.get('active_scans', []),
                    'queue_size': scanner_info.get('queue_size', 0),
                    'started_at': scanner_info.get('started_at'),
                    'folders': folders_info,
                })
        except Exception as e:
            logger.debug(f"Error sending scanner_init: {e}")

    @socketio.on("unsubscribe_scanner")
    def handle_unsubscribe_scanner():
        leave_room("scanner")
        logger.debug("Client left scanner room")

    # ------------------------------------------------------------------
    # Folder-level subscription
    # ------------------------------------------------------------------

    @socketio.on("subscribe_folder")
    def handle_subscribe_folder(data):
        """Subscribe to status changes for a specific folder."""
        folder = data.get("folder")
        if folder:
            room = f"folder:{folder}"
            join_room(room)
            logger.debug(f"Client joined folder room: {room}")
            emit("subscribed", {"room": room})

    @socketio.on("unsubscribe_folder")
    def handle_unsubscribe_folder(data):
        folder = data.get("folder")
        if folder:
            leave_room(f"folder:{folder}")

    # ------------------------------------------------------------------
    # Client-initiated cancellation  (future use)
    # ------------------------------------------------------------------

    @socketio.on("cancel_scan")
    def handle_cancel_scan(data):
        """Request cancellation of a running scan (placeholder for future use)."""
        session_id = data.get("session_id")
        logger.info(f"Scan cancellation requested for {session_id}")
        emit("cancel_ack", {"session_id": session_id, "status": "acknowledged"})


# ======================================================================
# Server-side emit helpers  (called from views.py / background_scanner)
# ======================================================================

_socketio_instance = None


def init_socketio_ref(socketio):
    """Store a module-level reference so helpers can emit without imports."""
    global _socketio_instance
    _socketio_instance = socketio


def emit_scan_progress(session_id: str, progress: dict) -> None:
    """Push a scan progress update to all clients watching *session_id*."""
    sio = _socketio_instance
    if sio is None:
        return
    try:
        sio.emit("scan_progress", progress, room=f"progress:{session_id}")
    except Exception as e:
        logger.debug(f"emit_scan_progress error: {e}")


def emit_scanner_status(status: dict) -> None:
    """Broadcast overall scanner status to all subscribed clients."""
    sio = _socketio_instance
    if sio is None:
        return
    try:
        sio.emit("scanner_status", status, room="scanner")
    except Exception as e:
        logger.debug(f"emit_scanner_status error: {e}")


def emit_folder_status(folder_name: str, status: dict) -> None:
    """Push a folder-level status change to watchers of that folder."""
    sio = _socketio_instance
    if sio is None:
        return
    try:
        sio.emit("folder_status", status, room=f"folder:{folder_name}")
    except Exception as e:
        logger.debug(f"emit_folder_status error: {e}")


def emit_folder_refresh(folder_name: str) -> None:
    """Notify clients that a folder has been rescanned and needs refresh."""
    sio = _socketio_instance
    if sio is None:
        return
    try:
        sio.emit("folder_refresh", {"folder": folder_name}, room=f"folder:{folder_name}")
        # Also broadcast to the general scanner room
        sio.emit("folder_refresh", {"folder": folder_name}, room="scanner")
    except Exception as e:
        logger.debug(f"emit_folder_refresh error: {e}")


def emit_folder_update(folder_name: str, state) -> None:
    """Push a single-folder status update to all scanner-room subscribers.

    Used for incremental dropdown updates — the client patches just this
    folder's ``<option>`` instead of rebuilding the entire list.
    """
    sio = _socketio_instance
    if sio is None:
        return
    try:
        payload = _serialize_folder_state(state)
        payload['folder'] = folder_name
        sio.emit("folder_update", payload, room="scanner")
    except Exception as e:
        logger.debug(f"emit_folder_update error: {e}")
