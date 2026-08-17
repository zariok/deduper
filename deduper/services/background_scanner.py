"""Background scanner service for pre-scanning folders.

This service runs in a background thread and continuously monitors folders for changes,
pre-scanning them so that when users navigate to a folder, the results are already cached.

Architecture (Phase 3.1):
- A *dispatcher* thread discovers folders, checks for changes, and enqueues scan
  tasks onto a priority queue.
- A configurable pool of *worker* threads pull tasks from the queue and run the
  actual DuplicateFinder scans concurrently.
- User-requested (prioritized) scans are enqueued at higher priority so they
  execute ahead of routine background work.
- Each queued task carries a cancellation event so in-flight scans can be
  cancelled when the scanner is stopped.
"""

import json
import os
import queue
import sys
import time
import logging
import logging.handlers
import threading
import traceback
from typing import Callable, Any
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from ..utils.logging_config import get_logger
from ..utils.hash_cache import HashCache
from .duplicate_finder import DuplicateFinder

# Get the standard logger
logger = get_logger(__name__)

# Create a dedicated file logger for background scanner
_scanner_file_logger: logging.Logger | None = None


def _setup_scanner_file_logger() -> logging.Logger:
    """Set up a dedicated file logger for background scanner operations."""
    global _scanner_file_logger

    if _scanner_file_logger is not None:
        return _scanner_file_logger

    _scanner_file_logger = logging.getLogger('deduper.background_scanner.file')
    _scanner_file_logger.setLevel(logging.DEBUG)
    _scanner_file_logger.propagate = False  # Don't send to root logger

    # Determine log file path
    # Try to use the data directory, fall back to current directory
    log_dir = os.environ.get('DEDUPER_DATA_DIR', './data')
    log_path = Path(log_dir) / 'logs'
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / 'background_scanner.log'

    # Create rotating file handler
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            str(log_file),
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        _scanner_file_logger.addHandler(file_handler)
        logger.info(f"Background scanner log file: {log_file}")
    except Exception as e:
        logger.warning(f"Could not create background scanner log file: {e}")

    return _scanner_file_logger


def scanner_log(level: str, message: str, exc_info: bool = False):
    """Log to both the main logger and the dedicated scanner file log."""
    # Log to main logger
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(message, exc_info=exc_info)

    # Also log to file logger
    file_logger = _setup_scanner_file_logger()
    if file_logger:
        file_log_func = getattr(file_logger, level.lower(), file_logger.info)
        if exc_info:
            # Manually format exception info for file
            message = f"{message}\n{traceback.format_exc()}"
        file_log_func(message)


class ScanStatus(Enum):
    """Status of a folder scan."""
    PENDING = "pending"
    SCANNING = "scanning"
    COMPLETE = "complete"
    ERROR = "error"
    STALE = "stale"


class ScannerState(Enum):
    """Overall state of the background scanner."""
    IDLE = "idle"
    SCANNING = "scanning"
    SLEEPING = "sleeping"
    WAITING = "waiting"


class ScanPriority(IntEnum):
    """Priority levels for scan tasks.  Lower numeric value = higher priority.

    PriorityQueue returns the smallest item first, so USER (0) beats
    PENDING (1) beats STALE (2) beats RETRY (3).
    """
    USER = 0       # User-requested / prioritized scan
    PENDING = 1    # Never-scanned folder
    STALE = 2      # Folder that changed since last scan
    RETRY = 3      # Previously failed folder being retried


@dataclass(order=True)
class ScanTask:
    """A unit of work for the scan worker pool.

    Ordering is by (priority, enqueued_at) so that higher-priority tasks
    run first and ties are broken by submission time (FIFO).
    """
    priority: ScanPriority
    enqueued_at: float = field(compare=True)
    folder_name: str = field(compare=False)
    cancel_event: threading.Event = field(default_factory=threading.Event, compare=False, repr=False)

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()


@dataclass
class FolderState:
    """Track state of a folder for background scanning."""
    path: str
    last_scan_time: float = 0
    last_modified_time: float = 0
    status: ScanStatus = ScanStatus.PENDING
    file_count: int = 0
    error_message: str = ""
    # Track when folder contents last changed (for 5-minute stability check)
    last_change_detected: float = 0
    # Track scan progress
    scan_progress: int = 0
    scan_total: int = 0
    scan_message: str = ""
    # Track failures for retry logic
    consecutive_failures: int = 0
    last_failure_time: float = 0
    # Track when scan started (for timeout detection)
    scan_start_time: float = 0
    # Track duplicate count: -1 = not yet scanned, 0+ = actual count
    duplicate_count: int = -1
    # When duplicate_count was computed.  Compared against last_modified_time to
    # tell a count that still matches the folder from one the folder has outrun.
    duplicate_count_time: float = 0


class BackgroundScanner:
    """Background service that pre-scans folders for duplicates.

    Features:
    - Scans unscanned folders on startup
    - Monitors folders for changes
    - Waits 5 minutes after last change before rescanning (to allow ongoing transfers)
    - Coordinates with user-initiated scans to avoid duplicate work
    - Provides real-time status updates
    - Timeout protection for stuck scans
    - Retry logic with exponential backoff for failed folders
    - Worker queue with priority scheduling (Phase 3.1)
    - Concurrent folder scanning via configurable worker pool
    """

    # Time to wait after folder changes before rescanning (5 minutes)
    STABILITY_WAIT_SECONDS = 300

    # How often to check for folder changes (30 seconds)
    CHECK_INTERVAL_SECONDS = 30

    # Minimum time between scans of the same folder (10 minutes)
    MIN_RESCAN_INTERVAL_SECONDS = 600

    # Timeout for a single folder scan (60 minutes for 70k+ file folders)
    SCAN_TIMEOUT_SECONDS = 3600

    # Maximum consecutive failures before giving up on a folder
    MAX_CONSECUTIVE_FAILURES = 3

    # Base retry delay after failure (doubles each failure: 5min, 10min, 20min)
    BASE_RETRY_DELAY_SECONDS = 300

    # Number of concurrent scan worker threads (Phase 3.1)
    NUM_SCAN_WORKERS = 2

    def __init__(
        self,
        data_dir: str,
        image_extensions: set[str],
        video_extensions: set[str]
    ):
        self.data_dir = data_dir
        self.image_extensions = image_extensions
        self.video_extensions = video_extensions

        self._folder_states: dict[str, FolderState] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._current_scan_folder: str | None = None

        # Scanner state for UI display
        self._scanner_state: ScannerState = ScannerState.IDLE
        self._state_message: str = "Initializing..."
        self._next_action_time: float = 0  # When the next action will happen

        # Track user-initiated scans to avoid conflicts
        self._user_scanning_folders: set[str] = set()

        # Track the folder currently being viewed in the UI
        # We skip rescanning this folder until user is idle for 5 minutes
        self._ui_active_folder: str | None = None
        self._ui_activity_time: float = 0  # Last time user interacted with the folder

        # Track folders that were rescanned in background and need UI refresh
        self._folders_needing_refresh: set[str] = set()

        # When the scanner thread was started (for UI: "running since app start")
        self._started_at: float = 0.0

        # Allow external progress callbacks to be registered
        self._progress_callbacks: dict[str, Callable] = {}

        # --- Worker queue infrastructure (Phase 3.1) ---
        # PriorityQueue ensures higher-priority tasks (user-requested) run first.
        self._scan_queue: queue.PriorityQueue[ScanTask] = queue.PriorityQueue()
        self._worker_threads: list[threading.Thread] = []
        # Track which folders are currently enqueued or being scanned by workers
        # so we don't enqueue duplicates.
        self._enqueued_folders: set[str] = set()
        # Track folders actively being scanned by worker threads (for UI status)
        self._active_worker_scans: dict[str, str] = {}  # thread_name -> folder_name
        # Pending ScanTask objects keyed by folder_name for cancellation
        self._pending_tasks: dict[str, ScanTask] = {}

    def start(self):
        """Start the dispatcher thread and worker pool."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Background scanner already running")
            return

        self._stop_event.clear()
        self._started_at = time.time()

        # Start worker threads first so they're ready when the dispatcher
        # begins enqueuing tasks.
        for i in range(self.NUM_SCAN_WORKERS):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"ScanWorker-{i}",
                daemon=True,
            )
            self._worker_threads.append(worker)
            worker.start()

        # Start the dispatcher (replaces old single _scanner_loop)
        self._thread = threading.Thread(
            target=self._scanner_loop,
            name="BackgroundScanner",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"Background scanner started ({self.NUM_SCAN_WORKERS} workers)")

    def stop(self):
        """Stop the dispatcher and all worker threads."""
        self._stop_event.set()

        # Cancel all pending tasks so workers unblock quickly
        with self._lock:
            for task in self._pending_tasks.values():
                task.cancel_event.set()
            self._pending_tasks.clear()

        # Drain the queue and post poison pills for each worker
        try:
            while True:
                self._scan_queue.get_nowait()
        except queue.Empty:
            pass

        for _ in self._worker_threads:
            # Sentinel value that workers recognise as a shutdown signal
            self._scan_queue.put(ScanTask(
                priority=ScanPriority.RETRY,   # lowest priority
                enqueued_at=time.time(),
                folder_name="__STOP__",
            ))

        # Wait for dispatcher
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                logger.warning("Dispatcher thread did not stop gracefully")
            else:
                logger.info("Dispatcher thread stopped")
        self._thread = None

        # Wait for workers
        for w in self._worker_threads:
            w.join(timeout=5)
            if w.is_alive():
                logger.warning(f"Worker {w.name} did not stop gracefully")
        self._worker_threads.clear()
        logger.info("Background scanner stopped")

    def is_running(self) -> bool:
        """Check if the background scanner is running."""
        return self._thread is not None and self._thread.is_alive()

    def get_scanner_status(self) -> dict:
        """Get the current overall scanner status for UI display."""
        with self._lock:
            active_scans = list(self._active_worker_scans.values())
            return {
                'state': self._scanner_state.value,
                'message': self._state_message,
                'current_folder': self._current_scan_folder,
                'next_action_in': max(0, self._next_action_time - time.time()) if self._next_action_time > 0 else 0,
                'started_at': self._started_at,
                'worker_count': self.NUM_SCAN_WORKERS,
                'active_scans': active_scans,
                'queue_size': self._scan_queue.qsize(),
            }

    def get_folder_status(self, folder_name: str) -> FolderState | None:
        """Get the current status of a folder."""
        with self._lock:
            return self._folder_states.get(folder_name)

    def get_all_folder_states(self) -> dict[str, FolderState]:
        """Get status of all tracked folders."""
        with self._lock:
            return dict(self._folder_states)

    def is_folder_ready(self, folder_name: str) -> bool:
        """Check if a folder has been pre-scanned and is ready for fast loading.

        Returns True for COMPLETE and STALE folders. STALE means the folder's
        mtime changed but cached results are still valid and displayable.
        The background scanner will rescan when the stability wait passes.
        """
        with self._lock:
            state = self._folder_states.get(folder_name)
            return state is not None and state.status in (ScanStatus.COMPLETE, ScanStatus.STALE)

    def is_folder_being_scanned(self, folder_name: str) -> bool:
        """Check if a folder is currently being scanned (by background or user)."""
        with self._lock:
            # Check if user is scanning this folder
            if folder_name in self._user_scanning_folders:
                return True
            # Check if background is scanning this folder (legacy field)
            if self._current_scan_folder == folder_name:
                return True
            # Check if a worker thread is scanning this folder
            if folder_name in self._active_worker_scans.values():
                return True
            # Check if it's enqueued and waiting
            if folder_name in self._enqueued_folders:
                return True
            return False

    def mark_user_scan_start(self, folder_name: str) -> bool:
        """Mark that a user has started scanning a folder.

        Returns True if the user can proceed, False if background is already scanning.
        If background is scanning, the user should wait for background results.
        """
        with self._lock:
            # If background is currently scanning this folder, return False
            # The user should wait for background results instead
            if self._current_scan_folder == folder_name:
                logger.info(f"Background already scanning {folder_name}, user will wait for results")
                return False

            # Mark this folder as being scanned by user
            self._user_scanning_folders.add(folder_name)
            logger.debug(f"User scan started for {folder_name}")
            return True

    def mark_user_scan_complete(self, folder_name: str):
        """Mark that a user has finished scanning a folder."""
        with self._lock:
            self._user_scanning_folders.discard(folder_name)
            # Update folder state to complete since user just scanned it
            if folder_name in self._folder_states:
                state = self._folder_states[folder_name]
                state.status = ScanStatus.COMPLETE
                state.last_scan_time = time.time()
            logger.debug(f"User scan complete for {folder_name}")

    def prioritize_folder(self, folder_name: str):
        """Mark a folder as high priority for immediate scanning.

        Call this when a user navigates to a folder to bump it to front of queue.
        Enqueues a USER-priority task so it runs ahead of routine background work.
        """
        with self._lock:
            if folder_name in self._folder_states:
                state = self._folder_states[folder_name]
                # If not currently scanning (by anyone), enqueue at high priority
                if (state.status != ScanStatus.SCANNING and
                    folder_name not in self._user_scanning_folders and
                    folder_name not in self._active_worker_scans.values()):
                    state.status = ScanStatus.PENDING
                    state.last_change_detected = 0  # Skip stability wait
                    self._enqueue_folder(folder_name, ScanPriority.USER)
                    logger.debug(f"Prioritized folder for scanning: {folder_name}")

    def set_ui_active_folder(self, folder_name: str | None):
        """Set the folder currently being viewed in the UI.

        The background scanner will skip rescanning this folder until the user
        has been idle for 5 minutes (STABILITY_WAIT_SECONDS).
        """
        with self._lock:
            if folder_name != self._ui_active_folder:
                scanner_log('debug', f"UI active folder changed: {self._ui_active_folder} -> {folder_name}")
            self._ui_active_folder = folder_name
            self._ui_activity_time = time.time()
            # Clear any pending refresh for the newly selected folder
            if folder_name:
                self._folders_needing_refresh.discard(folder_name)

    def mark_ui_activity(self):
        """Mark that the user has interacted with the current folder.

        Call this when user performs actions like deleting duplicates.
        """
        with self._lock:
            self._ui_activity_time = time.time()

    def update_folder_duplicate_count(self, folder_name: str, duplicate_count: int) -> None:
        """Record a duplicate count that was produced outside a background scan.

        The UI removes duplicates through /manage-duplicate, which edits the
        SQLite cache directly and never runs the scanner.  duplicate_count is
        otherwise only written by _scan_folder and _discover_folders, so without
        this the dropdown keeps showing the count from the last full scan - and a
        rescan is at least STABILITY_WAIT_SECONDS + MIN_RESCAN_INTERVAL_SECONDS
        away, longer while the user is still working in the folder.
        """
        with self._lock:
            state = self._folder_states.get(folder_name)
            if state is None:
                return
            # Stamp even when the number is unchanged: the folder mtime moved, so
            # without a fresh stamp the label would decay to "(rescan needed)"
            # despite the count having just been verified.
            state.duplicate_count = duplicate_count
            state.duplicate_count_time = time.time()

        # Emit outside the lock - every status endpoint contends on it.
        self._emit_folder_update(folder_name, state)

    def get_folders_needing_refresh(self) -> set[str]:
        """Get and clear the set of folders that were rescanned and need UI refresh."""
        with self._lock:
            folders = self._folders_needing_refresh.copy()
            self._folders_needing_refresh.clear()
            return folders

    def check_folder_needs_refresh(self, folder_name: str) -> bool:
        """Check if a specific folder needs refresh (and clear the flag)."""
        with self._lock:
            if folder_name in self._folders_needing_refresh:
                self._folders_needing_refresh.discard(folder_name)
                return True
            return False

    def register_progress_callback(self, folder_name: str, callback: Callable):
        """Register a callback to receive progress updates for a folder scan."""
        with self._lock:
            self._progress_callbacks[folder_name] = callback

    def unregister_progress_callback(self, folder_name: str):
        """Unregister a progress callback."""
        with self._lock:
            self._progress_callbacks.pop(folder_name, None)

    def get_folder_progress(self, folder_name: str) -> dict | None:
        """Get the current scan progress for a folder."""
        with self._lock:
            state = self._folder_states.get(folder_name)
            if not state:
                return None

            is_bg = (
                self._current_scan_folder == folder_name or
                folder_name in self._active_worker_scans.values()
            )
            return {
                'status': state.status.value,
                'progress': state.scan_progress,
                'total': state.scan_total,
                'message': state.scan_message,
                'is_background_scan': is_bg
            }

    def _set_scanner_state(self, state: ScannerState, message: str, next_action_time: float = 0):
        """Update the scanner state (thread-safe) and push via WebSocket."""
        with self._lock:
            self._scanner_state = state
            self._state_message = message
            self._next_action_time = next_action_time
            active_scans = list(self._active_worker_scans.values())
            queue_size = self._scan_queue.qsize()

        # Push scanner status to WebSocket subscribers
        try:
            from ..routes.socketio_events import emit_scanner_status
            emit_scanner_status({
                'state': state.value,
                'message': message,
                'next_action_in': max(0, next_action_time - time.time()) if next_action_time > 0 else 0,
                'active_scans': active_scans,
                'queue_size': queue_size,
                'started_at': self._started_at,
            })
        except Exception:
            pass  # WebSocket not yet initialized during startup

    def _emit_folder_update(self, folder_name: str, state=None):
        """Push a single-folder status change to WebSocket subscribers.

        If *state* is not provided, it will be looked up under the lock.
        Pass *state* directly when calling from within a locked section.
        """
        try:
            from ..routes.socketio_events import emit_folder_update
            if state is None:
                with self._lock:
                    state = self._folder_states.get(folder_name)
            if state:
                emit_folder_update(folder_name, state)
        except Exception:
            pass  # WebSocket not yet initialized

    # ------------------------------------------------------------------
    # Enqueue helper
    # ------------------------------------------------------------------

    def _enqueue_folder(self, folder_name: str, priority: ScanPriority) -> bool:
        """Add a folder to the scan queue if not already enqueued or running.

        Returns True if the task was enqueued, False if it was skipped.
        """
        # Must be called while holding self._lock or with knowledge that
        # the caller will manage thread safety.
        if folder_name in self._enqueued_folders:
            return False
        if folder_name in self._active_worker_scans.values():
            return False
        if folder_name in self._user_scanning_folders:
            return False

        task = ScanTask(
            priority=priority,
            enqueued_at=time.time(),
            folder_name=folder_name,
        )
        self._enqueued_folders.add(folder_name)
        self._pending_tasks[folder_name] = task
        self._scan_queue.put(task)
        scanner_log('debug', f"Enqueued {folder_name} at priority {priority.name}")
        return True

    # ------------------------------------------------------------------
    # Dispatcher loop  (was _scanner_loop — now only discovers & enqueues)
    # ------------------------------------------------------------------

    def _scanner_loop(self):
        """Dispatcher: discover folders, detect changes, and enqueue scan tasks."""
        scanner_log('info', "="*60)
        scanner_log('info', "Background scanner dispatcher STARTED")
        scanner_log('info', f"  Data directory: {self.data_dir}")
        scanner_log('info', f"  Workers: {self.NUM_SCAN_WORKERS}")
        scanner_log('info', f"  Scan timeout: {self.SCAN_TIMEOUT_SECONDS}s")
        scanner_log('info', f"  Max failures: {self.MAX_CONSECUTIVE_FAILURES}")
        scanner_log('info', f"  Base retry delay: {self.BASE_RETRY_DELAY_SECONDS}s")
        scanner_log('info', "="*60)

        # Initial discovery
        self._set_scanner_state(ScannerState.IDLE, "Discovering folders...")
        self._discover_folders()

        while not self._stop_event.is_set():
            try:
                # Check for folder changes
                self._check_folders_for_changes()

                # Collect all folders that need scanning and enqueue them
                enqueued_any = self._enqueue_ready_folders()

                # Update UI state
                with self._lock:
                    active_count = len(self._active_worker_scans)
                    queue_size = self._scan_queue.qsize()

                if active_count > 0:
                    self._set_scanner_state(
                        ScannerState.SCANNING,
                        f"Scanning {active_count} folder(s), {queue_size} queued",
                    )
                elif queue_size > 0:
                    self._set_scanner_state(
                        ScannerState.SCANNING,
                        f"{queue_size} folder(s) queued",
                    )
                else:
                    self._set_scanner_state(
                        ScannerState.SLEEPING,
                        "All folders up to date",
                        0,
                    )

                # Sleep before next check
                self._stop_event.wait(self.CHECK_INTERVAL_SECONDS)

            except Exception as e:
                logger.error(f"Error in scanner dispatcher: {e}", exc_info=True)
                self._set_scanner_state(ScannerState.IDLE, f"Error: {str(e)[:50]}")
                self._stop_event.wait(10)

        self._set_scanner_state(ScannerState.IDLE, "Stopped")
        logger.info("Background scanner dispatcher stopped")

    def _enqueue_ready_folders(self) -> bool:
        """Find folders that are ready to scan and enqueue them.

        Returns True if at least one folder was enqueued.
        """
        current_time = time.time()
        enqueued_any = False

        with self._lock:
            for folder_name, state in self._folder_states.items():
                # --- PENDING folders ---
                if state.status == ScanStatus.PENDING:
                    if folder_name in self._user_scanning_folders:
                        continue
                    if self._is_ui_folder_active(folder_name, current_time):
                        continue
                    # Retry delay for previously failed folders
                    if state.consecutive_failures > 0 and state.last_failure_time > 0:
                        retry_delay = self.BASE_RETRY_DELAY_SECONDS * (2 ** (state.consecutive_failures - 1))
                        if current_time - state.last_failure_time < retry_delay:
                            continue
                    if self._enqueue_folder(folder_name, ScanPriority.PENDING):
                        enqueued_any = True

                # --- STALE folders ---
                elif state.status == ScanStatus.STALE:
                    if folder_name in self._user_scanning_folders:
                        continue
                    if self._is_ui_folder_active(folder_name, current_time):
                        continue
                    time_since_change = current_time - state.last_change_detected
                    time_since_scan = current_time - state.last_scan_time
                    if (time_since_change >= self.STABILITY_WAIT_SECONDS and
                            time_since_scan >= self.MIN_RESCAN_INTERVAL_SECONDS):
                        if self._enqueue_folder(folder_name, ScanPriority.STALE):
                            enqueued_any = True

                # --- ERROR folders (retry) ---
                elif state.status == ScanStatus.ERROR:
                    if folder_name in self._user_scanning_folders:
                        continue
                    if self._is_ui_folder_active(folder_name, current_time):
                        continue
                    retry_delay = min(
                        self.BASE_RETRY_DELAY_SECONDS * (2 ** state.consecutive_failures),
                        3600,
                    )
                    if current_time - state.last_failure_time >= retry_delay:
                        scanner_log('info', f"Retrying previously failed folder: {folder_name} "
                                           f"(failed {state.consecutive_failures} times)")
                        if self._enqueue_folder(folder_name, ScanPriority.RETRY):
                            enqueued_any = True

        return enqueued_any

    # ------------------------------------------------------------------
    # Worker loop  (Phase 3.1: runs in each worker thread)
    # ------------------------------------------------------------------

    def _worker_loop(self):
        """Worker thread: pull scan tasks from the queue and execute them."""
        thread_name = threading.current_thread().name
        scanner_log('info', f"[{thread_name}] Worker started")

        while not self._stop_event.is_set():
            try:
                # Block with timeout so we re-check stop_event periodically
                try:
                    task: ScanTask = self._scan_queue.get(timeout=2)
                except queue.Empty:
                    continue

                # Poison pill
                if task.folder_name == "__STOP__":
                    self._scan_queue.task_done()
                    break

                folder_name = task.folder_name

                # Remove from pending bookkeeping
                with self._lock:
                    self._pending_tasks.pop(folder_name, None)

                # Check if the task was cancelled before we start
                if task.cancelled:
                    with self._lock:
                        self._enqueued_folders.discard(folder_name)
                    self._scan_queue.task_done()
                    continue

                # Mark as active
                with self._lock:
                    self._active_worker_scans[thread_name] = folder_name
                    self._enqueued_folders.discard(folder_name)

                try:
                    self._scan_folder(folder_name)
                finally:
                    with self._lock:
                        self._active_worker_scans.pop(thread_name, None)
                    self._scan_queue.task_done()

            except Exception as e:
                scanner_log('error', f"[{thread_name}] Unhandled error: {e}", exc_info=True)
                time.sleep(1)  # avoid tight loop on persistent errors

        scanner_log('info', f"[{thread_name}] Worker stopped")

    def _read_cache_metadata(self, folder_path: str) -> dict[str, Any]:
        """Read lightweight metadata from .deduper.db without loading full hash data.

        Returns a dict with:
        - status: ScanStatus.COMPLETE if valid cache exists, STALE if folder changed, else PENDING
        - last_scan_time: timestamp of last scan or 0
        - duplicate_count: number of duplicate groups or -1 if not scanned
        - file_count: number of files in groups or 0
        - folder_mtime: newest media mtime, so the caller can seed
          last_modified_time without a second scandir
        """
        default_result: dict[str, Any] = {
            'status': ScanStatus.PENDING,
            'last_scan_time': 0.0,
            'duplicate_count': -1,
            'file_count': 0,
            'folder_mtime': 0.0
        }
        status_map = {'pending': ScanStatus.PENDING, 'complete': ScanStatus.COMPLETE, 'stale': ScanStatus.STALE}
        folder_mtime = self._get_folder_mtime(folder_path)
        meta = HashCache.read_metadata(folder_path, folder_mtime)
        return {
            'status': status_map.get(meta['status'], ScanStatus.PENDING),
            'last_scan_time': meta['last_scan_time'],
            'duplicate_count': meta['duplicate_count'],
            'file_count': meta['file_count'],
            'folder_mtime': folder_mtime
        }

    def _discover_folders(self):
        """Discover all user folders in the data directory and initialize state from cache."""
        try:
            if not os.path.exists(self.data_dir):
                scanner_log('warning', f"Data directory does not exist: {self.data_dir}")
                return

            new_folders = []
            for disc_idx, folder_name in enumerate(os.listdir(self.data_dir)):
                folder_path = os.path.join(self.data_dir, folder_name)
                if os.path.isdir(folder_path) and not folder_name.startswith('.'):
                    with self._lock:
                        if folder_name not in self._folder_states:
                            # Try to read existing cache state
                            cache_meta = self._read_cache_metadata(folder_path)

                            self._folder_states[folder_name] = FolderState(
                                path=folder_path,
                                status=cache_meta['status'],
                                last_scan_time=cache_meta['last_scan_time'],
                                duplicate_count=cache_meta['duplicate_count'],
                                # The cached count is only as current as the scan
                                # that produced it; seeding both stamps lets the
                                # freshness test work before the first mtime check.
                                duplicate_count_time=cache_meta['last_scan_time'],
                                last_modified_time=cache_meta['folder_mtime'],
                                file_count=cache_meta['file_count']
                            )
                            new_folders.append(f"{folder_name} ({cache_meta['status'].value})")
                if disc_idx % 20 == 0:
                    time.sleep(0)  # release GIL for HTTP threads

            if new_folders:
                scanner_log('info', f"Discovered {len(new_folders)} folder(s): {', '.join(new_folders)}")

            scanner_log('debug', f"Total folders tracked: {len(self._folder_states)}")

        except Exception as e:
            scanner_log('error', f"Error discovering folders: {e}", exc_info=True)

    def _check_folders_for_changes(self):
        """Check all folders for file changes."""
        current_time = time.time()

        # Also check for new folders
        self._discover_folders()

        # Take a snapshot of folder info under the lock, then do I/O
        # (scandir for mtime) without holding it.  This avoids blocking
        # HTTP threads that call is_folder_ready() / get_scanner_status().
        with self._lock:
            folder_snapshot = [
                (name, state.path, state.last_scan_time, state.last_modified_time, state.status)
                for name, state in self._folder_states.items()
            ]

        for chk_idx, (folder_name, folder_path, last_scan, last_mtime, status) in enumerate(folder_snapshot):
            try:
                folder_mtime = self._get_folder_mtime(folder_path)

                if folder_mtime > last_scan and last_mtime != folder_mtime:
                    with self._lock:
                        state = self._folder_states.get(folder_name)
                        if state and state.last_modified_time != folder_mtime:
                            state.last_modified_time = folder_mtime
                            state.last_change_detected = current_time
                            if state.status == ScanStatus.COMPLETE:
                                state.status = ScanStatus.STALE
                                logger.debug(f"Folder marked stale: {folder_name}")
                                self._emit_folder_update(folder_name, state)
            except Exception as e:
                logger.warning(f"Error checking folder {folder_name}: {e}")

            if chk_idx % 20 == 0:
                time.sleep(0)  # release GIL for HTTP threads

    def _get_folder_mtime(self, folder_path: str) -> float:
        """Get the most recent modification time of media files in a folder.

        Only considers actual content files, ignoring metadata files (.json,
        .deduper.db, thumbnails) that may be updated by the app or external
        processes without representing real content changes.
        """
        try:
            # Don't use folder's own mtime as baseline — it changes when
            # any child file is added/removed (e.g. external .json updates).
            # Instead, only consider actual media file mtimes.
            max_mtime = 0

            # Check immediate children (files and subdirs).
            # Use context manager to close the directory handle promptly —
            # bare scandir() relies on GC and exhausts file descriptors
            # when called across 700+ folders in a tight loop.
            with os.scandir(folder_path) as entries:
                for entry in entries:
                    try:
                        # Skip non-media files when checking for changes.
                        # These are generated by the app or external processes
                        # and shouldn't trigger rescans.
                        entry_name = entry.name
                        if (entry_name.startswith("thumb") or
                            entry_name == ".deduper" or
                            entry_name.startswith(".deduper.db") or  # includes -wal, -shm, -journal
                            entry_name == ".json"):                   # external process metadata
                            continue

                        entry_mtime = entry.stat().st_mtime
                        if entry_mtime > max_mtime:
                            max_mtime = entry_mtime
                    except OSError:
                        continue

            return max_mtime

        except OSError as e:
            logger.warning(f"Error getting mtime for {folder_path}: {e}")
            return 0

    def _is_ui_folder_active(self, folder_name: str, current_time: float) -> bool:
        """Check if a folder is actively being used in the UI.

        Returns True if the folder is the UI active folder AND the user
        has been active within the last STABILITY_WAIT_SECONDS (5 minutes).
        """
        if folder_name != self._ui_active_folder:
            return False

        time_since_activity = current_time - self._ui_activity_time
        return time_since_activity < self.STABILITY_WAIT_SECONDS

    def _scan_folder(self, folder_name: str):
        """Scan a folder for duplicates with timeout protection.

        Called by worker threads.  The actual DuplicateFinder runs inside
        a sub-thread so we can implement a watchdog timeout without
        blocking the worker indefinitely.
        """
        thread_name = threading.current_thread().name

        with self._lock:
            state = self._folder_states.get(folder_name)
            if not state:
                return

            # Double-check user isn't scanning
            if folder_name in self._user_scanning_folders:
                logger.debug(f"Skipping {folder_name}, user is scanning")
                return

            state.status = ScanStatus.SCANNING
            state.scan_progress = 0
            state.scan_total = 0
            state.scan_message = "Starting scan..."
            state.scan_start_time = time.time()
            self._current_scan_folder = folder_name
            folder_path = state.path

        # Emit per-folder update (don't call _set_scanner_state here —
        # the dispatcher loop handles overall state on its next iteration)
        self._emit_folder_update(folder_name)
        scanner_log('info', f"[{thread_name}] Background scan STARTED: {folder_name} (path: {folder_path})")

        # Use a container to store results from the scan sub-thread
        scan_result: dict[str, Any] = {
            'success': False, 'images': None, 'videos': None,
            'error': None, 'traceback': None,
        }
        scan_complete = threading.Event()

        def run_scan():
            """Sub-thread: run the actual DuplicateFinder scan."""
            try:
                scanner_log('info', f"[{folder_name}] Scan sub-thread STARTED")

                last_log_time = [time.time()]

                def progress_callback(status: str, current: int, total: int, message: str):
                    with self._lock:
                        folder_state = self._folder_states.get(folder_name)
                        if folder_state:
                            folder_state.scan_progress = current
                            folder_state.scan_total = total
                            folder_state.scan_message = message

                    now = time.time()
                    should_log = (current > 0 and current % 100 == 0) or (now - last_log_time[0] > 30)
                    if should_log:
                        last_log_time[0] = now
                        scanner_log('info', f"[{folder_name}] Progress: {status} - {current}/{total} - {message}")

                    # Push folder progress via WebSocket (Phase 3.2)
                    try:
                        from ..routes.socketio_events import emit_folder_status
                        emit_folder_status(folder_name, {
                            'folder': folder_name,
                            'status': 'scanning',
                            'scan_progress': current,
                            'scan_total': total,
                            'scan_message': message,
                        })
                    except Exception:
                        pass

                    callback = self._progress_callbacks.get(folder_name)
                    if callback:
                        try:
                            callback(status, current, total, message)
                        except Exception as e:
                            logger.warning(f"Error in progress callback: {e}")

                finder = DuplicateFinder(self.image_extensions, self.video_extensions)
                duplicate_images, duplicate_videos = finder.find_duplicates(
                    folder_path,
                    progress_callback=progress_callback,
                )
                scan_result['success'] = True
                scan_result['images'] = duplicate_images
                scan_result['videos'] = duplicate_videos
                scanner_log('info', f"[{folder_name}] Scan sub-thread COMPLETED successfully")

            except Exception as e:
                scan_result['error'] = str(e)
                scan_result['traceback'] = traceback.format_exc()
                scanner_log('error',
                            f"EXCEPTION in scan for {folder_name}: {e}\n{traceback.format_exc()}")
            finally:
                scan_complete.set()

        # Start scan sub-thread
        sub = threading.Thread(target=run_scan, name=f"ScanRun-{folder_name}", daemon=True)
        sub.start()

        # Watchdog: log every 60s, enforce timeout
        start_time = time.time()
        while not scan_complete.wait(timeout=60):
            elapsed = time.time() - start_time
            if elapsed >= self.SCAN_TIMEOUT_SECONDS:
                break
            with self._lock:
                st = self._folder_states.get(folder_name)
                if st:
                    scanner_log('info',
                                f"[{folder_name}] WATCHDOG: {int(elapsed)}s elapsed - "
                                f"{st.scan_progress}/{st.scan_total} - {st.scan_message}")

        completed = scan_complete.is_set()
        if not completed:
            scanner_log('error',
                        f"TIMEOUT: scan for {folder_name} exceeded {self.SCAN_TIMEOUT_SECONDS}s")
            self._handle_scan_failure(
                folder_name,
                "Scan timed out - folder may have too many files or be inaccessible",
            )
            return

        # Process results
        if scan_result['success']:
            dup_images = scan_result['images']
            dup_videos = scan_result['videos']
            with self._lock:
                st = self._folder_states.get(folder_name)
                if st:
                    st.status = ScanStatus.COMPLETE
                    st.last_scan_time = time.time()
                    st.error_message = ""
                    st.scan_message = "Complete"
                    st.consecutive_failures = 0
                    image_count = sum(len(g.get('duplicate_files', [])) + 1 for g in (dup_images or []))
                    video_count = sum(len(g.get('duplicate_files', [])) + 1 for g in (dup_videos or []))
                    st.file_count = image_count + video_count
                    st.duplicate_count = len(dup_images or []) + len(dup_videos or [])
                    st.duplicate_count_time = st.last_scan_time

            with self._lock:
                if folder_name == self._ui_active_folder:
                    self._folders_needing_refresh.add(folder_name)
                    scanner_log('info', f"Folder {folder_name} marked for UI refresh")

            scanner_log('info',
                        f"Background scan COMPLETE: {folder_name} "
                        f"({len(dup_images or [])} image groups, "
                        f"{len(dup_videos or [])} video groups)")

            # Push completion via WebSocket
            self._emit_folder_update(folder_name)
            try:
                from ..routes.socketio_events import emit_folder_status, emit_folder_refresh
                emit_folder_status(folder_name, {
                    'folder': folder_name,
                    'status': 'complete',
                    'ready': True,
                    'duplicate_count': len(dup_images or []) + len(dup_videos or []),
                })
                emit_folder_refresh(folder_name)
            except Exception:
                pass
        else:
            error_msg = scan_result['error'] or "Unknown error"
            tb = scan_result.get('traceback', '')
            scanner_log('error', f"Background scan FAILED: {folder_name} - {error_msg}")
            if tb:
                scanner_log('debug', f"Traceback for {folder_name}:\n{tb}")
            self._handle_scan_failure(folder_name, error_msg)

            # Push failure via WebSocket
            self._emit_folder_update(folder_name)
            try:
                from ..routes.socketio_events import emit_folder_status
                emit_folder_status(folder_name, {
                    'folder': folder_name,
                    'status': 'error',
                    'error_message': error_msg,
                })
            except Exception:
                pass

        with self._lock:
            if self._current_scan_folder == folder_name:
                self._current_scan_folder = None

        # Close the SQLite connection for this folder to free file descriptors.
        # With 700+ folders, keeping all connections open exhausts the FD limit.
        # The connection will be re-opened on next access via get_hash_cache().
        try:
            from ..utils.hash_cache import close_hash_cache
            close_hash_cache(folder_path)
        except Exception:
            pass

    def _handle_scan_failure(self, folder_name: str, error_message: str):
        """Handle a scan failure with retry tracking."""
        with self._lock:
            state = self._folder_states.get(folder_name)
            if state:
                state.consecutive_failures += 1
                state.last_failure_time = time.time()
                state.error_message = error_message

                if state.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                    state.status = ScanStatus.ERROR
                    state.scan_message = f"Failed {state.consecutive_failures}x: {error_message[:40]}"
                    scanner_log('warning', f"GIVING UP on folder {folder_name} after {state.consecutive_failures} failures. "
                                          f"Error: {error_message}. Will retry in 1 hour.")
                else:
                    # Mark as pending to retry later
                    state.status = ScanStatus.PENDING
                    retry_delay = self.BASE_RETRY_DELAY_SECONDS * (2 ** (state.consecutive_failures - 1))
                    state.scan_message = f"Failed, retry in {retry_delay // 60}min"
                    scanner_log('warning', f"Folder {folder_name} FAILED (attempt {state.consecutive_failures}/{self.MAX_CONSECUTIVE_FAILURES}), "
                                          f"will retry in {retry_delay}s. Error: {error_message}")

            self._current_scan_folder = None


# Global instance for the background scanner
_background_scanner: BackgroundScanner | None = None


def get_background_scanner() -> BackgroundScanner | None:
    """Get the global background scanner instance."""
    return _background_scanner


def init_background_scanner(
    data_dir: str,
    image_extensions: set[str],
    video_extensions: set[str]
) -> BackgroundScanner:
    """Initialize and start the global background scanner."""
    global _background_scanner

    if _background_scanner is not None:
        _background_scanner.stop()

    _background_scanner = BackgroundScanner(
        data_dir=data_dir,
        image_extensions=image_extensions,
        video_extensions=video_extensions
    )
    _background_scanner.start()

    return _background_scanner


def stop_background_scanner():
    """Stop the global background scanner."""
    global _background_scanner

    if _background_scanner is not None:
        _background_scanner.stop()
        _background_scanner = None
