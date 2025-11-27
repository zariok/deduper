"""Background scanner service for pre-scanning folders.

This service runs in a background thread and continuously monitors folders for changes,
pre-scanning them so that when users navigate to a folder, the results are already cached.
"""

import os
import sys
import time
import logging
import logging.handlers
import threading
import traceback
from typing import Dict, Optional, Set, Callable, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from ..utils.logging_config import get_logger
from ..utils.hash_cache import HashCache
from .duplicate_finder import DuplicateFinder

# Get the standard logger
logger = get_logger(__name__)

# Create a dedicated file logger for background scanner
_scanner_file_logger: Optional[logging.Logger] = None


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
    """

    # Time to wait after folder changes before rescanning (5 minutes)
    STABILITY_WAIT_SECONDS = 300

    # How often to check for folder changes (30 seconds)
    CHECK_INTERVAL_SECONDS = 30

    # Minimum time between scans of the same folder (10 minutes)
    MIN_RESCAN_INTERVAL_SECONDS = 600

    # Timeout for a single folder scan (10 minutes)
    SCAN_TIMEOUT_SECONDS = 600

    # Maximum consecutive failures before giving up on a folder
    MAX_CONSECUTIVE_FAILURES = 3

    # Base retry delay after failure (doubles each failure: 5min, 10min, 20min)
    BASE_RETRY_DELAY_SECONDS = 300

    def __init__(
        self,
        data_dir: str,
        image_extensions: Set[str],
        video_extensions: Set[str]
    ):
        self.data_dir = data_dir
        self.image_extensions = image_extensions
        self.video_extensions = video_extensions

        self._folder_states: Dict[str, FolderState] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._current_scan_folder: Optional[str] = None

        # Scanner state for UI display
        self._scanner_state: ScannerState = ScannerState.IDLE
        self._state_message: str = "Initializing..."
        self._next_action_time: float = 0  # When the next action will happen

        # Track user-initiated scans to avoid conflicts
        self._user_scanning_folders: Set[str] = set()

        # Allow external progress callbacks to be registered
        self._progress_callbacks: Dict[str, Callable] = {}

    def start(self):
        """Start the background scanner thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Background scanner already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._scanner_loop,
            name="BackgroundScanner",
            daemon=True  # Thread will stop when main program exits
        )
        self._thread.start()
        logger.info("Background scanner started")

    def stop(self):
        """Stop the background scanner thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                logger.warning("Background scanner thread did not stop gracefully")
            else:
                logger.info("Background scanner stopped")
        self._thread = None

    def is_running(self) -> bool:
        """Check if the background scanner is running."""
        return self._thread is not None and self._thread.is_alive()

    def get_scanner_status(self) -> Dict:
        """Get the current overall scanner status for UI display."""
        with self._lock:
            return {
                'state': self._scanner_state.value,
                'message': self._state_message,
                'current_folder': self._current_scan_folder,
                'next_action_in': max(0, self._next_action_time - time.time()) if self._next_action_time > 0 else 0
            }

    def get_folder_status(self, folder_name: str) -> Optional[FolderState]:
        """Get the current status of a folder."""
        with self._lock:
            return self._folder_states.get(folder_name)

    def get_all_folder_states(self) -> Dict[str, FolderState]:
        """Get status of all tracked folders."""
        with self._lock:
            return dict(self._folder_states)

    def is_folder_ready(self, folder_name: str) -> bool:
        """Check if a folder has been pre-scanned and is ready for fast loading."""
        with self._lock:
            state = self._folder_states.get(folder_name)
            return state is not None and state.status == ScanStatus.COMPLETE

    def is_folder_being_scanned(self, folder_name: str) -> bool:
        """Check if a folder is currently being scanned (by background or user)."""
        with self._lock:
            # Check if user is scanning this folder
            if folder_name in self._user_scanning_folders:
                return True
            # Check if background is scanning this folder
            if self._current_scan_folder == folder_name:
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
        """
        with self._lock:
            if folder_name in self._folder_states:
                state = self._folder_states[folder_name]
                # If not currently scanning (by anyone), mark as pending to rescan soon
                if (state.status != ScanStatus.SCANNING and
                    folder_name not in self._user_scanning_folders and
                    self._current_scan_folder != folder_name):
                    state.status = ScanStatus.PENDING
                    state.last_change_detected = 0  # Skip stability wait
                    logger.debug(f"Prioritized folder for scanning: {folder_name}")

    def register_progress_callback(self, folder_name: str, callback: Callable):
        """Register a callback to receive progress updates for a folder scan."""
        with self._lock:
            self._progress_callbacks[folder_name] = callback

    def unregister_progress_callback(self, folder_name: str):
        """Unregister a progress callback."""
        with self._lock:
            self._progress_callbacks.pop(folder_name, None)

    def get_folder_progress(self, folder_name: str) -> Optional[Dict]:
        """Get the current scan progress for a folder."""
        with self._lock:
            state = self._folder_states.get(folder_name)
            if not state:
                return None

            return {
                'status': state.status.value,
                'progress': state.scan_progress,
                'total': state.scan_total,
                'message': state.scan_message,
                'is_background_scan': self._current_scan_folder == folder_name
            }

    def _set_scanner_state(self, state: ScannerState, message: str, next_action_time: float = 0):
        """Update the scanner state (thread-safe)."""
        with self._lock:
            self._scanner_state = state
            self._state_message = message
            self._next_action_time = next_action_time

    def _scanner_loop(self):
        """Main loop for the background scanner."""
        scanner_log('info', "="*60)
        scanner_log('info', "Background scanner loop STARTED")
        scanner_log('info', f"  Data directory: {self.data_dir}")
        scanner_log('info', f"  Scan timeout: {self.SCAN_TIMEOUT_SECONDS}s")
        scanner_log('info', f"  Max failures: {self.MAX_CONSECUTIVE_FAILURES}")
        scanner_log('info', f"  Base retry delay: {self.BASE_RETRY_DELAY_SECONDS}s")
        scanner_log('info', "="*60)

        # Initial scan of all folders
        self._set_scanner_state(ScannerState.IDLE, "Discovering folders...")
        self._discover_folders()

        while not self._stop_event.is_set():
            try:
                # Check for folder changes and queue scans
                self._check_folders_for_changes()

                # Find next folder to scan
                folder_to_scan, wait_reason = self._get_next_folder_to_scan()

                if folder_to_scan:
                    self._scan_folder(folder_to_scan)
                else:
                    # No folders need scanning
                    if wait_reason:
                        self._set_scanner_state(
                            ScannerState.WAITING,
                            wait_reason,
                            time.time() + self.CHECK_INTERVAL_SECONDS
                        )
                    else:
                        self._set_scanner_state(
                            ScannerState.SLEEPING,
                            "All folders up to date",
                            0
                        )

                    # Wait before checking again
                    self._stop_event.wait(self.CHECK_INTERVAL_SECONDS)

            except Exception as e:
                logger.error(f"Error in background scanner loop: {e}", exc_info=True)
                self._set_scanner_state(ScannerState.IDLE, f"Error: {str(e)[:50]}")
                # Wait before retrying to avoid tight error loops
                self._stop_event.wait(10)

        self._set_scanner_state(ScannerState.IDLE, "Stopped")
        logger.info("Background scanner loop stopped")

    def _discover_folders(self):
        """Discover all user folders in the data directory."""
        try:
            if not os.path.exists(self.data_dir):
                scanner_log('warning', f"Data directory does not exist: {self.data_dir}")
                return

            new_folders = []
            for folder_name in os.listdir(self.data_dir):
                folder_path = os.path.join(self.data_dir, folder_name)
                if os.path.isdir(folder_path) and not folder_name.startswith('.'):
                    with self._lock:
                        if folder_name not in self._folder_states:
                            self._folder_states[folder_name] = FolderState(
                                path=folder_path,
                                status=ScanStatus.PENDING
                            )
                            new_folders.append(folder_name)

            if new_folders:
                scanner_log('info', f"Discovered {len(new_folders)} new folder(s): {', '.join(new_folders)}")

            scanner_log('debug', f"Total folders tracked: {len(self._folder_states)}")

        except Exception as e:
            scanner_log('error', f"Error discovering folders: {e}", exc_info=True)

    def _check_folders_for_changes(self):
        """Check all folders for file changes."""
        current_time = time.time()

        # Also check for new folders
        self._discover_folders()

        with self._lock:
            for folder_name, state in self._folder_states.items():
                try:
                    # Get current modification time of folder
                    folder_mtime = self._get_folder_mtime(state.path)

                    # Check if folder has changed since last scan
                    if folder_mtime > state.last_scan_time:
                        if state.last_modified_time != folder_mtime:
                            # Folder contents changed - update tracking
                            state.last_modified_time = folder_mtime
                            state.last_change_detected = current_time

                            if state.status == ScanStatus.COMPLETE:
                                state.status = ScanStatus.STALE
                                logger.debug(f"Folder marked stale: {folder_name}")

                except Exception as e:
                    logger.warning(f"Error checking folder {folder_name}: {e}")

    def _get_folder_mtime(self, folder_path: str) -> float:
        """Get the most recent modification time in a folder.

        Checks the folder itself and immediate children for changes.
        """
        try:
            # Start with folder's own mtime
            max_mtime = os.path.getmtime(folder_path)

            # Check immediate children (files and subdirs)
            for entry in os.scandir(folder_path):
                try:
                    entry_mtime = entry.stat().st_mtime
                    if entry_mtime > max_mtime:
                        max_mtime = entry_mtime
                except OSError:
                    continue

            return max_mtime

        except OSError as e:
            logger.warning(f"Error getting mtime for {folder_path}: {e}")
            return 0

    def _get_next_folder_to_scan(self) -> Tuple[Optional[str], Optional[str]]:
        """Get the next folder that needs scanning.

        Returns (folder_name, wait_reason) where wait_reason explains why we're waiting.

        Priority order:
        1. PENDING folders (never scanned) - respecting retry delays for failed ones
        2. STALE folders that have been stable for 5+ minutes
        3. ERROR folders that have waited long enough for retry
        """
        current_time = time.time()
        wait_reason = None
        earliest_ready = None

        with self._lock:
            # First priority: unscanned folders (not being scanned by user)
            for folder_name, state in self._folder_states.items():
                if state.status == ScanStatus.PENDING:
                    # Skip if user is currently scanning this folder
                    if folder_name in self._user_scanning_folders:
                        continue

                    # Check if this folder has failed before and needs to wait
                    if state.consecutive_failures > 0 and state.last_failure_time > 0:
                        retry_delay = self.BASE_RETRY_DELAY_SECONDS * (2 ** (state.consecutive_failures - 1))
                        time_since_failure = current_time - state.last_failure_time

                        if time_since_failure < retry_delay:
                            # Still waiting for retry delay
                            wait_time = retry_delay - time_since_failure
                            if earliest_ready is None or wait_time < earliest_ready:
                                earliest_ready = wait_time
                                wait_reason = f"Retry in {int(wait_time)}s for {folder_name}"
                            continue

                    return folder_name, None

            # Second priority: stale folders that have been stable
            for folder_name, state in self._folder_states.items():
                if state.status == ScanStatus.STALE:
                    # Skip if user is currently scanning this folder
                    if folder_name in self._user_scanning_folders:
                        continue

                    time_since_change = current_time - state.last_change_detected
                    time_since_scan = current_time - state.last_scan_time

                    # Must be stable for 5 minutes AND not scanned recently
                    if time_since_change >= self.STABILITY_WAIT_SECONDS:
                        if time_since_scan >= self.MIN_RESCAN_INTERVAL_SECONDS:
                            return folder_name, None
                        else:
                            # Folder is stable but was scanned too recently
                            ready_in = self.MIN_RESCAN_INTERVAL_SECONDS - time_since_scan
                            if earliest_ready is None or ready_in < earliest_ready:
                                earliest_ready = ready_in
                                wait_reason = f"Waiting {int(ready_in)}s before rescanning"
                    else:
                        # Folder changed recently, waiting for stability
                        stable_in = self.STABILITY_WAIT_SECONDS - time_since_change
                        if earliest_ready is None or stable_in < earliest_ready:
                            earliest_ready = stable_in
                            wait_reason = f"Folder changed, waiting {int(stable_in)}s for stability"

            # Third priority: ERROR folders that have waited long enough
            for folder_name, state in self._folder_states.items():
                if state.status == ScanStatus.ERROR:
                    # Skip if user is currently scanning this folder
                    if folder_name in self._user_scanning_folders:
                        continue

                    # Calculate retry delay with exponential backoff
                    # After max failures, use a long delay (1 hour)
                    retry_delay = min(
                        self.BASE_RETRY_DELAY_SECONDS * (2 ** state.consecutive_failures),
                        3600  # Max 1 hour between retries
                    )
                    time_since_failure = current_time - state.last_failure_time

                    if time_since_failure >= retry_delay:
                        # Reset to pending and try again
                        scanner_log('info', f"Retrying previously failed folder: {folder_name} (failed {state.consecutive_failures} times)")
                        return folder_name, None
                    else:
                        wait_time = retry_delay - time_since_failure
                        if earliest_ready is None or wait_time < earliest_ready:
                            earliest_ready = wait_time
                            wait_reason = f"Retry failed folder in {int(wait_time)}s"

        return None, wait_reason

    def _scan_folder(self, folder_name: str):
        """Scan a folder for duplicates with timeout protection."""
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

        self._set_scanner_state(ScannerState.SCANNING, f"Scanning {folder_name}")
        scanner_log('info', f"Background scan STARTED: {folder_name} (path: {folder_path})")

        # Use a container to store results from the worker thread
        scan_result = {'success': False, 'images': None, 'videos': None, 'error': None, 'traceback': None}
        scan_complete = threading.Event()

        def run_scan():
            """Worker function to run the actual scan."""
            try:
                scanner_log('debug', f"Worker thread started for: {folder_name}")

                # Create progress callback that updates state
                def progress_callback(status: str, current: int, total: int, message: str):
                    with self._lock:
                        folder_state = self._folder_states.get(folder_name)
                        if folder_state:
                            folder_state.scan_progress = current
                            folder_state.scan_total = total
                            folder_state.scan_message = message

                        self._state_message = f"{folder_name}: {message}"

                    # Log progress periodically
                    if current > 0 and current % 100 == 0:
                        scanner_log('debug', f"Progress [{folder_name}]: {status} - {current}/{total} - {message}")

                        # Forward to registered callback if any
                        callback = self._progress_callbacks.get(folder_name)
                        if callback:
                            try:
                                callback(status, current, total, message)
                            except Exception as e:
                                logger.warning(f"Error in progress callback: {e}")

                # Run the duplicate finder
                finder = DuplicateFinder(self.image_extensions, self.video_extensions)
                duplicate_images, duplicate_videos = finder.find_duplicates(
                    folder_path,
                    progress_callback=progress_callback
                )

                scan_result['success'] = True
                scan_result['images'] = duplicate_images
                scan_result['videos'] = duplicate_videos
                scanner_log('debug', f"Worker thread completed successfully for: {folder_name}")

            except Exception as e:
                scan_result['error'] = str(e)
                scan_result['traceback'] = traceback.format_exc()
                scanner_log('error', f"EXCEPTION in scan worker for {folder_name}: {e}\n{traceback.format_exc()}")

            finally:
                scan_complete.set()

        # Start the scan in a worker thread
        worker = threading.Thread(target=run_scan, name=f"ScanWorker-{folder_name}", daemon=True)
        worker.start()

        # Wait for completion with timeout
        completed = scan_complete.wait(timeout=self.SCAN_TIMEOUT_SECONDS)

        if not completed:
            # Scan timed out
            scanner_log('error', f"TIMEOUT: Background scan for {folder_name} exceeded {self.SCAN_TIMEOUT_SECONDS}s")
            self._handle_scan_failure(folder_name, "Scan timed out - folder may have too many files or be inaccessible")
            return

        # Process results
        if scan_result['success']:
            duplicate_images = scan_result['images']
            duplicate_videos = scan_result['videos']

            with self._lock:
                state = self._folder_states.get(folder_name)
                if state:
                    state.status = ScanStatus.COMPLETE
                    state.last_scan_time = time.time()
                    state.error_message = ""
                    state.scan_message = "Complete"
                    state.consecutive_failures = 0  # Reset failure count on success
                    # Count total files for stats
                    image_count = sum(len(g.get('duplicate_files', [])) + 1 for g in (duplicate_images or []))
                    video_count = sum(len(g.get('duplicate_files', [])) + 1 for g in (duplicate_videos or []))
                    state.file_count = image_count + video_count

            scanner_log('info', f"Background scan COMPLETE: {folder_name} "
                       f"({len(duplicate_images or [])} image groups, "
                       f"{len(duplicate_videos or [])} video groups)")
        else:
            error_msg = scan_result['error'] or "Unknown error"
            tb = scan_result.get('traceback', '')
            scanner_log('error', f"Background scan FAILED: {folder_name} - {error_msg}")
            if tb:
                scanner_log('debug', f"Traceback for {folder_name}:\n{tb}")
            self._handle_scan_failure(folder_name, error_msg)

        with self._lock:
            self._current_scan_folder = None

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
_background_scanner: Optional[BackgroundScanner] = None


def get_background_scanner() -> Optional[BackgroundScanner]:
    """Get the global background scanner instance."""
    return _background_scanner


def init_background_scanner(
    data_dir: str,
    image_extensions: Set[str],
    video_extensions: Set[str]
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
