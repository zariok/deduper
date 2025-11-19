import os
import urllib.parse
from pathlib import Path
from flask import Blueprint, render_template, jsonify, request, send_from_directory, abort, url_for, Response
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException
import json
import threading
import time
from typing import Dict, Any, Optional
from ..services.duplicate_finder import DuplicateFinder
from ..config import Config
from ..utils.setup import create_example_folder
from ..utils.media import extract_video_thumbnail
from ..utils.hash_cache import HashCache
from ..utils.logging_config import get_logger
from ..utils.metrics import metrics, timer, increment_counter, set_gauge

logger = get_logger(__name__)

bp = Blueprint('main', __name__)

# Global progress tracking
progress_data = {}
progress_lock = threading.Lock()

def update_progress(session_id: str, status: str, current: int = 0, total: int = 0, message: str = "") -> None:
    """Update progress for a specific session."""
    logger.debug(f"update_progress called - session_id: {session_id}, status: {status}, current: {current}, total: {total}, message: {message}")
    with progress_lock:
        progress_data[session_id] = {
            'status': status,  # 'scanning', 'processing', 'grouping', 'complete', 'error'
            'current': current,
            'total': total,
            'message': message,
            'timestamp': time.time()
        }

def get_progress(session_id: str) -> Dict[str, Any]:
    """Get current progress for a session."""
    with progress_lock:
        progress = progress_data.get(session_id, {
            'status': 'idle',
            'current': 0,
            'total': 0,
            'message': '',
            'timestamp': time.time()
        })
        logger.debug(f"get_progress called for {session_id}, returning: {progress}")
        return progress

@bp.route('/')
@timer('page_index')
def index():
    """Render the main page with user folders."""
    try:
        logger.info("Rendering main page")
        increment_counter('page_requests', tags={'page': 'index'})
        
        user_folders = [d for d in os.listdir(Config.DATA_DIR) 
                       if os.path.isdir(os.path.join(Config.DATA_DIR, d))]
        user_folders.sort(key=str.lower)
        
        # Create example folder if no folders exist
        if not user_folders:
            logger.info("No user folders found, creating example folder")
            example_folder = create_example_folder()
            user_folders = [os.path.basename(example_folder)]
            
        logger.info(f"Found {len(user_folders)} user folders")
        set_gauge('user_folders_count', len(user_folders))
        return render_template('index.html', user_folders=user_folders)
    except Exception as e:
        logger.error(f"Error rendering main page: {e}", exc_info=True)
        increment_counter('page_errors', tags={'page': 'index'})
        return render_template('index.html', user_folders=[], error=str(e))

@bp.route('/progress/<session_id>')
def progress_stream(session_id):
    """Stream progress updates via Server-Sent Events."""
    def generate():
        while True:
            progress = get_progress(session_id)
            if progress['status'] in ['complete', 'error']:
                # Send final update and close
                yield f"data: {json.dumps(progress)}\n\n"
                break
            else:
                yield f"data: {json.dumps(progress)}\n\n"
            time.sleep(0.5)  # Update every 500ms
    
    return Response(generate(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*'
    })

@bp.route('/scan/<path:user_folder>')
@timer('scan_folder')
def scan_folder(user_folder: str):
    """Scan a user folder for duplicates."""
    # URL decode the folder name to handle special characters
    decoded_folder = urllib.parse.unquote(user_folder)
    
    # Get session ID from request headers or generate one
    session_id = request.headers.get('X-Session-ID', f"{decoded_folder}_{int(time.time())}")
    
    # Debug logging
    logger.debug(f"Received folder name: {user_folder}")
    logger.debug(f"Decoded folder name: {decoded_folder}")
    logger.debug(f"Data directory: {Config.DATA_DIR}")
    
    # Security check: ensure the folder name doesn't contain path traversal attempts
    if '..' in decoded_folder or '/' in decoded_folder or '\\' in decoded_folder:
        logger.warning(f"Invalid folder name detected: {decoded_folder}")
        increment_counter('scan_errors', tags={'error': 'invalid_folder_name'})
        return jsonify({'error': 'Invalid folder name'}), 400
    
    folder_path = os.path.join(Config.DATA_DIR, decoded_folder)
    logger.debug(f"Full folder path: {folder_path}")
    logger.debug(f"Folder exists: {os.path.exists(folder_path)}")
    
    if not os.path.exists(folder_path):
        logger.warning(f"Folder not found: {decoded_folder}")
        increment_counter('scan_errors', tags={'error': 'folder_not_found'})
        return jsonify({'error': f'Folder not found: {decoded_folder}'}), 404
        
    try:
        # Start progress tracking
        increment_counter('scan_requests', tags={'folder': decoded_folder})
        update_progress(session_id, 'scanning', 0, 0, 'Starting scan...')
        
        finder = DuplicateFinder(Config.IMAGE_EXTENSIONS, Config.VIDEO_EXTENSIONS)
        duplicate_images, duplicate_videos = finder.find_duplicates(folder_path, progress_callback=lambda status, current, total, message: update_progress(session_id, status, current, total, message))
        
        # Ensure we have valid lists
        if duplicate_images is None:
            duplicate_images = []
        if duplicate_videos is None:
            duplicate_videos = []
        
        logger.info(f"Found {len(duplicate_images)} image groups and {len(duplicate_videos)} video groups")
        
        # Mark as complete
        update_progress(session_id, 'complete', 100, 100, f'Found {len(duplicate_images)} image groups and {len(duplicate_videos)} video groups')
        
        # Convert file paths to URLs
        def convert_to_urls(duplicates):
            for group in duplicates:
                # Get the full file URL for best file (path is now relative)
                best_file_path = group['best_file']['path']
                full_file = url_for('main.serve_file', 
                    filename=os.path.join(decoded_folder, best_file_path))
                
                # Get the thumbnail URL for best file
                thumb_file = url_for('main.serve_thumbnail', 
                    filename=os.path.join(decoded_folder, best_file_path))
                
                group['best_file'].update({
                    'full': full_file,
                    'thumb': thumb_file,
                    'original_path': group['best_file']['path']  # Keep relative path
                })
                
                # Convert duplicate files
                for duplicate in group['duplicate_files']:
                    duplicate_path = duplicate['path']
                    duplicate.update({
                        'full': url_for('main.serve_file', 
                            filename=os.path.join(decoded_folder, duplicate_path)),
                        'thumb': url_for('main.serve_thumbnail', 
                            filename=os.path.join(decoded_folder, duplicate_path)),
                        'original_path': duplicate_path  # Keep relative path
                    })
            return duplicates
        
        return jsonify({
            'duplicate_images': convert_to_urls(duplicate_images),
            'duplicate_videos': convert_to_urls(duplicate_videos),
            'session_id': session_id
        })
    except Exception as e:
        update_progress(session_id, 'error', 0, 0, f'Error: {str(e)}')
        return jsonify({'error': str(e)}), 500

@bp.route('/manage-duplicate', methods=['POST'])
def manage_duplicate():
    """Handle duplicate file management."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        action = data.get('action')
        file_path = data.get('file_path')
        
        if not action:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # file_path is only required for certain actions
        if action in ['delete', 'set_best'] and not file_path:
            return jsonify({'error': 'Missing required parameters'}), 400
            
        if action == 'delete':
            best_file_path = data.get('best_file_path')
            folder = data.get('folder')
            if not best_file_path:
                return jsonify({'error': 'Best file path required for symlink creation'}), 400
            if not folder:
                return jsonify({'error': 'Folder required for file operations'}), 400
            
            # Convert relative paths to absolute paths for file operations
            folder_path = os.path.join(Config.DATA_DIR, folder)
            full_file_path = os.path.join(folder_path, file_path)
            full_best_file_path = os.path.join(folder_path, best_file_path)
            
            # Debug logging
            logger.debug(f"Attempting to delete file: {full_file_path}")
            logger.debug(f"File exists check: {os.path.exists(full_file_path)}")
            logger.debug(f"Best file path: {full_best_file_path}")
            logger.debug(f"Best file exists: {os.path.exists(full_best_file_path)}")
            
            # Calculate relative path for symlink
            if os.path.exists(full_file_path):
                duplicate_dir = os.path.dirname(full_file_path)
                best_file_dir = os.path.dirname(full_best_file_path)
                best_file_name = os.path.basename(full_best_file_path)
                
                if duplicate_dir == best_file_dir:
                    relative_path = best_file_name
                else:
                    try:
                        relative_path = os.path.relpath(full_best_file_path, duplicate_dir)
                    except ValueError:
                        relative_path = full_best_file_path
                
                logger.debug(f"Will create symlink with relative path: {relative_path}")
                
            if os.path.exists(full_file_path):
                # Remove the original duplicate file
                os.remove(full_file_path)
                
                # Create relative symlink to the best file
                try:
                    # Calculate relative path from duplicate to best file
                    duplicate_dir = os.path.dirname(full_file_path)
                    best_file_dir = os.path.dirname(full_best_file_path)
                    best_file_name = os.path.basename(full_best_file_path)
                    
                    # If they're in the same directory, just use the filename
                    if duplicate_dir == best_file_dir:
                        relative_path = best_file_name
                    else:
                        # Calculate relative path between directories
                        try:
                            relative_path = os.path.relpath(full_best_file_path, duplicate_dir)
                        except ValueError:
                            # If we can't calculate relative path (different drives on Windows), use absolute
                            relative_path = full_best_file_path
                    
                    os.symlink(relative_path, full_file_path)
                except OSError as e:
                    return jsonify({'error': f'Failed to create symlink: {str(e)}'}), 500
                
                # Also remove deduper thumbnail if it exists (we'll let the system regenerate it if needed)
                directory = os.path.dirname(full_file_path)
                basename = os.path.basename(full_file_path)
                basename_stem = Path(basename).stem
                deduper_thumb_path = os.path.join(directory, f"thumb-deduper.{basename_stem}.jpg")
                if os.path.exists(deduper_thumb_path):
                    os.remove(deduper_thumb_path)
                
                # Update cache to reflect the file deletion and symlink creation
                try:
                    from ..utils.hash_cache import HashCache
                    cache = HashCache(folder_path)
                    
                    # Remove the deleted file from cache hashes and file_stats
                    cache.update_file_stats(full_file_path)
                    
                    # Remove the deleted file from cached groups
                    cache.remove_file_from_groups(full_file_path)
                    
                    logger.debug(f"Updated cache after deleting {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to update cache after deletion: {e}")
                    
                return jsonify({'message': 'File replaced with symlink to best file successfully'})
            return jsonify({'error': f'File not found: {file_path}'}), 404
        elif action == 'set_best':
            best_file_path = data.get('best_file_path')
            group_id = data.get('group_id')
            folder = data.get('folder')
            if not best_file_path:
                return jsonify({'error': 'Best file path required'}), 400
            if not folder:
                return jsonify({'error': 'Folder required for file operations'}), 400
            
            # Convert relative paths to absolute paths for validation
            folder_path = os.path.join(Config.DATA_DIR, folder)
            full_file_path = os.path.join(folder_path, file_path)
            full_best_file_path = os.path.join(folder_path, best_file_path)
            
            logger.debug(f"Setting new best file: {full_file_path}")
            logger.debug(f"Previous best file: {full_best_file_path}")
            logger.debug(f"Group ID: {group_id}")
            
            # Validate that both files exist
            if not os.path.exists(full_file_path):
                return jsonify({'error': f'New best file not found: {full_file_path}'}), 404
            if not os.path.exists(full_best_file_path):
                return jsonify({'error': f'Previous best file not found: {full_best_file_path}'}), 404
            
            try:
                # Instead of physically swapping files, just update the cache to remember
                # which file should be considered the "best" for this group
                from ..utils.hash_cache import HashCache
                cache = HashCache(folder_path)
                
                # Store the new best file selection in cache (using absolute path for cache)
                if group_id:
                    cache.set_best_file(group_id, full_file_path)  # full_file_path is the new best file
                
                cache.save()
                
                logger.debug(f"Successfully updated best file selection in cache")
                logger.debug(f"New best file is now: {file_path}")
                logger.debug(f"Previous best file is now: {best_file_path}")
                logger.debug(f"No files were physically moved - only cache updated")
                
                return jsonify({'message': 'Best file selection updated successfully'})
                
            except Exception as e:
                logger.error(f"Error updating best file selection: {e}")
                return jsonify({'error': f'Failed to update best file selection: {str(e)}'}), 500
        elif action == 'process_all_duplicates':
            best_file_path = data.get('best_file_path')
            group_id = data.get('group_id')
            folder = data.get('folder')
            if not best_file_path or not group_id or not folder:
                return jsonify({'error': 'Missing required parameters: best_file_path, group_id, and folder are required'}), 400
            
            logger.debug(f"Processing all duplicates for group {group_id}")
            logger.debug(f"Best file (relative): {best_file_path}")
            logger.debug(f"Folder: {folder}")
            
            try:
                from ..utils.hash_cache import HashCache
                
                # Construct the full folder path
                folder_path = os.path.join(Config.DATA_DIR, folder)
                logger.debug(f"Full folder path: {folder_path}")
                
                # Validate that the folder exists
                if not os.path.exists(folder_path):
                    return jsonify({'error': f'Folder not found: {folder}'}), 404
                
                # Construct the full path to the best file
                full_best_file_path = os.path.join(folder_path, best_file_path)
                logger.debug(f"Full best file path: {full_best_file_path}")
                
                # Validate that the best file exists
                if not os.path.exists(full_best_file_path):
                    return jsonify({'error': f'Best file not found: {full_best_file_path}'}), 404
            
                # Get all files in this group
                cache = HashCache(folder_path)
                group_files = cache.get_group_files(group_id)
                if not group_files:
                    # Additional debugging information
                    logger.debug(f"No files found for group {group_id}")
                    logger.debug(f"Available groups in cache: {list(cache.cache_data.get('groups', {}).keys())}")
                    logger.debug(f"Available grouping_results: {list(cache.cache_data.get('grouping_results', {}).keys())}")

                    # Try to refresh the cache and search again
                    logger.debug(f"Attempting to refresh cache and search again...")
                    cache = HashCache(folder_path)  # Reload cache
                    group_files = cache.get_group_files(group_id)
                    
                    if not group_files:
                        logger.debug(f"Still no files found after cache refresh")
                        return jsonify({'error': f'No files found for group {group_id}'}), 404
                    else:
                        logger.debug(f"Found {len(group_files)} files after cache refresh")
 
                    return jsonify({'error': f'No files found for group {group_id}'}), 404
                
                logger.debug(f"Found {len(group_files)} files in group {group_id}")
                
                # Find and update any existing symlinks that point to the old best file
                from ..utils.helpers import find_symlinks_pointing_to
                old_symlinks = []
                for file_path in group_files:
                    if file_path != full_best_file_path and os.path.islink(file_path):
                        try:
                            link_target = os.readlink(file_path)
                            if not os.path.isabs(link_target):
                                link_target = os.path.join(os.path.dirname(file_path), link_target)
                            link_target = os.path.abspath(link_target)
                            
                            # If this symlink points to another file in the group (old best), we need to update it
                            if link_target in group_files and link_target != full_best_file_path:
                                old_symlinks.append(file_path)
                        except OSError:
                            continue
                
                logger.debug(f"Found {len(old_symlinks)} existing symlinks to update")
                
                processed_count = 0
                errors = []
                
                # Update existing symlinks to point to the new best file
                for symlink_path in old_symlinks:
                    try:
                        # Remove the old symlink
                        os.remove(symlink_path)
                        
                        # Create new symlink pointing to the new best file
                        symlink_dir = os.path.dirname(symlink_path)
                        best_file_dir = os.path.dirname(full_best_file_path)
                        best_file_name = os.path.basename(full_best_file_path)
                        
                        # Calculate relative path from symlink to best file
                        if symlink_dir == best_file_dir:
                            relative_path = best_file_name
                        else:
                            try:
                                relative_path = os.path.relpath(full_best_file_path, symlink_dir)
                            except ValueError:
                                # If we can't calculate relative path (different drives on Windows), use absolute
                                relative_path = full_best_file_path
                        
                        os.symlink(relative_path, symlink_path)
                        logger.debug(f"Updated symlink {symlink_path} to point to {full_best_file_path}")
                        
                    except OSError as e:
                        error_msg = f'Failed to update symlink {symlink_path}: {str(e)}'
                        logger.error(error_msg)
                        errors.append(error_msg)
                
                # Process each file in the group (except the best file)
                for file_path in group_files:
                    if file_path == full_best_file_path:
                        continue  # Skip the best file itself
                    
                    if not os.path.exists(file_path):
                        logger.debug(f"File not found, skipping: {file_path}")
                        continue
                    
                    try:
                        # Remove the original duplicate file
                        os.remove(file_path)
                        
                        # Create relative symlink to the best file
                        duplicate_dir = os.path.dirname(file_path)
                        best_file_dir = os.path.dirname(full_best_file_path)
                        best_file_name = os.path.basename(full_best_file_path)
                        
                        # If they're in the same directory, just use the filename
                        if duplicate_dir == best_file_dir:
                            relative_path = best_file_name
                        else:
                            # Calculate relative path between directories
                            try:
                                relative_path = os.path.relpath(full_best_file_path, duplicate_dir)
                            except ValueError:
                                # If we can't calculate relative path (different drives on Windows), use absolute
                                relative_path = full_best_file_path
                        
                        os.symlink(relative_path, file_path)
                        
                        # Also remove deduper thumbnail if it exists
                        basename = os.path.basename(file_path)
                        basename_stem = Path(basename).stem
                        deduper_thumb_path = os.path.join(duplicate_dir, f"thumb-deduper.{basename_stem}.jpg")
                        if os.path.exists(deduper_thumb_path):
                            os.remove(deduper_thumb_path)
                        
                        # Update cache to reflect the file deletion and symlink creation
                        try:
                            # Remove the deleted file from cache hashes and file_stats
                            cache.update_file_stats(file_path)
                            
                            # Remove the deleted file from cached groups
                            cache.remove_file_from_groups(file_path)
                            
                            logger.debug(f"Updated cache for processed file: {file_path}")
                        except Exception as cache_e:
                            logger.warning(f"Failed to update cache for {file_path}: {cache_e}")
                        
                        processed_count += 1
                        logger.debug(f"Successfully processed: {file_path}")
                        
                    except OSError as e:
                        error_msg = f'Failed to process {file_path}: {str(e)}'
                        logger.error(f"{error_msg}")
                        errors.append(error_msg)
                
                # Update cache to mark this group as processed
                cache.mark_group_processed(group_id)
                cache.save()
                
                result_message = f'Successfully processed {processed_count} duplicate files'
                if errors:
                    result_message += f'. Errors: {len(errors)} files failed to process'
                
                logger.info(f"{result_message}")
                return jsonify({'message': result_message, 'processed_count': processed_count, 'errors': errors})
                
            except Exception as e:
                logger.error(f"Error processing all duplicates: {e}")
                return jsonify({'error': f'Failed to process all duplicates: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Invalid action'}), 400
            
    except Exception as e:
        increment_counter('manage_duplicate_errors')
        return jsonify({'error': str(e)}), 500


    

@bp.route('/data/<path:filename>')
def serve_file(filename):
    """Serve files from the data directory."""
    try:
        return send_from_directory(Config.DATA_DIR, filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@bp.route('/thumb/<path:filename>')
def serve_thumbnail(filename):
    """Serve thumbnails from the data directory."""
    try:
        # For videos, look for the deduper thumbnail with thumb-deduper prefix
        if any(filename.lower().endswith(ext) for ext in Config.VIDEO_EXTENSIONS):
            # Get the directory and filename parts
            directory = os.path.dirname(filename)
            basename = os.path.basename(filename)
            basename_stem = Path(basename).stem
            deduper_thumb_path = os.path.join(Config.DATA_DIR, directory, f"thumb-deduper.{basename_stem}.jpg")
            if os.path.exists(deduper_thumb_path):
                return send_from_directory(Config.DATA_DIR, os.path.join(directory, f"thumb-deduper.{basename_stem}.jpg"))
            else:
                # Try to generate thumbnail on-demand for videos
                from deduper.utils.media import extract_video_thumbnail
                video_path = os.path.join(Config.DATA_DIR, filename)
                if os.path.exists(video_path):
                    thumbnail_path = extract_video_thumbnail(video_path)
                    if thumbnail_path and os.path.exists(thumbnail_path):
                        # Return the generated thumbnail
                        return send_from_directory(Config.DATA_DIR, os.path.relpath(thumbnail_path, Config.DATA_DIR))
                # If thumbnail generation fails, return 404
                return jsonify({'error': 'Thumbnail not found'}), 404
        
        # For images, just serve the file
        return send_from_directory(Config.DATA_DIR, filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

    

@bp.route('/cache/<path:user_folder>/clear', methods=['POST'])
def clear_cache(user_folder):
    """Clear cache for a folder."""
    try:
        decoded_folder = urllib.parse.unquote(user_folder)
        folder_path = os.path.join(Config.DATA_DIR, decoded_folder)
        
        if not os.path.exists(folder_path):
            return jsonify({'error': f'Folder not found: {decoded_folder}'}), 404
        
        cache = HashCache(folder_path)
        cache.invalidate_cache()
        return jsonify({'message': 'Cache cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/cache/<path:user_folder>/live-stats')
def live_cache_stats(user_folder):
    """Get live cache statistics with real-time updates."""
    try:
        decoded_folder = urllib.parse.unquote(user_folder)
        folder_path = os.path.join(Config.DATA_DIR, decoded_folder)
        
        if not os.path.exists(folder_path):
            return jsonify({'error': f'Folder not found: {decoded_folder}'}), 404
        
        cache = HashCache(folder_path)
        stats = cache.get_cache_stats()
        
        # Add additional statistics
        stats['total_groups'] = 0
        stats['duplicate_groups'] = 0
        stats['processed_groups'] = 0
        stats['remaining_duplicates'] = 0
        
        # Count groups from cached grouping results
        if 'grouping_results' in cache.cache_data:
            all_groups = cache.cache_data['grouping_results']
            stats['total_groups'] = len(all_groups)
            
            # Count duplicate groups (more than 1 file) and remaining duplicates
            for group_files in all_groups.values():
                if len(group_files) > 1:
                    stats['duplicate_groups'] += 1
                    # Count non-symlink files as remaining duplicates
                    for rel_path in group_files:
                        abs_path = cache._get_absolute_path(rel_path)
                        if os.path.exists(abs_path) and not os.path.islink(abs_path):
                            stats['remaining_duplicates'] += 1
        
        # Count processed groups
        if 'groups' in cache.cache_data:
            for group_data in cache.cache_data['groups'].values():
                if group_data.get('processed', False):
                    stats['processed_groups'] += 1
        
        # Add timestamp for cache freshness
        stats['timestamp'] = time.time()
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.errorhandler(Exception)
def handle_error(e):
    """Handle all errors."""
    if isinstance(e, HTTPException):
        return jsonify({'error': e.description}), e.code
    return jsonify({'error': str(e)}), 500 