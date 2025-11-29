import os
import json
import time
import psutil
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import threading
from typing import Dict, List
import traceback

from util.logger import get_logger
import config

# Get logger
logger = get_logger(__name__)

# List of UUIDs to ignore during cleanup
ignore_uuid_list = ["d8fc4f61-4624-4142-99d9-c24c6dff0b5b"]

class MonitorThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True  # Make thread daemon so it exits when main program exits
        self.running = True
        # Create cleanup directory if it doesn't exist
        cleanup_dir = Path('./cleanup')
        cleanup_dir.mkdir(exist_ok=True)
        self.cleanup_record_path = cleanup_dir / "cleanup_record.json"
        self.cleanup_record: Dict[str, List[Dict]] = self._load_cleanup_record()

        if not self.cleanup_record_path.exists():
            self.cleanup_record = {"deleted_files": []}
            self._save_cleanup_record()

    def _load_cleanup_record(self) -> Dict[str, List[Dict]]:
        """Load existing cleanup record or create new one."""
        if self.cleanup_record_path.exists():
            try:
                with open(self.cleanup_record_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {"deleted_files": []}
        return {"deleted_files": []}

    def _save_cleanup_record(self):
        """Save cleanup record to file."""
        try:
            self.cleanup_record_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cleanup_record_path, 'w', encoding='utf-8') as f:
                json.dump(self.cleanup_record, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving cleanup record: {e}")

    def _record_deletion(self, uuid: str, files: List[Path], reason: str):
        """Record deleted files in the cleanup record."""
        deletion_record = {
            "uuid": uuid,
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "files": [str(f) for f in files]
        }

        if "deleted_files" not in self.cleanup_record:
            self.cleanup_record["deleted_files"] = []    
        self.cleanup_record["deleted_files"].append(deletion_record)
        self._save_cleanup_record()

    def _is_already_deleted(self, file_path: Path) -> bool:
        """Check if a file has already been recorded as deleted in the cleanup_record."""
        str_path = str(file_path)
        for deletion_record in self.cleanup_record.get("deleted_files", []):
            if str_path in deletion_record.get("files", []):
                return True
        return False

    def cleanup_old_files(self):
        """Check and cleanup files older than 7 days."""
        try:
            # 음성 파일 확장자 목록 정의
            AUDIO_EXTENSIONS = ['.wav', '.mp3', '.ogg', '.m4a', '.flac']
            
            # Check uploads directory for analysis work files
            uploads_dir = Path(config.UPLOADS_DIR)
            if not uploads_dir.exists():
                return
                
            total_deleted_files = []
            
            for uuid_dir in uploads_dir.iterdir():
                if not uuid_dir.is_dir():
                    continue
                
                # Skip ignored UUIDs
                if uuid_dir.name in ignore_uuid_list:
                    logger.info(f"Ignoring UUID {uuid_dir.name} during cleanup (in ignore_uuid_list)")
                    continue

                analysis_work_path = uuid_dir / f"id[{uuid_dir.name}]_analysiswork.json"
                if not analysis_work_path.exists():
                    continue

                try:
                    logger.debug(f"Checking {analysis_work_path}")
                    text_in_file = ""
                    with open(analysis_work_path, 'r', encoding='utf-8') as f:
                        text_in_file = f.read()
                    if text_in_file == "":
                        logger.debug(f"Skipping {analysis_work_path} because it is empty")
                        continue

                    analysis_work = json.loads(text_in_file)
                    
                    created_at = datetime.fromisoformat(analysis_work['created_at'])
                    if datetime.now() - created_at > timedelta(days=7):
                        
                        # Initialize files_to_delete for this UUID directory
                        files_to_delete = []
                        skipped_files = []
                        
                        # Files in uploads directory - 음성 파일만 삭제
                        for file in uuid_dir.glob(f"id[{uuid_dir.name}]*"):
                            # 파일 확장자 확인
                            if file.suffix.lower() in AUDIO_EXTENSIONS:
                                if not self._is_already_deleted(file):
                                    files_to_delete.append(file)
                                else:
                                    skipped_files.append(file)
                        
                        # Files in temp directory
                        temp_dirs = [
                            config.TEMP_DIR / "uploading" / uuid_dir.name,
                            config.TEMP_DIR / "splits" / uuid_dir.name,
                            config.TEMP_DIR / "transcripts" / uuid_dir.name
                        ]
                        
                        for temp_dir in temp_dirs:
                            if temp_dir.exists():
                                for file in temp_dir.glob(f"id[{uuid_dir.name}]*"):
                                    if not self._is_already_deleted(file):
                                        files_to_delete.append(file)
                                    else:
                                        skipped_files.append(file)
                                        
                                if not self._is_already_deleted(temp_dir):
                                    files_to_delete.append(temp_dir)
                                else:
                                    skipped_files.append(temp_dir)

                        if not files_to_delete:
                            if skipped_files:
                                logger.debug(f"All files for UUID {uuid_dir.name} already deleted previously. Skipped {len(skipped_files)} files.")
                            continue
                            
                        # Delete all collected files
                        for file in files_to_delete:
                            try:
                                if file.is_file():
                                    file.unlink()
                                elif file.is_dir():
                                    shutil.rmtree(file)
                            except Exception as e:
                                logger.error(f"Error deleting {file}: {e}")

                        # Record the deletion
                        self._record_deletion(
                            uuid_dir.name,
                            files_to_delete,
                            f"Audio files older than 7 days (created at {created_at})"
                        )
                        
                        logger.info(f"Cleaned up old files for UUID: {uuid_dir.name}, {len(files_to_delete)} files deleted, {len(skipped_files)} files already deleted previously")
                        total_deleted_files.extend(files_to_delete)

                except Exception as e:
                    # logger.error(f"Error processing : {e}")
                    logger.error(f"Error in cleanup processing the path {analysis_work_path} - {e}\n{traceback.format_exc()}")

            if len(total_deleted_files) > 0:
                logger.info(f"Total cleanup: {len(total_deleted_files)} files across {len(set(f.parent for f in total_deleted_files if hasattr(f, 'parent')))} directories")
            else:
                logger.info("No old files to clean up")

        except Exception as e:
            logger.error(f"Error in cleanup_old_files: {e}\n{traceback.format_exc()}")

    def report_usage(self):
        """Report memory and disk usage."""

        MB, GB = 1024**2, 1024**3
        try:
            process = psutil.Process()
            

            # Memory usage (system-wide and process-specific)
            system_memory = psutil.virtual_memory()
            process_memory = process.memory_info()
            
            # CPU usage
            cpu_percent = process.cpu_percent(interval=1)
            
            # Disk usage for temp and uploads directories
            temp_usage = None
            if Path(config.TEMP_DIR).exists():
                temp_usage = shutil.disk_usage(config.TEMP_DIR)
                uploads_usage = shutil.disk_usage(config.UPLOADS_DIR)
            
            if temp_usage is not None:
                print(
                    f"Resource Usage: "
                    f"Memory (Process): {process_memory.rss / MB:.1f}MB ({process.memory_percent():.1f}%) , Memory (System): {system_memory.used / GB:.1f}GB used / {system_memory.total / GB:.1f}GB total ({system_memory.percent}%) , "
                    f"CPU: {cpu_percent:.1f}% , "
                    f"Disk: {temp_usage.used / GB:.1f}GB used, {temp_usage.free / GB:.1f}GB free of {temp_usage.total / GB:.1f}GB"
                )
        
        except Exception as e:
            logger.error(f"Error in report_usage: {e}\n{traceback.format_exc()}")

    def run(self):
        """Main thread loop."""
        while self.running:
            try:
                # Report usage every minute
                self.report_usage()
                
                # Check for old files every minute
                self.cleanup_old_files()
                
                # Wait for 10 minute
                time.sleep(600)
                
            except Exception as e:
                logger.error(f"Error in monitor thread: {e}\n{traceback.format_exc()}")
                time.sleep(600)  # Wait before retrying

    def stop(self):
        """Stop the monitor thread."""
        self.running = False 