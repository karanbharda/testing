import os
import json
from datetime import datetime
from typing import Any, Dict
from threading import Lock

class KarmaLogger:
    """
    Append-only Audit Trail (Karma).
    Records observations without authority.
    """
    def __init__(self, output_dir: str = "karma_logs"):
        self.output_dir = output_dir
        self.lock = Lock()
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Current Log Files
        self.current_minute = datetime.now().minute
        self.files = {} # type: Dict[str, Any]

    def _get_file_handle(self, bucket: str):
        """
        Rotates file handles every minute (simple bucketing).
        """
        now = datetime.now()
        if now.minute != self.current_minute:
            # Close old files
            for f in self.files.values():
                f.close()
            self.files = {}
            self.current_minute = now.minute
            
        if bucket not in self.files:
            filename = f"{self.output_dir}/karma_{bucket}_{now.strftime('%Y%m%d_%H%M')}.jsonl"
            self.files[bucket] = open(filename, "a", encoding="utf-8")
            
        return self.files[bucket]

    def log(self, bucket: str, data: Dict[str, Any]):
        """
        Logs an event to the specified bucket.
        """
        with self.lock:
            try:
                f = self._get_file_handle(bucket)
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "data": data
                }
                f.write(json.dumps(entry) + "\n")
                f.flush()
            except Exception as e:
                print(f"KARMA_ERROR: Failed to log to {bucket}: {e}")

    def log_tick(self, tick_data: Dict[str, Any]):
        self.log("ticks", tick_data)

    def log_trade(self, trade_data: Dict[str, Any]):
        self.log("trades", trade_data)

    def log_limit_check(self, check_data: Dict[str, Any]):
        self.log("risk_checks", check_data)
        
    def close(self):
        with self.lock:
            for f in self.files.values():
                f.close()
            self.files = {}
