from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import hashlib

@dataclass(frozen=True)
class KarmaEntry:
    entry_id: str
    timestamp: str
    observation_type: str
    payload: Dict[str, Any]
    prev_hash: str
    
    @property
    def hash(self) -> str:
        """Compute SHA256 hash of this entry for integrity."""
        data = f"{self.entry_id}|{self.timestamp}|{self.observation_type}|{json.dumps(self.payload, sort_keys=True)}|{self.prev_hash}"
        return hashlib.sha256(data.encode()).hexdigest()

class KarmaLog:
    """
    Append-Only Immutable Log ("Karma").
    Stores critical events with cryptographic linking (simple blockchain-like structure).
    No updates, no deletes.
    """
    def __init__(self):
        self._log: List[KarmaEntry] = []
        self._genesis_hash = "0" * 64

    def append(self, observation_type: str, payload: Dict[str, Any]) -> str:
        """
        Appends a new entry to the log.
        Returns the entry hash.
        """
        prev_hash = self._log[-1].hash if self._log else self._genesis_hash
        timestamp = datetime.now().isoformat()
        entry_id = f"K{len(self._log):09d}"
        
        entry = KarmaEntry(
            entry_id=entry_id,
            timestamp=timestamp,
            observation_type=observation_type,
            payload=payload,
            prev_hash=prev_hash
        )
        
        self._log.append(entry)
        return entry.hash

    def get_log(self) -> List[Dict[str, Any]]:
        """Returns a copy of the log as dicts, including computed hash."""
        result = []
        for e in self._log:
            d = asdict(e)
            d['hash'] = e.hash
            result.append(d)
        return result

    def verify_integrity(self) -> bool:
        """Verifies the hash chain."""
        if not self._log:
            return True
            
        params_hash = self._genesis_hash
        for entry in self._log:
            if entry.prev_hash != params_hash:
                return False
            params_hash = entry.hash
        return True
