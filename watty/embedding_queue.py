"""
Watty Async Embedding Pipeline
Stores text immediately, embeds in a background thread.
Single operations embed inline. Bulk operations enqueue.
"""

import threading
import sqlite3
from collections import deque

import numpy as np

from watty.embeddings_loader import embed_text
from watty.config import EMBEDDING_DIMENSION


class EmbeddingQueue:
    def __init__(self, db_path: str, batch_threshold: int = 5):
        self.db_path = db_path
        self.batch_threshold = batch_threshold
        self._queue = deque()
        self._lock = threading.Lock()
        self._worker = None
        self._running = False

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._queue)

    def enqueue(self, chunk_id: int, text: str):
        with self._lock:
            self._queue.append((chunk_id, text))
            if len(self._queue) >= self.batch_threshold and not self._running:
                self._start_worker()

    def embed_inline(self, text: str) -> np.ndarray:
        return embed_text(text)

    def flush(self):
        """Process all pending embeddings synchronously."""
        while True:
            with self._lock:
                if not self._queue:
                    return
                batch = []
                while self._queue and len(batch) < 50:
                    batch.append(self._queue.popleft())
            self._process_batch(batch)

    def _start_worker(self):
        self._running = True
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self):
        try:
            while True:
                with self._lock:
                    if not self._queue:
                        self._running = False
                        return
                    batch = []
                    while self._queue and len(batch) < 50:
                        batch.append(self._queue.popleft())
                self._process_batch(batch)
        except Exception:
            self._running = False

    def _process_batch(self, batch: list):
        from watty.crypto import connect as crypto_connect
        conn = crypto_connect(self.db_path)
        for chunk_id, text in batch:
            try:
                vec = embed_text(text)
                conn.execute("UPDATE chunks SET embedding = ? WHERE id = ?", (vec.tobytes(), chunk_id))
            except Exception:
                pass
        conn.commit()
        conn.close()
