
# """
# Vector Database Manager for Face Embeddings using SQLite
# Provides persistent storage with incremental enrollment support
# Lightweight, no external dependencies (built-in with Python)
# """
# import numpy as np
# from typing import List, Dict, Tuple, Optional
# import pickle
# import sqlite3
# import json
# import os
# import config
# from scipy.spatial.distance import cosine


# class VectorDBManager:
#     """
#     Manages face embeddings with SQLite for persistent storage
#     Supports incremental enrollment by checking existing employees
#     Uses built-in Python SQLite - no external dependencies needed
#     """

#     def __init__(self, persist_directory: str = None):
#         if persist_directory is None:
#             persist_directory = os.path.join(config.BASE_DIR, "vector_db")

#         os.makedirs(persist_directory, exist_ok=True)
#         self.persist_directory = persist_directory

#         # Initialize SQLite database
#         self.db_path = os.path.join(persist_directory, "embeddings.db")

#         # Create initial connection to set up database
#         conn = self._get_connection()
#         cursor = conn.cursor()

#         # Create table for embeddings if not exists
#         cursor.execute('''
#             CREATE TABLE IF NOT EXISTS face_embeddings (
#                 employee_id TEXT PRIMARY KEY,
#                 name TEXT NOT NULL,
#                 embedding BLOB NOT NULL,
#                 metadata TEXT,
#                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#                 updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#             )
#         ''')
#         conn.commit()

#         # Get count of existing embeddings
#         cursor.execute("SELECT COUNT(*) FROM face_embeddings")
#         existing_count = cursor.fetchone()[0]
#         conn.close()

#         print(f"✓ Vector database (SQLite) initialized at: {self.db_path}")
#         print(f"  Embeddings in database: {existing_count}")

#     def _get_connection(self) -> sqlite3.Connection:
#         """
#         Get a thread-safe database connection
#         Creates a new connection each time to avoid threading issues
#         """
#         conn = sqlite3.connect(self.db_path, check_same_thread=False)
#         return conn

#     def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
#         """Convert numpy array to bytes for storage"""
#         return pickle.dumps(embedding)

#     def _deserialize_embedding(self, blob: bytes) -> np.ndarray:
#         """Convert bytes back to numpy array"""
#         return pickle.loads(blob)

#     def employee_exists(self, employee_id: str) -> bool:
#         """Check if an employee already has embeddings in the database"""
#         try:
#             conn = self._get_connection()
#             cursor = conn.cursor()
#             cursor.execute(
#                 "SELECT COUNT(*) FROM face_embeddings WHERE employee_id = ?",
#                 (str(employee_id),)
#             )
#             count = cursor.fetchone()[0]
#             conn.close()
#             return count > 0
#         except Exception as e:
#             print(f"Error checking employee existence: {e}")
#             return False

#     def get_existing_employee_ids(self) -> List[str]:
#         """Get list of all employee IDs that have embeddings"""
#         try:
#             conn = self._get_connection()
#             cursor = conn.cursor()
#             cursor.execute("SELECT employee_id FROM face_embeddings")
#             result = [row[0] for row in cursor.fetchall()]
#             conn.close()
#             return result
#         except Exception as e:
#             print(f"Error getting employee IDs: {e}")
#             return []

#     def add_employee_embedding(self, employee_id: str, name: str,
#                                face_embedding: np.ndarray,
#                                metadata: Dict = None) -> bool:
#         """Add or update employee face embedding to SQLite database"""
#         try:
#             if metadata is None:
#                 metadata = {}

#             metadata['name'] = name
#             metadata['employee_id'] = str(employee_id)
            
#             # CRITICAL: Ensure embedding is normalized before storing
#             # This guarantees consistent similarity calculations
#             embedding_norm = np.linalg.norm(face_embedding)
#             if embedding_norm > 0:
#                 face_embedding_normalized = face_embedding / embedding_norm
#             else:
#                 print(f"Error: Cannot normalize zero-norm embedding for {employee_id}")
#                 return False

#             # Serialize normalized embedding
#             embedding_blob = self._serialize_embedding(face_embedding_normalized)
#             metadata_json = json.dumps(metadata)

#             # Insert or replace (upsert)
#             conn = self._get_connection()
#             cursor = conn.cursor()
#             cursor.execute('''
#                 INSERT OR REPLACE INTO face_embeddings 
#                 (employee_id, name, embedding, metadata, updated_at)
#                 VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
#             ''', (str(employee_id), name, embedding_blob, metadata_json))

#             conn.commit()
#             conn.close()
#             return True

#         except Exception as e:
#             print(f"Error adding embedding: {e}")
#             import traceback
#             traceback.print_exc()
#             return False

#     def add_face_encoding(self, employee_id: int, encoding: np.ndarray,
#                           metadata: Dict = None) -> bool:
#         """Alias for add_employee_embedding"""
#         name = metadata.get(
#             'name', f'Employee {employee_id}') if metadata else f'Employee {employee_id}'
#         return self.add_employee_embedding(str(employee_id), name, encoding, metadata)

#     def search_similar_faces(self, query_embedding: np.ndarray,
#                              n_results: int = 1,
#                              distance_threshold: float = 0.5) -> List[Dict]:
#         """
#         Search for similar faces using cosine similarity

#         Args:
#             query_embedding: Face embedding to search for
#             n_results: Number of results to return
#             distance_threshold: Maximum distance for a match

#         Returns:
#             List of dicts with employee_id, distance, and metadata
#         """
#         try:
#             # Reduce verbosity - only log if matches found
#             # print(f"[DEBUG VectorDB] Searching with threshold: {distance_threshold}")
#             # print(f"[DEBUG VectorDB] Query embedding shape: {query_embedding.shape}")

#             # Get all embeddings from database
#             conn = self._get_connection()
#             cursor = conn.cursor()
#             cursor.execute(
#                 "SELECT employee_id, embedding, metadata FROM face_embeddings"
#             )
#             rows = cursor.fetchall()
#             conn.close()

#             if not rows:
#                 # print(f"[DEBUG VectorDB] No embeddings in database!")
#                 return []

#             # print(f"[DEBUG VectorDB] Comparing against {len(rows)} stored embeddings")
#             results = []

#             # Calculate cosine distance to all embeddings
#             for row in rows:
#                 emp_id = row[0]
#                 stored_embedding = self._deserialize_embedding(row[1])
#                 metadata_json = row[2]

#                 # Calculate cosine distance
#                 try:
#                     distance = cosine(query_embedding, stored_embedding)
#                 except:
#                     # Fallback calculation
#                     dot_product = np.dot(query_embedding, stored_embedding)
#                     norm_query = np.linalg.norm(query_embedding)
#                     norm_stored = np.linalg.norm(stored_embedding)
#                     distance = 1 - (dot_product / (norm_query * norm_stored))

#                 # Filter by threshold
#                 if distance <= distance_threshold:
#                     result = {
#                         'employee_id': emp_id,
#                         'distance': float(distance),
#                     }
#                     # Add metadata
#                     if metadata_json:
#                         metadata = json.loads(metadata_json)
#                         result.update(metadata)
#                     results.append(result)

#             # Sort by distance (closest first) and return top n
#             results.sort(key=lambda x: x['distance'])

#             if results:
#                 # Only log the best match with cleaner output
#                 print(
#                     f"✓ Match: {results[0]['employee_id']} - {results[0].get('name', 'Unknown')} ({(1-results[0]['distance'])*100:.1f}% confidence)")

#             return results[:n_results]

#         except Exception as e:
#             print(f"[DEBUG VectorDB] Error searching faces: {e}")
#             import traceback
#             traceback.print_exc()
#             return []

#     def get_all_encodings(self) -> List[Tuple[str, np.ndarray, Dict]]:
#         """Get all encodings from SQLite database"""
#         try:
#             conn = self._get_connection()
#             cursor = conn.cursor()
#             cursor.execute(
#                 "SELECT employee_id, embedding, metadata FROM face_embeddings"
#             )
#             rows = cursor.fetchall()
#             conn.close()

#             encodings = []

#             for row in rows:
#                 emp_id = row[0]
#                 embedding = self._deserialize_embedding(row[1])
#                 metadata = json.loads(row[2]) if row[2] else {}
#                 encodings.append((emp_id, embedding, metadata))

#             return encodings
#         except Exception as e:
#             print(f"Error getting encodings: {e}")
#             return []

#     def update_face_encoding(self, employee_id: int, new_encoding: np.ndarray) -> bool:
#         """Update an existing employee's face encoding"""
#         try:
#             emp_id_str = str(employee_id)

#             # Check if exists
#             conn = self._get_connection()
#             cursor = conn.cursor()
#             cursor.execute(
#                 "SELECT name, metadata FROM face_embeddings WHERE employee_id = ?",
#                 (emp_id_str,)
#             )
#             row = cursor.fetchone()

#             if not row:
#                 conn.close()
#                 return False

#             name = row[0]
#             metadata = row[1]

#             # Update embedding
#             embedding_blob = self._serialize_embedding(new_encoding)
#             cursor.execute('''
#                 UPDATE face_embeddings 
#                 SET embedding = ?, updated_at = CURRENT_TIMESTAMP
#                 WHERE employee_id = ?
#             ''', (embedding_blob, emp_id_str))

#             conn.commit()
#             conn.close()
#             return True
#         except Exception as e:
#             print(f"Error updating encoding: {e}")
#             return False

#     def delete_face_encoding(self, employee_id: int) -> bool:
#         """Delete an employee's face encoding"""
#         try:
#             emp_id_str = str(employee_id)

#             # Delete from SQLite
#             conn = self._get_connection()
#             cursor = conn.cursor()
#             cursor.execute(
#                 "DELETE FROM face_embeddings WHERE employee_id = ?",
#                 (emp_id_str,)
#             )
#             conn.commit()
#             conn.close()
#             return True
#         except Exception as e:
#             print(f"Error deleting encoding: {e}")
#             return False

#     def clear_all(self) -> bool:
#         """Clear all embeddings from SQLite database"""
#         try:
#             conn = self._get_connection()
#             cursor = conn.cursor()
#             cursor.execute("DELETE FROM face_embeddings")
#             conn.commit()
#             conn.close()
#             return True
#         except Exception as e:
#             print(f"Error clearing database: {e}")
#             return False

#     def count(self) -> int:
#         """Get count of embeddings in SQLite database"""
#         try:
#             conn = self._get_connection()
#             cursor = conn.cursor()
#             cursor.execute("SELECT COUNT(*) FROM face_embeddings")
#             result = cursor.fetchone()[0]
#             conn.close()
#             return result
#         except:
#             return 0

#     def close(self):
#         """Close SQLite database connection (no-op for thread-safe implementation)"""
#         # With the new thread-safe approach, connections are opened/closed per operation
#         # This method is kept for backward compatibility but does nothing
#         pass


"""
Vector DB Manager - Optimized
==============================
CHANGES FROM ORIGINAL:
1. CRITICAL: search_similar_faces() was loading ALL embeddings from SQLite
   on EVERY face recognition call. For 100 employees = 100 DB reads per face
   per frame. At 5fps with 2 cameras = 1000+ DB reads/sec. WRONG.
   
   FIX: Load all embeddings into memory as a numpy matrix at startup.
   Recognition becomes a single np.dot() = microseconds instead of milliseconds.
   
2. Added reload_embeddings() method for after enrollment
3. Thread-safe in-memory cache with RWLock pattern
4. search_similar_faces now uses vectorized numpy operations (same as recognize_face
   in face_recognition_engine_onnx.py which was already doing this correctly,
   but vector_db.py was NOT - it was doing per-row cosine from scipy)
5. Removed debug print statements from hot path
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import pickle
import sqlite3
import json
import os
import threading
import config
from scipy.spatial.distance import cosine


class VectorDBManager:
    """
    Face embedding store with in-memory cache for fast inference.
    
    Architecture:
    - SQLite: persistent storage (source of truth)
    - numpy matrix: in-memory cache for fast similarity search
    
    All recognition queries hit the numpy matrix (microseconds).
    SQLite is only read at startup and after enrollment changes.
    """

    def __init__(self, persist_directory: str = None):
        if persist_directory is None:
            persist_directory = os.path.join(config.BASE_DIR, "vector_db")

        os.makedirs(persist_directory, exist_ok=True)
        self.persist_directory = persist_directory
        self.db_path = os.path.join(persist_directory, "embeddings.db")

        # In-memory cache
        self._employee_ids: List[str] = []
        self._employee_names: List[str] = []
        self._embedding_matrix: Optional[np.ndarray] = None  # Shape: (N, 512)
        self._cache_lock = threading.RLock()

        # Initialize DB
        self._init_db()

        # Load into memory
        self._load_cache()

        count = len(self._employee_ids)
        print(f"✓ VectorDB initialized: {count} embeddings in memory cache")

    def _init_db(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_embeddings (
                employee_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return conn

    def _load_cache(self):
        """
        Load ALL embeddings from SQLite into numpy matrix.
        Called once at startup and after any enrollment change.
        """
        with self._cache_lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT employee_id, embedding, metadata FROM face_embeddings")
                rows = cursor.fetchall()
                conn.close()

                if not rows:
                    self._employee_ids = []
                    self._employee_names = []
                    self._embedding_matrix = None
                    return

                ids = []
                names = []
                embeddings = []

                for emp_id, emb_blob, metadata_json in rows:
                    emb = pickle.loads(emb_blob)
                    # Ensure normalized
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    else:
                        continue

                    metadata = json.loads(
                        metadata_json) if metadata_json else {}
                    ids.append(emp_id)
                    names.append(metadata.get('name', 'Unknown'))
                    embeddings.append(emb)

                self._employee_ids = ids
                self._employee_names = names
                # Stack into matrix for vectorized search: shape (N, 512)
                self._embedding_matrix = np.array(embeddings, dtype=np.float32)

            except Exception as e:
                print(f"Error loading embedding cache: {e}")
                self._employee_ids = []
                self._employee_names = []
                self._embedding_matrix = None

    def reload_embeddings(self):
        """Call this after enrolling new employees"""
        self._load_cache()
        print(
            f"✓ Embedding cache reloaded: {len(self._employee_ids)} employees")

    # =========================================================================
    # FAST SEARCH - uses in-memory numpy matrix
    # =========================================================================

    def search_similar_faces(self, query_embedding: np.ndarray,
                             n_results: int = 1,
                             distance_threshold: float = 0.5) -> List[Dict]:
        """
        Fast similarity search using in-memory numpy matrix.
        
        REPLACES: per-row scipy cosine distance (slow, DB reads on every call)
        WITH: single np.dot() on full matrix (fast, all in RAM)
        
        For 100 employees: original = 100 scipy calls
                           optimized = 1 numpy matmul
        """
        with self._cache_lock:
            if self._embedding_matrix is None or len(self._employee_ids) == 0:
                return []

            # Normalize query
            norm = np.linalg.norm(query_embedding)
            if norm == 0:
                return []
            q = query_embedding.astype(np.float32) / norm

            # Vectorized cosine similarity: shape (N,)
            # Since all embeddings are normalized: dot product = cosine similarity
            # (N,512) @ (512,) = (N,)
            similarities = self._embedding_matrix @ q

            # Convert similarity to distance
            distances = 1.0 - similarities

            # Filter by threshold
            mask = distances <= distance_threshold
            if not np.any(mask):
                return []

            # Get indices of matches, sorted by distance
            match_indices = np.where(mask)[0]
            match_distances = distances[match_indices]
            sort_order = np.argsort(match_distances)
            sorted_indices = match_indices[sort_order]

            results = []
            for idx in sorted_indices[:n_results]:
                emp_id = self._employee_ids[idx]
                dist = float(distances[idx])
                results.append({
                    'employee_id': emp_id,
                    'name': self._employee_names[idx],
                    'distance': dist,
                })

            if results:
                best = results[0]
                print(f"✓ Match: {best['employee_id']} - {best['name']} "
                      f"({(1-best['distance'])*100:.1f}% confidence)")

            return results

    # =========================================================================
    # CRUD - writes go to SQLite then reload cache
    # =========================================================================

    def employee_exists(self, employee_id: str) -> bool:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM face_embeddings WHERE employee_id = ?",
                           (str(employee_id),))
            count = cursor.fetchone()[0]
            conn.close()
            return count > 0
        except Exception:
            return False

    def get_existing_employee_ids(self) -> List[str]:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT employee_id FROM face_embeddings")
            result = [row[0] for row in cursor.fetchall()]
            conn.close()
            return result
        except Exception:
            return []

    def add_employee_embedding(self, employee_id: str, name: str,
                               face_embedding: np.ndarray,
                               metadata: Dict = None) -> bool:
        try:
            if metadata is None:
                metadata = {}
            metadata['name'] = name
            metadata['employee_id'] = str(employee_id)

            # Normalize
            norm = np.linalg.norm(face_embedding)
            if norm == 0:
                return False
            embedding_normalized = face_embedding / norm

            embedding_blob = pickle.dumps(
                embedding_normalized.astype(np.float32))
            metadata_json = json.dumps(metadata)

            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO face_embeddings
                (employee_id, name, embedding, metadata, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (str(employee_id), name, embedding_blob, metadata_json))
            conn.commit()
            conn.close()

            # Reload cache so recognition picks up the new employee immediately
            self._load_cache()
            return True

        except Exception as e:
            print(f"Error adding embedding: {e}")
            return False

    def add_face_encoding(self, employee_id: int, encoding: np.ndarray,
                          metadata: Dict = None) -> bool:
        name = metadata.get(
            'name', f'Employee {employee_id}') if metadata else f'Employee {employee_id}'
        return self.add_employee_embedding(str(employee_id), name, encoding, metadata)

    def get_all_encodings(self) -> List[Tuple[str, np.ndarray, Dict]]:
        """Used by face_recognition_engine_onnx.py to load encodings into its own array"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT employee_id, embedding, metadata FROM face_embeddings")
            rows = cursor.fetchall()
            conn.close()

            return [
                (row[0], pickle.loads(row[1]),
                 json.loads(row[2]) if row[2] else {})
                for row in rows
            ]
        except Exception as e:
            print(f"Error getting encodings: {e}")
            return []

    def update_face_encoding(self, employee_id: int, new_encoding: np.ndarray) -> bool:
        try:
            emp_id_str = str(employee_id)
            norm = np.linalg.norm(new_encoding)
            if norm > 0:
                new_encoding = new_encoding / norm

            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE face_embeddings SET embedding = ?, updated_at = CURRENT_TIMESTAMP
                WHERE employee_id = ?
            ''', (pickle.dumps(new_encoding.astype(np.float32)), emp_id_str))
            conn.commit()
            conn.close()
            self._load_cache()
            return True
        except Exception as e:
            print(f"Error updating encoding: {e}")
            return False

    def delete_face_encoding(self, employee_id) -> bool:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM face_embeddings WHERE employee_id = ?",
                           (str(employee_id),))
            conn.commit()
            conn.close()
            self._load_cache()
            return True
        except Exception as e:
            print(f"Error deleting encoding: {e}")
            return False

    def clear_all(self) -> bool:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM face_embeddings")
            conn.commit()
            conn.close()
            self._load_cache()
            return True
        except Exception as e:
            print(f"Error clearing: {e}")
            return False

    def count(self) -> int:
        with self._cache_lock:
            return len(self._employee_ids)

    def close(self):
        pass  # No persistent connection to close
