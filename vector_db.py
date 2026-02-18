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

#             # Serialize embedding
#             embedding_blob = self._serialize_embedding(face_embedding)
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
#             print(
#                 f"[DEBUG VectorDB] Searching with threshold: {distance_threshold}")
#             print(
#                 f"[DEBUG VectorDB] Query embedding shape: {query_embedding.shape}")

#             # Get all embeddings from database
#             conn = self._get_connection()
#             cursor = conn.cursor()
#             cursor.execute(
#                 "SELECT employee_id, embedding, metadata FROM face_embeddings"
#             )
#             rows = cursor.fetchall()
#             conn.close()

#             if not rows:
#                 print(f"[DEBUG VectorDB] No embeddings in database!")
#                 return []

#             print(
#                 f"[DEBUG VectorDB] Comparing against {len(rows)} stored embeddings")
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
#                     print(
#                         f"[DEBUG VectorDB] Match: {emp_id}, distance: {distance:.4f}")

#             # Sort by distance (closest first) and return top n
#             results.sort(key=lambda x: x['distance'])

#             if results:
#                 print(
#                     f"[DEBUG VectorDB] Best match: {results[0]['employee_id']} (distance: {results[0]['distance']:.4f})")
#             else:
#                 print(f"[DEBUG VectorDB] No matches within threshold")

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
Vector Database Manager for Face Embeddings using SQLite
Provides persistent storage with incremental enrollment support
Lightweight, no external dependencies (built-in with Python)
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
import pickle
import sqlite3
import json
import os
import config
from scipy.spatial.distance import cosine


class VectorDBManager:
    """
    Manages face embeddings with SQLite for persistent storage
    Supports incremental enrollment by checking existing employees
    Uses built-in Python SQLite - no external dependencies needed
    """

    def __init__(self, persist_directory: str = None):
        if persist_directory is None:
            persist_directory = os.path.join(config.BASE_DIR, "vector_db")

        os.makedirs(persist_directory, exist_ok=True)
        self.persist_directory = persist_directory

        # Initialize SQLite database
        self.db_path = os.path.join(persist_directory, "embeddings.db")

        # Create initial connection to set up database
        conn = self._get_connection()
        cursor = conn.cursor()

        # Create table for embeddings if not exists
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

        # Get count of existing embeddings
        cursor.execute("SELECT COUNT(*) FROM face_embeddings")
        existing_count = cursor.fetchone()[0]
        conn.close()

        print(f"✓ Vector database (SQLite) initialized at: {self.db_path}")
        print(f"  Embeddings in database: {existing_count}")

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a thread-safe database connection
        Creates a new connection each time to avoid threading issues
        """
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return conn

    def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """Convert numpy array to bytes for storage"""
        return pickle.dumps(embedding)

    def _deserialize_embedding(self, blob: bytes) -> np.ndarray:
        """Convert bytes back to numpy array"""
        return pickle.loads(blob)

    def employee_exists(self, employee_id: str) -> bool:
        """Check if an employee already has embeddings in the database"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM face_embeddings WHERE employee_id = ?",
                (str(employee_id),)
            )
            count = cursor.fetchone()[0]
            conn.close()
            return count > 0
        except Exception as e:
            print(f"Error checking employee existence: {e}")
            return False

    def get_existing_employee_ids(self) -> List[str]:
        """Get list of all employee IDs that have embeddings"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT employee_id FROM face_embeddings")
            result = [row[0] for row in cursor.fetchall()]
            conn.close()
            return result
        except Exception as e:
            print(f"Error getting employee IDs: {e}")
            return []

    def add_employee_embedding(self, employee_id: str, name: str,
                               face_embedding: np.ndarray,
                               metadata: Dict = None) -> bool:
        """Add or update employee face embedding to SQLite database"""
        try:
            if metadata is None:
                metadata = {}

            metadata['name'] = name
            metadata['employee_id'] = str(employee_id)
            
            # CRITICAL: Ensure embedding is normalized before storing
            # This guarantees consistent similarity calculations
            embedding_norm = np.linalg.norm(face_embedding)
            if embedding_norm > 0:
                face_embedding_normalized = face_embedding / embedding_norm
            else:
                print(f"Error: Cannot normalize zero-norm embedding for {employee_id}")
                return False

            # Serialize normalized embedding
            embedding_blob = self._serialize_embedding(face_embedding_normalized)
            metadata_json = json.dumps(metadata)

            # Insert or replace (upsert)
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO face_embeddings 
                (employee_id, name, embedding, metadata, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (str(employee_id), name, embedding_blob, metadata_json))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            print(f"Error adding embedding: {e}")
            import traceback
            traceback.print_exc()
            return False

    def add_face_encoding(self, employee_id: int, encoding: np.ndarray,
                          metadata: Dict = None) -> bool:
        """Alias for add_employee_embedding"""
        name = metadata.get(
            'name', f'Employee {employee_id}') if metadata else f'Employee {employee_id}'
        return self.add_employee_embedding(str(employee_id), name, encoding, metadata)

    def search_similar_faces(self, query_embedding: np.ndarray,
                             n_results: int = 1,
                             distance_threshold: float = 0.5) -> List[Dict]:
        """
        Search for similar faces using cosine similarity

        Args:
            query_embedding: Face embedding to search for
            n_results: Number of results to return
            distance_threshold: Maximum distance for a match

        Returns:
            List of dicts with employee_id, distance, and metadata
        """
        try:
            # Reduce verbosity - only log if matches found
            # print(f"[DEBUG VectorDB] Searching with threshold: {distance_threshold}")
            # print(f"[DEBUG VectorDB] Query embedding shape: {query_embedding.shape}")

            # Get all embeddings from database
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT employee_id, embedding, metadata FROM face_embeddings"
            )
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                # print(f"[DEBUG VectorDB] No embeddings in database!")
                return []

            # print(f"[DEBUG VectorDB] Comparing against {len(rows)} stored embeddings")
            results = []

            # Calculate cosine distance to all embeddings
            for row in rows:
                emp_id = row[0]
                stored_embedding = self._deserialize_embedding(row[1])
                metadata_json = row[2]

                # Calculate cosine distance
                try:
                    distance = cosine(query_embedding, stored_embedding)
                except:
                    # Fallback calculation
                    dot_product = np.dot(query_embedding, stored_embedding)
                    norm_query = np.linalg.norm(query_embedding)
                    norm_stored = np.linalg.norm(stored_embedding)
                    distance = 1 - (dot_product / (norm_query * norm_stored))

                # Filter by threshold
                if distance <= distance_threshold:
                    result = {
                        'employee_id': emp_id,
                        'distance': float(distance),
                    }
                    # Add metadata
                    if metadata_json:
                        metadata = json.loads(metadata_json)
                        result.update(metadata)
                    results.append(result)

            # Sort by distance (closest first) and return top n
            results.sort(key=lambda x: x['distance'])

            if results:
                # Only log the best match with cleaner output
                print(
                    f"✓ Match: {results[0]['employee_id']} - {results[0].get('name', 'Unknown')} ({(1-results[0]['distance'])*100:.1f}% confidence)")

            return results[:n_results]

        except Exception as e:
            print(f"[DEBUG VectorDB] Error searching faces: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_all_encodings(self) -> List[Tuple[str, np.ndarray, Dict]]:
        """Get all encodings from SQLite database"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT employee_id, embedding, metadata FROM face_embeddings"
            )
            rows = cursor.fetchall()
            conn.close()

            encodings = []

            for row in rows:
                emp_id = row[0]
                embedding = self._deserialize_embedding(row[1])
                metadata = json.loads(row[2]) if row[2] else {}
                encodings.append((emp_id, embedding, metadata))

            return encodings
        except Exception as e:
            print(f"Error getting encodings: {e}")
            return []

    def update_face_encoding(self, employee_id: int, new_encoding: np.ndarray) -> bool:
        """Update an existing employee's face encoding"""
        try:
            emp_id_str = str(employee_id)

            # Check if exists
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name, metadata FROM face_embeddings WHERE employee_id = ?",
                (emp_id_str,)
            )
            row = cursor.fetchone()

            if not row:
                conn.close()
                return False

            name = row[0]
            metadata = row[1]

            # Update embedding
            embedding_blob = self._serialize_embedding(new_encoding)
            cursor.execute('''
                UPDATE face_embeddings 
                SET embedding = ?, updated_at = CURRENT_TIMESTAMP
                WHERE employee_id = ?
            ''', (embedding_blob, emp_id_str))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error updating encoding: {e}")
            return False

    def delete_face_encoding(self, employee_id: int) -> bool:
        """Delete an employee's face encoding"""
        try:
            emp_id_str = str(employee_id)

            # Delete from SQLite
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM face_embeddings WHERE employee_id = ?",
                (emp_id_str,)
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error deleting encoding: {e}")
            return False

    def clear_all(self) -> bool:
        """Clear all embeddings from SQLite database"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM face_embeddings")
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error clearing database: {e}")
            return False

    def count(self) -> int:
        """Get count of embeddings in SQLite database"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM face_embeddings")
            result = cursor.fetchone()[0]
            conn.close()
            return result
        except:
            return 0

    def close(self):
        """Close SQLite database connection (no-op for thread-safe implementation)"""
        # With the new thread-safe approach, connections are opened/closed per operation
        # This method is kept for backward compatibility but does nothing
        pass
