import redis
import numpy as np
import json
from typing import Optional, List, Dict, Tuple
import base64
import time

class RedisClient:
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        """
        Initialize Redis client with separate connections for string and binary data.
        
        Args:
            host: Redis server hostname
            port: Redis server port
            db: Redis database number
        """
        # Client for string data with automatic decoding
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True
        )
        
        # Client for binary data without decoding
        self.face_db = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=False
        )
        
        # Test connection
        try:
            self.redis_client.ping()
        except redis.ConnectionError as e:
            raise ConnectionError(f"Could not connect to Redis: {str(e)}")

    def store_face_embedding(self, 
                           person_id: str, 
                           embedding: np.ndarray,
                           metadata: Optional[Dict] = None) -> bool:
        """
        Store face embedding and metadata in Redis.
        
        Args:
            person_id: Unique identifier for the person
            embedding: Face embedding numpy array
            metadata: Additional information about the person
            
        Returns:
            bool: True if storage was successful
        """
        try:
            # Store the embedding
            embedding_key = f"face:embedding:{person_id}"
            embedding_bytes = base64.b64encode(embedding.tobytes())
            self.face_db.set(embedding_key, embedding_bytes)
            
            # Store metadata if provided
            if metadata:
                metadata_key = f"face:metadata:{person_id}"
                # Convert all values to strings for Redis compatibility
                string_metadata = {k: str(v) for k, v in metadata.items()}
                self.redis_client.hmset(metadata_key, string_metadata)
            
            return True
        except Exception as e:
            print(f"Error storing face embedding: {str(e)}")
            return False

    def get_face_embedding(self, person_id: str) -> Optional[np.ndarray]:
        """
        Retrieve face embedding for a person.
        
        Args:
            person_id: Unique identifier for the person
            
        Returns:
            Optional[np.ndarray]: Face embedding if found, None otherwise
        """
        try:
            embedding_key = f"face:embedding:{person_id}"
            embedding_data = self.face_db.get(embedding_key)
            
            if embedding_data:
                embedding_bytes = base64.b64decode(embedding_data)
                return np.frombuffer(embedding_bytes, dtype=np.float64)
            return None
        except Exception as e:
            print(f"Error retrieving face embedding: {str(e)}")
            return None

    def get_metadata(self, person_id: str) -> Optional[Dict]:
        """
        Retrieve metadata for a person.
        
        Args:
            person_id: Unique identifier for the person
            
        Returns:
            Optional[Dict]: Metadata dictionary if found, None otherwise
        """
        try:
            metadata_key = f"face:metadata:{person_id}"
            metadata = self.redis_client.hgetall(metadata_key)
            return metadata if metadata else None
        except Exception as e:
            print(f"Error retrieving metadata: {str(e)}")
            return None

    def update_last_seen(self, person_id: str, camera_id: str, timestamp: float):
        """
        Update last seen information for a person.
        
        Args:
            person_id: Unique identifier for the person
            camera_id: Identifier for the camera that saw the person
            timestamp: Unix timestamp of the sighting
        """
        try:
            key = f"face:lastseen:{person_id}"
            self.redis_client.hset(key, camera_id, str(timestamp))
        except Exception as e:
            print(f"Error updating last seen: {str(e)}")

    def get_all_persons(self) -> List[str]:
        """
        Get list of all person IDs in the database.
        
        Returns:
            List[str]: List of person IDs
        """
        try:
            pattern = "face:embedding:*"
            keys = self.redis_client.keys(pattern)
            return [key.split(':')[-1] for key in keys]
        except Exception as e:
            print(f"Error getting person list: {str(e)}")
            return []

    def find_similar_faces(self, 
                         query_embedding: np.ndarray, 
                         threshold: float = 0.6) -> List[Tuple[str, float]]:
        """
        Find similar faces in the database.
        
        Args:
            query_embedding: Face embedding to search for
            threshold: Similarity threshold (0-1)
            
        Returns:
            List[Tuple[str, float]]: List of (person_id, similarity) tuples
        """
        matches = []
        try:
            for person_id in self.get_all_persons():
                stored_embedding = self.get_face_embedding(person_id)
                if stored_embedding is not None:
                    similarity = self._compute_similarity(query_embedding, stored_embedding)
                    if similarity >= threshold:
                        matches.append((person_id, float(similarity)))
            
            # Sort by similarity in descending order
            matches.sort(key=lambda x: x[1], reverse=True)
            return matches
        except Exception as e:
            print(f"Error finding similar faces: {str(e)}")
            return []

    def _compute_similarity(self, 
                          embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        """
        Compute similarity between two face embeddings.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            float: Similarity score (0-1)
        """
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            embedding1_normalized = embedding1 / norm1
            embedding2_normalized = embedding2 / norm2
            
            # Compute cosine similarity
            similarity = np.dot(embedding1_normalized, embedding2_normalized)
            return float(similarity)
        except Exception as e:
            print(f"Error computing similarity: {str(e)}")
            return 0.0

    def clear_person_data(self, person_id: str) -> bool:
        """
        Remove all data associated with a person.
        
        Args:
            person_id: Unique identifier for the person
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            keys_to_delete = [
                f"face:embedding:{person_id}",
                f"face:metadata:{person_id}",
                f"face:lastseen:{person_id}"
            ]
            
            for key in keys_to_delete:
                self.redis_client.delete(key)
            return True
        except Exception as e:
            print(f"Error clearing person data: {str(e)}")
            return False  