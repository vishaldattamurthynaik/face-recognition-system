"""
Enrolled Face Database manager.
Stores identity metadata, multi-sample embeddings, and computes identity centroids.
Supports JSON persistence, vector indexing, and dynamic updates.
"""
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATABASE_PATH, ENROLLED_DIR


@dataclass
class EnrolledPerson:
    """
    Data representation of an enrolled identity.
    """
    name: str
    person_id: str
    metadata: Dict[str, str]
    created_at: float
    num_samples: int
    embeddings: List[List[float]]  # List of 128-d float vectors
    centroid: List[float]          # Mean normalized 128-d float vector


class EnrolledFaceDatabase:
    """
    Persistent Database of Enrolled Face Embeddings.
    """
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or DATABASE_PATH)
        self.people: Dict[str, EnrolledPerson] = {}  # key: person_id or name
        self.load()

    def enroll(
        self,
        name: str,
        embedding: np.ndarray,
        person_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> EnrolledPerson:
        """
        Enrolls a new sample embedding for an identity. If identity exists, appends sample and updates centroid.

        Args:
            name: Identity name (e.g. "Alice Smith").
            embedding: 128-d numpy array.
            person_id: Unique identifier (defaults to sanitized name).
            metadata: Optional dictionary with role, department, notes, etc.

        Returns:
            Updated EnrolledPerson record.
        """
        pid = person_id or name.strip().lower().replace(" ", "_")
        name = name.strip()
        meta = metadata or {}
        emb_list = embedding.flatten().tolist()

        if pid in self.people:
            # Person already exists: append embedding and recalculate centroid
            person = self.people[pid]
            person.name = name  # update display name if modified
            person.embeddings.append(emb_list)
            person.num_samples = len(person.embeddings)
            person.metadata.update(meta)

            # Recalculate normalized centroid vector
            all_embs = np.array(person.embeddings, dtype=np.float32)
            mean_vec = np.mean(all_embs, axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 0:
                mean_vec = mean_vec / norm
            person.centroid = mean_vec.tolist()
        else:
            # New person
            person = EnrolledPerson(
                name=name,
                person_id=pid,
                metadata=meta,
                created_at=time.time(),
                num_samples=1,
                embeddings=[emb_list],
                centroid=emb_list
            )
            self.people[pid] = person

        self.save()
        return person

    def delete_person(self, person_id: str) -> bool:
        """
        Removes an identity from the database.
        """
        if person_id in self.people:
            del self.people[person_id]
            self.save()
            return True
        return False

    def clear(self):
        """
        Clears all enrolled records.
        """
        self.people.clear()
        self.save()

    def get_person(self, person_id: str) -> Optional[EnrolledPerson]:
        """
        Gets a person record by person_id.
        """
        return self.people.get(person_id)

    def list_people(self) -> List[EnrolledPerson]:
        """
        Returns a list of all enrolled people.
        """
        return list(self.people.values())

    def count(self) -> int:
        """
        Returns total number of enrolled identities.
        """
        return len(self.people)

    def get_all_centroids_matrix(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Returns matrix of all identity centroids for fast vectorized matching.

        Returns:
            centroids_matrix: np.ndarray of shape (N, 128)
            person_ids: List of person_ids of length N
            names: List of display names of length N
        """
        if not self.people:
            return np.empty((0, 128), dtype=np.float32), [], []

        person_ids = []
        names = []
        centroids = []

        for pid, person in self.people.items():
            person_ids.append(pid)
            names.append(person.name)
            centroids.append(person.centroid)

        return np.array(centroids, dtype=np.float32), person_ids, names

    def get_all_embeddings_matrix(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Returns matrix of all individual sample embeddings across all people.
        Useful for multi-exemplar nearest-neighbor search.

        Returns:
            embeddings_matrix: np.ndarray of shape (M, 128)
            person_ids: List of person_ids of length M
            names: List of display names of length M
        """
        if not self.people:
            return np.empty((0, 128), dtype=np.float32), [], []

        person_ids = []
        names = []
        embeddings = []

        for pid, person in self.people.items():
            for emb in person.embeddings:
                person_ids.append(pid)
                names.append(person.name)
                embeddings.append(emb)

        return np.array(embeddings, dtype=np.float32), person_ids, names

    def save(self):
        """
        Persists database to disk in JSON format.
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        data = {pid: asdict(person) for pid, person in self.people.items()}
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self):
        """
        Loads database from disk.
        """
        if not self.db_path.exists():
            return

        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.people = {
                pid: EnrolledPerson(
                    name=p["name"],
                    person_id=p["person_id"],
                    metadata=p.get("metadata", {}),
                    created_at=p.get("created_at", 0.0),
                    num_samples=p.get("num_samples", len(p.get("embeddings", []))),
                    embeddings=p["embeddings"],
                    centroid=p["centroid"]
                )
                for pid, p in data.items()
            }
        except Exception as e:
            print(f"[!] Warning: Failed to load database from {self.db_path}: {e}")
            self.people = {}
