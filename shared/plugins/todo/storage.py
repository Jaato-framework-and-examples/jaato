"""Storage backends for the TODO plugin.

Provides in-memory and file-based persistence for plans.
"""

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

from .models import TodoPlan


class TodoStorage(ABC):
    """Base class for plan storage backends."""

    @abstractmethod
    def save_plan(self, plan: TodoPlan) -> None:
        """Save or update a plan."""
        ...

    @abstractmethod
    def get_plan(self, plan_id: str) -> Optional[TodoPlan]:
        """Get a plan by ID."""
        ...

    @abstractmethod
    def get_all_plans(self) -> List[TodoPlan]:
        """Get all stored plans."""
        ...

    @abstractmethod
    def delete_plan(self, plan_id: str) -> bool:
        """Delete a plan. Returns True if deleted."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all plans."""
        ...


class InMemoryStorage(TodoStorage):
    """In-memory storage for plans.

    Plans are stored in a dict and lost when the process exits.
    Thread-safe using a lock.
    """

    def __init__(self):
        self._plans: Dict[str, TodoPlan] = {}
        self._lock = Lock()

    def save_plan(self, plan: TodoPlan) -> None:
        """Save or update a plan."""
        with self._lock:
            self._plans[plan.plan_id] = plan

    def get_plan(self, plan_id: str) -> Optional[TodoPlan]:
        """Get a plan by ID."""
        with self._lock:
            return self._plans.get(plan_id)

    def get_all_plans(self) -> List[TodoPlan]:
        """Get all stored plans."""
        with self._lock:
            return list(self._plans.values())

    def delete_plan(self, plan_id: str) -> bool:
        """Delete a plan. Returns True if deleted."""
        with self._lock:
            if plan_id in self._plans:
                del self._plans[plan_id]
                return True
            return False

    def clear(self) -> None:
        """Clear all plans."""
        with self._lock:
            self._plans.clear()


class FileStorage(TodoStorage):
    """File-based persistent storage for plans.

    Plans are stored as JSON in a single file or directory structure.
    Thread-safe using a lock.
    """

    def __init__(self, path: str, use_directory: bool = False):
        """Initialize file storage.

        Args:
            path: Path to storage file or directory
            use_directory: If True, store each plan in separate file
        """
        self._path = Path(path)
        self._use_directory = use_directory
        self._lock = Lock()

        # Ensure parent directory exists
        if use_directory:
            self._path.mkdir(parents=True, exist_ok=True)
        else:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def save_plan(self, plan: TodoPlan) -> None:
        """Save or update a plan."""
        with self._lock:
            if self._use_directory:
                self._save_plan_to_file(plan)
            else:
                self._save_plan_to_single_file(plan)

    def _save_plan_to_file(self, plan: TodoPlan) -> None:
        """Save plan to individual file in directory."""
        plan_file = self._path / f"{plan.plan_id}.json"
        with open(plan_file, 'w', encoding='utf-8') as f:
            json.dump(plan.to_dict(), f, indent=2)

    def _save_plan_to_single_file(self, plan: TodoPlan) -> None:
        """Save plan to single JSON file."""
        plans = self._load_all_plans_from_file()
        plans[plan.plan_id] = plan.to_dict()
        with open(self._path, 'w', encoding='utf-8') as f:
            json.dump(plans, f, indent=2)

    def _load_all_plans_from_file(self) -> Dict[str, dict]:
        """Load all plans from single file."""
        if not self._path.exists():
            return {}
        try:
            with open(self._path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def get_plan(self, plan_id: str) -> Optional[TodoPlan]:
        """Get a plan by ID."""
        with self._lock:
            if self._use_directory:
                return self._get_plan_from_file(plan_id)
            else:
                return self._get_plan_from_single_file(plan_id)

    def _get_plan_from_file(self, plan_id: str) -> Optional[TodoPlan]:
        """Get plan from individual file."""
        plan_file = self._path / f"{plan_id}.json"
        if not plan_file.exists():
            return None
        try:
            with open(plan_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return TodoPlan.from_dict(data)
        except (json.JSONDecodeError, IOError):
            return None

    def _get_plan_from_single_file(self, plan_id: str) -> Optional[TodoPlan]:
        """Get plan from single file."""
        plans = self._load_all_plans_from_file()
        if plan_id not in plans:
            return None
        return TodoPlan.from_dict(plans[plan_id])

    def get_all_plans(self) -> List[TodoPlan]:
        """Get all stored plans."""
        with self._lock:
            if self._use_directory:
                return self._get_all_plans_from_directory()
            else:
                return self._get_all_plans_from_single_file()

    def _get_all_plans_from_directory(self) -> List[TodoPlan]:
        """Get all plans from directory."""
        plans = []
        if not self._path.exists():
            return plans
        for plan_file in self._path.glob("*.json"):
            try:
                with open(plan_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                plans.append(TodoPlan.from_dict(data))
            except (json.JSONDecodeError, IOError):
                continue
        return plans

    def _get_all_plans_from_single_file(self) -> List[TodoPlan]:
        """Get all plans from single file."""
        plans_data = self._load_all_plans_from_file()
        return [TodoPlan.from_dict(p) for p in plans_data.values()]

    def delete_plan(self, plan_id: str) -> bool:
        """Delete a plan. Returns True if deleted."""
        with self._lock:
            if self._use_directory:
                return self._delete_plan_from_directory(plan_id)
            else:
                return self._delete_plan_from_single_file(plan_id)

    def _delete_plan_from_directory(self, plan_id: str) -> bool:
        """Delete plan file from directory."""
        plan_file = self._path / f"{plan_id}.json"
        if plan_file.exists():
            plan_file.unlink()
            return True
        return False

    def _delete_plan_from_single_file(self, plan_id: str) -> bool:
        """Delete plan from single file."""
        plans = self._load_all_plans_from_file()
        if plan_id not in plans:
            return False
        del plans[plan_id]
        with open(self._path, 'w', encoding='utf-8') as f:
            json.dump(plans, f, indent=2)
        return True

    def clear(self) -> None:
        """Clear all plans."""
        with self._lock:
            if self._use_directory:
                for plan_file in self._path.glob("*.json"):
                    plan_file.unlink()
            else:
                if self._path.exists():
                    with open(self._path, 'w', encoding='utf-8') as f:
                        json.dump({}, f)


class HybridStorage(TodoStorage):
    """Hybrid storage combining in-memory cache with file persistence.

    Provides fast in-memory access with automatic file backup.
    """

    def __init__(self, file_path: str, use_directory: bool = False):
        """Initialize hybrid storage.

        Args:
            file_path: Path to storage file or directory
            use_directory: If True, store each plan in separate file
        """
        self._memory = InMemoryStorage()
        self._file = FileStorage(file_path, use_directory)
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Load plans from file into memory if not already loaded."""
        if not self._loaded:
            for plan in self._file.get_all_plans():
                self._memory.save_plan(plan)
            self._loaded = True

    def save_plan(self, plan: TodoPlan) -> None:
        """Save to both memory and file."""
        self._ensure_loaded()
        self._memory.save_plan(plan)
        self._file.save_plan(plan)

    def get_plan(self, plan_id: str) -> Optional[TodoPlan]:
        """Get from memory (fast)."""
        self._ensure_loaded()
        return self._memory.get_plan(plan_id)

    def get_all_plans(self) -> List[TodoPlan]:
        """Get all from memory."""
        self._ensure_loaded()
        return self._memory.get_all_plans()

    def delete_plan(self, plan_id: str) -> bool:
        """Delete from both memory and file."""
        self._ensure_loaded()
        deleted_memory = self._memory.delete_plan(plan_id)
        deleted_file = self._file.delete_plan(plan_id)
        return deleted_memory or deleted_file

    def clear(self) -> None:
        """Clear both memory and file storage."""
        self._memory.clear()
        self._file.clear()


def create_storage(
    storage_type: str = "memory",
    path: Optional[str] = None,
    use_directory: bool = False
) -> TodoStorage:
    """Factory function to create a storage backend.

    Args:
        storage_type: One of "memory", "file", "hybrid"
        path: Path for file-based storage (required for file/hybrid)
        use_directory: Use directory structure for file storage

    Returns:
        TodoStorage instance

    Raises:
        ValueError: If storage_type is unknown or path missing for file storage
    """
    if storage_type == "memory":
        return InMemoryStorage()

    elif storage_type == "file":
        if not path:
            raise ValueError("File storage requires 'path' parameter")
        return FileStorage(path, use_directory)

    elif storage_type == "hybrid":
        if not path:
            # Default path
            path = os.environ.get("TODO_STORAGE_PATH", "./todo_plans.json")
        return HybridStorage(path, use_directory)

    else:
        raise ValueError(f"Unknown storage type: {storage_type}. "
                        f"Available: memory, file, hybrid")
