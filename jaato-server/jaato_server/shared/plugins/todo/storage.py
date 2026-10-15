"""Storage backends for the TODO plugin.

Provides in-memory and file-based persistence for plans.

File persistence is **YAML only** (#1195): ``{plan_id}.yaml`` per plan in
directory mode, one ``todo_plans.yaml`` mapping in single-file mode.  The
plan document is :meth:`TodoPlan.to_dict`, which is also the shape of a
hand-authored predefined plan (``<config_root>/plans/<id>.yaml``), so a
saved plan and an authored one are the same format.  Files the plugin
wrote as JSON before are not read — there is no fallback reader.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import yaml

from jaato_sdk.plugins.todo.models import TodoPlan

#: Extension of every file :class:`FileStorage` writes or reads.
PLAN_FILE_SUFFIX = ".yaml"

#: Default single-file path for ``hybrid`` storage when neither a path nor
#: ``TODO_STORAGE_PATH`` is given.  Relative to the process cwd — never under
#: ``.jaato/plans/``, which holds AUTHORED plans a confined runner may not
#: write.
DEFAULT_STORAGE_PATH = "./todo_plans.yaml"


def _dump(data, fh) -> None:
    """Write *data* as YAML (``safe_dump``; key order kept for readability)."""
    yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)


def _load(fh):
    """Read one YAML document (``safe_load`` — plan files are data)."""
    return yaml.safe_load(fh)


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

    Plans are stored as YAML — one ``{plan_id}.yaml`` per plan in a
    directory, or a single file mapping plan id to plan.  Thread-safe using
    a lock.  An unreadable or malformed file reads as "no plan", as it did
    when the format was JSON.
    """

    def __init__(self, path: str, use_directory: bool = False,
                 owner: Optional[Tuple[int, int]] = None):
        """Initialize file storage.

        Args:
            path: Path to storage file or directory
            use_directory: If True, store each plan in separate file
            owner: ``(uid, gid)`` every directory this storage creates and
                every file it writes is created as (#1703).  The daemon
                configures its TODO mirror on
                ``<ws>/.jaato/sessions/<id>/plans/``, inside a workspace whose
                owner must be able to remove it; ``None`` (the default)
                writes as the calling process, as before.
        """
        self._path = Path(path)
        self._use_directory = use_directory
        self._owner = owner
        self._lock = Lock()

        # Ensure parent directory exists
        self._ensure_dir(self._path if use_directory else self._path.parent)

    def _ensure_dir(self, directory: Path) -> None:
        """``mkdir -p`` *directory*, each directory created owned by ``_owner``."""
        from jaato_server.shared.workspace_ownership import make_dirs_owned
        make_dirs_owned(str(directory), self._owner)

    def _open_for_write(self, path: Path):
        """Open *path* for writing, handed to ``_owner`` before a byte is written.

        A file the calling process owns (a new one, or one an earlier
        process left behind) is ``fchown``-ed; one another account owns is
        left with it.
        """
        f = open(path, 'w', encoding='utf-8')
        if self._owner is not None:
            try:
                if os.fstat(f.fileno()).st_uid == os.geteuid():
                    os.fchown(f.fileno(), self._owner[0], self._owner[1])
            except OSError:
                f.close()
                raise
        return f

    def save_plan(self, plan: TodoPlan) -> None:
        """Save or update a plan."""
        with self._lock:
            if self._use_directory:
                self._save_plan_to_file(plan)
            else:
                self._save_plan_to_single_file(plan)

    def _save_plan_to_file(self, plan: TodoPlan) -> None:
        """Save plan to individual file in directory."""
        # Ensure directory exists before writing (handles CWD changes, deletions, etc.)
        self._ensure_dir(self._path)
        plan_file = self._path / f"{plan.plan_id}{PLAN_FILE_SUFFIX}"
        with self._open_for_write(plan_file) as f:
            _dump(plan.to_dict(), f)

    def _save_plan_to_single_file(self, plan: TodoPlan) -> None:
        """Save plan to the single YAML file."""
        # Ensure parent directory exists before writing
        self._ensure_dir(self._path.parent)
        plans = self._load_all_plans_from_file()
        plans[plan.plan_id] = plan.to_dict()
        with self._open_for_write(self._path) as f:
            _dump(plans, f)

    def _load_all_plans_from_file(self) -> Dict[str, dict]:
        """Load all plans from single file."""
        if not self._path.exists():
            return {}
        try:
            with open(self._path, 'r', encoding='utf-8') as f:
                data = _load(f)
        except (yaml.YAMLError, IOError):
            return {}
        return data if isinstance(data, dict) else {}

    def get_plan(self, plan_id: str) -> Optional[TodoPlan]:
        """Get a plan by ID."""
        with self._lock:
            if self._use_directory:
                return self._get_plan_from_file(plan_id)
            else:
                return self._get_plan_from_single_file(plan_id)

    def _get_plan_from_file(self, plan_id: str) -> Optional[TodoPlan]:
        """Get plan from individual file."""
        plan_file = self._path / f"{plan_id}{PLAN_FILE_SUFFIX}"
        if not plan_file.exists():
            return None
        try:
            with open(plan_file, 'r', encoding='utf-8') as f:
                data = _load(f)
        except (yaml.YAMLError, IOError):
            return None
        return TodoPlan.from_dict(data) if isinstance(data, dict) else None

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
        for plan_file in self._path.glob(f"*{PLAN_FILE_SUFFIX}"):
            try:
                with open(plan_file, 'r', encoding='utf-8') as f:
                    data = _load(f)
            except (yaml.YAMLError, IOError):
                continue
            if isinstance(data, dict):
                plans.append(TodoPlan.from_dict(data))
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
        plan_file = self._path / f"{plan_id}{PLAN_FILE_SUFFIX}"
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
        # Ensure parent directory exists before writing
        self._ensure_dir(self._path.parent)
        with self._open_for_write(self._path) as f:
            _dump(plans, f)
        return True

    def clear(self) -> None:
        """Clear all plans."""
        with self._lock:
            if self._use_directory:
                if self._path.exists():
                    for plan_file in self._path.glob(f"*{PLAN_FILE_SUFFIX}"):
                        plan_file.unlink()
            else:
                # Ensure parent directory exists before writing
                self._ensure_dir(self._path.parent)
                with self._open_for_write(self._path) as f:
                    _dump({}, f)


class HybridStorage(TodoStorage):
    """Hybrid storage combining in-memory cache with file persistence.

    Provides fast in-memory access with automatic file backup.
    """

    def __init__(self, file_path: str, use_directory: bool = False,
                 owner: Optional[Tuple[int, int]] = None):
        """Initialize hybrid storage.

        Args:
            file_path: Path to storage file or directory
            use_directory: If True, store each plan in separate file
            owner: passed to :class:`FileStorage` (#1703)
        """
        self._memory = InMemoryStorage()
        self._file = FileStorage(file_path, use_directory, owner=owner)
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
    use_directory: bool = False,
    owner: Optional[Tuple[int, int]] = None,
) -> TodoStorage:
    """Factory function to create a storage backend.

    Args:
        storage_type: One of "memory", "file", "hybrid"
        path: Path for file-based storage (required for file/hybrid)
        use_directory: Use directory structure for file storage
        owner: ``(uid, gid)`` file storage writes as (#1703); ignored
            for memory storage

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
        return FileStorage(path, use_directory, owner=owner)

    elif storage_type == "hybrid":
        if not path:
            # Default path
            path = os.environ.get("TODO_STORAGE_PATH", DEFAULT_STORAGE_PATH)  # env: where the todo plugin persists plans (default ./todo_plans.yaml)
        return HybridStorage(path, use_directory, owner=owner)

    else:
        raise ValueError(f"Unknown storage type: {storage_type}. "
                        f"Available: memory, file, hybrid")
