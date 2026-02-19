"""Tests for TODO plugin storage backends."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from jaato_sdk.plugins.todo.models import TodoPlan, StepStatus
from ..storage import (
    InMemoryStorage,
    FileStorage,
    HybridStorage,
    create_storage,
)


class TestInMemoryStorage:
    """Tests for InMemoryStorage."""

    def test_save_and_get_plan(self):
        storage = InMemoryStorage()
        plan = TodoPlan.create("Test", ["A", "B"])

        storage.save_plan(plan)
        retrieved = storage.get_plan(plan.plan_id)

        assert retrieved is not None
        assert retrieved.plan_id == plan.plan_id
        assert retrieved.title == "Test"

    def test_get_nonexistent_plan(self):
        storage = InMemoryStorage()

        retrieved = storage.get_plan("nonexistent")

        assert retrieved is None

    def test_get_all_plans(self):
        storage = InMemoryStorage()
        plan1 = TodoPlan.create("Plan 1", ["A"])
        plan2 = TodoPlan.create("Plan 2", ["B"])

        storage.save_plan(plan1)
        storage.save_plan(plan2)
        plans = storage.get_all_plans()

        assert len(plans) == 2
        titles = {p.title for p in plans}
        assert "Plan 1" in titles
        assert "Plan 2" in titles

    def test_delete_plan(self):
        storage = InMemoryStorage()
        plan = TodoPlan.create("Test", ["A"])
        storage.save_plan(plan)

        deleted = storage.delete_plan(plan.plan_id)

        assert deleted is True
        assert storage.get_plan(plan.plan_id) is None

    def test_delete_nonexistent_plan(self):
        storage = InMemoryStorage()

        deleted = storage.delete_plan("nonexistent")

        assert deleted is False

    def test_clear(self):
        storage = InMemoryStorage()
        storage.save_plan(TodoPlan.create("Plan 1", ["A"]))
        storage.save_plan(TodoPlan.create("Plan 2", ["B"]))

        storage.clear()

        assert len(storage.get_all_plans()) == 0

    def test_update_plan(self):
        storage = InMemoryStorage()
        plan = TodoPlan.create("Test", ["A", "B"])
        storage.save_plan(plan)

        # Modify and save again
        plan.steps[0].complete("Done")
        storage.save_plan(plan)

        retrieved = storage.get_plan(plan.plan_id)
        assert retrieved.steps[0].status == StepStatus.COMPLETED


class TestFileStorage:
    """Tests for FileStorage."""

    def test_save_and_get_plan_single_file(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            storage = FileStorage(path, use_directory=False)
            plan = TodoPlan.create("Test", ["A"])

            storage.save_plan(plan)
            retrieved = storage.get_plan(plan.plan_id)

            assert retrieved is not None
            assert retrieved.title == "Test"
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_save_and_get_plan_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileStorage(tmpdir, use_directory=True)
            plan = TodoPlan.create("Test", ["A"])

            storage.save_plan(plan)
            retrieved = storage.get_plan(plan.plan_id)

            assert retrieved is not None
            assert retrieved.title == "Test"

            # Check file exists
            plan_file = Path(tmpdir) / f"{plan.plan_id}.json"
            assert plan_file.exists()

    def test_get_all_plans_single_file(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            storage = FileStorage(path, use_directory=False)
            plan1 = TodoPlan.create("Plan 1", ["A"])
            plan2 = TodoPlan.create("Plan 2", ["B"])

            storage.save_plan(plan1)
            storage.save_plan(plan2)
            plans = storage.get_all_plans()

            assert len(plans) == 2
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_get_all_plans_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileStorage(tmpdir, use_directory=True)
            plan1 = TodoPlan.create("Plan 1", ["A"])
            plan2 = TodoPlan.create("Plan 2", ["B"])

            storage.save_plan(plan1)
            storage.save_plan(plan2)
            plans = storage.get_all_plans()

            assert len(plans) == 2

    def test_delete_plan_single_file(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            storage = FileStorage(path, use_directory=False)
            plan = TodoPlan.create("Test", ["A"])
            storage.save_plan(plan)

            deleted = storage.delete_plan(plan.plan_id)

            assert deleted is True
            assert storage.get_plan(plan.plan_id) is None
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_delete_plan_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileStorage(tmpdir, use_directory=True)
            plan = TodoPlan.create("Test", ["A"])
            storage.save_plan(plan)

            deleted = storage.delete_plan(plan.plan_id)

            assert deleted is True
            plan_file = Path(tmpdir) / f"{plan.plan_id}.json"
            assert not plan_file.exists()

    def test_clear_single_file(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            storage = FileStorage(path, use_directory=False)
            storage.save_plan(TodoPlan.create("Test", ["A"]))

            storage.clear()

            assert len(storage.get_all_plans()) == 0
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_clear_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileStorage(tmpdir, use_directory=True)
            storage.save_plan(TodoPlan.create("Test", ["A"]))

            storage.clear()

            assert len(list(Path(tmpdir).glob("*.json"))) == 0


class TestHybridStorage:
    """Tests for HybridStorage."""

    def test_save_persists_to_file(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            storage = HybridStorage(path)
            plan = TodoPlan.create("Test", ["A"])

            storage.save_plan(plan)

            # Verify file was created
            with open(path, 'r') as f:
                data = json.load(f)
            assert plan.plan_id in data
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_get_from_memory_after_save(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            storage = HybridStorage(path)
            plan = TodoPlan.create("Test", ["A"])
            storage.save_plan(plan)

            # Get should come from memory (fast)
            retrieved = storage.get_plan(plan.plan_id)

            assert retrieved is not None
            assert retrieved.title == "Test"
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_loads_from_file_on_first_access(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            # Create and save with first storage instance
            storage1 = HybridStorage(path)
            plan = TodoPlan.create("Test", ["A"])
            storage1.save_plan(plan)

            # Create new storage instance (simulating restart)
            storage2 = HybridStorage(path)

            # Should load from file
            retrieved = storage2.get_plan(plan.plan_id)

            assert retrieved is not None
            assert retrieved.title == "Test"
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestCreateStorage:
    """Tests for create_storage factory function."""

    def test_create_memory_storage(self):
        storage = create_storage("memory")

        assert isinstance(storage, InMemoryStorage)

    def test_create_file_storage(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            storage = create_storage("file", path=path)

            assert isinstance(storage, FileStorage)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_create_hybrid_storage(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            storage = create_storage("hybrid", path=path)

            assert isinstance(storage, HybridStorage)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_create_file_storage_requires_path(self):
        with pytest.raises(ValueError, match="path"):
            create_storage("file")

    def test_create_unknown_storage_type(self):
        with pytest.raises(ValueError, match="Unknown storage type"):
            create_storage("invalid_type")
