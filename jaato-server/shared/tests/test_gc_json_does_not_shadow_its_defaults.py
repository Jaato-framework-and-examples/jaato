"""``gc.json`` supplies what it says and nothing else.

``load_gc_from_file`` built its :class:`GCConfig` with ``data.get(key,
<literal>)`` for the trigger keys, so a file that never mentioned them still
decided them.  Two of those literals are the defaults ``GCConfig`` derives
from the environment, whose own docstrings say *"Can be overridden via
JAATO_GC_THRESHOLD"* / ``JAATO_GC_TARGET`` -- and they could not be, for any
session that had a ``gc.json`` at all.

``pressure_percent`` was worse, because its default is not a number.
``data.get('pressure_percent')`` answers ``None`` for a file that omits it,
``None`` is how ``GCConfig`` spells CONTINUOUS mode (GC after every turn
above ``target_percent``, ``threshold_percent`` ignored), and passing it
explicitly beat the env-derived 90.0.  So omitting one key silently selected
a different operating mode.  The ``== 0`` test that sat immediately below the
read is the evidence it was never meant to: that line exists to make a
literal ``0`` mean continuous, which is only worth writing if *absent* does
not.

The profile route into the same dataclass never had any of this --
``GCProfileConfig`` carries real dataclass defaults, ``pressure_percent =
90.0`` among them -- so the two routes disagreed about what an omitted key
means.  ``_media_settings`` had already fixed exactly this for the three
media keys and stated the rule in its docstring; these tests pin the rule
now that it covers the rest of the file.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.plugins.gc import load_gc_from_file


def _write(tmp_path: Path, data: dict) -> str:
    (tmp_path / ".jaato").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".jaato" / "gc.json").write_text(json.dumps(data))
    return str(tmp_path)


def _config(tmp_path: Path, data: dict):
    result = load_gc_from_file(workspace_root=_write(tmp_path, data))
    assert result is not None
    return result[1]


def test_an_omitted_threshold_leaves_the_env_var_in_charge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JAATO_GC_THRESHOLD", "55.0")
    assert _config(tmp_path, {"type": "truncate"}).threshold_percent == 55.0


def test_an_omitted_target_leaves_the_env_var_in_charge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JAATO_GC_TARGET", "42.0")
    assert _config(tmp_path, {"type": "truncate"}).target_percent == 42.0


def test_a_declared_value_still_wins_over_the_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The file is the more specific layer; omission is the only change."""
    monkeypatch.setenv("JAATO_GC_THRESHOLD", "55.0")
    monkeypatch.setenv("JAATO_GC_TARGET", "42.0")
    config = _config(
        tmp_path, {"type": "truncate", "threshold_percent": 70, "target_percent": 30},
    )
    assert (config.threshold_percent, config.target_percent) == (70.0, 30.0)


def test_an_omitted_pressure_is_not_continuous_mode(tmp_path: Path) -> None:
    """The regression that changes an operating MODE, not just a number."""
    config = _config(tmp_path, {"type": "budget"})
    assert config.pressure_percent == 90.0
    assert not config.continuous_mode


def test_an_omitted_pressure_leaves_the_env_var_in_charge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JAATO_GC_PRESSURE", "75.0")
    assert _config(tmp_path, {"type": "budget"}).pressure_percent == 75.0


@pytest.mark.parametrize("declared", [0, None])
def test_a_declared_zero_or_null_still_opts_into_continuous_mode(
    tmp_path: Path, declared,
) -> None:
    """The documented opt-in, which the fix must not take away.

    ``JAATO_GC_PRESSURE=0`` selects it too, via ``_get_pressure_percent``;
    this is the per-file spelling of the same choice.
    """
    config = _config(tmp_path, {"type": "budget", "pressure_percent": declared})
    assert config.pressure_percent is None
    assert config.continuous_mode


def test_the_file_and_the_profile_agree_about_an_omitted_key(
    tmp_path: Path,
) -> None:
    """Two routes into one dataclass; an omitted key must mean one thing.

    ``GCProfileConfig``'s defaults are the reference, because that route was
    always right.
    """
    from shared.plugins.subagent.config import (
        GCProfileConfig, gc_profile_to_plugin_config,
    )
    _, from_profile = gc_profile_to_plugin_config(GCProfileConfig(type="budget"))
    from_file = _config(tmp_path, {"type": "budget"})
    for field in ("threshold_percent", "target_percent", "pressure_percent",
                  "preserve_recent_turns"):
        assert getattr(from_file, field) == getattr(from_profile, field), field
