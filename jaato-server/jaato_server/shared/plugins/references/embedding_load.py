"""Loading an embedding model without asking the network first (#1482).

The provider (out of tree, e.g. jaato-premium's SentenceTransformer one)
owns the load. What this module decides is HOW it is asked: when the model
is already in the local Hugging Face cache, the first attempt runs offline
(``local_files_only=True`` when ``load_model`` accepts it, and the hub's own
offline switches for the duration of the call), so a cached
``all-MiniLM-L6-v2`` costs no HEAD requests. Only when that attempt fails
does the ordinary online load run. The model is never changed.

Stdlib only; nothing here imports the hub.
"""

from __future__ import annotations

import contextlib
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional

#: Hub environment switches honoured by huggingface_hub / transformers.
OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


def _hub_cache_dirs() -> List[Path]:
    """Every directory the hub (or sentence-transformers) may cache into."""
    dirs: List[Path] = []
    if os.environ.get("HF_HUB_CACHE"):
        dirs.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        dirs.append(Path(os.environ["HF_HOME"]) / "hub")
    xdg = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    dirs.append(Path(xdg) / "huggingface" / "hub")
    if os.environ.get("SENTENCE_TRANSFORMERS_HOME"):
        dirs.append(Path(os.environ["SENTENCE_TRANSFORMERS_HOME"]))
    return dirs


def _repo_ids(model_name: str) -> List[str]:
    """Repo ids a bare model name may be cached under."""
    if "/" in model_name:
        return [model_name]
    return [model_name, f"sentence-transformers/{model_name}"]


def model_cached_locally(model_name: str) -> bool:
    """Whether ``model_name`` can load with no network at all.

    True for a local directory path, or a hub repo with at least one
    snapshot in a cache directory. Best effort: an unreadable cache reads
    as not cached, which only costs the online load it always cost.
    """
    if not model_name:
        return False
    try:
        if os.path.isdir(model_name):
            return True
        for cache in _hub_cache_dirs():
            for repo in _repo_ids(model_name):
                snapshots = cache / ("models--" + repo.replace("/", "--")) / "snapshots"
                if snapshots.is_dir() and any(snapshots.iterdir()):
                    return True
                # sentence-transformers' legacy flat cache layout
                if (cache / repo.replace("/", "_")).is_dir():
                    return True
    except OSError:
        return False
    return False


@contextlib.contextmanager
def hub_offline() -> Iterator[None]:
    """Make the hub refuse network access for the duration of the block.

    Sets the environment switches (read when the hub is first imported)
    and, when the hub is already imported, its module-level constant.
    Everything is restored on exit.
    """
    saved_env = {k: os.environ.get(k) for k in OFFLINE_ENV_VARS}
    constants = sys.modules.get("huggingface_hub.constants")
    saved_const = getattr(constants, "HF_HUB_OFFLINE", None) if constants else None
    for k in OFFLINE_ENV_VARS:
        os.environ[k] = "1"
    if constants is not None and saved_const is not None:
        constants.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if constants is not None and saved_const is not None:
            constants.HF_HUB_OFFLINE = saved_const


def _accepts_local_files_only(load: Callable[..., Any]) -> bool:
    try:
        params = inspect.signature(load).parameters
    except (TypeError, ValueError):
        return False
    return "local_files_only" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _attempt(provider: Any, **kwargs: Any) -> bool:
    try:
        ok = provider.load_model(**kwargs)
    except Exception:
        return False
    return bool(ok) and bool(getattr(provider, "available", False))


def load_model_offline_first(
    provider: Any, trace: Optional[Callable[[str], None]] = None,
) -> bool:
    """Load ``provider``'s model, offline first when it is cached.

    Returns whether the provider is available afterwards. Never raises.
    """
    if getattr(provider, "available", False):
        return True
    log = trace or (lambda _msg: None)
    name = str(getattr(provider, "model_name", "") or "")
    if model_cached_locally(name):
        kwargs = ({"local_files_only": True}
                  if _accepts_local_files_only(provider.load_model) else {})
        with hub_offline():
            if _attempt(provider, **kwargs):
                log(f"embedding model '{name}' loaded from the local cache (offline)")
                return True
        log(f"embedding model '{name}': offline load failed; loading online")
    return _attempt(provider)
