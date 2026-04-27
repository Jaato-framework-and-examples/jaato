"""Harbor adapter that runs a jaato-server inside the eval environment.

Architecture::

    host (Harbor)                      |  container (BaseEnvironment)
    ----------------------------------- + -----------------------------------
    JaatoAgent(BaseAgent)               |  /opt/jaato/bin/python -m server
      setup(env):                       |    └─ daemon, /tmp/jaato.sock
        - pip install jaato-server,     |
          jaato-sdk, harbor_jaato       |  /opt/jaato/bin/python -m
        - upload .jaato/profiles/*.json |     harbor_jaato.harness
        - upload instruction text       |       └─ IPCRecoveryClient
      run(instruction, env, ctx):       |       └─ writes result.json
        - env.exec("python -m           |
            harbor_jaato.harness ...")  |
        - download result.json          |
        - copy fields onto ctx          |

The host BaseAgent is a thin shim. All real work — model selection,
tool execution, permission handling, token accounting — happens inside
the container, driven by ``harness.py`` against the local Unix socket.
That keeps WS auth, port forwarding, and cross-host event streaming out
of the picture.
"""

from __future__ import annotations

import json
import logging
import shlex
import tempfile
from importlib import metadata, resources
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .result import HarnessResult

logger = logging.getLogger(__name__)

CONTAINER_VENV = "/opt/jaato"
CONTAINER_PYTHON = f"{CONTAINER_VENV}/bin/python"
CONTAINER_PIP = f"{CONTAINER_VENV}/bin/pip"
CONTAINER_JAATO_DIR = "/workspace/.jaato"
CONTAINER_PKG_DIR = "/opt/harbor_jaato"
CONTAINER_INSTRUCTION = f"{CONTAINER_JAATO_DIR}/instruction.txt"
CONTAINER_RESULT = f"{CONTAINER_JAATO_DIR}/result.json"
CONTAINER_PROFILE = f"{CONTAINER_JAATO_DIR}/profiles/harbor.json"
CONTAINER_AGENT = f"{CONTAINER_JAATO_DIR}/agents/harbor-shell.md"

AGENT_NAME = "harbor-shell"
AGENT_RESOURCE = "data/harbor-shell.md"

DEFAULT_PLUGINS = [
    "cli",
    "file_edit",
    "filesystem_query",
    "todo",
    "interactive_shell",
]

PROVIDER_ENV_KEYS: dict[str, list[str]] = {
    "anthropic": ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"],
    "google_genai": [
        "PROJECT_ID",
        "LOCATION",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ],
    "github_models": ["GITHUB_TOKEN", "JAATO_GITHUB_ORGANIZATION"],
    "ollama": ["OLLAMA_HOST"],
    "lmstudio": ["LMSTUDIO_HOST", "LMSTUDIO_API_TOKEN"],
    "nim": ["JAATO_NIM_API_KEY", "JAATO_NIM_BASE_URL"],
}


class JaatoAgent(BaseAgent):
    """Harbor BaseAgent backed by a jaato-server running inside the env."""

    SUPPORTS_ATIF = False
    SUPPORTS_WINDOWS = False

    @staticmethod
    def name() -> str:
        return "jaato"

    def version(self) -> str | None:
        try:
            return metadata.version("jaato-server")
        except metadata.PackageNotFoundError:
            return None

    async def setup(self, environment: BaseEnvironment) -> None:
        """Install jaato + the harness into the env and write the profile."""
        await self._install_jaato(environment)
        await self._upload_harness(environment)
        await self._write_profile(environment)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Drive the in-container harness, then transfer results onto context."""
        await environment.exec(f"mkdir -p {CONTAINER_JAATO_DIR}")
        await self._upload_text(environment, instruction, CONTAINER_INSTRUCTION)

        env_overrides = self._provider_env()
        cmd = (
            f"PYTHONPATH={shlex.quote(CONTAINER_PKG_DIR)} "
            f"{shlex.quote(CONTAINER_PYTHON)} -m harbor_jaato.harness "
            f"--instruction-file {shlex.quote(CONTAINER_INSTRUCTION)} "
            f"--profile harbor "
            f"--result {shlex.quote(CONTAINER_RESULT)}"
        )
        try:
            await environment.exec(cmd, cwd="/workspace", env=env_overrides)
        finally:
            # Always try to pull whatever the harness wrote, even on
            # nonzero exit / timeout — partial token counts are still
            # useful in AgentContext.
            await self._populate_from_result(environment, context)

    async def cleanup(self, environment: BaseEnvironment) -> None:
        """Stop the daemon and remove its socket.

        Container teardown handles this anyway, but graceful shutdown
        avoids stragglers on long-lived eval hosts and surfaces daemon
        crashes deterministically (a crash leaves the socket behind;
        explicit removal turns the next ``connect()`` into a clean
        auto-start instead of an EADDRINUSE).
        """
        try:
            await environment.exec(
                "pkill -f 'jaato.*--daemon' || true; rm -f /tmp/jaato.sock"
            )
        except Exception as e:
            logger.warning("cleanup exec failed: %s", e)

    async def _install_jaato(self, environment: BaseEnvironment) -> None:
        cmd = (
            f"python3 -m venv {CONTAINER_VENV} && "
            f"{CONTAINER_PIP} install --quiet --upgrade pip && "
            f"{CONTAINER_PIP} install --quiet "
            "  --extra-index-url https://test.pypi.org/simple/ "
            "  jaato-server jaato-sdk"
        )
        await environment.exec(cmd)

    async def _upload_harness(self, environment: BaseEnvironment) -> None:
        """Upload this package's source into the container at PYTHONPATH.

        We ship our own source rather than ``pip install``ing the host's
        copy, because the host package may not be on a public index.
        """
        pkg_root = Path(__file__).resolve().parent
        await environment.exec(f"mkdir -p {CONTAINER_PKG_DIR}/harbor_jaato")
        for source in pkg_root.glob("*.py"):
            await environment.upload_file(
                source, f"{CONTAINER_PKG_DIR}/harbor_jaato/{source.name}"
            )

    async def _write_profile(self, environment: BaseEnvironment) -> None:
        """Materialize the profile and the ``harbor-shell`` agent definition.

        The profile pins model + provider + plugin set; the agent
        markdown supplies system instructions. ``create_session`` then
        receives ``profile="harbor", agent="harbor-shell"`` and the
        rendered agent body becomes the session's system prompt.
        """
        provider, model = self._split_model_name()
        profile = {
            "name": "harbor",
            "description": "Harbor eval profile",
            "model": model,
            "provider": provider,
            "plugins": list(DEFAULT_PLUGINS),
            "plugin_configs": {},
        }
        await environment.exec(
            f"mkdir -p {CONTAINER_JAATO_DIR}/profiles "
            f"{CONTAINER_JAATO_DIR}/agents"
        )
        await self._upload_text(
            environment, json.dumps(profile, indent=2), CONTAINER_PROFILE
        )
        agent_md = resources.files("harbor_jaato").joinpath(
            AGENT_RESOURCE
        ).read_text()
        await self._upload_text(environment, agent_md, CONTAINER_AGENT)

    async def _upload_text(
        self, environment: BaseEnvironment, text: str, dest: str
    ) -> None:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".tmp", delete=False
        ) as fh:
            fh.write(text)
            local = fh.name
        try:
            await environment.upload_file(local, dest)
        finally:
            Path(local).unlink(missing_ok=True)

    async def _populate_from_result(
        self,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False
        ) as fh:
            local = Path(fh.name)
        try:
            try:
                await environment.download_file(CONTAINER_RESULT, local)
            except Exception as e:
                logger.warning("could not download result.json: %s", e)
                context.metadata = {
                    **(context.metadata or {}),
                    "harbor_jaato_error": f"missing result.json: {e}",
                }
                return

            result = HarnessResult.read(local)
            context.n_input_tokens = result.n_input_tokens
            context.n_output_tokens = result.n_output_tokens
            context.n_cache_tokens = result.n_cache_tokens
            if result.cost_usd is not None:
                context.cost_usd = result.cost_usd
            context.metadata = {
                **(context.metadata or {}),
                "model": self.model_name,
                "turns": result.turns,
                "finished": result.finished,
                "trajectory": result.trajectory,
                **({"error": result.error} if result.error else {}),
            }
        finally:
            local.unlink(missing_ok=True)

    def _split_model_name(self) -> tuple[str, str]:
        """Harbor passes ``model_name`` as ``provider/model``."""
        if "/" not in self.model_name:
            raise ValueError(
                f"model_name must be 'provider/model', got {self.model_name!r}"
            )
        provider, model = self.model_name.split("/", 1)
        return provider, model

    def _provider_env(self) -> dict[str, str]:
        """Forward only the env keys the chosen provider needs.

        Harbor doesn't define a credential-passing protocol, so we mine
        ``os.environ`` for the keys jaato's provider plugin expects and
        forward them through ``environment.exec(env=...)``.
        """
        import os

        provider, _ = self._split_model_name()
        keys = PROVIDER_ENV_KEYS.get(provider, [])
        out: dict[str, str] = {}
        for key in keys:
            value = os.environ.get(key)
            if value:
                out[key] = value
        return out
