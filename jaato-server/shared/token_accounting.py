import datetime
import json
import logging
import os
import random
import ssl
import time
import traceback
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from .ssl_helper import log_ssl_guidance, is_ssl_cert_failure

logger = logging.getLogger(__name__)

from google.api_core import exceptions as google_exceptions

if TYPE_CHECKING:
    from google.genai import Client

# Shared token accounting utilities
# Usage pattern:
#   from google import genai
#   from shared.token_accounting import TokenLedger
#   client = genai.Client(vertexai=True, project=..., location=...)
#   ledger = TokenLedger()            # or TokenLedger(path="ledger.jsonl")
#   response = ledger.generate_with_accounting(client, model_name, prompt)
#   ledger.write_ledger()             # flushes what per-record append did not
#   summary = ledger.summarize()
#
# Note on **gen_kwargs in generate_with_accounting:
# We forward arbitrary generation parameters to the google-genai client.
# This keeps TokenLedger focused on accounting (token counts, retries, logging)
# while remaining automatically compatible with new / optional model arguments.

#: The env var naming the ledger file.  Session-scoped; its typed home is the
#: profile's ``trace.ledger`` key, which seeds it (``TRACE_ENV_VARS``).
LEDGER_PATH_ENV = "LEDGER_PATH"

#: The env var carrying the ledger's INTEGRITY posture.  Session-scoped,
#: and seeded from the typed home ``record_keeping.integrity`` exactly as
#: ``LEDGER_PATH`` is seeded from ``trace.ledger`` -- the env var is the
#: transport, not the place an author writes it.  It is a transport at all
#: because the ledger is constructed before any profile is resolved, and
#: the runner-side session reads the same session-scoped context the path
#: comes from.
LEDGER_INTEGRITY_ENV = "JAATO_LEDGER_INTEGRITY"

#: How much of a ledger's tail is read to recover the chain pointer.
#: One record is a few hundred bytes, so this is generous by two
#: orders of magnitude and still bounded -- a ledger grows without
#: limit and this runs per chained append.
_TAIL_READ_BYTES = 65536


class TokenLedger:
    """The per-runtime record of every model round trip and permission verdict.

    Every ``response`` (tokens, cost, the user the session runs as), every
    ``permission-check`` (verdict, method, approver) and the auxiliary
    stages land here through :meth:`_record`.  Until this change the only
    way any of it reached DISK was :meth:`write_ledger`, and that method had
    **no caller outside its tests** -- on the daemon path the ledger was an
    in-memory list that died with the runner, so a ``permission-check`` row
    carrying an approver existed nowhere after the process exited.  That is
    the record-keeping gap ``docs/design/eu-ai-act.md`` §4.4 names first
    (Regulation (EU) 2024/1689 Arts. 12 and 19 ask for logs that outlive
    the run).

    So a record is now APPENDED TO DISK THE MOMENT IT IS RECORDED, whenever
    a path is configured -- ``path=`` at construction, else the
    ``LEDGER_PATH`` env var, which a profile's ``trace.ledger`` key seeds
    per session.  A process that dies mid-turn has written what it
    recorded.  :meth:`write_ledger` survives as the flush of whatever was
    NOT yet appended (a ledger with no path configured until the end), so
    no record is written twice.

    Lifecycle:
        ``_record`` -> in-memory list -> (path configured) one JSONL line
        appended and flushed -> ``_flushed`` advances.  ``write_ledger``
        writes ``_events[_flushed:]`` and advances the same cursor.

    Args:
        path: Explicit ledger file.  Outranks the env var -- the inversion
            where ``LEDGER_PATH`` beat the argument its caller passed is
            gone.  ``""`` means "no ledger file", explicitly.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        integrity: Optional[str] = None,
    ):
        self._events: List[Dict[str, Any]] = []
        self._path = path
        #: ``"sha256-chain"`` or ``None``/``"none"``.  Explicit argument
        #: wins over :data:`LEDGER_INTEGRITY_ENV`, the inversion
        #: ``ledger_path`` already avoids.
        self._integrity = integrity
        #: The previous chained record's digest, or ``None`` before the
        #: first.  A CACHE of what this instance last wrote, never the
        #: authority: the chain belongs to the FILE, so every chained
        #: append re-reads the tail digest off disk under the lock (see
        #: :meth:`_tail_digest`).  Holding it in memory alone was the
        #: #1120 defect -- a daemon restart, or a second session sharing
        #: an absolute ``trace.ledger``, appended a record linked to
        #: ``genesis`` in the middle of a file, and the verifier
        #: correctly reported an untouched file as tampered.
        self._prev_digest: Optional[str] = None
        #: How many leading events are already on disk.
        self._flushed = 0
        #: Whether an append failure has been reported -- once per ledger,
        #: because a record is made per round trip and per tool call.
        self._append_failed = False
        #: Whether "this file's existing records are not chained" has been
        #: said.  Once per ledger, for the same reason.
        self._unchained_tail_reported = False

    def ledger_path(self) -> Optional[str]:
        """Where records are appended, or ``None`` when nowhere.

        ``path`` given at construction wins; else ``LEDGER_PATH``.  A
        RELATIVE path resolves against ``JAATO_WORKSPACE_ROOT`` when the
        framework has set it -- the rule ``jaato_sdk.trace`` applies to the
        two trace paths, so ``trace.ledger: .jaato/logs/ledger.jsonl`` is
        one file per session exactly as ``trace.session_log`` is.  An empty
        value means "no ledger", not "the default name".
        """
        # Session-scoped reads go through the per-session context first,
        # so two sessions on one daemon do not read each other's paths.
        from .session_context import get_session_env
        raw = self._path if self._path is not None else get_session_env(LEDGER_PATH_ENV)
        if not raw:
            return None
        if os.path.isabs(raw):
            return raw
        workspace = get_session_env("JAATO_WORKSPACE_ROOT")
        return os.path.join(workspace, raw) if workspace else raw

    def chains(self) -> bool:
        """Whether records carry tamper-evidence links (Art. 73(6), #1120).

        Resolved the way :meth:`ledger_path` is: the constructor argument
        wins, else the session-scoped env var seeded from the profile's
        ``record_keeping.integrity``.  Anything other than
        ``"sha256-chain"`` -- including a typo -- is ``none``, because the
        only alternative is silently writing digests a verifier does not
        expect, and an unchained file is reported as unchained rather
        than as intact.
        """
        from .session_context import get_session_env
        raw = (self._integrity if self._integrity is not None
               else get_session_env(LEDGER_INTEGRITY_ENV))
        return (raw or "none").strip().lower() == "sha256-chain"

    def _line(self, index: int) -> str:
        """The JSONL line for event ``index``.

        The ONE place a record becomes bytes, so the per-record append
        and :meth:`write_ledger`'s flush cannot chain differently -- the
        failure #1120 names, since ``write_ledger`` exists precisely to
        write records the append path did not.

        Chaining advances :attr:`_prev_digest`, so this method is
        ORDER-DEPENDENT and callers must walk indices in order.  Both do.
        It is also WRITE-dependent: the pointer it leaves behind names a
        record the caller has not written yet, so a caller whose write
        fails must not reuse it.  Both callers go through
        :meth:`_write_pending`, which re-reads the pointer from the file
        on every chained append -- so a failed write self-heals rather
        than chaining the retry to a record that reached no disk.
        """
        record = self._enrich(self._events[index], index)
        if not self.chains():
            return json.dumps(record)
        from jaato_sdk.audit_chain import chain
        chained = chain(record, self._prev_digest)
        self._prev_digest = chained["digest"]
        return json.dumps(chained)

    def _record(self, stage: str, details: Dict[str, Any]) -> None:
        details["stage"] = stage
        details["ts"] = time.time()
        self._events.append(details)
        self._append_to_disk()

    @staticmethod
    def _enrich(ev: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """The on-disk form of one event: ISO timestamp, index, derived tokens."""
        enriched = dict(ev)
        enriched["iso_ts"] = datetime.datetime.utcfromtimestamp(ev.get("ts", time.time())).isoformat() + "Z"
        enriched["event_index"] = idx
        if "prompt_tokens" in ev and "output_tokens" in ev and "total_tokens" in ev:
            pt = ev.get("prompt_tokens") or 0
            ot = ev.get("output_tokens") or 0
            tt = ev.get("total_tokens") or 0
            enriched["internal_tokens"] = tt - (pt + ot)
        return enriched

    def _tail_digest(self, path: str) -> Optional[str]:
        """The digest of the last chained record already in ``path``.

        The chain is a property of the FILE, not of whichever process is
        appending, so this is where a chained append gets its link:
        without it a restart or a second writer begins a fresh chain
        mid-file and the verifier -- rightly, on the evidence it has --
        reports the file as tampered with.

        Reads the tail only (:data:`_TAIL_READ_BYTES`), because a ledger
        grows without bound and this runs per chained append.  ``None``
        means *nothing to link to*: an absent or empty file (the first
        record, which links to ``genesis``), or a tail that carries no
        chain fields -- an unchained file being appended to with chaining
        now on, which is announced, because the verifier will report the
        older half as carrying no evidence either way.
        """
        try:
            size = os.path.getsize(path)
        except OSError:
            return None
        if size == 0:
            return None
        try:
            with open(path, "rb") as fh:
                if size > _TAIL_READ_BYTES:
                    fh.seek(size - _TAIL_READ_BYTES)
                    fh.readline()          # drop the partial first line
                tail = fh.read().decode("utf-8", errors="replace")
        except OSError:
            return None
        for raw in reversed(tail.splitlines()):
            text = raw.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except ValueError:
                continue
            if isinstance(record, dict) and record.get("digest"):
                return str(record["digest"])
            break
        if not self._unchained_tail_reported:
            self._unchained_tail_reported = True
            logger.warning(
                "ledger: %s already holds records with no chain fields; the "
                "records appended from now on are chained and the earlier "
                "ones will verify as 'carries no chain fields' -- evidence "
                "of nothing either way, rather than of tampering", path)
        return None

    def _write_pending(self, path: str, *, fsync: bool) -> None:
        """Append every not-yet-flushed event to ``path``.  May raise.

        The ONE writer, so the per-record append and
        :meth:`write_ledger`'s flush cannot chain or advance differently
        -- the same argument :meth:`_line` makes about rendering.

        Two properties the chained path depends on:

        * **The link comes from the file, under an exclusive lock.**  A
          restart, a second session on a shared absolute ``trace.ledger``
          and two concurrent appenders all continue the one chain instead
          of starting rival ones.  The lock is the #683 primitive, so
          there is one flock mechanism in the tree rather than two.
        * **The cursors advance PER LINE, after that line is flushed.**
          Advancing them for the whole batch up front meant a failure on
          line 3 of 5 left lines 1-2 on disk and unrecorded, so the retry
          wrote them again -- with different digests, since the pointer
          had moved.  A duplicated record in an audit log is the thing
          the log exists to make impossible.
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if not self.chains():
            self._flush_lines(path, fsync=fsync)
            return
        from .credential_lock import credential_lock
        with credential_lock(path):
            self._prev_digest = self._tail_digest(path)
            self._flush_lines(path, fsync=fsync)

    def _flush_lines(self, path: str, *, fsync: bool) -> None:
        """Render and write each pending event, advancing as each lands."""
        with open(path, "a", encoding="utf-8") as f:
            for idx in range(self._flushed, len(self._events)):
                f.write(self._line(idx) + "\n")
                f.flush()
                self._flushed = idx + 1
            if fsync:
                try:
                    os.fsync(f.fileno())
                except OSError:
                    # Not supported on every filesystem; the data is at
                    # least in the OS buffer cache.
                    pass

    def _append_to_disk(self) -> None:
        """Append every not-yet-flushed event to the configured path, if any.

        Best-effort and never raises: a ledger that cannot be written must
        not fail the model round trip it is recording.  The failure is
        logged once per ledger at WARNING and the cursor is left where it
        was, so :meth:`write_ledger` can still flush the same events to a
        path that works.  One ``flush()`` per record, no ``fsync`` -- a
        record per tool call is the hot path; ``write_ledger`` fsyncs.
        """
        path = self.ledger_path()
        if path is None or self._flushed >= len(self._events):
            return
        try:
            self._write_pending(path, fsync=False)
        except Exception as exc:  # noqa: BLE001
            if not self._append_failed:
                self._append_failed = True
                logger.warning(
                    "ledger: cannot append to %s (%s); records stay in memory "
                    "until write_ledger() -- reported once", path, exc)

    def generate_with_accounting(self, client: 'Client', model_name: str, prompt: str, **gen_kwargs):
        """Generate content with token accounting.

        Args:
            client: google.genai.Client instance
            model_name: Model name (e.g., 'gemini-2.5-flash')
            prompt: The prompt text
            **gen_kwargs: Additional generation parameters (passed to GenerateContentConfig)
        """
        from google.genai import types

        try:
            count_info = client.models.count_tokens(model=model_name, contents=prompt)
            self._record("pre-count", {"total_tokens": getattr(count_info, "total_tokens", None)})
        except Exception as exc:
            self._record("pre-count-error", {"error": str(exc)})
            if is_ssl_cert_failure(exc):
                silent = os.environ.get('AI_RETRY_LOG_SILENT', '').lower() in ('1','true','yes')
                log_ssl_guidance('Pre-count', exc, silent=silent, pre_count=True)
                raise
        # Retry loop for transient quota / rate-limit errors (HTTP 429 / ResourceExhausted)
        max_attempts = int(os.environ.get("AI_RETRY_ATTEMPTS", "5"))
        base_delay = float(os.environ.get("AI_RETRY_BASE_DELAY", "1.0"))
        max_delay = float(os.environ.get("AI_RETRY_MAX_DELAY", "30.0"))
        last_exc: Optional[Exception] = None
        response = None

        transient_classes = (
            google_exceptions.TooManyRequests,
            google_exceptions.ResourceExhausted,
            google_exceptions.ServiceUnavailable,
            google_exceptions.InternalServerError,
            google_exceptions.DeadlineExceeded,
            google_exceptions.Aborted,
        )

        def _is_transient(exc: Exception) -> Dict[str, bool]:
            rate_like = False
            infra_like = False
            if isinstance(exc, transient_classes):
                if isinstance(exc, (google_exceptions.TooManyRequests, google_exceptions.ResourceExhausted)):
                    rate_like = True
                else:
                    infra_like = True
            else:
                lower = str(exc).lower()
                if any(p in lower for p in ["429", "too many requests", "resource exhausted"]):
                    rate_like = True
                if any(p in lower for p in ["503", "service unavailable", "temporarily unavailable", "internal error"]):
                    infra_like = True
            return {"transient": rate_like or infra_like, "rate_limit": rate_like, "infra": infra_like}

        # Build config from gen_kwargs
        config = types.GenerateContentConfig(**gen_kwargs) if gen_kwargs else None

        for attempt in range(1, max_attempts + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config
                )
                break
            except Exception as exc:
                last_exc = exc
                # SSL certificate guidance detection
                if is_ssl_cert_failure(exc):
                    self._record('ssl-error', {'attempt': attempt, 'error': str(exc)})
                    silent = os.environ.get('AI_RETRY_LOG_SILENT', '').lower() in ('1','true','yes')
                    log_ssl_guidance('Generate', exc, silent=silent, pre_count=False)
                    raise
                classification = _is_transient(exc)
                self._record("api-error", {"attempt": attempt, "error": str(exc), **classification})
                if not classification["transient"] or attempt == max_attempts:
                    raise
                sleep_sec = min(max_delay, base_delay * (2 ** (attempt - 1))) * random.uniform(0.5, 1.5)
                if os.environ.get("AI_RETRY_LOG_SILENT", "").lower() not in ("1", "true", "yes"):
                    try:
                        err_cls = exc.__class__.__name__
                    except Exception:
                        err_cls = "Exception"
                    tag = "rate-limit" if classification["rate_limit"] else "transient"
                    exc_msg = str(exc)[:140].replace('\n', ' ')
                    logger.info(f"[AI Retry {attempt}/{max_attempts}] {tag}: {err_cls}: {exc_msg} | sleep {sleep_sec:.2f}s")
                time.sleep(sleep_sec)
        usage = getattr(response, "usage_metadata", None)
        if usage:
            self._record(
                "response",
                {
                    "prompt_tokens": getattr(usage, "prompt_token_count", None),
                    "output_tokens": getattr(usage, "candidates_token_count", None),
                    "total_tokens": getattr(usage, "total_token_count", None),
                },
            )
        else:
            self._record("response", {"prompt_tokens": None, "output_tokens": None, "total_tokens": None})
        return response

    def summarize(self) -> Dict[str, Any]:
        total_prompt = sum(e.get("prompt_tokens") or 0 for e in self._events if e.get("stage") == "response")
        total_output = sum(e.get("output_tokens") or 0 for e in self._events if e.get("stage") == "response")
        total = sum(e.get("total_tokens") or 0 for e in self._events if e.get("stage") == "response")
        api_errors = [e for e in self._events if e.get("stage") == "api-error"]
        rate_errors = [e for e in api_errors if e.get("rate_limit")]
        retry_attempts = len(api_errors)
        rate_limit_retries = len(rate_errors)
        last_rate_error = rate_errors[-1]["error"] if rate_errors else None
        max_attempt = max((e.get("attempt", 0) for e in api_errors), default=0)
        return {
            "calls": len([e for e in self._events if e.get("stage") == "response"]),
            "total_prompt_tokens": total_prompt,
            "total_output_tokens": total_output,
            "total_tokens": total,
            "events": self._events,
            "retry_attempts": retry_attempts,
            "rate_limit_retries": rate_limit_retries,
            "last_rate_limit_error": last_rate_error,
            "max_retry_attempt_index": max_attempt,
        }

    def write_ledger(self, filepath: Optional[str] = None) -> Optional[str]:
        """Flush every event NOT yet appended to the JSONL ledger.

        Each event is written as a single line.  After all events are
        written we flush + fsync so a process crash or power loss
        can't leave a partial line in the ledger (which would corrupt
        downstream JSONL parsers).

        Args:
            filepath: Where to write.  Given, it wins; else the configured
                :meth:`ledger_path`; else the legacy default
                ``token_events_ledger.jsonl`` in the working directory.

        Returns:
            The path written, or ``None`` on failure.  Writes only the
            events :meth:`_append_to_disk` has not already landed, so a
            ledger that appended per record has nothing left to flush and
            a record is never on disk twice.
        """
        path = filepath or self.ledger_path() or "token_events_ledger.jsonl"
        try:
            self._write_pending(path, fsync=True)
            return path
        except Exception as exc:
            logger.error(f"Ledger write failed: {exc}", exc_info=True)
            return None

    def events(self) -> List[Dict[str, Any]]:
        return list(self._events)


def generate_with_ledger(client: 'Client', model_name: str, prompt: str, ledger: Optional[TokenLedger] = None, **kwargs):
    """Generate content with a ledger for token accounting.

    Args:
        client: google.genai.Client instance
        model_name: Model name (e.g., 'gemini-2.5-flash')
        prompt: The prompt text
        ledger: Optional TokenLedger instance (created if not provided)
        **kwargs: Additional generation parameters
    """
    if ledger is None:
        ledger = TokenLedger()
    return ledger.generate_with_accounting(client, model_name, prompt, **kwargs), ledger
