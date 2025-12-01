import os
import time
import random
import json
import datetime
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import ssl
from .ssl_helper import log_ssl_guidance, is_ssl_cert_failure

from google.api_core import exceptions as google_exceptions

if TYPE_CHECKING:
    from google.genai import Client

# Shared token accounting utilities
# Usage pattern:
#   from google import genai
#   from shared.token_accounting import TokenLedger
#   client = genai.Client(vertexai=True, project=..., location=...)
#   ledger = TokenLedger()
#   response = ledger.generate_with_accounting(client, model_name, prompt)
#   ledger.write_ledger()
#   summary = ledger.summarize()
#
# Note on **gen_kwargs in generate_with_accounting:
# We forward arbitrary generation parameters to the google-genai client.
# This keeps TokenLedger focused on accounting (token counts, retries, logging)
# while remaining automatically compatible with new / optional model arguments.

class TokenLedger:
    def __init__(self):
        self._events: List[Dict[str, Any]] = []

    def _record(self, stage: str, details: Dict[str, Any]) -> None:
        details["stage"] = stage
        details["ts"] = time.time()
        self._events.append(details)

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
                    print(f"[AI Retry {attempt}/{max_attempts}] {tag}: {err_cls}: {exc_msg} | sleep {sleep_sec:.2f}s")
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

    def write_ledger(self, filepath: str = "token_events_ledger.jsonl") -> Optional[str]:
        path = os.environ.get("LEDGER_PATH", filepath)
        try:
            with open(path, "a", encoding="utf-8") as f:
                for idx, ev in enumerate(self._events):
                    enriched = dict(ev)
                    enriched["iso_ts"] = datetime.datetime.utcfromtimestamp(ev.get("ts", time.time())).isoformat() + "Z"
                    enriched["event_index"] = idx
                    if "prompt_tokens" in ev and "output_tokens" in ev and "total_tokens" in ev:
                        pt = ev.get("prompt_tokens") or 0
                        ot = ev.get("output_tokens") or 0
                        tt = ev.get("total_tokens") or 0
                        enriched["internal_tokens"] = tt - (pt + ot) if (pt is not None and ot is not None and tt is not None) else None
                    f.write(json.dumps(enriched) + "\n")
            return path
        except Exception as exc:
            print(f"(Ledger write failed: {exc})")
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
