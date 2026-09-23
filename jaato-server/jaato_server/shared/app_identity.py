"""Who is the application making these requests?

WHAT THIS ANSWERS.  Everything built on jaato used to introduce itself to
upstream services as *jaato* — the framework's own name and its GitHub URL,
hardcoded in ``model_provider/openrouter/env.py``.  On an OpenRouter dashboard
that means every harness anyone builds with the SDK collapses into one row
called ``jaato``: an integrator cannot see their own app's spend, and the
framework's app-ranking entry is an aggregate of other people's products.
The framework's name is not the application's name, and the two were the same
string.

This module separates them.  :class:`AppIdentity` is *the application* —
the product an integrator ships — and the framework is the thing it is built
with.  Both reach the upstream: a title of ``"Acme Copilot (powered by
jaato)"`` names the app and keeps the attribution jaato was getting before.

HOW AN AUTHOR SETS IT.  Three surfaces, highest precedence first:

1. **Provider knob** — ``plugin_configs.openrouter.app_title`` /
   ``http_referer`` in a profile.  Per-session, provider-specific, and
   already existed; nothing here changes it.  Use it when *one* session
   needs to attribute differently from the rest.
2. **Programmatic** — ``JaatoRuntime(app_identity=AppIdentity(...))``, for
   a product that embeds the framework in its own process.
3. **Environment** — :data:`ENV_APP_NAME` / :data:`ENV_APP_URL` /
   :data:`ENV_APP_VERSION` / :data:`ENV_APP_POWERED_BY` /
   :data:`ENV_APP_CATEGORIES`.  The deployment surface: a daemon started by
   an app, or a workspace ``.env``.

Absent all three the identity is the framework's own (:data:`FRAMEWORK_NAME`),
so an unconfigured checkout keeps reporting exactly as it did before.

WHY ``host`` SCOPE.  The env vars are tagged ``host`` in
:mod:`shared.env_scope`: *which application this is* is a property of the
deployment, not of a conversation, and two sessions in one process disagreeing
about it would be a lie about who is spending the money.  Per-session
attribution is a real need — it is what tier 1 above serves.

WHAT CONSUMES IT.  Today: all three of the OpenRouter provider's
app-attribution headers (``HTTP-Referer`` / ``X-OpenRouter-Title`` /
``X-OpenRouter-Categories``) — no part of "who is this app" is left living in
the provider.  Note that :meth:`AppIdentity.attribution_categories` does NOT
fall back to the framework's category the way ``attribution_url`` falls back
to its URL; the reason is on that method.  :meth:`AppIdentity.user_agent`
is the general form for any provider or HTTP client that wants to identify the
caller; it is deliberately shaped like a conventional UA string
(``Acme-Copilot/1.4.0 (powered by jaato/0.7.0)``) so nothing has to parse the
attribution convention to use it.

HEADER SAFETY.  Every field is sanitised at construction: control characters
(CR/LF included) are stripped and the value is length-capped, because these
strings are written verbatim into HTTP headers.  An app name is attacker-
influenced in exactly the deployments where it matters most — a hosted product
naming itself after a tenant.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

__all__ = [
    "AppIdentity",
    "FRAMEWORK_CATEGORIES",
    "FRAMEWORK_IDENTITY",
    "FRAMEWORK_NAME",
    "FRAMEWORK_URL",
    "framework_version",
    "resolve_app_identity",
]

# ============================================================
# The framework's own identity
# ============================================================

#: Display name jaato itself reports under.  An :class:`AppIdentity` whose
#: name matches this is the framework, not an application built on it, and
#: never gets the "powered by" suffix (``jaato (powered by jaato)``).
FRAMEWORK_NAME = "jaato"

#: Canonical URL for the framework — the fallback ``HTTP-Referer`` when an
#: application supplies a name but no URL of its own.
FRAMEWORK_URL = "https://github.com/Jaato-framework-and-examples/jaato"

#: Marketplace categories jaato itself claims.  ``cli-agent`` is the closest
#: fit in OpenRouter's taxonomy ("terminal-based coding assistants") for a
#: terminal-driven agentic tool orchestrator.  Deliberately NOT inherited by
#: an application that names itself — see
#: :meth:`AppIdentity.attribution_categories`.
FRAMEWORK_CATEGORIES = ("cli-agent",)

#: Distribution whose version is the framework version.
_FRAMEWORK_DISTRIBUTION = "jaato-server"

# ============================================================
# Environment variable names
# ============================================================

ENV_APP_NAME = "JAATO_APP_NAME"
ENV_APP_URL = "JAATO_APP_URL"
ENV_APP_VERSION = "JAATO_APP_VERSION"
ENV_APP_POWERED_BY = "JAATO_APP_POWERED_BY"
ENV_APP_CATEGORIES = "JAATO_APP_CATEGORIES"

# ============================================================
# Sanitisation limits
# ============================================================

#: Longest accepted name / version.  OpenRouter truncates long titles itself;
#: the cap here exists so a runaway string cannot be smuggled into a header.
MAX_NAME_LENGTH = 128

#: URLs are longer by nature but still bounded.
MAX_URL_LENGTH = 512

#: Per-category cap.  Category taxonomies are short slugs everywhere they
#: appear; the consuming provider applies its own (usually tighter) rules.
MAX_CATEGORY_LENGTH = 64

_FALSE_WORDS = frozenset({"0", "false", "no", "off", "none", ""})


def _sanitise(value: Optional[str], *, limit: int) -> Optional[str]:
    """Make ``value`` safe to write into an HTTP header value.

    Drops every C0/C1 control character — CR and LF above all, which is what
    turns a header value into a header *injection* — collapses surrounding
    whitespace, and truncates to ``limit``.  Returns ``None`` for anything
    that is empty once cleaned, so callers can treat "unset" and "set to
    whitespace" identically.
    """
    if value is None:
        return None
    text = str(value)
    cleaned = "".join(ch for ch in text if ch.isprintable()).strip()
    if not cleaned:
        return None
    return cleaned[:limit]


def _as_flag(raw: Optional[str], default: bool) -> bool:
    """Interpret an env-var string as a boolean, ``None`` meaning ``default``.

    Takes the already-read value rather than the variable name on purpose:
    the env catalog's AST scan (``shared/scaffold/introspect.py``) reads
    ``os.environ.get`` call sites, so a var read behind a helper would be
    invisible to it and fail ``test_no_stale_catalog_entries``.
    """
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSE_WORDS


def _as_categories(value: Any) -> Tuple[str, ...]:
    """Coerce a categories value to a tuple of slugs.

    Accepts the two shapes categories arrive in: a comma-separated string
    (the env-var form, which is also the wire form of the header) and any
    iterable of strings (the programmatic and JSON forms).  ``None`` and
    anything unrecognised become the empty tuple — a malformed category
    list should cost the listing, not the request.

    Entry-level cleaning (trim / drop empties / cap) happens in
    ``AppIdentity.__post_init__``, so every construction path shares it.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(value.split(","))
    if isinstance(value, Iterable):
        return tuple(str(entry) for entry in value)
    return ()


_framework_version_cache: Optional[str] = None


def framework_version() -> Optional[str]:
    """Version of the installed framework, or ``None`` when unknowable.

    Read from installed distribution metadata (the same source as the
    daemon's ``server_version``) rather than a literal, so it cannot drift
    from the package.  A source checkout that was never installed has no
    metadata; that is not an error — the version is simply omitted from
    attribution strings.
    """
    global _framework_version_cache
    if _framework_version_cache is None:
        try:
            from importlib.metadata import version as pkg_version
            _framework_version_cache = pkg_version(_FRAMEWORK_DISTRIBUTION)
        except Exception:  # noqa: BLE001 — metadata absent / unreadable
            _framework_version_cache = ""
    return _framework_version_cache or None


@dataclass(frozen=True)
class AppIdentity:
    """The application making requests through the framework.

    Immutable and sanitised at construction: ``__post_init__`` rewrites every
    field through :func:`_sanitise`, so an instance is always safe to render
    into an HTTP header regardless of where its values came from (env var,
    profile, a tenant name in a hosted product).

    Attributes:
        name: Display name of the application, e.g. ``"Acme Copilot"``.
            Defaults to :data:`FRAMEWORK_NAME` — an unconfigured deployment
            reports as jaato, exactly as it did before this type existed.
        url: The application's own site / repository.  Becomes the
            ``HTTP-Referer`` OpenRouter attributes rankings to; falls back to
            :data:`FRAMEWORK_URL` when the application supplies none.
        version: The application's version, not the framework's.  Optional;
            only :meth:`user_agent` uses it.
        powered_by: Whether attribution strings append ``(powered by jaato)``.
            ``True`` by default so the framework keeps the credit an
            integrator's app would otherwise take from it; set ``False`` for
            a white-labelled product.  Ignored when this *is* the framework
            identity — ``jaato (powered by jaato)`` helps nobody.
        categories: Marketplace categories the application claims, as a
            tuple of slugs — what an upstream app directory files it under
            (OpenRouter's ``X-OpenRouter-Categories`` today).  Empty by
            default, and NOT inherited from the framework: see
            :meth:`attribution_categories`.  Only lightly sanitised here;
            each consumer enforces its own taxonomy rules, because the legal
            slugs are the upstream's to define, not the framework's.
    """

    name: str = FRAMEWORK_NAME
    url: Optional[str] = None
    version: Optional[str] = None
    powered_by: bool = True
    categories: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # frozen dataclass: normalise through object.__setattr__.
        name = _sanitise(self.name, limit=MAX_NAME_LENGTH) or FRAMEWORK_NAME
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self, "url", _sanitise(self.url, limit=MAX_URL_LENGTH),
        )
        object.__setattr__(
            self, "version", _sanitise(self.version, limit=MAX_NAME_LENGTH),
        )
        object.__setattr__(self, "powered_by", bool(self.powered_by))
        # A tuple, so the dataclass stays frozen-and-hashable and a caller
        # cannot mutate an identity out from under a provider that kept it.
        object.__setattr__(self, "categories", tuple(
            cleaned
            for cleaned in (
                _sanitise(entry, limit=MAX_CATEGORY_LENGTH)
                for entry in (self.categories or ())
            )
            if cleaned
        ))

    # -- Identity ---------------------------------------------------------

    @property
    def is_framework(self) -> bool:
        """True when this identity is jaato itself rather than an app.

        Compared case-insensitively on the name alone: an integrator who sets
        ``JAATO_APP_NAME=jaato`` has named their app after the framework and
        gets the framework's presentation, which is the honest reading.
        """
        return self.name.casefold() == FRAMEWORK_NAME.casefold()

    # -- Attribution ------------------------------------------------------

    def attribution_title(self) -> str:
        """Display name for upstream app attribution.

        ``"Acme Copilot (powered by jaato)"`` for an application,
        ``"jaato"`` for the framework itself, and the bare app name when
        ``powered_by`` is off.  Consumed by OpenRouter's
        ``X-OpenRouter-Title`` header; suitable for any upstream that wants a
        human-readable caller name.
        """
        if self.is_framework or not self.powered_by:
            return self.name
        return f"{self.name} (powered by {FRAMEWORK_NAME})"

    def attribution_url(self) -> str:
        """Site URL for upstream app attribution.

        The application's own URL when it has one, else the framework's.
        Never empty: OpenRouter keys its app rankings on this value, so an
        app with no URL is attributed to jaato rather than to nothing.
        """
        return self.url or FRAMEWORK_URL

    def attribution_categories(self) -> Tuple[str, ...]:
        """Marketplace categories to attribute this application under.

        The application's own when it declared any; the framework's
        (:data:`FRAMEWORK_CATEGORIES`) only when this *is* the framework;
        empty otherwise.

        That last clause is the one worth stating out loud, because it is
        the opposite of :meth:`attribution_url`, which DOES hand a nameless
        app the framework's URL.  The asymmetry is deliberate:

        * a referer is what rankings key on, so an app with none is better
          served landing in jaato's row than in no row at all;
        * a category is a claim about *what the application is*, and once an
          app has told us it is not jaato, jaato's claim about itself does
          not transfer.  A Slack bot silently filed under "cli-agent" is
          worse than one filed nowhere — it mis-files the app and pollutes
          the directory for everyone reading it.

        So an application that wants a listing names its own categories.
        """
        if self.categories:
            return self.categories
        return FRAMEWORK_CATEGORIES if self.is_framework else ()

    def user_agent(self) -> str:
        """Conventional ``User-Agent`` string naming app and framework.

        ``Acme-Copilot/1.4.0 (powered by jaato/0.7.0)``; the version of
        either half is omitted when unknown, and the framework identity
        renders as plain ``jaato/0.7.0``.  Whitespace in the name becomes
        ``-`` so the product token stays a single UA token.

        Nothing in the framework sends this yet — it is the general form for
        providers and HTTP clients that want to identify the caller without
        re-deriving the attribution convention.
        """
        fw_version = framework_version()
        fw_token = (
            f"{FRAMEWORK_NAME}/{fw_version}" if fw_version else FRAMEWORK_NAME
        )
        product = self.name.replace(" ", "-")
        token = f"{product}/{self.version}" if self.version else product
        if self.is_framework:
            return fw_token
        if not self.powered_by:
            return token
        return f"{token} (powered by {fw_token})"

    # -- Serialisation ----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict form, for stamping onto ``ProviderConfig.extra``.

        Only set fields are emitted, so a round-trip through
        :meth:`from_dict` reproduces the same instance and a stamped identity
        stays small on the wire.
        """
        data: Dict[str, Any] = {"name": self.name, "powered_by": self.powered_by}
        if self.url:
            data["url"] = self.url
        if self.version:
            data["version"] = self.version
        if self.categories:
            # A list, not a tuple: this dict is JSON on the wire.
            data["categories"] = list(self.categories)
        return data

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "AppIdentity":
        """Rebuild from :meth:`to_dict` output (or any partial mapping).

        Unknown keys are ignored and missing keys take their defaults, so a
        newer producer talking to an older consumer degrades to the fields
        both understand rather than raising.
        """
        if not data:
            return AppIdentity()
        return cls(
            name=str(data.get("name") or FRAMEWORK_NAME),
            url=data.get("url"),
            version=data.get("version"),
            powered_by=bool(data.get("powered_by", True)),
            categories=_as_categories(data.get("categories")),
        )


#: The framework's own identity — the resolution default.
FRAMEWORK_IDENTITY = AppIdentity(
    name=FRAMEWORK_NAME, url=FRAMEWORK_URL, categories=FRAMEWORK_CATEGORIES,
)


def resolve_app_identity(
    overrides: Optional[Union["AppIdentity", Mapping[str, Any]]] = None,
) -> AppIdentity:
    """Resolve the application identity for this process.

    Precedence, highest first:

    1. ``overrides`` — an :class:`AppIdentity` (returned as-is) or a partial
       mapping whose set keys win over the environment.  This is the
       programmatic surface (``JaatoRuntime(app_identity=...)``) and the
       carrier for an identity stamped onto a provider config.
    2. :data:`ENV_APP_NAME` / :data:`ENV_APP_URL` / :data:`ENV_APP_VERSION` /
       :data:`ENV_APP_POWERED_BY` / :data:`ENV_APP_CATEGORIES`.
    3. :data:`FRAMEWORK_IDENTITY` — jaato's own name, URL and categories.

    Note that tier 2 is read per call rather than cached: the daemon overlays
    a session's ``env`` onto ``os.environ`` for the duration of a turn, so a
    value cached at import would be the wrong one for every session that
    overrides it.

    Args:
        overrides: Explicit identity or partial field mapping.

    Returns:
        A sanitised :class:`AppIdentity`; never ``None``.
    """
    if isinstance(overrides, AppIdentity):
        return overrides

    name = _sanitise(os.environ.get(ENV_APP_NAME), limit=MAX_NAME_LENGTH)  # env: display name of the application embedding jaato (app attribution)
    url = _sanitise(os.environ.get(ENV_APP_URL), limit=MAX_URL_LENGTH)  # env: the embedding application's site/repo URL (app attribution)
    version = _sanitise(os.environ.get(ENV_APP_VERSION), limit=MAX_NAME_LENGTH)  # env: the embedding application's own version string (app attribution)
    powered_by = _as_flag(
        os.environ.get(ENV_APP_POWERED_BY), True,  # env: append "(powered by jaato)" to app attribution (default true)
    )
    categories = _as_categories(
        os.environ.get(ENV_APP_CATEGORIES),  # env: comma-separated marketplace categories the application claims (app attribution)
    )

    fields: Dict[str, Any] = {
        "name": name,
        "url": url,
        "version": version,
        "powered_by": powered_by,
        "categories": categories,
    }
    for key in fields:
        if overrides and overrides.get(key) is not None:
            fields[key] = overrides[key]

    if not fields["name"]:
        # Nobody named an application: this is the framework itself, which
        # does have a URL and categories of its own.  An app that names
        # itself but gives no URL keeps ``url=None`` and is attributed to
        # FRAMEWORK_URL by attribution_url() — the distinction survives a
        # to_dict() round-trip; categories are NOT filled in the same way,
        # per attribution_categories().
        fields["name"] = FRAMEWORK_NAME
        fields["url"] = fields["url"] or FRAMEWORK_URL
        fields["categories"] = fields["categories"] or FRAMEWORK_CATEGORIES

    return AppIdentity(**fields)
