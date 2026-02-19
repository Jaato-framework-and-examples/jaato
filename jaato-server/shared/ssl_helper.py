import os
from typing import Optional, Iterable, Tuple

_OPTIONAL_CA_VARS = ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE")

def normalize_ca_env_vars(prefer_order: Optional[Iterable[str]] = None) -> None:
    """Normalize optional CA env vars by expanding user paths and making them absolute.

    This function only mutates environment variables; it does not validate file existence
    unless `ENV_VALIDATE_CA` is set to a truthy value in the environment.
    """
    for var in (prefer_order or _OPTIONAL_CA_VARS):
        val = os.environ.get(var)
        if not val:
            continue
        expanded = os.path.abspath(os.path.expanduser(val))
        os.environ[var] = expanded

    # Optional validation (opt-in)
    if os.environ.get('ENV_VALIDATE_CA', '').lower() in ('1', 'true', 'yes'):
        for var in (prefer_order or _OPTIONAL_CA_VARS):
            val = os.environ.get(var)
            if not val:
                continue
            if not os.path.isfile(val):
                print(f'[env][warn] {var} points to missing file: {val}')

def active_cert_bundle(prefer_order: Optional[Iterable[str]] = None, verbose: bool = False) -> Optional[str]:
    """Return the active CA bundle path, preferring variables in `prefer_order`.

    If verbose, prints which variable is used. Returns None if none are set.
    """
    for var in (prefer_order or _OPTIONAL_CA_VARS):
        val = os.environ.get(var)
        if val:
            if verbose:
                print(f'[env] Using custom CA bundle via {var}: {val}')
            return val
    return None

__all__ = [
    'normalize_ca_env_vars',
    'active_cert_bundle',
]
import os
import os.path
import ssl
from typing import Optional

def log_ssl_guidance(prefix: str, exc: Exception, silent: bool = False, pre_count: bool = False) -> None:
    """Log standardized SSL certificate verification failure guidance.
    prefix: short context string ('Pre-count' or 'Generate').
    exc: the original exception.
    silent: if True, suppress printing.
    pre_count: whether this occurred during the count_tokens phase.
    """
    if silent:
        return
    bundle = os.environ.get('REQUESTS_CA_BUNDLE') or os.environ.get('SSL_CERT_FILE')
    print(f'[SSL] {prefix} certificate verification failed.')
    print(f'[SSL] Exception: {str(exc)}')
    if bundle:
        exists = os.path.isfile(bundle)
        print(f'[SSL] CA bundle: {bundle} (exists={exists})')
        if exists:
            try:
                with open(bundle, 'r', encoding='utf-8', errors='ignore') as f:
                    head = ''.join(f.readlines()[:5])
                if 'BEGIN CERTIFICATE' not in head:
                    print('[SSL][warn] Bundle head lacks PEM header; confirm PEM encoding.')
            except Exception:
                pass
    else:
        print('[SSL] No REQUESTS_CA_BUNDLE / SSL_CERT_FILE set. Set one to combined certifi + corporate root PEM.')
    print('[SSL] Guidance:')
    print('  1. Export corporate/Zscaler root (and intermediates if any) as Base64 PEM.')
    print('  2. Append to certifi bundle creating a combined file.')
    print('  3. Set REQUESTS_CA_BUNDLE (and optionally SSL_CERT_FILE) before starting Python.')
    if pre_count:
        print('[SSL] Aborting before model.generate_content; fix trust chain then retry.')
    else:
        print('[SSL] Aborting call; fix trust chain then retry.')


def is_ssl_cert_failure(exc: Exception) -> bool:
    text = str(exc)
    return isinstance(exc, ssl.SSLError) or 'CERTIFICATE_VERIFY_FAILED' in text

__all__ = ['log_ssl_guidance', 'is_ssl_cert_failure']
