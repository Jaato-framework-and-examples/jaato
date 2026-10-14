"""A service names the CA bundle its TLS is verified against (#1671).

kbwiki's connectors reach corporate hosts whose certificates chain to a
private root.  ``call_service`` could reach them only by turning
verification off (``ssl_trusted`` / ``insecure=true``) or by trusting the
root daemon-wide (``SSL_CERT_FILE``, the system store).  ``ca_file`` in
``<service>/_service.yaml`` trusts it for that service alone.

These tests run a real HTTPS server on 127.0.0.1 whose certificate is
signed by a CA minted here, and drive the real plugin's ``call_service``.
"""

from __future__ import annotations

import datetime
import http.server
import ipaddress
import ssl
import threading
from pathlib import Path

import pytest
import yaml

from jaato_server.shared.tests.reversion import Reversion

_PLUGIN = "jaato-server/jaato_server/shared/plugins/service_connector/plugin.py"
_VALIDATE = "jaato-server/jaato_server/shared/scaffold/validate.py"

REVERSIONS = [
    Reversion(
        target=_PLUGIN,
        find="                ca_file=ca_file,\n            )",
        replace="                ca_file=None,\n            )",
        test="test_a_privately_signed_host_is_reached_through_its_bundle",
        because="the bundle is never handed to the client: system store only",
    ),
    Reversion(
        target=_PLUGIN,
        find="        if ca_file:\n            return True\n        if insecure:",
        replace="        if insecure:",
        test="test_ssl_trusted_does_not_switch_verification_off_beside_a_bundle",
        because="ssl_trusted beside a ca_file turns verification off again",
    ),
    Reversion(
        target=_PLUGIN,
        find="        if insecure:\n            return None, {",
        replace="        if False:\n            return None, {",
        test="test_insecure_is_refused_for_a_service_with_a_bundle",
        because="insecure=true quietly unverifies a service that names its root",
    ),
    Reversion(
        target=_VALIDATE,
        find="    out.extend(validate_services(str(ws)))\n",
        replace="",
        test="test_validate_reports_a_bundle_that_does_not_load",
        because="validate_workspace never asks about service bundles",
    ),
]


# ----------------------------------------------------------------- fixtures

def _mint_ca(name: str):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(subject).issuer_name(subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=30))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None),
                           critical=True)
            .add_extension(x509.KeyUsage(
                digital_signature=True, key_cert_sign=True, crl_sign=True,
                content_commitment=False, key_encipherment=False,
                data_encipherment=False, key_agreement=False,
                encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(
                key.public_key()), critical=False)
            .sign(key, hashes.SHA256()))
    return key, cert


def _mint_leaf(ca_key, ca_cert):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,
                                                        "127.0.0.1")]))
            .issuer_name(ca_cert.subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=30))
            .add_extension(x509.SubjectAlternativeName(
                [x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
                critical=False)
            .add_extension(x509.ExtendedKeyUsage(
                [ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(
                ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256()))
    return key, cert


def _pem(cert) -> bytes:
    from cryptography.hazmat.primitives import serialization
    return cert.public_bytes(serialization.Encoding.PEM)


def _key_pem(key) -> bytes:
    from cryptography.hazmat.primitives import serialization
    return key.private_bytes(serialization.Encoding.PEM,
                             serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def tls_host(tmp_path_factory):
    """A 127.0.0.1 HTTPS server signed by a private root, plus a stranger CA."""
    pytest.importorskip("cryptography")
    d = tmp_path_factory.mktemp("pki")
    ca_key, ca_cert = _mint_ca("private-root")
    _, other_cert = _mint_ca("some-other-root")
    leaf_key, leaf_cert = _mint_leaf(ca_key, ca_cert)
    (d / "root.pem").write_bytes(_pem(ca_cert))
    (d / "other.pem").write_bytes(_pem(other_cert))
    (d / "leaf.pem").write_bytes(_pem(leaf_cert))
    (d / "leaf.key").write_bytes(_key_pem(leaf_key))

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(d / "leaf.pem"), str(d / "leaf.key"))
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield {"url": f"https://127.0.0.1:{srv.server_address[1]}",
           "root": (d / "root.pem").read_bytes(),
           "other": (d / "other.pem").read_bytes()}
    srv.shutdown()


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "SSL_CERT_DIR"):
        monkeypatch.delenv(var, raising=False)
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


def _define(ws: Path, name: str, url: str, pem: bytes = None,
            ca_file: str = None, **extra) -> Path:
    sdir = ws / ".jaato" / "services" / name
    sdir.mkdir(parents=True)
    doc = {"name": name, "base_url": url, "proxy_bypass": True, **extra}
    if pem is not None:
        (sdir / "root.pem").write_bytes(pem)
        doc["ca_file"] = ca_file or "root.pem"
    elif ca_file is not None:
        doc["ca_file"] = ca_file
    (sdir / "_service.yaml").write_text(yaml.safe_dump(doc))
    return sdir


def _plugin(ws: Path):
    from jaato_server.shared.plugins.service_connector.plugin import (
        ServiceConnectorPlugin)
    p = ServiceConnectorPlugin()
    p.set_workspace_path(str(ws))
    p.initialize({})
    return p


def _call(p, service: str, **extra):
    out = p._execute_call_service(
        {"service": service, "method": "GET", "path": "/x", **extra})
    return out[1] if isinstance(out, tuple) else out


# -------------------------------------------------------------------- tests

def test_a_privately_signed_host_is_reached_through_its_bundle(tls_host, workspace):
    _define(workspace, "lda", tls_host["url"], tls_host["root"])
    result = _call(_plugin(workspace), "lda")
    assert result.get("status") == 200, result


def test_the_same_host_without_a_bundle_still_fails_verification(tls_host, workspace):
    """The root is trusted for the service that names it, not for the host."""
    _define(workspace, "lda", tls_host["url"], tls_host["root"])
    _define(workspace, "plain", tls_host["url"])
    p = _plugin(workspace)
    assert _call(p, "lda").get("status") == 200
    plain = _call(p, "plain")
    assert plain.get("ssl_error") is True, plain


def test_the_bundle_is_the_only_trust_root(tls_host, workspace):
    from jaato_server.shared.plugins.service_connector.ca_bundle import (
        ssl_context_for)
    sdir = _define(workspace, "lda", tls_host["url"], tls_host["root"])
    ctx = ssl_context_for(sdir / "root.pem")
    assert len(ctx.get_ca_certs()) == 1      # no system store loaded
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname


def test_a_wrong_bundle_fails_and_does_not_suggest_insecure(tls_host, workspace):
    _define(workspace, "lda", tls_host["url"], tls_host["other"])
    result = _call(_plugin(workspace), "lda")
    assert result.get("ssl_error") is True, result
    assert result.get("ca_file")
    assert "insecure=true is not available" in result["hint"]


def test_ssl_trusted_does_not_switch_verification_off_beside_a_bundle(
        tls_host, workspace):
    _define(workspace, "lda", tls_host["url"], tls_host["other"],
            ssl_trusted=True)
    result = _call(_plugin(workspace), "lda")
    assert result.get("ssl_error") is True, result


def test_insecure_is_refused_for_a_service_with_a_bundle(tls_host, workspace):
    _define(workspace, "lda", tls_host["url"], tls_host["other"])
    p = _plugin(workspace)
    result = _call(p, "lda", insecure=True)
    assert "status" not in result, result
    assert "insecure=true is refused" in result["error"]
    # ...and nothing was persisted that would unverify it later
    assert not p._schema_store.load_service_config("lda").ssl_trusted


def test_a_missing_bundle_refuses_the_call_naming_it(tls_host, workspace):
    _define(workspace, "lda", tls_host["url"], ca_file="nope.pem")
    result = _call(_plugin(workspace), "lda")
    assert "status" not in result, result
    assert "nope.pem" in result["error"]


def test_a_relative_bundle_stays_pinned_when_the_definition_moves(
        tls_host, workspace):
    """A user-tier service saved into the workspace keeps finding its bundle."""
    from jaato_server.shared.plugins.service_connector.schema_store import (
        SchemaStore)
    home = Path.home()
    _define(home, "lda", tls_host["url"], tls_host["root"])
    store = SchemaStore(str(workspace))
    cfg = store.load_service_config("lda")
    assert cfg.ca_file == "root.pem"
    store.save_service_config(cfg)
    saved = yaml.safe_load(
        (workspace / ".jaato/services/lda/_service.yaml").read_text())
    assert saved["ca_file"] == str(home / ".jaato/services/lda/root.pem")
    assert _call(_plugin(workspace), "lda").get("status") == 200


def test_validate_reports_a_bundle_that_does_not_load(workspace):
    from jaato_server.shared.scaffold.validate import validate_workspace
    _define(workspace, "lda", "https://127.0.0.1:1", ca_file="nope.pem",
            ssl_trusted=True)
    codes = {d.code for d in validate_workspace(str(workspace))}
    assert "service_ca_file_invalid" in codes
    assert "service_ca_file_with_ssl_trusted" in codes


def test_validate_warns_about_a_bundle_confinement_cannot_read(
        tls_host, workspace, tmp_path):
    from jaato_server.shared.scaffold.validate import validate_services
    outside = tmp_path / "elsewhere.pem"
    outside.write_bytes(tls_host["root"])
    _define(workspace, "lda", tls_host["url"], ca_file=str(outside))
    codes = {d.code for d in validate_services(str(workspace))}
    assert codes == {"service_ca_file_unconfined"}


def test_explain_services_names_the_bundle(tls_host, workspace):
    from jaato_server.shared.scaffold.explain import services
    sdir = _define(workspace, "lda", tls_host["url"], tls_host["root"])
    data, text = services(str(workspace))
    row = next(k for k in data["known"] if k["service"] == "lda")
    assert row["ca_file"] == str(sdir / "root.pem")
    assert "ca_file" in text and str(sdir / "root.pem") in text
