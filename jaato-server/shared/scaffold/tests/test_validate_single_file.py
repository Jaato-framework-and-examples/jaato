"""Scaffold: validating a STANDALONE profile file actually checks it.

Regression guard. ``jaato-scaffold validate <file.yaml>`` for a profile OUTSIDE
the canonical ``<ws>/.jaato/profiles/`` layout used to resolve to a bogus
workspace where ``discover_profiles`` found nothing — so the per-profile checks
never ran and the file was silently reported "valid — no findings" (a false
pass). ``validate_profile_file`` loads and checks the file directly.
"""
import textwrap

from shared.scaffold.validate import validate_profile_file


def _write(tmp_path, body):
    p = tmp_path / "web-fetch-secure.yaml"
    p.write_text(textwrap.dedent(body))
    return str(p)


_GOOD = """\
    name: web-fetch-secure
    provider: anthropic
    model: claude-sonnet-4-6
    plugins: [web_fetch]
    plugin_configs:
      web_fetch:
        allow_url_var_expansion: false
        block_private_networks: true
        secret_host_bindings:
          GITHUB_TOKEN: ["api.github.com"]
"""


def _unknown_knobs(diags):
    return [d.where for d in diags if d.code == "unknown_knob"]


def test_standalone_valid_profile_has_no_unknown_knobs(tmp_path):
    diags = validate_profile_file(_write(tmp_path, _GOOD))
    assert _unknown_knobs(diags) == [], diags


def test_standalone_typo_knob_is_caught(tmp_path):
    # The regression: previously silently passed; now flagged.
    body = _GOOD.replace("allow_url_var_expansion:", "allow_url_var_expansionX:")
    diags = validate_profile_file(_write(tmp_path, body))
    hits = _unknown_knobs(diags)
    assert any("allow_url_var_expansionX" in w for w in hits), diags
    # It's a warn (plugin schemas may be incomplete), not a hard error.
    assert all(d.severity == "warn" for d in diags
               if d.code == "unknown_knob")


def test_standalone_unknown_plugin_is_flagged(tmp_path):
    body = _GOOD.replace("plugins: [web_fetch]", "plugins: [web_fetch, made_up_plugin]")
    diags = validate_profile_file(_write(tmp_path, body))
    assert any(d.code == "unknown_plugin" and "made_up_plugin" in d.message
               for d in diags), diags


def test_non_profile_file_reports_parse_error(tmp_path):
    p = tmp_path / "notaprofile.yaml"
    p.write_text("- just\n- a\n- list\n")   # not a mapping
    diags = validate_profile_file(str(p))
    assert diags and diags[0].code == "parse_error"
