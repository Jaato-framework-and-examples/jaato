"""Tests for _validate_template_index() in the template plugin."""

import pytest

from shared.plugins.template.plugin import TemplatePlugin


@pytest.fixture
def plugin():
    p = TemplatePlugin()
    p.initialize()
    return p


class TestValidateTemplateIndex:
    """Tests for the template index validator."""

    def test_valid_index(self, plugin):
        data = {
            "generated_at": "2026-02-11T12:00:00",
            "template_count": 2,
            "templates": {
                "Entity.java.tpl": {
                    "name": "Entity.java.tpl",
                    "source_path": "templates/Entity.java.tpl",
                    "syntax": "mustache",
                    "variables": ["Entity", "basePackage"],
                    "origin": "standalone",
                },
                "Service.java.tpl": {
                    "name": "Service.java.tpl",
                    "source_path": "templates/Service.java.tpl",
                    "syntax": "jinja2",
                    "variables": ["service_name"],
                    "origin": "embedded",
                },
            },
        }
        is_valid, errors, warnings = plugin._validate_template_index(data)
        assert is_valid is True
        assert errors == []
        assert warnings == []

    def test_missing_generated_at(self, plugin):
        data = {"template_count": 0, "templates": {}}
        is_valid, errors, warnings = plugin._validate_template_index(data)
        assert is_valid is False
        assert any("'generated_at' is required" in e for e in errors)

    def test_generated_at_not_string(self, plugin):
        data = {"generated_at": 12345, "template_count": 0, "templates": {}}
        is_valid, errors, warnings = plugin._validate_template_index(data)
        assert is_valid is False
        assert any("'generated_at' must be a string" in e for e in errors)

    def test_missing_template_count(self, plugin):
        data = {"generated_at": "2026-01-01", "templates": {}}
        is_valid, errors, warnings = plugin._validate_template_index(data)
        assert is_valid is False
        assert any("'template_count' is required" in e for e in errors)

    def test_template_count_not_int(self, plugin):
        data = {"generated_at": "2026-01-01", "template_count": "two", "templates": {}}
        is_valid, errors, warnings = plugin._validate_template_index(data)
        assert is_valid is False
        assert any("'template_count' must be an integer" in e for e in errors)

    def test_missing_templates(self, plugin):
        data = {"generated_at": "2026-01-01", "template_count": 0}
        is_valid, errors, warnings = plugin._validate_template_index(data)
        assert is_valid is False
        assert any("'templates' is required" in e for e in errors)

    def test_templates_not_dict(self, plugin):
        data = {"generated_at": "2026-01-01", "template_count": 0, "templates": []}
        is_valid, errors, warnings = plugin._validate_template_index(data)
        assert is_valid is False
        assert any("'templates' must be an object" in e for e in errors)

    def test_count_mismatch_warning(self, plugin):
        data = {
            "generated_at": "2026-01-01",
            "template_count": 5,
            "templates": {
                "a.tpl": {
                    "name": "a.tpl",
                    "source_path": "/tmp/a.tpl",
                    "syntax": "jinja2",
                    "origin": "standalone",
                }
            },
        }
        is_valid, errors, warnings = plugin._validate_template_index(data)
        assert is_valid is True
        assert any("template_count (5) does not match" in w for w in warnings)

    def test_entry_missing_name(self, plugin):
        data = {
            "generated_at": "2026-01-01",
            "template_count": 1,
            "templates": {
                "bad.tpl": {
                    "source_path": "/tmp/bad.tpl",
                    "syntax": "jinja2",
                    "origin": "standalone",
                }
            },
        }
        is_valid, errors, warnings = plugin._validate_template_index(data)
        assert is_valid is False
        assert any("'name' is required" in e for e in errors)

    def test_entry_missing_source_path(self, plugin):
        data = {
            "generated_at": "2026-01-01",
            "template_count": 1,
            "templates": {
                "bad.tpl": {
                    "name": "bad.tpl",
                    "syntax": "jinja2",
                    "origin": "standalone",
                }
            },
        }
        is_valid, errors, warnings = plugin._validate_template_index(data)
        assert is_valid is False
        assert any("'source_path' is required" in e for e in errors)

    def test_entry_invalid_syntax(self, plugin):
        data = {
            "generated_at": "2026-01-01",
            "template_count": 1,
            "templates": {
                "bad.tpl": {
                    "name": "bad.tpl",
                    "source_path": "/tmp/bad.tpl",
                    "syntax": "handlebars",
                    "origin": "standalone",
                }
            },
        }
        is_valid, errors, warnings = plugin._validate_template_index(data)
        assert is_valid is False
        assert any("invalid syntax 'handlebars'" in e for e in errors)

    def test_entry_invalid_origin(self, plugin):
        data = {
            "generated_at": "2026-01-01",
            "template_count": 1,
            "templates": {
                "bad.tpl": {
                    "name": "bad.tpl",
                    "source_path": "/tmp/bad.tpl",
                    "syntax": "jinja2",
                    "origin": "discovered",
                }
            },
        }
        is_valid, errors, warnings = plugin._validate_template_index(data)
        assert is_valid is False
        assert any("invalid origin 'discovered'" in e for e in errors)

    def test_entry_variables_not_list(self, plugin):
        data = {
            "generated_at": "2026-01-01",
            "template_count": 1,
            "templates": {
                "bad.tpl": {
                    "name": "bad.tpl",
                    "source_path": "/tmp/bad.tpl",
                    "syntax": "jinja2",
                    "origin": "standalone",
                    "variables": "not-a-list",
                }
            },
        }
        is_valid, errors, warnings = plugin._validate_template_index(data)
        assert is_valid is False
        assert any("'variables' must be an array" in e for e in errors)

    def test_entry_variables_not_strings(self, plugin):
        data = {
            "generated_at": "2026-01-01",
            "template_count": 1,
            "templates": {
                "bad.tpl": {
                    "name": "bad.tpl",
                    "source_path": "/tmp/bad.tpl",
                    "syntax": "jinja2",
                    "origin": "standalone",
                    "variables": ["ok", 42],
                }
            },
        }
        is_valid, errors, warnings = plugin._validate_template_index(data)
        assert is_valid is False
        assert any("'variables' must contain only strings" in e for e in errors)

    def test_not_a_dict(self, plugin):
        is_valid, errors, warnings = plugin._validate_template_index("not a dict")
        assert is_valid is False
        assert any("JSON object" in e for e in errors)

    def test_entry_not_a_dict(self, plugin):
        data = {
            "generated_at": "2026-01-01",
            "template_count": 1,
            "templates": {"bad": "not-an-object"},
        }
        is_valid, errors, warnings = plugin._validate_template_index(data)
        assert is_valid is False
        assert any("must be an object" in e for e in errors)

    def test_missing_syntax(self, plugin):
        data = {
            "generated_at": "2026-01-01",
            "template_count": 1,
            "templates": {
                "no-syntax.tpl": {
                    "name": "no-syntax.tpl",
                    "source_path": "/tmp/no-syntax.tpl",
                    "origin": "standalone",
                }
            },
        }
        is_valid, errors, warnings = plugin._validate_template_index(data)
        assert is_valid is False
        assert any("'syntax' is required" in e for e in errors)

    def test_missing_origin(self, plugin):
        data = {
            "generated_at": "2026-01-01",
            "template_count": 1,
            "templates": {
                "no-origin.tpl": {
                    "name": "no-origin.tpl",
                    "source_path": "/tmp/no-origin.tpl",
                    "syntax": "jinja2",
                }
            },
        }
        is_valid, errors, warnings = plugin._validate_template_index(data)
        assert is_valid is False
        assert any("'origin' is required" in e for e in errors)
