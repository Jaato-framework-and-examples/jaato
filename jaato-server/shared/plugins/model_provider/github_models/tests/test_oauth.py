"""Tests for GitHub OAuth token management and refresh."""

import time
import pytest
from unittest.mock import patch, MagicMock

from ..oauth import (
    CopilotToken,
    OAuthTokens,
    get_stored_access_token,
    load_copilot_token,
    load_tokens,
    save_copilot_token,
    clear_copilot_token,
)


class TestCopilotToken:
    """Tests for CopilotToken expiration and refresh logic."""

    def test_is_expired_returns_false_when_valid(self):
        """Token should not be expired when expires_at is in the future."""
        token = CopilotToken(
            token="test_token",
            expires_at=int(time.time()) + 3600,  # 1 hour from now
        )
        assert token.is_expired() is False

    def test_is_expired_returns_true_when_past_expiry(self):
        """Token should be expired when expires_at is in the past."""
        token = CopilotToken(
            token="test_token",
            expires_at=int(time.time()) - 60,  # 1 minute ago
        )
        assert token.is_expired() is True

    def test_needs_refresh_returns_false_when_plenty_of_time(self):
        """Token should not need refresh when more than 5 minutes remain."""
        token = CopilotToken(
            token="test_token",
            expires_at=int(time.time()) + 3600,  # 1 hour from now
        )
        assert token.needs_refresh() is False

    def test_needs_refresh_returns_true_when_less_than_5_minutes(self):
        """Token should need refresh when less than 5 minutes remain."""
        token = CopilotToken(
            token="test_token",
            expires_at=int(time.time()) + 200,  # ~3 minutes from now
        )
        assert token.needs_refresh() is True

    def test_needs_refresh_uses_refresh_in_value(self):
        """Token should use refresh_in value if provided."""
        token = CopilotToken(
            token="test_token",
            expires_at=int(time.time()) + 600,  # 10 minutes from now
            refresh_in=900,  # Refresh when less than 15 min remaining
        )
        # With refresh_in=900, should refresh when time < expires_at - 900
        # 600 seconds remaining < 900 threshold, so should need refresh
        assert token.needs_refresh() is True

    def test_needs_refresh_returns_true_when_expired(self):
        """Token should need refresh when already expired."""
        token = CopilotToken(
            token="test_token",
            expires_at=int(time.time()) - 60,  # Already expired
        )
        assert token.needs_refresh() is True


class TestGetStoredAccessToken:
    """Tests for get_stored_access_token() proactive refresh."""

    @patch('shared.plugins.model_provider.github_models.oauth.load_copilot_token')
    @patch('shared.plugins.model_provider.github_models.oauth._oauth_trace')
    def test_returns_token_when_not_needing_refresh(self, mock_trace, mock_load):
        """Should return existing token when it doesn't need refresh."""
        valid_token = CopilotToken(
            token="valid_token",
            expires_at=int(time.time()) + 3600,  # 1 hour from now
        )
        mock_load.return_value = valid_token

        result = get_stored_access_token()

        assert result == "valid_token"
        mock_load.assert_called_once()

    @patch('shared.plugins.model_provider.github_models.oauth.save_copilot_token')
    @patch('shared.plugins.model_provider.github_models.oauth.exchange_oauth_for_copilot_token')
    @patch('shared.plugins.model_provider.github_models.oauth.load_tokens')
    @patch('shared.plugins.model_provider.github_models.oauth.load_copilot_token')
    @patch('shared.plugins.model_provider.github_models.oauth._oauth_trace')
    def test_refreshes_token_when_needs_refresh(
        self, mock_trace, mock_load_copilot, mock_load_oauth, mock_exchange, mock_save
    ):
        """Should refresh token when needs_refresh() returns True."""
        # Token that will expire in 2 minutes (needs refresh)
        expiring_token = CopilotToken(
            token="expiring_token",
            expires_at=int(time.time()) + 120,  # 2 minutes from now
        )
        mock_load_copilot.return_value = expiring_token

        # OAuth token to use for exchange
        mock_load_oauth.return_value = OAuthTokens(access_token="oauth_token")

        # New token from exchange
        new_token = CopilotToken(
            token="fresh_token",
            expires_at=int(time.time()) + 3600,
        )
        mock_exchange.return_value = new_token

        result = get_stored_access_token()

        assert result == "fresh_token"
        mock_exchange.assert_called_once_with("oauth_token")
        mock_save.assert_called_once_with(new_token)

    @patch('shared.plugins.model_provider.github_models.oauth.load_tokens')
    @patch('shared.plugins.model_provider.github_models.oauth.load_copilot_token')
    @patch('shared.plugins.model_provider.github_models.oauth._oauth_trace')
    def test_returns_none_when_no_oauth_token(
        self, mock_trace, mock_load_copilot, mock_load_oauth
    ):
        """Should return None when no OAuth token available for refresh."""
        # No copilot token (or expired)
        mock_load_copilot.return_value = None
        mock_load_oauth.return_value = None

        result = get_stored_access_token()

        assert result is None

    @patch('shared.plugins.model_provider.github_models.oauth.exchange_oauth_for_copilot_token')
    @patch('shared.plugins.model_provider.github_models.oauth.load_tokens')
    @patch('shared.plugins.model_provider.github_models.oauth.load_copilot_token')
    @patch('shared.plugins.model_provider.github_models.oauth._oauth_trace')
    def test_returns_none_when_exchange_fails(
        self, mock_trace, mock_load_copilot, mock_load_oauth, mock_exchange
    ):
        """Should return None when token exchange fails."""
        # No copilot token
        mock_load_copilot.return_value = None
        mock_load_oauth.return_value = OAuthTokens(access_token="oauth_token")

        # Exchange fails
        mock_exchange.side_effect = RuntimeError("Exchange failed")

        result = get_stored_access_token()

        assert result is None


class TestCopilotClientTokenRefresh:
    """Tests for CopilotClient token refresh behavior."""

    @patch('shared.plugins.model_provider.github_models.copilot_client.CopilotClient._create_session')
    def test_ensure_valid_token_updates_when_refreshed(self, mock_session):
        """_ensure_valid_token should update token when get_stored_access_token returns new one."""
        from ..copilot_client import CopilotClient

        mock_session.return_value = MagicMock()
        client = CopilotClient("old_token")

        with patch(
            'shared.plugins.model_provider.github_models.oauth.get_stored_access_token',
            return_value="new_token"
        ):
            with patch('shared.plugins.model_provider.github_models.oauth._oauth_trace'):
                client._ensure_valid_token()

        assert client._token == "new_token"

    @patch('shared.plugins.model_provider.github_models.copilot_client.CopilotClient._create_session')
    def test_force_token_refresh_clears_and_exchanges(self, mock_session):
        """_force_token_refresh should clear copilot token and get fresh one."""
        from ..copilot_client import CopilotClient

        mock_session.return_value = MagicMock()
        client = CopilotClient("old_token")

        with patch('shared.plugins.model_provider.github_models.oauth.clear_copilot_token') as mock_clear:
            with patch(
                'shared.plugins.model_provider.github_models.oauth.get_stored_access_token',
                return_value="fresh_token"
            ):
                with patch('shared.plugins.model_provider.github_models.oauth._oauth_trace'):
                    client._force_token_refresh()

        mock_clear.assert_called_once()
        assert client._token == "fresh_token"

    @patch('shared.plugins.model_provider.github_models.copilot_client.CopilotClient._create_session')
    def test_force_token_refresh_keeps_old_token_on_failure(self, mock_session):
        """_force_token_refresh should keep old token if refresh returns None."""
        from ..copilot_client import CopilotClient

        mock_session.return_value = MagicMock()
        client = CopilotClient("old_token")

        with patch('shared.plugins.model_provider.github_models.oauth.clear_copilot_token'):
            with patch(
                'shared.plugins.model_provider.github_models.oauth.get_stored_access_token',
                return_value=None  # Refresh failed
            ):
                with patch('shared.plugins.model_provider.github_models.oauth._oauth_trace'):
                    client._force_token_refresh()

        assert client._token == "old_token"
