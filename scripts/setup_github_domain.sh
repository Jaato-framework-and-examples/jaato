#!/usr/bin/env bash
set -e

echo "=== GitHub Domain Setup for jaato ==="

# Check if gh CLI is installed
if command -v gh &> /dev/null; then
    echo "[OK] GitHub CLI (gh) is already installed: $(gh --version | head -1)"
else
    echo "[INSTALL] Installing GitHub CLI..."
    # Detect OS and install
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Ubuntu/Debian
        curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
        sudo apt update && sudo apt install gh -y
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install gh
    else
        echo "[ERROR] Unsupported OS. Please install gh manually: https://cli.github.com/"
        exit 1
    fi
fi

# Check gh authentication
if gh auth status &> /dev/null; then
    echo "[OK] GitHub CLI is authenticated"
else
    echo "[AUTH] Please authenticate with GitHub..."
    gh auth login
fi

# Check if github-mcp-server is installed
if command -v github-mcp-server &> /dev/null; then
    echo "[OK] github-mcp-server is already installed"
else
    echo "[INSTALL] Installing github-mcp-server..."
    npm install -g @modelcontextprotocol/server-github
fi

# Check for GITHUB_TOKEN in .env
if grep -q "GITHUB_TOKEN=" .env 2>/dev/null; then
    echo "[OK] GITHUB_TOKEN found in .env"
else
    echo "[SETUP] Creating GitHub token..."
    echo "Please create a Personal Access Token at: https://github.com/settings/tokens"
    echo "Required scopes: repo, read:org"
    read -p "Enter your GitHub token: " token
    echo "GITHUB_TOKEN=${token}" >> .env
    echo "[OK] GITHUB_TOKEN added to .env"
fi

echo ""
echo "=== Setup Complete ==="
echo "You can now run: .venv/bin/python cli_vs_mcp/cli_mcp_harness.py --domain github ..."
