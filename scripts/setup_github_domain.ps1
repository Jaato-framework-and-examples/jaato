# setup_github_domain.ps1
# GitHub Domain Setup for jaato (Windows)

Write-Host "=== GitHub Domain Setup for jaato ===" -ForegroundColor Cyan

# Check gh CLI
if (Get-Command gh -ErrorAction SilentlyContinue) {
    $ghVersion = gh --version | Select-Object -First 1
    Write-Host "[OK] GitHub CLI (gh) is installed: $ghVersion" -ForegroundColor Green
} else {
    Write-Host "[INSTALL] Installing GitHub CLI via winget..." -ForegroundColor Yellow
    winget install --id GitHub.cli
}

# Check authentication
$authStatus = gh auth status 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] GitHub CLI is authenticated" -ForegroundColor Green
} else {
    Write-Host "[AUTH] Please authenticate..." -ForegroundColor Yellow
    gh auth login
}

# Check github-mcp-server
if (Get-Command github-mcp-server -ErrorAction SilentlyContinue) {
    Write-Host "[OK] github-mcp-server is installed" -ForegroundColor Green
} else {
    Write-Host "[INSTALL] Installing github-mcp-server..." -ForegroundColor Yellow
    npm install -g @modelcontextprotocol/server-github
}

# Check .env for GITHUB_TOKEN
if (Test-Path .env) {
    if (Select-String -Path .env -Pattern "GITHUB_TOKEN=" -Quiet) {
        Write-Host "[OK] GITHUB_TOKEN found in .env" -ForegroundColor Green
    } else {
        Write-Host "[SETUP] Add GITHUB_TOKEN to .env" -ForegroundColor Yellow
        Write-Host "Please create a Personal Access Token at: https://github.com/settings/tokens"
        Write-Host "Required scopes: repo, read:org"
        $token = Read-Host "Enter your GitHub token"
        Add-Content -Path .env -Value "GITHUB_TOKEN=$token"
        Write-Host "[OK] GITHUB_TOKEN added to .env" -ForegroundColor Green
    }
} else {
    Write-Host "[SETUP] Creating .env file with GITHUB_TOKEN..." -ForegroundColor Yellow
    Write-Host "Please create a Personal Access Token at: https://github.com/settings/tokens"
    Write-Host "Required scopes: repo, read:org"
    $token = Read-Host "Enter your GitHub token"
    "GITHUB_TOKEN=$token" | Out-File -FilePath .env -Encoding utf8
    Write-Host "[OK] .env created with GITHUB_TOKEN" -ForegroundColor Green
}

Write-Host "`n=== Setup Complete ===" -ForegroundColor Cyan
Write-Host "You can now run: .venv\Scripts\python cli_vs_mcp\cli_mcp_harness.py --domain github ..."
