Param(
    [Parameter(Mandatory=$true)][string]$RemoteUrl,
    [string]$Branch = "main",
    [string]$CommitMessage = "chore: update",
    [switch]$Force
)

Write-Host "Running git push helper..."

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "git is not installed or not in PATH."
    exit 1
}

if (-not (Test-Path ".git")) {
    Write-Host "No .git found — initializing repository."
    git init
}

$existing = & git remote get-url origin 2>$null
if ($LASTEXITCODE -ne 0 -or $existing -ne $RemoteUrl) {
    if ($existing) {
        Write-Host "Updating existing remote 'origin'."
        git remote remove origin
    } else {
        Write-Host "Adding remote 'origin'."
    }
    git remote add origin $RemoteUrl
}

git add .
$commitOutput = git commit -m $CommitMessage 2>&1
if ($LASTEXITCODE -ne 0) {
    if ($commitOutput -match "nothing to commit") {
        Write-Host "No changes to commit."
    } else {
        Write-Host $commitOutput
    }
} else {
    Write-Host "Committed changes."
}

if ($Force) {
    git push -u origin $Branch --force
} else {
    git push -u origin $Branch
}

exit $LASTEXITCODE
