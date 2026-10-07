# Run from the PyTAG repository root: . .\scripts\setup_pandemic.ps1
# Dot sourcing keeps PYTAG_JAR_PATH and the TAG data working directory in this shell.
$ErrorActionPreference = 'Stop'

function Assert-CommandSucceeded([string] $action) {
    if ($LASTEXITCODE -ne 0) {
        throw "$action failed (exit code $LASTEXITCODE)."
    }
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$tagRoot = Join-Path $repoRoot 'pytag\TabletopGames'
$tagCommit = 'a5b8608cf53239b254f4e4a4f1b943c39794ddef'
$patchPath = Join-Path $repoRoot 'patches\pandemic-tag.patch'
$venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
$jarPath = Join-Path $tagRoot 'target\TAG-pytag.jar'

foreach ($tool in @('py', 'git', 'mvn')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "$tool is required on PATH. Install Python 3, Git, Maven, and Java 21 first."
    }
}
if (-not (Test-Path -LiteralPath $patchPath -PathType Leaf)) {
    throw "Missing TAG patch: $patchPath"
}

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    & py -3 -m venv (Join-Path $repoRoot '.venv')
    Assert-CommandSucceeded 'Create Python virtual environment'
}
& $venvPython -m pip install -e $repoRoot
Assert-CommandSucceeded 'Install PyTAG'

if (-not (Test-Path -LiteralPath $tagRoot)) {
    & git clone --no-checkout https://github.com/GAIGResearch/TabletopGames.git $tagRoot
    Assert-CommandSucceeded 'Clone TAG'
    & git -C $tagRoot checkout --detach $tagCommit
    Assert-CommandSucceeded 'Check out pinned TAG commit'
} elseif (-not (Test-Path -LiteralPath (Join-Path $tagRoot '.git'))) {
    throw "TAG path exists but is not a Git checkout: $tagRoot"
}

$actualCommit = (& git -C $tagRoot rev-parse HEAD).Trim()
Assert-CommandSucceeded 'Read TAG commit'
if ($actualCommit -ne $tagCommit) {
    throw "TAG checkout is at $actualCommit; expected $tagCommit. Use a clean checkout of the pinned commit."
}

& git -C $tagRoot apply --reverse --check --quiet $patchPath 2>$null
if ($LASTEXITCODE -ne 0) {
    & git -C $tagRoot apply --check $patchPath
    Assert-CommandSucceeded 'Check TAG patch'
    & git -C $tagRoot apply $patchPath
    Assert-CommandSucceeded 'Apply TAG patch'
}

Push-Location $tagRoot
try {
    & mvn -q -DskipTests package
    Assert-CommandSucceeded 'Build TAG-pytag.jar'
} finally {
    Pop-Location
}

if (-not (Test-Path -LiteralPath $jarPath -PathType Leaf)) {
    throw "Maven did not create $jarPath"
}
if (-not (Test-Path -LiteralPath (Join-Path $tagRoot 'data\pandemic') -PathType Container)) {
    throw "Missing Pandemic data under $tagRoot"
}

$env:PYTAG_JAR_PATH = (Resolve-Path -LiteralPath $jarPath).Path
Set-Location -LiteralPath $tagRoot
Write-Host "PyTAG installed: $venvPython"
Write-Host "Pinned TAG commit: $tagCommit"
Write-Host "PYTAG_JAR_PATH=$env:PYTAG_JAR_PATH"
Write-Host "Pandemic data working directory: $(Get-Location)"
