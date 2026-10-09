param([switch]$Gpu)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) { throw 'Install Python 3.12 first (python.org), with Add Python to PATH enabled.' }
    & $pythonCommand.Source -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create Python environment.' }
}
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Could not install pip.' }
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
if ($Gpu) {
    & $venvPython -m pip uninstall -y onnxruntime onnxruntime-gpu
    & $venvPython -m pip install 'onnxruntime-gpu[cuda,cudnn]>=1.21,<2'
    if ($LASTEXITCODE -ne 0) { throw 'GPU runtime installation failed.' }
}
& $venvPython setup_models.py
if ($LASTEXITCODE -ne 0) { throw 'Model setup failed. Read README.md for manual model installation.' }
Write-Host 'Ready. Double-click start.bat.'
