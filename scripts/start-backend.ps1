$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
  throw "Python virtual environment not found at .venv. Create it first."
}

$listener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($listener) {
  $pids = $listener | Select-Object -ExpandProperty OwningProcess -Unique
  foreach ($procId in $pids) {
    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
  }
  Start-Sleep -Milliseconds 300
}

& $venvPython -m uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload
