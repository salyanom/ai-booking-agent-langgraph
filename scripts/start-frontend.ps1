$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$ports = @(8080, 8081, 8082)
foreach ($port in $ports) {
	$listeners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
	if ($listeners) {
		$pids = $listeners | Select-Object -ExpandProperty OwningProcess -Unique
		foreach ($procId in $pids) {
			Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
		}
	}
}
Start-Sleep -Milliseconds 300

npm run dev
