$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$runtimeDir = Join-Path $repoRoot '.runtime'
New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
try {
    $health = Invoke-RestMethod 'http://127.0.0.1:8765/health' -TimeoutSec 2
    if ($health.model -eq 'Qwen/Qwen3-Embedding-8B') {
        if ($health.runtime_active) {
            Write-Output 'Embedding bridge is already running.'
            exit 0
        }
        # A new GPU/model may have been provisioned after idle auto-release.
        $bridgeProcessId = [int](Get-Content -LiteralPath (Join-Path $runtimeDir 'bridge.pid'))
        $process = Get-CimInstance Win32_Process -Filter "ProcessId=$bridgeProcessId"
        if (-not $process -or $process.CommandLine -notlike '*src.colab_bridge*') {
            throw 'Could not identify the inactive bridge process safely.'
        }
        Stop-Process -Id $bridgeProcessId
    } else {
        throw 'Port 8765 is used by a different service.'
    }
} catch [System.Net.WebException] {
    # No local bridge yet.
}
$pythonExe = (& py -c 'import sys; print(sys.executable)').Trim()
Start-Process -FilePath $pythonExe -ArgumentList @('-u', '-m', 'src.colab_bridge') `
    -WorkingDirectory $repoRoot -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $runtimeDir 'bridge.stdout.log') `
    -RedirectStandardError (Join-Path $runtimeDir 'bridge.stderr.log')
for ($attempt = 0; $attempt -lt 20; $attempt++) {
    Start-Sleep -Milliseconds 500
    try {
        $health = Invoke-RestMethod 'http://127.0.0.1:8765/health' -TimeoutSec 2
        Write-Output "Bridge ready: $($health.model), $($health.dimensions) dimensions."
        exit 0
    } catch [System.Net.WebException] { }
}
throw "Bridge did not start; inspect $runtimeDir/bridge.stderr.log"
