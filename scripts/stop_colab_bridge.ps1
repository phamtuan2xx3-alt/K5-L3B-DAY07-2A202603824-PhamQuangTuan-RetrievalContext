$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
& wsl.exe -d Debian -- /opt/colab-cli/bin/colab stop -s day07-embedding
if ($LASTEXITCODE -ne 0) { throw 'Could not confirm Colab runtime release. Inspect colab sessions/usage.' }
$pidPath = Join-Path $repoRoot '.runtime/bridge.pid'
if (Test-Path -LiteralPath $pidPath) {
    $bridgeProcessId = [int](Get-Content -LiteralPath $pidPath)
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$bridgeProcessId"
    if ($process -and $process.CommandLine -like '*src.colab_bridge*') {
        Stop-Process -Id $bridgeProcessId
    }
}
Write-Output 'Colab runtime released and local bridge stopped.'
