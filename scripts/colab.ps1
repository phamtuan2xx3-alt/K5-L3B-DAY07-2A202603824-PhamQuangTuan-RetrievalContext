# Forward arguments as separate values to the isolated Linux installation.
# Example: .\scripts\colab.ps1 version
& wsl.exe -d Debian -- /opt/colab-cli/bin/colab @args
exit $LASTEXITCODE
