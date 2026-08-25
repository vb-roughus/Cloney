# Cloney einrichten (Windows).
#
# Legt eine virtuelle Umgebung an und uebergibt an scripts\setup.py, wo die
# eigentliche Arbeit passiert. Alle Argumente werden durchgereicht, etwa
# --skip-torch oder --dry-run.
#
# Aufruf:
#   powershell -ExecutionPolicy Bypass -File .\install.ps1
#
# Ohne -ExecutionPolicy Bypass lehnt Windows unsignierte Skripte in der
# Voreinstellung ab.

[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$SetupArgs
)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

function Find-Python {
    # Der Python-Launcher kennt alle installierten Versionen; ohne ihn bleibt
    # nur, was gerade im Suchpfad liegt.
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($version in @('-3.12', '-3.11')) {
            & py $version -c 'import sys' 2>$null
            if ($LASTEXITCODE -eq 0) { return @('py', $version) }
        }
    }
    foreach ($name in @('python3', 'python')) {
        $found = Get-Command $name -ErrorAction SilentlyContinue
        if ($found) {
            & $name -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>$null
            if ($LASTEXITCODE -eq 0) { return @($name) }
        }
    }
    return $null
}

$python = Find-Python
if (-not $python) {
    Write-Host 'Kein passendes Python gefunden. Cloney braucht Python 3.11 oder neuer.' -ForegroundColor Red
    Write-Host 'Installieren mit:  winget install --id=Python.Python.3.12' -ForegroundColor Yellow
    exit 1
}

$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    Write-Host "Virtuelle Umgebung anlegen mit $($python -join ' ') ..."
    & $python[0] @($python[1..($python.Length - 1)]) -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Die virtuelle Umgebung konnte nicht angelegt werden.' -ForegroundColor Red
        exit 1
    }
}

# torchaudio laedt ab Version 2.10 ausschliesslich ueber torchcodec, und das
# verlangt die FFmpeg-Bibliotheken. Entscheidend sind die DLLs, nicht ffmpeg.exe:
# der statische Build legt nur die ausfuehrbare Datei ab und nuetzt hier nichts.
$ffmpegLibs = $env:PATH -split ';' | Where-Object { $_ } | ForEach-Object {
    Get-ChildItem -Path (Join-Path $_ 'avcodec-*.dll') -ErrorAction SilentlyContinue
}
if (-not $ffmpegLibs) {
    Write-Host ''
    Write-Host 'Hinweis: Die FFmpeg-Bibliotheken fehlen im Suchpfad.' -ForegroundColor DarkYellow
    Write-Host '         Ohne sie kann torchaudio keine Audiodateien oeffnen.' -ForegroundColor DarkGray
    Write-Host '         Der Shared-Build wird gebraucht, nicht der statische:' -ForegroundColor DarkGray
    Write-Host '           winget install --id Gyan.FFmpeg.Shared' -ForegroundColor DarkGray
    Write-Host '         Danach die Konsole neu oeffnen.' -ForegroundColor DarkGray
}

& $venvPython (Join-Path 'scripts' 'setup.py') @SetupArgs
exit $LASTEXITCODE
