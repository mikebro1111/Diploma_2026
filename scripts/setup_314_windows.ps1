<#
.SYNOPSIS
    Setup Python 3.14 (GIL + Free-threaded) on Windows for benchmarking.
.DESCRIPTION
    1. Downloads Python 3.14.0 standard installer and free-threaded zip
    2. Installs GIL build via silent installer
    3. Extracts free-threaded build from zip
    4. Creates venv_gil_314 and venv_nogil_314
    5. Installs dependencies from requirements files
.NOTES
    Run as: powershell -ExecutionPolicy Bypass -File scripts\setup_314_windows.ps1
#>

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

$GilInstallDir   = "C:\Python314"
$NogilInstallDir  = "C:\Python314t"
$TempDir          = Join-Path $ProjectRoot "_setup_temp"

$GilInstallerUrl  = "https://www.python.org/ftp/python/3.14.0/python-3.14.0-amd64.exe"
$NogilZipUrl      = "https://www.python.org/ftp/python/3.14.0/python-3.14.0t-amd64.zip"

$GilInstallerPath = Join-Path $TempDir "python-3.14.0-amd64.exe"
$NogilZipPath     = Join-Path $TempDir "python-3.14.0t-amd64.zip"

# ── Helpers ──────────────────────────────────────────────────────────────
function Download-File($url, $dest) {
    if (Test-Path $dest) {
        Write-Host "  Already downloaded: $(Split-Path -Leaf $dest)" -ForegroundColor DarkGray
        return
    }
    Write-Host "  Downloading $(Split-Path -Leaf $dest)..." -ForegroundColor Cyan
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $wc = New-Object System.Net.WebClient
    $wc.DownloadFile($url, $dest)
    $sizeMB = [math]::Round((Get-Item $dest).Length / 1MB, 1)
    Write-Host "  Done ($sizeMB MB)" -ForegroundColor Green
}

# ── Step 0: Temp dir ────────────────────────────────────────────────────
Write-Host "`n=== Setup Python 3.14 for Benchmarking ===" -ForegroundColor Yellow
if (-not (Test-Path $TempDir)) { New-Item -ItemType Directory -Path $TempDir | Out-Null }

# ── Step 1: Download ────────────────────────────────────────────────────
Write-Host "`n--- Step 1: Downloading Python 3.14 builds ---" -ForegroundColor Yellow
Download-File $GilInstallerUrl $GilInstallerPath
Download-File $NogilZipUrl $NogilZipPath

# ── Step 2: Install GIL build ───────────────────────────────────────────
Write-Host "`n--- Step 2: Installing Python 3.14 (GIL) ---" -ForegroundColor Yellow
$gilPython = Join-Path $GilInstallDir "python.exe"

if (Test-Path $gilPython) {
    $ver = & $gilPython --version 2>&1
    Write-Host "  Already installed: $ver" -ForegroundColor DarkGray
} else {
    Write-Host "  Installing to $GilInstallDir (silent)..." -ForegroundColor Cyan
    # /quiet = silent, InstallAllUsers=0 = current user, TargetDir = custom path
    $proc = Start-Process -FilePath $GilInstallerPath -ArgumentList `
        "/quiet", "InstallAllUsers=0", "PrependPath=0", "Include_launcher=0", `
        "TargetDir=$GilInstallDir" `
        -Wait -PassThru
    if ($proc.ExitCode -ne 0) {
        Write-Host "  ERROR: Installer exited with code $($proc.ExitCode)" -ForegroundColor Red
        exit 1
    }
    Write-Host "  Installed: $(& $gilPython --version 2>&1)" -ForegroundColor Green
}

# ── Step 3: Extract free-threaded build ─────────────────────────────────
Write-Host "`n--- Step 3: Extracting Python 3.14t (Free-threaded) ---" -ForegroundColor Yellow
$nogilPython = Join-Path $NogilInstallDir "python.exe"

if (Test-Path $nogilPython) {
    $ver = & $nogilPython --version 2>&1
    Write-Host "  Already extracted: $ver" -ForegroundColor DarkGray
} else {
    Write-Host "  Extracting to $NogilInstallDir..." -ForegroundColor Cyan
    Expand-Archive -Path $NogilZipPath -DestinationPath $NogilInstallDir -Force

    # The zip may have a nested folder — flatten if needed
    $nested = Get-ChildItem -Path $NogilInstallDir -Directory
    if ($nested.Count -eq 1 -and -not (Test-Path (Join-Path $NogilInstallDir "python.exe"))) {
        $nestedPath = $nested[0].FullName
        Get-ChildItem -Path $nestedPath | Move-Item -Destination $NogilInstallDir -Force
        Remove-Item $nestedPath -Recurse -Force
    }

    if (Test-Path $nogilPython) {
        Write-Host "  Extracted: $(& $nogilPython --version 2>&1)" -ForegroundColor Green
    } else {
        # Try python3.14t.exe naming
        $altNames = @("python3.14t.exe", "python3.14.exe", "python3.exe")
        foreach ($alt in $altNames) {
            $altPath = Join-Path $NogilInstallDir $alt
            if (Test-Path $altPath) {
                Write-Host "  Found as $alt, creating python.exe symlink..." -ForegroundColor Cyan
                Copy-Item $altPath $nogilPython
                break
            }
        }
        if (Test-Path $nogilPython) {
            Write-Host "  Extracted: $(& $nogilPython --version 2>&1)" -ForegroundColor Green
        } else {
            Write-Host "  ERROR: python.exe not found in $NogilInstallDir" -ForegroundColor Red
            Write-Host "  Contents:" -ForegroundColor Red
            Get-ChildItem $NogilInstallDir | Select-Object Name
            exit 1
        }
    }
}

# ── Step 4: Ensure pip for free-threaded build ──────────────────────────
Write-Host "`n--- Step 4: Ensuring pip ---" -ForegroundColor Yellow
# The zip build may not have pip — bootstrap it
$nogilPip = Join-Path $NogilInstallDir "Scripts\pip.exe"
if (-not (Test-Path $nogilPip)) {
    Write-Host "  Bootstrapping pip for free-threaded build..." -ForegroundColor Cyan
    $getPipUrl = "https://bootstrap.pypa.io/get-pip.py"
    $getPipPath = Join-Path $TempDir "get-pip.py"
    if (-not (Test-Path $getPipPath)) {
        Download-File $getPipUrl $getPipPath
    }
    & $nogilPython $getPipPath
}

# ── Step 5: Create venvs ────────────────────────────────────────────────
Write-Host "`n--- Step 5: Creating virtual environments ---" -ForegroundColor Yellow

$venvGil   = Join-Path $ProjectRoot "venv_gil_314"
$venvNogil = Join-Path $ProjectRoot "venv_nogil_314"

if (Test-Path (Join-Path $venvGil "Scripts\python.exe")) {
    Write-Host "  venv_gil_314 already exists" -ForegroundColor DarkGray
} else {
    Write-Host "  Creating venv_gil_314..." -ForegroundColor Cyan
    & $gilPython -m venv $venvGil
}

if (Test-Path (Join-Path $venvNogil "Scripts\python.exe")) {
    Write-Host "  venv_nogil_314 already exists" -ForegroundColor DarkGray
} else {
    Write-Host "  Creating venv_nogil_314..." -ForegroundColor Cyan
    & $nogilPython -m venv $venvNogil
}

# ── Step 6: Install dependencies ────────────────────────────────────────
Write-Host "`n--- Step 6: Installing dependencies ---" -ForegroundColor Yellow

$gilPip   = Join-Path $venvGil "Scripts\pip.exe"
$nogilPipVenv = Join-Path $venvNogil "Scripts\pip.exe"

Write-Host "  Installing into venv_gil_314..." -ForegroundColor Cyan
& $gilPip install -r (Join-Path $ProjectRoot "requirements_gil.txt")

Write-Host "  Installing into venv_nogil_314..." -ForegroundColor Cyan
& $nogilPipVenv install -r (Join-Path $ProjectRoot "requirements_nogil.txt")

# ── Step 7: Verify ──────────────────────────────────────────────────────
Write-Host "`n--- Step 7: Verification ---" -ForegroundColor Yellow

$gilVenvPy   = Join-Path $venvGil "Scripts\python.exe"
$nogilVenvPy = Join-Path $venvNogil "Scripts\python.exe"

Write-Host "  GIL build:" -ForegroundColor Cyan
& $gilVenvPy -c "import sys; print(f'  Version: {sys.version}'); print(f'  GIL enabled: {sys._is_gil_enabled()}')"
& $gilVenvPy -c "import numpy, pandas, sklearn, PIL, psutil; print('  All core packages: OK')"

Write-Host "  Free-threaded build:" -ForegroundColor Cyan
& $nogilVenvPy -c "import sys; print(f'  Version: {sys.version}'); print(f'  GIL enabled: {sys._is_gil_enabled()}')"
& $nogilVenvPy -c "import numpy, pandas, sklearn, PIL, psutil; print('  All core packages: OK')"

# ── Cleanup ─────────────────────────────────────────────────────────────
Write-Host "`n--- Cleanup ---" -ForegroundColor Yellow
Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "  Temp files removed" -ForegroundColor Green

# ── Done ────────────────────────────────────────────────────────────────
Write-Host "`n=== Setup Complete ===" -ForegroundColor Green
Write-Host "  GIL:       $gilVenvPy"
Write-Host "  Free-thr:  $nogilVenvPy"
Write-Host ""
Write-Host "  Run benchmarks:" -ForegroundColor Yellow
Write-Host "    $gilVenvPy scripts\run_multiple_benchmarks.py 5 3.14"
Write-Host ""
