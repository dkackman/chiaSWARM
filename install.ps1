$ErrorActionPreference = "Stop"

# Check for 64-bit Windows installation
if (-not [Environment]::Is64BitOperatingSystem) {
    Write-Error "chiaSWARM requires a 64-bit Windows installation"
    Exit 1
}

# Check for Visual C++ Runtime DLLs
$vcRuntime = Get-Item -ErrorAction SilentlyContinue "$env:windir\System32\msvcp140.dll"
if (-not $vcRuntime.Exists) {
    $vcRuntimeUrl = "https://visualstudio.microsoft.com/downloads/#microsoft-visual-c-redistributable-for-visual-studio-2019"
    Write-Error "Unable to find Visual C++ Runtime DLLs"
    Write-Output "Download and install the Visual C++ Redistributable for Visual Studio 2019 package from: $vcRuntimeUrl"
    Exit 1
}

# Check for Python
try {
    $pythonVersion = (python --version).split(" ")[1]
}
catch {
    Write-Error "Unable to find python"
    $pythonUrl = "https://docs.python.org/3/using/windows.html#installation-steps"
    Write-Output "Note the check box during installation of Python to install the Python Launcher for Windows."
    Write-Output "Install Python from: $pythonUrl"
    Exit 1
}

# Check for supported Python version
$supportedPythonVersions = "3.12", "3.11", "3.10"
if ($env:INSTALL_PYTHON_VERSION) {
    $pythonVersion = $env:INSTALL_PYTHON_VERSION
}
else {
    $pythonVersion = $null
    foreach ($version in $supportedPythonVersions) {
        try {
            $pver = (python --version).split(" ")[1]
            $result = $pver.StartsWith($version)
        }
        catch {
            $result = $false
        }
        if ($result) {
            $pythonVersion = $version
            break
        }
    }
}

if (-not $pythonVersion) {
    $supportedPythonVersions = ($supportedPythonVersions | ForEach-Object { "Python $_" }) -join ", "
    Write-Error "No usable Python version found, supported versions are: $supportedPythonVersions"
    Exit 1
}

# Print Python version
Write-Output "Python version is: $pythonVersion"

# remove the venv if it exists
if (Test-Path -Path ".\venv" -PathType Container) {
    Remove-Item -LiteralPath ".\venv" -Recurse -Force
}

python -m venv venv

.\venv\scripts\activate 

python.exe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

pip3 install torch diffusers[torch] transformers accelerate scipy ftfy safetensors moviepy opencv-python sentencepiece peft --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

pip3 install aiohttp matplotlib moviepy opencv-python concurrent-log-handler qrcode protobuf imageio imageio-ffmpeg beautifulsoup4 soundfile
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output "done"
exit
