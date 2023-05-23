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
$supportedPythonVersions = "3.11", "3.10", "3.9", "3.8", "3.7"
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

python -m pip install wheel setuptools
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python.exe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m pip install diffusers[torch] transformers accelerate scipy ftfy safetensors moviepy opencv-python sentencepiece
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

#python -m pip install xformers
#if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m pip install aiohttp concurrent-log-handler pydub controlnet_aux
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m pip install git+https://github.com/suno-ai/bark.git@main
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# ControlNet Auxiliary Library for Image Registration
# Required for a specific feature of chiaSWARM
python -m pip install controlnet_aux==0.0.3  # pinned mediapipe dpenendency not found
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output "Audio conversion to mp3 requires ffmpeg"
Write-Output "Install ffmpeg from an elevated command prompt with the following command:"
Write-Output "choco install ffmpeg"

Write-Output ""
Write-Output "chiaSWARM worker installation is now complete."
Write-Output ""
Write-Output "Type '.\venv\scripts\activate' and then 'python -m swarm.initialize' to begin."
