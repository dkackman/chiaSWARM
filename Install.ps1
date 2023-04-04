$ErrorActionPreference = "Stop"

if ([Environment]::Is64BitOperatingSystem -eq $false) {
    Write-Output "chiaSWARM requires a 64-bit Windows installation"
    Exit 1
}

if (-not (Get-Item -ErrorAction SilentlyContinue "$env:windir\System32\msvcp140.dll").Exists) {
    Write-Output "Unable to find Visual C++ Runtime DLLs"
    Write-Output ""
    Write-Output "Download and install the Visual C++ Redistributable for Visual Studio 2019 package from:"
    Write-Output "https://visualstudio.microsoft.com/downloads/#microsoft-visual-c-redistributable-for-visual-studio-2019"
    Exit 1
}

if ($null -eq (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Output "Unable to find python"
    Write-Output "Note the check box during installation of Python to install the Python Launcher for Windows."
    Write-Output ""
    Write-Output "https://docs.python.org/3/using/windows.html#installation-steps"
    Exit 1
}

$supportedPythonVersions = "3.10", "3.9", "3.8", "3.7"
if ("$env:INSTALL_PYTHON_VERSION" -ne "") {
    $pythonVersion = $env:INSTALL_PYTHON_VERSION
}
else {
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

    if (-not $pythonVersion) {
        $reversedPythonVersions = $supportedPythonVersions.clone()
        [array]::Reverse($reversedPythonVersions)
        $reversedPythonVersions = $reversedPythonVersions -join ", "
        Write-Output "No usable Python version found, supported versions are: $reversedPythonVersions"
        Exit 1
    }
}

$fullPythonVersion = (python --version).split(" ")[1]

Write-Output "Python version is: $fullPythonVersion"

# remove the venv if it exists
if (Test-Path -Path ".\venv" -PathType Container) {
    Remove-Item -LiteralPath ".\venv" -Recurse -Force
}

python -m venv venv

venv\scripts\python -m pip install --upgrade pip setuptools wheel

venv\scripts\pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
venv\scripts\pip install diffusers[torch] transformers accelerate scipy ftfysafetensors moviepy opencv-python xformers
venv\scripts\pip install aiohttp concurrent-log-handler 


Write-Output ""
Write-Output "chiaSWARM worker installation is now complete."
Write-Output ""
Write-Output "Type '.\venv\scripts\activate' and then 'python -m swarm.initialize' to begin."
