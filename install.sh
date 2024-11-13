#!/bin/bash

# 100% credit to chia and chia dev team - https://github.com/Chia-Network/chia-blockchain

set -o errexit


UBUNTU=false
DEBIAN=false
if [ "$(uname)" = "Linux" ]; then
  #LINUX=1
  if command -v apt-get >/dev/null; then
    OS_ID=$(lsb_release -is)
    if [ "$OS_ID" = "Debian" ]; then
      DEBIAN=true
    else
      UBUNTU=true
    fi
  fi
fi

# Check for non 64 bit ARM64/Raspberry Pi installs
if [ "$(uname -m)" = "armv7l" ]; then
  echo ""
  echo "WARNING:"
  echo "The chiaSWARM requires a 64 bit OS and this is 32 bit armv7l"
  echo "Exiting."
  exit 1
fi

UBUNTU_PRE_20=0
UBUNTU_20=0
UBUNTU_21=0
UBUNTU_22=0

if $UBUNTU; then
  LSB_RELEASE=$(lsb_release -rs)
  # In case Ubuntu minimal does not come with bc
  if ! command -v bc > /dev/null 2>&1; then
    sudo apt install bc -y
  fi
  # Mint 20.04 responds with 20 here so 20 instead of 20.04
  if [ "$(echo "$LSB_RELEASE<20" | bc)" = "1" ]; then
    UBUNTU_PRE_20=1
  elif [ "$(echo "$LSB_RELEASE<21" | bc)" = "1" ]; then
    UBUNTU_20=1
  elif [ "$(echo "$LSB_RELEASE<22" | bc)" = "1" ]; then
    UBUNTU_21=1
  else
    UBUNTU_22=1
  fi
fi

install_python3_from_source_with_yum() {
  CURRENT_WD=$(pwd)
  TMP_PATH=/tmp

  # Preparing installing Python
  echo 'yum groupinstall -y "Development Tools"'
  sudo yum groupinstall -y "Development Tools"

  echo "cd $TMP_PATH"
  cd "$TMP_PATH"
  # '| stdbuf ...' seems weird but this makes command outputs stay in single line.
  ./configure --prefix=/usr/local | stdbuf -o0 cut -b1-"$(tput cols)" | sed -u 'i\\o033[2K' | stdbuf -o0 tr '\n' '\r'; echo
  echo "make -j$(nproc)"
  make -j"$(nproc)" | stdbuf -o0 cut -b1-"$(tput cols)" | sed -u 'i\\o033[2K' | stdbuf -o0 tr '\n' '\r'; echo
  echo "sudo make install"
  sudo make install | stdbuf -o0 cut -b1-"$(tput cols)" | sed -u 'i\\o033[2K' | stdbuf -o0 tr '\n' '\r'; echo
  # yum install python3 brings Python3.6 which is not supported
  cd ..
  echo "wget https://www.python.org/ftp/python/3.9.11/Python-3.9.11.tgz"
  wget https://www.python.org/ftp/python/3.9.11/Python-3.9.11.tgz
  tar xf Python-3.9.11.tgz
  echo "cd Python-3.9.11"
  cd Python-3.9.11
  echo "LD_RUN_PATH=/usr/local/lib ./configure --prefix=/usr/local"
  # '| stdbuf ...' seems weird but this makes command outputs stay in single line.
  LD_RUN_PATH=/usr/local/lib ./configure --prefix=/usr/local | stdbuf -o0 cut -b1-"$(tput cols)" | sed -u 'i\\o033[2K' | stdbuf -o0 tr '\n' '\r'; echo
  echo "LD_RUN_PATH=/usr/local/lib make -j$(nproc)"
  LD_RUN_PATH=/usr/local/lib make -j"$(nproc)" | stdbuf -o0 cut -b1-"$(tput cols)" | sed -u 'i\\o033[2K' | stdbuf -o0 tr '\n' '\r'; echo
  echo "LD_RUN_PATH=/usr/local/lib sudo make altinstall"
  LD_RUN_PATH=/usr/local/lib sudo make altinstall | stdbuf -o0 cut -b1-"$(tput cols)" | sed -u 'i\\o033[2K' | stdbuf -o0 tr '\n' '\r'; echo
  cd "$CURRENT_WD"
}

# You can specify preferred python version by exporting `INSTALL_PYTHON_VERSION`
# e.g. `export INSTALL_PYTHON_VERSION=3.8`
INSTALL_PYTHON_PATH=
PYTHON_MAJOR_VER=
PYTHON_MINOR_VER=

find_python() {
  set +e
  unset BEST_VERSION
  for V in 312 3.12 311 3.11 310 3.10; do
    if command -v python$V >/dev/null; then
      if [ "$BEST_VERSION" = "" ]; then
        BEST_VERSION=$V
      fi
    fi
  done

  if [ -n "$BEST_VERSION" ]; then
    INSTALL_PYTHON_VERSION="$BEST_VERSION"
    INSTALL_PYTHON_PATH=python${INSTALL_PYTHON_VERSION}
    PY3_VER=$($INSTALL_PYTHON_PATH --version | cut -d ' ' -f2)
    PYTHON_MAJOR_VER=$(echo "$PY3_VER" | cut -d'.' -f1)
    PYTHON_MINOR_VER=$(echo "$PY3_VER" | cut -d'.' -f2)
  fi
  set -e
}

if [ "$INSTALL_PYTHON_VERSION" = "" ]; then
  echo "Searching available python executables..."
  find_python
else
  echo "Python $INSTALL_PYTHON_VERSION is requested"
  INSTALL_PYTHON_PATH=python${INSTALL_PYTHON_VERSION}
  PY3_VER=$($INSTALL_PYTHON_PATH --version | cut -d ' ' -f2)
  PYTHON_MAJOR_VER=$(echo "$PY3_VER" | cut -d'.' -f1)
  PYTHON_MINOR_VER=$(echo "$PY3_VER" | cut -d'.' -f2)
fi

if [ "$(uname)" = "Linux" ]; then
  #LINUX=1
  if [ "$UBUNTU_PRE_20" = "1" ]; then
    # Ubuntu
    echo "Installing on Ubuntu pre 20.*."
    sudo apt-get update
    # distutils must be installed as well to avoid a complaint about ensurepip while
    # creating the venv.  This may be related to a mis-check while using or
    # misconfiguration of the secondary Python version 3.7.  The primary is Python 3.6.
    sudo apt-get install -y python3.7-venv python3.7-distutils ffmpeg
  elif [ "$UBUNTU_20" = "1" ]; then
    echo "Installing on Ubuntu 20.*."
    sudo apt-get update
    sudo apt-get install -y python3.8-venv ffmpeg
  elif [ "$UBUNTU_21" = "1" ]; then
    echo "Installing on Ubuntu 21.*."
    sudo apt-get update
    sudo apt-get install -y python3.9-venv ffmpeg
  elif [ "$UBUNTU_22" = "1" ]; then
    echo "Installing on Ubuntu 22.* or newer."
    sudo apt-get update
    if [ "$PYTHON_MINOR_VER" -eq "11" ]; then
      sudo apt-get install -y python3.11-venv ffmpeg
    else
      sudo apt-get install -y python3.10-venv ffmpeg
    fi
  elif [ "$DEBIAN" = "true" ]; then
    echo "Installing on Debian."
    sudo apt-get update
    sudo apt-get install -y python3-venv ffmpeg
  elif type yum >/dev/null 2>&1 && [ ! -f "/etc/redhat-release" ] && [ ! -f "/etc/centos-release" ] && [ ! -f "/etc/fedora-release" ]; then
    # AMZN 2
    echo "Installing on Amazon Linux 2."
    if ! command -v python3.9 >/dev/null 2>&1; then
      install_python3_from_source_with_yum
    fi
  elif type yum >/dev/null 2>&1 && [ -f "/etc/centos-release" ]; then
    # CentOS
    echo "Install on CentOS."
    if ! command -v python3.9 >/dev/null 2>&1; then
      install_python3_from_source_with_yum
    fi
  elif type yum >/dev/null 2>&1 && [ -f "/etc/redhat-release" ] && grep Rocky /etc/redhat-release; then
    echo "Installing on Rocky."
    # TODO: make this smarter about getting the latest version
    sudo yum install --assumeyes python39
  elif type yum >/dev/null 2>&1 && [ -f "/etc/redhat-release" ] || [ -f "/etc/fedora-release" ]; then
    # Redhat or Fedora
    echo "Installing on Redhat/Fedora."
    if ! command -v python3.9 >/dev/null 2>&1; then
      sudo yum install -y python39
    fi
  fi
elif [ "$(uname)" = "Darwin" ]; then
  echo "Installing on macOS."
  if ! type brew >/dev/null 2>&1; then
    echo "Installation currently requires brew on macOS - https://brew.sh/"
    exit 1
  fi
fi

if ! command -v "$INSTALL_PYTHON_PATH" >/dev/null; then
  echo "${INSTALL_PYTHON_PATH} was not found"
  exit 1
fi

if [ "$PYTHON_MAJOR_VER" -ne "3" ] || [ "$PYTHON_MINOR_VER" -lt "7" ] || [ "$PYTHON_MINOR_VER" -ge "12" ]; then
  echo "The chiaSWARM requires Python version >= 3.7 and  <= 3.11.0" >&2
  echo "Current Python version = $INSTALL_PYTHON_VERSION" >&2
  # If Arch, direct to Arch Wiki
  if type pacman >/dev/null 2>&1 && [ -f "/etc/arch-release" ]; then
    echo "Please see https://wiki.archlinux.org/title/python#Old_versions for support." >&2
  fi

  exit 1
fi
echo "Python version is $INSTALL_PYTHON_VERSION"

# delete the venv folder if present
if [ -d "venv" ]; then
  rm ./venv -rf
fi

# create the venv and add soft link to activate
$INSTALL_PYTHON_PATH -m venv venv
if [ ! -f "activate" ]; then
  ln -s venv/bin/activate .
fi

# shellcheck disable=SC1091
. ./activate

python -m pip install --upgrade pip

pip install torch diffusers[torch] peft transformers accelerate safetensors controlnet_aux sentencepiece mediapipe bitsandbytes
pip install aiohttp matplotlib moviepy opencv-python concurrent-log-handler qrcode protobuf imageio imageio-ffmpeg beautifulsoup4 soundfile
