#!/usr/bin/env bash

set -e

# Check OS
if [ "$(uname)" != "Linux" ]; then
    echo "This script is only for Linux"
    exit 1
fi

# Get distro to determine package manager (pacman or apt)
distro=$(cat /etc/os-release | grep -oP '(?<=^ID=).+' | tr -d '"')

# Install Rust and Cargo
which cargo > /dev/null 2>&1 || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install xdotool and uinput kernel module
if [ "$distro" = "arch" ]; then
    pacman -Q xdotool > /dev/null 2>&1 || sudo pacman -Syu xdotool
elif [ "$distro" = "debian" || "$distro" = "ubuntu" ]; then
    apt list --installed | grep -q xdotool || sudo apt-get update && sudo apt-get install xdotool
fi

lsmod | grep -q uinput || sudo modprobe uinput

# Build the Rust code
(cd inputter && cargo build --release)

# Install Python dependencies
pip3 install -r requirements.txt
