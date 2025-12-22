#!/usr/bin/env python3
"""
Build script for creating a standalone executable of the GTA V Casino Fingerprint Solver.

This script uses PyInstaller to package the Python script and all dependencies
into a single executable file with templates bundled as resources.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def check_pyinstaller():
    """Check if PyInstaller is installed, install if missing."""
    try:
        import PyInstaller

        print(f"[INFO] PyInstaller found: {PyInstaller.__version__}")
        return True
    except ImportError:
        print("[INFO] PyInstaller not found. Installing...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "pyinstaller"]
            )
            print("[INFO] PyInstaller installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to install PyInstaller: {e}")
            return False


def clean_build_artifacts():
    """Remove previous build artifacts."""
    dirs_to_remove = ["build", "dist"]
    files_to_remove = ["casino-solver.spec"]

    for dir_name in dirs_to_remove:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"[INFO] Removing {dir_name}/ directory...")
            shutil.rmtree(dir_path)

    for file_name in files_to_remove:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"[INFO] Removing {file_name}...")
            file_path.unlink()


def verify_templates():
    """Verify that template directory exists and contains required files."""
    template_dir = Path("1920x1080")
    if not template_dir.exists():
        print(f"[ERROR] Template directory not found: {template_dir}")
        return False

    # Check for required template files (1.bmp through 16.bmp)
    missing_files = []
    for i in range(1, 17):
        template_file = template_dir / f"{i}.bmp"
        if not template_file.exists():
            missing_files.append(str(template_file))

    if missing_files:
        print(f"[ERROR] Missing template files: {', '.join(missing_files)}")
        return False

    print(f"[INFO] Template directory verified: {template_dir}")
    return True


def build_executable():
    """Build the standalone executable using PyInstaller."""
    script_path = Path("casino-solver.py")
    if not script_path.exists():
        print(f"[ERROR] Main script not found: {script_path}")
        return False

    # PyInstaller command arguments
    # --onefile: Create a single executable file
    # --add-data: Bundle template directory (platform-specific separator)
    # --hidden-import: Ensure these modules are included
    # --console: Keep console window for debug output
    # --name: Output executable name

    # Use platform-specific path separator for --add-data
    # Windows uses ';', Unix (Linux/macOS) uses ':'
    data_separator = ";" if sys.platform == "win32" else ":"

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--console",
        "--name",
        "casino-solver",
        "--add-data",
        f"1920x1080{data_separator}1920x1080",
        "--hidden-import",
        "tkinter",
        "--hidden-import",
        "cv2",
        "--hidden-import",
        "numpy",
        "--hidden-import",
        "mss",
        "--hidden-import",
        "pynput",
        "--hidden-import",
        "pynput.keyboard",
        str(script_path),
    ]

    print("[INFO] Building executable with PyInstaller...")
    print(f"[INFO] Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("[INFO] Build completed successfully!")

        # PyInstaller adds .exe on Windows automatically
        exe_name = "casino-solver.exe" if sys.platform == "win32" else "casino-solver"
        exe_path = Path("dist") / exe_name
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print(f"[INFO] Executable created: {exe_path} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"[WARNING] Expected executable not found: {exe_path}")
            return False

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Build failed: {e}")
        return False


def main():
    """Main build process."""
    print("=" * 60)
    print("GTA V Casino Fingerprint Solver - Build Script")
    print("=" * 60)
    print()

    # Warn if building on non-Windows (project is Windows-only)
    if sys.platform != "win32":
        print("[WARNING] Building on non-Windows platform.")
        print("[WARNING] This project uses Windows-specific APIs (DirectInput).")
        print("[WARNING] The resulting executable will NOT work on Windows.")
        print(
            "[WARNING] For Windows executables, build on Windows or use GitHub Actions."
        )
        print()
        response = input("Continue anyway? (yes/no): ").strip().lower()
        if response not in ("yes", "y"):
            print("[INFO] Build cancelled.")
            return 1
        print()

    # Change to script directory
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)
    print(f"[INFO] Working directory: {script_dir}")
    print()

    # Step 1: Check PyInstaller
    if not check_pyinstaller():
        print("[ERROR] Cannot proceed without PyInstaller")
        return 1
    print()

    # Step 2: Verify templates
    if not verify_templates():
        print("[ERROR] Cannot proceed without template files")
        return 1
    print()

    # Step 3: Clean previous builds
    clean_build_artifacts()
    print()

    # Step 4: Build executable
    if not build_executable():
        print("[ERROR] Build failed")
        return 1

    print()
    print("=" * 60)
    print("Build completed successfully!")
    print("=" * 60)
    exe_name = "casino-solver.exe" if sys.platform == "win32" else "casino-solver"
    print(f"Executable location: {Path('dist') / exe_name}")
    print()
    print("You can now run the executable standalone without Python.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
