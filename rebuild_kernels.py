#!/usr/bin/env python3
import shutil
import subprocess
import os
from pathlib import Path

# paths
repo_root = Path(__file__).parent.resolve()
build_dir = repo_root / "build"
package_so_dirs = repo_root.glob("pfvs/c_kernels/**/*.so")

def remove_old_builds():
    print("Cleaning old build directories...")
    if build_dir.exists():
        shutil.rmtree(build_dir)
        print(f"Removed {build_dir}")

    for so_file in package_so_dirs:
        so_file.unlink()
        print(f"Removed {so_file}")

def rebuild():
    print("Rebuilding Cython extensions in place...")
    subprocess.check_call([os.environ.get("PYTHON", "python"), "setup.py", "build_ext", "--inplace"])

if __name__ == "__main__":
    remove_old_builds()
    rebuild()
    print("Done!")
