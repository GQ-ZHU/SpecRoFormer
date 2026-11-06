"""
Environment self-check for SpecRoFormer.

Run this script after installation to verify that all required
dependencies are available and that the SpecRoFormer package can
be imported.

Usage:
    python scripts/check_env.py
"""

import importlib
import sys


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MIN_PYTHON = (3, 9)

# (import name, display name, required?)
REQUIRED_PACKAGES = [
    ("torch", "torch", True),
    ("torchvision", "torchvision", False),
    ("numpy", "numpy", True),
    ("scipy", "scipy", False),
    ("pandas", "pandas", True),
    ("sklearn", "scikit-learn", True),
    ("tqdm", "tqdm", True),
    ("matchms", "matchms", True),
    ("rdkit", "rdkit", True),
    ("numba", "numba", True),
    ("llvmlite", "llvmlite", False),
    ("optuna", "optuna", False),
    ("matplotlib", "matplotlib", True),
    ("seaborn", "seaborn", True),
    ("openpyxl", "openpyxl", False),
    ("xlsxwriter", "xlsxwriter", False),
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def check_python():
    ok = sys.version_info >= MIN_PYTHON
    print(
        f"[{'OK' if ok else 'FAIL'}] Python "
        f"{sys.version_info.major}.{sys.version_info.minor}."
        f"{sys.version_info.micro} "
        f"(>= {MIN_PYTHON[0]}.{MIN_PYTHON[1]} required)"
    )
    return ok


def check_packages():
    all_ok = True
    for mod_name, display, required in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(mod_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"[OK]   {display:15s} {version}")
        except ImportError:
            tag = "FAIL" if required else "WARN"
            print(f"[{tag}] {display:15s} not installed")
            if required:
                all_ok = False
    return all_ok


def check_cuda():
    try:
        import torch
    except ImportError:
        print("[WARN] torch not installed, skip CUDA check")
        return True

    available = torch.cuda.is_available()
    if available:
        print(
            f"[OK]   CUDA available, "
            f"device count = {torch.cuda.device_count()}"
        )
        print(f"       torch.version.cuda = {torch.version.cuda}")
        print(
            f"       device 0 = "
            f"{torch.cuda.get_device_name(0)}"
        )
    else:
        print("[WARN] CUDA not available, will fall back to CPU")
    return True


def check_package_import():
    try:
        import SpecRoFormer
        print(
            f"[OK]   SpecRoFormer import OK "
            f"({SpecRoFormer.__file__})"
        )
        return True
    except ImportError as e:
        print(f"[FAIL] Cannot import SpecRoFormer: {e}")
        print(
            "       Make sure you run this script from the "
            "repository root, or install the package with "
            "'pip install -e .'"
        )
        return False


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("=" * 60)
    print("SpecRoFormer environment check")
    print("=" * 60)

    results = []
    results.append(check_python())
    results.append(check_packages())
    results.append(check_cuda())
    results.append(check_package_import())

    print("=" * 60)
    if all(results):
        print("All required checks passed.")
        return 0
    print("Some checks failed. Please review the output above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())