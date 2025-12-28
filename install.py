import argparse
import platform
import re
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Elastix + IMPACT binary assets hosted on GitHub Releases.
#
# Key format: (OS, ARCH, FLAVOR)
#   - OS     : platform.system() -> "Linux", "Windows", "Darwin"
#   - ARCH   : normalized architecture -> "x86_64"
#   - FLAVOR : "cpu" or "cu128"
#
# CPU assets are standalone (LibTorch CPU bundled).
# CUDA assets do NOT bundle LibTorch (too large for GitHub limits).
# -----------------------------------------------------------------------------
ELX_ASSET_TEMPLATE = {
    ("Linux", "x86_64", "cpu"): "elastix-impact-linux-x86_64-shared-with-deps-cpu.zip",
    ("Linux", "x86_64", "cu128"): "elastix-impact-linux-x86_64-cu128.zip",
    ("Windows", "x86_64", "cpu"): "elastix-impact-windows-x86_64-shared-with-deps-cpu.zip",
    ("Windows", "x86_64", "cu128"): "elastix-impact-windows-x86_64-cu128.zip",
    ("Darwin", "x86_64", "cpu"): "elastix-impact-macos-14-x86_64-shared-with-deps-cpu.zip",
}

# -----------------------------------------------------------------------------
# Official LibTorch downloads (PyTorch).
#
# IMPORTANT:
# - Version MUST match the one used at build time (ABI compatibility).
# - Using "shared-with-deps" ensures CUDA runtime libraries are included
#   (except the NVIDIA driver, which must be installed system-wide).
# -----------------------------------------------------------------------------

LIBTORCH_URL = {
    (
        "Linux",
        "x86_64",
        "cu128",
    ): "https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.8.0%2Bcu128.zip",
    (
        "Windows",
        "x86_64",
        "cu128",
    ): "https://download.pytorch.org/libtorch/cu128/libtorch-win-shared-with-deps-2.8.0%2Bcu128.zip",
}

# -----------------------------------------------------------------------------
# Minimum NVIDIA driver versions required for CUDA 12.8.
#
# The CUDA Toolkit itself is NOT required.
# Only a sufficiently recent NVIDIA driver must be installed.
# -----------------------------------------------------------------------------
CUDA128_MIN_DRIVER_LINUX = (570, 26)
CUDA128_MIN_DRIVER_WINDOWS = (570, 65)

GITHUB_OWNER = "vboussot"
GITHUB_REPO = "ImpactElastix"
GITHUB_TAG = "1.0.0"


DEFAULT_PREFIX = Path.cwd() / "elastix-impact"


def run_cmd(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", errors="replace")


def detect_nvidia_driver() -> tuple[bool, tuple[int, int] | None]:
    """
    Detect presence of an NVIDIA GPU and extract the driver version.
    Returns:
        (True, (major, minor)) if detected
        (False, None) if nvidia-smi is not available
    """
    try:
        out = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return (False, None)

    # Exemple: "575.64"
    m = re.match(r"([0-9]+)\.([0-9]+)", out)
    if not m:
        return (True, None)

    return (True, (int(m.group(1)), int(m.group(2))))


def driver_ok_for_cuda(os_name: str, drv: tuple[int, int] | None) -> bool:
    """
    Check whether the detected NVIDIA driver satisfies the minimum
    requirement for CUDA 12.8 on the given operating system.
    """
    if drv is None:
        return False
    if os_name == "Linux":
        return drv >= CUDA128_MIN_DRIVER_LINUX
    if os_name == "Windows":
        return drv >= CUDA128_MIN_DRIVER_WINDOWS
    return False


def normalize_arch(machine: str) -> str:
    """
    Normalize platform.machine() output across operating systems.

    Examples:
        AMD64   -> x86_64
        aarch64 -> arm64
    """
    m = machine.lower()
    if m in ("x86_64", "amd64"):
        return "x86_64"
    if m in ("aarch64", "arm64"):
        return "arm64"
    return machine


def download_file(url: str, dst: Path) -> None:
    """
    Download a file from the given URL with a progress indicator.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {url}", flush=True)
    print(f"       to: {dst}", flush=True)

    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dst, "wb") as f:
                with tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {dst.name}",
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
    except Exception as e:
        raise e


def extract_archive(archive: Path, dst_dir: Path) -> None:
    """
    Extract a ZIP archive to the destination directory.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting: {archive} -> {dst_dir}", flush=True)
    with zipfile.ZipFile(archive, "r") as z:
        z.extractall(dst_dir)
    archive.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=Path, default=DEFAULT_PREFIX, help="Install directory (default: ./elastix-impact)")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--force-cpu", action="store_true", help="Force CPU install even if NVIDIA present")
    group.add_argument("--force-cuda", action="store_true", help="Force CUDA install (fails if no suitable driver)")
    args = ap.parse_args()

    os_name = platform.system()
    arch = normalize_arch(platform.machine())
    has_nvidia, drv = detect_nvidia_driver()

    if os_name not in ("Linux", "Windows", "Darwin"):
        raise NameError(f"Unsupported OS: {os_name}")

    if arch != "x86_64":
        raise NameError(f"Unsupported arch: {arch} (expected x86_64)")

    flavor = "cpu"
    if args.force_cuda:
        if not has_nvidia or not driver_ok_for_cuda(os_name, drv):
            raise NameError(
                "CUDA forced but NVIDIA driver/GPU not suitable. Detected: has_nvidia={has_nvidia}, driver={drv}"
            )

        flavor = "cu128"
    elif not args.force_cpu:
        if has_nvidia and driver_ok_for_cuda(os_name, drv):
            flavor = "cu128"
        else:
            flavor = "cpu"

    print(f"System: {os_name} {arch}", flush=True)
    print(f"NVIDIA: {has_nvidia}, driver={drv}", flush=True)
    print(f"Selected flavor: {flavor}", flush=True)

    key = (os_name, arch, flavor)
    if key not in ELX_ASSET_TEMPLATE:
        raise NameError(f"No elastix asset configured for {key}")
    if flavor != "cpu" and key not in LIBTORCH_URL:
        raise NameError(f"No libtorch url configured for {key}")

    prefix: Path = args.prefix.resolve()
    prefix.mkdir(parents=True, exist_ok=True)

    elx_asset = ELX_ASSET_TEMPLATE[key]
    elx_url = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/download/{GITHUB_TAG}/{elx_asset}"
    elx_archive = prefix / elx_asset
    download_file(elx_url, elx_archive)
    extract_archive(elx_archive, prefix)

    # -------------------------------------------------------------------------
    # ZIP archives may drop executable permissions.
    # Ensure elastix and transformix are executable on Unix platforms.
    # -------------------------------------------------------------------------
    if os_name in ("Linux", "Darwin"):
        for exe in ("elastix", "transformix"):
            p = prefix / "bin" / exe
            if p.exists():
                p.chmod(p.stat().st_mode | stat.S_IEXEC)

    if flavor != "cpu":
        # ---------------------------------------------------------------------
        # CUDA build:
        # - Download matching LibTorch cu128
        # - Extract it
        # - Move runtime libraries to a location visible to the dynamic loader
        #
        # Linux / macOS : prefix/lib
        # Windows       : prefix (next to elastix.exe)
        # ---------------------------------------------------------------------
        lt_url = LIBTORCH_URL[key]
        lt_archive = prefix / Path(lt_url).name
        download_file(lt_url, lt_archive)
        extract_archive(lt_archive, prefix)
        for p in (prefix / "libtorch" / "lib").iterdir():
            shutil.move(p, (prefix if os_name == "Windows" else prefix / "lib") / p.name)
        shutil.rmtree(prefix / "libtorch")
        if os_name == "Linux":
            shutil.copy(prefix / "lib" / "libcudart-c3a75b33.so.12", prefix / "lib" / "libcudart.so.12")

    # Installation completed successfully
    return 0


if __name__ == "__main__":
    main()
