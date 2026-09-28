import glob
import os
import platform
import shlex
import shutil
import subprocess

from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def get_gmp_paths():
    """Find GMP from a conda environment, else rely on system default paths."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        return [os.path.join(conda_prefix, "include")], [os.path.join(conda_prefix, "lib")]
    return [], []


compile_args = ["-O3"]

# Machine-specific build is opt-in via PFVS_NATIVE=1 (the default build must
# run on any machine of the target architecture, e.g. for wheels)
if os.environ.get("PFVS_NATIVE") == "1":
    if platform.machine() in ("arm64", "aarch64"):
        compile_args += ["-mcpu=native"]
    else:
        compile_args += ["-march=native", "-mtune=native"]

gmp_include, gmp_lib = get_gmp_paths()


# Optional GPU backend
# --------------------
# pfvs/fp_kernel/libpfvs_gpu.so (loaded with ctypes by pfvs.gpu), built from
# pfvs/fp_kernel/cuda/ for NVIDIA (CUDA) or AMD (HIP/ROCm) GPUs. Environment:
#     PFVS_GPU        auto (default): build for whichever toolchain is found,
#                     never failing the install over it; cuda / hip: that one,
#                     errors are fatal; none: skip. (PFVS_CUDA=1 / PFVS_HIP=1
#                     are accepted as cuda / hip.)
#     NVCC, HIPCC     the compilers (default: on PATH; ROCm also ROCM_PATH or
#                     /opt/rocm, and pip-installed ROCm SDKs in this Python)
#     PFVS_CUDA_ARCH  nvcc -arch value(s), comma-separated (default: native,
#                     else a portable list when no GPU is visible)
#     PFVS_HIP_ARCH   --offload-arch value(s) (same defaults)
#     PFVS_ROCM_PATH  a ROCm root to build against explicitly
#     PFVS_GPU_FLAGS  extra compiler flags
#     NVCC_CCBIN      host compiler for nvcc (read by nvcc itself)
GPU_SRC = os.path.join("pfvs", "fp_kernel", "cuda", "pfvs_gpu.cu")
CUDA_ARCHS = ["sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_100", "sm_120"]
HIP_ARCHS = ["gfx90a", "gfx942", "gfx1030", "gfx1100", "gfx1101", "gfx1151", "gfx1200", "gfx1201"]


def _run_ok(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _cuda_cmd(out):
    nvcc = os.environ.get("NVCC") or shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError("nvcc not found (set NVCC)")
    archs = [a.strip() for a in os.environ.get("PFVS_CUDA_ARCH", "native").split(",") if a.strip()]
    if archs == ["native"] and not _run_ok(["nvidia-smi", "-L"]):   # no GPU visible
        supported = subprocess.run([nvcc, "--list-gpu-arch"], capture_output=True, text=True).stdout
        archs = [a for a in CUDA_ARCHS if a.replace("sm_", "compute_") in supported]
    gen = ["-arch=native"] if archs == ["native"] else \
        [f"-gencode=arch={a.replace('sm_', 'compute_')},code={a}" for a in archs] + \
        [f"-gencode=arch={archs[-1].replace('sm_', 'compute_')},code={archs[-1].replace('sm_', 'compute_')}"]
    return [nvcc, "-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", *gen,
            *shlex.split(os.environ.get("PFVS_GPU_FLAGS", "")), GPU_SRC, "-o", out]


def _rocm_sdk_root():
    """Root of a pip-installed ROCm SDK (_rocm_sdk_core) in this Python, if any."""
    import importlib.util
    spec = importlib.util.find_spec("_rocm_sdk_core")
    return os.path.dirname(spec.origin) if spec and spec.origin else None


def _hip_cmd(out):
    archs = [a.strip() for a in os.environ.get("PFVS_HIP_ARCH", "native").split(",") if a.strip()]
    root = os.environ.get("PFVS_ROCM_PATH") or _rocm_sdk_root()
    extra = shlex.split(os.environ.get("PFVS_GPU_FLAGS", ""))
    if root and os.path.exists(os.path.join(root, "lib", "llvm", "bin", "clang++")):
        # a self-contained ROCm tree (e.g. the pip SDK): drive its clang
        # directly, so no older system HIP headers or runtime get mixed in
        clang = os.path.join(root, "lib", "llvm", "bin", "clang++")
        libs = sorted(glob.glob(os.path.join(root, "lib", "libamdhip64.so*")))
        if not libs:
            raise RuntimeError(f"no libamdhip64 under {root}/lib")
        if archs == ["native"] and not _run_ok([os.path.join(root, "lib", "llvm", "bin", "amdgpu-arch")]):
            archs = HIP_ARCHS
        return [clang, "-x", "hip", *[f"--offload-arch={a}" for a in archs],
                f"--rocm-path={root}",
                f"--rocm-device-lib-path={os.path.join(root, 'lib', 'llvm', 'amdgcn', 'bitcode')}",
                "-nogpuinc", f"-I{os.path.join(root, 'include')}",
                "-include", "__clang_hip_runtime_wrapper.h", "-no-hip-rt",
                "-O3", "-std=c++17", "-shared", "-fPIC", *extra, GPU_SRC, "-o", out,
                f"-L{os.path.join(root, 'lib')}", f"-l:{os.path.basename(libs[-1])}",
                f"-Wl,-rpath,{os.path.join(root, 'lib')}"]
    hipcc = os.environ.get("HIPCC") or shutil.which("hipcc")
    if not hipcc:
        for r in filter(None, [root, os.environ.get("ROCM_PATH"), "/opt/rocm"]):
            if os.path.exists(os.path.join(r, "bin", "hipcc")):
                hipcc = os.path.join(r, "bin", "hipcc")
                break
    if not hipcc:
        raise RuntimeError("hipcc not found (set HIPCC or PFVS_ROCM_PATH)")
    if archs == ["native"] and not (_run_ok(["amdgpu-arch"]) or _run_ok(["rocm_agent_enumerator"])):
        archs = HIP_ARCHS
    return [hipcc, "-x", "hip", *[f"--offload-arch={a}" for a in archs],
            "-O3", "-std=c++17", "-shared", "-fPIC", *extra, GPU_SRC, "-o", out]


class BuildExtGpu(build_ext):
    """build_ext that also builds the optional GPU backend (see above)."""

    def run(self):
        super().run()
        mode = os.environ.get("PFVS_GPU", "").lower()
        if not mode:
            mode = "cuda" if os.environ.get("PFVS_CUDA") == "1" else \
                   "hip" if os.environ.get("PFVS_HIP") == "1" else "auto"
        if mode not in ("auto", "cuda", "hip", "none"):
            raise RuntimeError(f"PFVS_GPU must be auto, cuda, hip or none, got {mode!r}")
        if mode == "none":
            return
        out_dir = os.path.join("pfvs", "fp_kernel") if self.inplace \
            else os.path.join(self.build_lib, "pfvs", "fp_kernel")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "libpfvs_gpu.so")
        if mode == "auto":
            # prefer the toolchain whose GPUs are present
            order = ["cuda", "hip"]
            if not _run_ok(["nvidia-smi", "-L"]) and (_run_ok(["rocminfo"]) or _rocm_sdk_root()):
                order = ["hip", "cuda"]
            for m in order:
                try:
                    cmd = _cuda_cmd(out) if m == "cuda" else _hip_cmd(out)
                except RuntimeError:
                    continue
                print("building the GPU backend:", " ".join(shlex.quote(c) for c in cmd))
                if subprocess.run(cmd).returncode == 0:
                    return
                print(f"warning: the {m} GPU backend failed to build; continuing without it")
            print("note: no GPU backend built (pfvs runs on the CPU; see setup.py for PFVS_GPU)")
            return
        cmd = _cuda_cmd(out) if mode == "cuda" else _hip_cmd(out)
        print("building the GPU backend:", " ".join(shlex.quote(c) for c in cmd))
        subprocess.check_call(cmd)


# One kernel (fp_kernel.h) serves both pipelines; `pfvs.conipfv_kernel` and
# `pfvs.pfv_kernel` re-export its two entry points.
setup(
    cmdclass={"build_ext": BuildExtGpu},
    ext_modules=cythonize(
        [Extension(
            "pfvs.fp_kernel.fp_kernel",
            sources=["pfvs/fp_kernel/fp_kernel.pyx"],
            include_dirs=["pfvs/fp_kernel"] + gmp_include,
            library_dirs=gmp_lib,
            runtime_library_dirs=gmp_lib,
            libraries=["gmp"],
            define_macros=[("FP_KERNEL_IMPLEMENTATION", None),
                           ("PFV_LATTICE_IMPLEMENTATION", None)],
            extra_compile_args=compile_args,
            language="c",
        )],
        compiler_directives={"language_level": "3"},
    ),
)
