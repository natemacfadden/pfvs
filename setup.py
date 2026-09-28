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


class BuildExtCuda(build_ext):
    """build_ext that also builds the optional CUDA backend (PFVS_CUDA=1).

    The backend is a plain shared library (pfvs/fp_kernel/libpfvs_gpu.so,
    loaded with ctypes by pfvs.gpu), compiled by nvcc. Environment:
        PFVS_CUDA=1       build it (otherwise skipped; pfvs works without it)
        NVCC              the nvcc to use (default: nvcc on PATH)
        PFVS_CUDA_ARCH    nvcc -arch value(s), comma-separated
                          (default: native, the GPUs of this machine)
        NVCC_CCBIN        host compiler for nvcc (read by nvcc itself), for
                          when the default gcc is newer than nvcc supports
    """

    def run(self):
        super().run()
        if os.environ.get("PFVS_CUDA") != "1":
            return
        nvcc = os.environ.get("NVCC") or shutil.which("nvcc")
        if not nvcc:
            raise RuntimeError("PFVS_CUDA=1 but nvcc was not found (set NVCC)")
        archs = os.environ.get("PFVS_CUDA_ARCH", "native").split(",")
        src = os.path.join("pfvs", "fp_kernel", "cuda", "pfvs_gpu.cu")
        out_dir = os.path.join("pfvs", "fp_kernel") if self.inplace \
            else os.path.join(self.build_lib, "pfvs", "fp_kernel")
        os.makedirs(out_dir, exist_ok=True)
        cmd = [nvcc, "-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC",
               *[f"-arch={a.strip()}" for a in archs if a.strip()],
               src, "-o", os.path.join(out_dir, "libpfvs_gpu.so")]
        print("building the CUDA backend:", " ".join(shlex.quote(c) for c in cmd))
        subprocess.check_call(cmd)


# One kernel (fp_kernel.h) serves both pipelines; `pfvs.conipfv_kernel` and
# `pfvs.pfv_kernel` re-export its two entry points.
setup(
    cmdclass={"build_ext": BuildExtCuda},
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
