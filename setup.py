import os
import platform

from Cython.Build import cythonize
from setuptools import Extension, setup


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

# One kernel (fp_kernel.h) serves both pipelines; `pfvs.conipfv_kernel` and
# `pfvs.pfv_kernel` re-export its two entry points.
setup(
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
