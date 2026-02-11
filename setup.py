from setuptools import setup, Extension
from Cython.Build import cythonize
import os

kernels = [
    {
        "name": "coni_kernel",
        "pyx": "coni_kernel.pyx",
        "include": ".",
        "impl": "CONI_KERNEL_IMPLEMENTATION",
    },
    {
        "name": "pvec_kernel",
        "pyx": "pvec_kernel.pyx",
        "include": ".",
        "impl": "PVEC_KERNEL_IMPLEMENTATION",
    },
]

extensions = []
package_path = "pfvs/c_kernels"

for k in kernels:
    ext = Extension(
        f"pfvs.c_kernels.{k['name']}",
        sources=[
            os.path.join(package_path, k["pyx"]),
        ],
        include_dirs=[
            os.path.join(package_path, k["include"]),
        ],
        define_macros=[
            (k["impl"], None),
        ],
        language="c",
        extra_compile_args=["-O3"],
    )
    extensions.append(ext)

setup(
    name="c_kernels",
    packages=["pfvs", "pfvs.c_kernels"],
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)
