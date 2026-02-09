# setup.py
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import os

package_dir = "pfvs/c_kernels"
c_src_dir = os.path.join(package_dir, "c_code")

extensions = [
    Extension(
        name=f"pfvs.c_kernels.coni_kernel",
        sources=[os.path.join(package_dir, "coni_kernel.pyx"),
                 os.path.join(c_src_dir, "coni_kernel.c")],
        include_dirs=[c_src_dir],
        language="c",
        extra_compile_args=["-O3"],
    )
]

setup(
    name="coni_kernel",
    packages=find_packages(),
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
    zip_safe=False,
)
