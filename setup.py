# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import os

package_dir = "coni_kernel"
c_src_dir = os.path.join(package_dir, "c_code")

extensions = [
    Extension(
        name=f"{package_dir}.coni_kernel",
        sources=[os.path.join(package_dir, "coni_kernel.pyx"),
                 os.path.join(c_src_dir, "coni_kernel.c")],
        include_dirs=[c_src_dir],
        language="c",
        extra_compile_args=["-O3"],
    )
]

setup(
    name="coni_kernel",
    packages=[package_dir],
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)
