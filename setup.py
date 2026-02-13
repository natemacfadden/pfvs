from setuptools import setup, Extension
from setuptools.command.develop import develop
from setuptools.command.build_ext import build_ext as _build_ext
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
    extensions.append(
        Extension(
            f"pfvs.c_kernels.{k['name']}",
            sources=[os.path.join(package_path, k["pyx"])],
            include_dirs=[os.path.join(package_path, k["include"])],
            define_macros=[(k["impl"], None)],
            language="c",
            extra_compile_args=["-O3"],
        )
    )


class build_ext(_build_ext):
    """Custom build_ext to ensure extensions are built in-place for editable installs."""
    def run(self):
        self.inplace = 1  # Force in-place build
        super().run()


class _develop(develop):
    """Custom develop command to build extensions in-place."""
    def run(self):
        # Build extensions in-place before running develop
        self.reinitialize_command('build_ext', inplace=1)
        self.run_command('build_ext')
        super().run()


setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
    cmdclass={
        'build_ext': build_ext,
        'develop': _develop,
    },
)