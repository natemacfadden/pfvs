from setuptools import setup, Extension
from setuptools.command.develop import develop
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize
import os

def get_gmp_paths():
    """Find GMP from conda environment or system."""
    conda_prefix = os.environ.get('CONDA_PREFIX')
    
    if conda_prefix:
        return {
            'include': os.path.join(conda_prefix, 'include'),
            'library': os.path.join(conda_prefix, 'lib'),
        }
    
    # Fallback to system default paths (empty = use system defaults)
    return None

kernels = [
    {
        "name": "coni_kernel",
        "pyx": "coni_kernel.pyx",
        "include": ".",
        "impl": "CONI_KERNEL_IMPLEMENTATION",
        "libraries": ["gmp"],
    },
]

extensions = []
package_path = "pfvs/coni_kernel"
gmp_paths = get_gmp_paths()

for k in kernels:
    include_dirs = [os.path.join(package_path, k["include"])]
    library_dirs = []
    
    # Add GMP paths if needed and available
    if "gmp" in k.get("libraries", []) and gmp_paths:
        include_dirs.append(gmp_paths['include'])
        library_dirs.append(gmp_paths['library'])
    
    extensions.append(
        Extension(
            f"pfvs.coni_kernel.{k['name']}",
            sources=[os.path.join(package_path, k["pyx"])],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            define_macros=[(k["impl"], None)],
            libraries=k.get("libraries", []),
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