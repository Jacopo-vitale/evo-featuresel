from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "evo.core",
        ["evo/core.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['/openmp'],
        extra_link_args=['/openmp'],
    )
]

setup(
    name="evo-featuresel",
    ext_modules=cythonize(extensions, language_level="3"),
)
