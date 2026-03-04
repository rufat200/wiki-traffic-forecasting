from setuptools import setup
from Cython.Build import cythonize
import numpy as np


setup(
    name="slope_lib",
    packages=[],
    ext_modules=cythonize(
        "slope.pyx",
        compiler_directives={"language_level": "3"}
    ),
    include_dirs=[np.get_include()],
)
