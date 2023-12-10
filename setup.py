from setuptools import setup

from torch.utils import cpp_extension

setup(
    name="ece277_final",
    version="0.0.1",
    author="Eric Simpson",
    author_email="ersimpson@ucsd.edu",
    description="ECE 277 Final Project",
    packages=["src"],
    ext_modules=[
        cpp_extension.CppExtension(
            name="mnist_model_cpp",
            sources=[
                "src/cxx/mnist_model.cpp",
            ]
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    entry_points={
        "console_scripts": [
            "ece277final = src.cli:entrypoint",
        ]
    },
)