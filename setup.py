from setuptools import setup
import distutils.command.build

class BuildCommand(distutils.command.build.build):
    def initialize_options(self):
        distutils.command.build.build.initialize_options(self)
        self.build_base = "py_build"

setup(
    name="ece277_final",
    version="0.0.1",
    author="Eric Simpson",
    author_email="ersimpson@ucsd.edu",
    description="ECE 277 Final Project",
    packages=["py_src"],
    entry_points={
        "console_scripts": [
            "ece277final = py_src.cli:entrypoint",
        ]
    },
    cmdclass={ "build": BuildCommand },
    include_package_data=True,
)
