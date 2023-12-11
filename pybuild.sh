cp build/src/cuda_model/Debug/*.pyd py_src
cp build/src/cuda_model/Release/*.pyd py_src
winpty "/c/Program Files/Python39/python.exe" setup.py sdist
winpty "/c/Program Files/Python39/python.exe" -m pip install dist/ece277_final-0.0.1.tar.gz --user
