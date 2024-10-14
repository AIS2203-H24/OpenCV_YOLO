# OpenCV YOLO example (c++ edition)

Simple demo using YOLO v3/4/7 in C++.

### Getting OpenCV

To retrieve OpenCV, either:

#### Manually fetch OpenCV
1. Download from https://opencv.org/releases/
    1. Run self-extracting archive and extract to a suitable location
    2. Create an environmental ENV named `OpenCV_DIR` pointing to `/build`
    3. Add `/bin` folder to PATH (e.g. `.../build/x64/vc16/bin`)

#### Utilize vcpkg (using manifest mode)
Call CMake with `-DCMAKE_TOOLCHAIN_FILE=[path to vcpkg]/scripts/buildsystems/vcpkg.cmake`

> Consider removing "contrib", "cuda" features from vcpkg.json if you don't need them.

## CUDA

You can leverage CUDA if you have CUDA hardware and have compiled OpenCV with CUDA support. 
Replace `DNN_TARGET_CPU` with `DNN_TARGET_CUDA`.

> Note, building OpenCV with CUDA may take a loooooong time (many hours).
