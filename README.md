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
