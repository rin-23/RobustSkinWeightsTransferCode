# C++ implementation of the Robust Skin Weights Transfer Algorithm

Follows the logic of the python implementation one to one.

## Compile

Compile this project using the standard cmake routine:

    mkdir build
    cd build
    cmake ..
    make -j8

This should find and build the dependencies and create a `skintransfer` binary.

## Run

From within the `build` directory just issue:

    ./skintransfer

This will run **main** function on a sample data and print out the resulting inpainted weights matrix.
You can read the **main** function to learn how to use the **robust_skin_weights_transfer**
function.

## Dependencies

The only dependencies are STL, Eigen, [libigl](http://libigl.github.io/libigl/).

The CMake build system will automatically download libigl and its dependencies using
[CMake FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html),
thus requiring no setup on your part.

### Use a local copy of libigl
You can use the CMake cache variable `FETCHCONTENT_SOURCE_DIR_LIBIGL` when configuring your CMake project for
the first time to aim it at a local copy of libigl instead.
```
cmake -DFETCHCONTENT_SOURCE_DIR_LIBIGL=<path-to-libigl> ..
```
When changing this value, do not forget to clear your `CMakeCache.txt`, or to update the cache variable
via `cmake-gui` or `ccmake`.
