Windows build :
cmake -S . -B build
cmake --build build --target main --config Release

Linux build :
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build