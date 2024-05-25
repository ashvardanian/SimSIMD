#!/bin/sh

echo $1 > VERSION &&
    sed -i "s/^\(#define SIMSIMD_VERSION_MAJOR \).*/\1$(echo "$1" | cut -d. -f1)/" include/simsimd/simsimd.h &&
    sed -i "s/^\(#define SIMSIMD_VERSION_MINOR \).*/\1$(echo "$1" | cut -d. -f2)/" include/simsimd/simsimd.h &&
    sed -i "s/^\(#define SIMSIMD_VERSION_PATCH \).*/\1$(echo "$1" | cut -d. -f3)/" include/simsimd/simsimd.h &&
    sed -i "s/^version = \".*\"/version = \"$1\"/" Cargo.toml &&
    sed -i "s/\"version\": \".*\"/\"version\": \"$1\"/" package.json &&
    sed -i "s/VERSION [0-9]\+\.[0-9]\+\.[0-9]\+/VERSION $1/" CMakeLists.txt
