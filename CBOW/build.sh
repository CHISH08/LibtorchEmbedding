#!/bin/bash
set -e

BUILD_DIR="build"

# Удаляем папку build, если она существует
if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"
fi

mkdir "$BUILD_DIR"

cd "$BUILD_DIR"
cmake .. 
make
clear
./cbow
