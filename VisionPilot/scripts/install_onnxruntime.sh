#!/usr/bin/env bash
# Install ONNX Runtime (GPU, CUDA 12) for VisionPilot from the NuGet package
# Microsoft.ML.OnnxRuntime.Gpu.Linux, laid out like the former GitHub release tarball:
#
#   <install_dir>/onnxruntime-linux-x64-gpu_cuda12-<version>/
#   ├── include/   (buildTransitive/native/include/*)
#   └── lib/       (runtimes/linux-x64/native/*.so + libonnxruntime.so.1 -> libonnxruntime.so)
#
# The GitHub release asset onnxruntime-linux-x64-gpu_cuda12-<version>.tgz is no longer available.
# Notes:
#   - Microsoft.ML.OnnxRuntime (without .Gpu.Linux) is the CPU-only package: it has no CUDA provider.
#   - On nuget.org the Gpu.Linux package is a CUDA 12 build up to 1.26.0 and a CUDA 13 build from
#     1.27.0. The CARLA simulation setup uses CUDA 12, so this script refuses anything that is not CUDA 12.
#
# Usage: install_onnxruntime.sh <version> [install_dir=/opt/onnx_runtime] [sha256 of the .nupkg]
# Requires: curl, unzip, readelf (binutils) (run as a user that can write install_dir, e.g. with sudo)
set -euo pipefail

version=${1:?"usage: $0 <version> [install_dir] [sha256]"}
install_dir=${2:-/opt/onnx_runtime}
sha256=${3:-}

package=microsoft.ml.onnxruntime.gpu.linux
dest="$install_dir/onnxruntime-linux-x64-gpu_cuda12-$version"

if [ -e "$dest/lib/libonnxruntime.so.1" ]; then
    echo "ONNX Runtime $version is already installed at $dest"
    exit 0
fi

work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT

echo "Downloading $package $version from nuget.org"
curl -fsSL --retry 3 -o "$work_dir/ort.nupkg" \
    "https://api.nuget.org/v3-flatcontainer/${package}/${version}/${package}.${version}.nupkg"

if [ -n "$sha256" ]; then
    echo "$sha256  $work_dir/ort.nupkg" | sha256sum -c -
fi

unzip -q "$work_dir/ort.nupkg" 'buildTransitive/native/include/*' 'runtimes/linux-x64/native/*' -d "$work_dir/pkg"
native="$work_dir/pkg/runtimes/linux-x64/native"

if [ ! -f "$native/libonnxruntime_providers_cuda.so" ]; then
    echo "ERROR: $package $version has no CUDA execution provider" >&2
    exit 1
fi
# Check the real dynamic dependency (the binary also contains other CUDA version strings).
if ! readelf -d "$native/libonnxruntime_providers_cuda.so" | grep -q 'NEEDED.*\[libcudart\.so\.12\]'; then
    echo "ERROR: $package $version is not a CUDA 12 build (nuget.org ships CUDA 13 builds from 1.27.0)." >&2
    echo "       Use 1.26.0 or older with CUDA 12." >&2
    exit 1
fi

stage="$work_dir/onnxruntime-linux-x64-gpu_cuda12-$version"
mkdir -p "$stage/include" "$stage/lib"
cp -a "$work_dir/pkg/buildTransitive/native/include/." "$stage/include/"
cp -a "$native/." "$stage/lib/"
# The NuGet package ships only libonnxruntime.so, whose SONAME is libonnxruntime.so.1.
ln -s libonnxruntime.so "$stage/lib/libonnxruntime.so.1"
chmod -R a+rX "$stage"

mkdir -p "$install_dir"
rm -rf "$dest"
mv "$stage" "$dest"
echo "ONNX Runtime $version installed at $dest"
echo "  export ONNXRUNTIME_ROOT=$dest"
