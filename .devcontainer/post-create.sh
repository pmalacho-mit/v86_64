#!/usr/bin/env bash
# Runs once when the dev container is created. Installs the build and test
# dependencies of v86_64 (the same ones as its CI, .github/workflows/ci.yml),
# picking cross compilers instead of 32-bit multilib on ARM hosts.
set -euo pipefail
cd "$(dirname "$0")/.."

# Behind an intercepting proxy (e.g. desolate), the proxy CA's env vars live in
# a profile script that non-login shells don't read. Pick them up if present.
for f in /etc/profile.d/*ca*.sh; do [ -f "$f" ] && . "$f"; done

arch="$(dpkg --print-architecture)"
echo "== system packages ($arch)"
pkgs=(
  make clang lld          # clang compiles lib/softfloat and lib/zstd to wasm32
  openjdk-21-jre-headless # Closure Compiler (downloaded by make into closure-compiler/)
  nasm                    # test programs in tests/nasm, tests/longmode, tests/devices
  qemu-system-x86         # reference output for make qemutests
  gdb wget curl unzip p7zip-full python3 time
)
case "$arch" in
  amd64) pkgs+=(gcc-multilib libc6-dev-i386) ;;            # gcc -m32 works natively
  *)     pkgs+=(gcc-i686-linux-gnu gcc-x86-64-linux-gnu) ;; # cross compilers for 32/64-bit x86 test programs
esac
sudo apt-get update -qq
sudo apt-get install -y -qq --no-install-recommends "${pkgs[@]}" >/dev/null

echo "== Rust targets"
# The rust feature installs these; repeat in case the toolchain was replaced.
rustup target add wasm32-unknown-unknown i686-unknown-linux-musl x86_64-unknown-linux-musl >/dev/null
rustup component add rustfmt >/dev/null

if [ -f package.json ]; then
  echo "== npm dependencies"
  npm install --no-audit --no-fund >/dev/null
fi

cat <<EOF

Ready ($arch). Common commands:
  make all-debug                 debug build (build/libv86-debug.mjs, build/v86-debug.wasm)
  make all                       release build (build/libv86.mjs, build/v86.wasm, ...)
  ./tools/serve.mjs --port 8000  test pages
  make api-tests                 needs images/ (see the "Download uncached images" step in .github/workflows/ci.yml)
  make longmode-tests            64-bit guests
  ALPINE_ISO=images/alpine-virt-3.19.1-x86_64.iso ./tests/longmode/alpine.js
EOF
if [ "$arch" != amd64 ]; then cat <<'EOF'
On this ARM host:
  make qemutests CC=i686-linux-gnu-gcc       (and jitpagingtests) use the x86 cross compiler
  make nasmtests                             needs a real x86 CPU (runs tests natively under gdb); CI covers it
EOF
fi
