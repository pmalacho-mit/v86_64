#!/usr/bin/env bash
# PROVISIONAL nasmtests fixtures from qemu-user + gdb-multiarch.
#
# For hosts that are not x86 (e.g. aarch64), where gen_fixtures.js cannot run
# the test binaries under native gdb. The test binaries are run under
# qemu-i386 with its gdbstub, and the same gdb-extract-def script is used to
# dump the state at the `loop` label.
#
# These fixtures are NOT hardware truth (see
# docs/testing-aarch64.md). Never commit them, and treat failures in the known-bad
# categories as "unknown", not as v86 bugs. Real fixtures come from running
# gen_fixtures.js on any x86 machine (Intel or AMD give identical results).
#
# Requires: qemu-i386 (or qemu-i386-static) and gdb-multiarch on PATH.
# Usage: tests/nasm/gen_fixtures_qemu.sh [name-regex]   (default: all tests
#        in tests/nasm/build/ that have no up-to-date .fixture)
# Env:   JOBS (default: nproc), BASE_PORT (default: 12340; each test uses
#        its own port, BASE_PORT + test index)

set -u

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build="$dir/build"
filter="${1:-.}"
jobs="${JOBS:-$(nproc)}"
base_port="${BASE_PORT:-12340}"

qemu="$(command -v qemu-i386-static || command -v qemu-i386 || true)"
gdb="$(command -v gdb-multiarch || true)"
if [ -z "$qemu" ] || [ -z "$gdb" ]; then
    echo "gen_fixtures_qemu.sh: need qemu-i386 (qemu-user) and gdb-multiarch on PATH" >&2
    exit 1
fi

export qemu gdb dir build base_port

gen_one() {
    local name="$1" slot="$2"
    local img="$build/$name.img" out="$build/$name.fixture"
    local port=$((base_port + slot))
    local script
    script="$(mktemp)"

    # Reuse gdb-extract-def unchanged, except that `run` becomes "attach to
    # the qemu gdbstub and continue to the breakpoint on `loop`".
    # Also: decide "special" x87 registers (printed as "invalid") and the tag
    # word from the registers' contents, since qemu reports the tag word as 0
    # (see gdb-qemu-x87.py).
    sed -e "s|^  run\$|  target remote :$port\n  continue|" \
        -e "s|^  file \$arg0\$|  file \$arg0\n  break loop|" \
        -e 's|^  if (\$ftag >> (2 \* ((\$top + \([0-7]\)) \& 7)) \& 3) != 2$|  if $qemu_st_special(\1) == 0|' \
        -e 's|^  printf "    %d,\\n", \$ftag$|  printf "    %d,\\n", $qemu_ftag()|' \
        "$dir/gdb-extract-def" > "$script"

    "$qemu" -g "$port" "$img" >/dev/null 2>&1 &
    local qpid=$!
    # Wait for the gdbstub to listen. Don't probe by connecting: the stub
    # accepts exactly one connection, which must be gdb's.
    local hex
    hex="$(printf '%04X' "$port")"
    for _ in $(seq 1 100); do
        grep -qi ":$hex [0-9A-F:]* 0A " /proc/net/tcp /proc/net/tcp6 2>/dev/null && break
        sleep 0.1
    done

    timeout 120 "$gdb" -batch \
        --eval-command="set architecture i386" \
        --eval-command="set disable-randomization off" \
        --command="$dir/gdb-qemu-x87.py" \
        --command="$script" \
        --eval-command="set breakpoint pending on" \
        --eval-command="catch signal SIGFPE" \
        --eval-command="catch signal SIGILL" \
        --eval-command="catch signal SIGSEGV" \
        --eval-command="catch signal SIGBUS" \
        --eval-command="extract-state $img $out" >/dev/null 2>&1
    local status=$?
    kill "$qpid" 2>/dev/null
    wait "$qpid" 2>/dev/null
    rm -f "$script"
    if [ $status -ne 0 ] || ! grep -q -- '---END JSON---' "$out" 2>/dev/null; then
        # a fixture without JSON is only valid if the test raised a signal
        if ! grep -q 'signal SIG' "$out" 2>/dev/null; then
            rm -f "$out"
        fi
    fi
    return 0
}
export -f gen_one

: > "$build/PROVISIONAL-QEMU-FIXTURES"

todo="$(mktemp)"
for img in "$build"/*.img; do
    name="$(basename "$img" .img)"
    [[ "$name" =~ $filter ]] || continue
    fixture="$build/$name.fixture"
    if [ -e "$fixture" ] && [ ! "$img" -nt "$fixture" ]; then
        continue
    fi
    echo "$name"
done > "$todo"

# One port per test, never reused: a qemu that outlives its test must not be
# attached to as the next one's.
awk '{ print $0, NR }' "$todo" | xargs -r -P "$jobs" -L1 bash -c 'gen_one "$@"' _

# Under load, gdb sometimes connects before qemu listens and times out (seen
# for ~0.5% of tests at 8 jobs). Retry those one at a time, on fresh ports.
port_offset=$(( $(wc -l < "$todo") + 1 ))
for attempt in 1 2 3; do
    while read -r name; do
        [ -e "$build/$name.fixture" ] && continue
        gen_one "$name" "$port_offset" < /dev/null # gdb and qemu must not eat the list
        port_offset=$((port_offset + 1))
    done < "$todo"
done

failed=0
while read -r name; do
    if [ ! -e "$build/$name.fixture" ]; then
        echo "gen_fixtures_qemu.sh: no usable state for $name" >&2
        failed=$((failed + 1))
    fi
done < "$todo"
total=$(wc -l < "$todo")
rm -f "$todo"

echo "[+] $((total - failed)) of $total provisional (qemu) fixtures written to $build; do not commit them"
[ "$failed" -eq 0 ]
