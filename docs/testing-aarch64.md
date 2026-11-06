# Testing v86_64 on an aarch64 (non-x86) host

## Summary

- **You almost never need x86 hardware.** Every v86 test suite exercises v86,
  and v86 runs natively on aarch64.
- **The one exception is generating nasmtests fixtures.** `gen_fixtures.js`
  runs each test binary under gdb on a real CPU. Running v86 *against*
  fixtures needs no x86 at all.
- **So: get fixtures from x86 CI (or any x86 machine) and run everything
  locally.** Don't generate fixtures under qemu, except as an explicitly
  *provisional* pre-check for tests you've just written (see
  [Provisional fixtures under qemu](#provisional-fixtures-under-qemu)).

> **For AI agents:** if a change adds or modifies nasm tests (anything under
> `tests/nasm/` that changes the generated `.img` files, or the instruction
> table in `gen/x86_table.js` that `create_tests.js` reads) and you are not on
> an x86 host, you cannot produce trustworthy fixtures yourself. **Tell the
> human user to re-run the fixture generator on an x86 machine**
> (`make nasmtests`, or `tests/nasm/create_tests.js && tests/nasm/gen_fixtures.js`)
> and to hand you the resulting `tests/nasm/build/*.fixture` files, or to let
> CI regenerate them and download the artifact. Any x86 CPU will do: Intel and
> AMD produced bit-identical fixtures for all 16,529 tests. Until then, report
> any nasmtests result you got from qemu fixtures as provisional.

## Suite by suite

| Suite | Needs real x86? | Notes for aarch64 |
|---|---|---|
| `nasmtests`, `nasmtests-force-jit` | Only to generate fixtures | `create_tests.js` needs `nasm`, plus an `ld` that handles `-m elf_i386`: install `binutils-x86-64-linux-gnu` and put `x86_64-linux-gnu-ld` on PATH as `ld` (e.g. `mkdir -p ~/bin && ln -s /usr/bin/x86_64-linux-gnu-ld ~/bin/ld`). The native aarch64 `ld` fails with `unrecognised emulation mode: elf_i386`. `run.js` is node only. |
| `qemutests`, `qemutests-release` | No | The reference is `qemu-system-x86_64` with TCG and `-cpu Westmere`, in upstream's x86 CI too, so it's the same oracle here. Building `tests/qemu/test-i386` needs `gcc -m32`. |
| `jitpagingtests` | No | Needs i386 binaries built with `gcc -m32`. |
| `kvm-unit-test` | No | `build.sh` compiles x86 `.flat` images with `gcc`. |
| `longmode-tests` | No | Expected values are hard-coded in `run.js`. See [64-bit](#64-bit) for where *new* ones should come from. Needs nasm 3.x, see the [verification log](#verification-log). |
| `api-tests`, `tests`, `devices-test`, `expect-tests`, `fuzz-tests`, `rust-test` | No | node, wasm, cargo. |

**Building x86 artifacts:** do it in a `linux/amd64` container
(`docker run --platform linux/amd64 debian:bookworm …`). Compilers and
assemblers are deterministic, and emulation is reliable for building. It's
slow, but correct.

**One caveat on `qemutests`:** qemu computes x87 `fsin`/`fcos`/`fptan`/`fsincos`
with the host's libm. If the comparison against the reference ever differs only
on x87 transcendental output, suspect the aarch64 libm before suspecting v86.

## nasmtests: the measured facts

These were measured on v86_64 `c394574`, all 16,529 tests, against an Intel
Xeon Gold 6140, an AMD EPYC 9V74, and qemu-i386 8.2.2 on an aarch64 VM:

- **The fixtures are a pure function of the inputs** (`tests/nasm/*.asm`,
  `create_tests.js` and its fixed seed, `header.inc`/`footer.inc`,
  `gdb-extract-def`, nasm/ld versions). All three machines built byte-identical
  `.img` files.
- **Intel and AMD produced bit-identical fixtures** for all 16,529. Any real
  x86 CPU is a valid reference.
- **qemu-emulated fixtures disagree with real hardware on 1,038 tests**, even
  under `run.js`'s own tolerance and masks:

  | Cause | Tests |
  |---|---|
  | x87 register-stack overflow/underflow: real CPUs set IE+SF and load the indefinite NaN; qemu doesn't | 920 |
  | `FCOMI`/`FUCOMI`/`FCOMIP`/`FUCOMIP`: qemu doesn't clear OF/SF/AF | 56 |
  | `F6`/`F7 /1` and `0F 18` hint NOPs: qemu raises #UD | 33 |
  | Register forms of `MOVNTQ`/`MOVNTDQ`/`MOVLPD`/`MOVHPD`: qemu executes them instead of #UD | 20 |
  | 16-bit `ENTER`, `FSAVE` reserved bits, `fprem` C3, `fyl2xp1` out of range | 9 |

- **Hidden by `run.js`'s tolerance:** qemu computes `fsin`, `fcos`, `fptan` and
  `fsincos` in double precision (`fptan(π)`: qemu `-1.2e-16`, real `+5.4e-20`).
- **Emulated fixtures also have a defect the comparison didn't measure.**
  qemu's gdb server reports the x87 tag word as 0. `gdb-extract-def` therefore
  records stale values for empty st registers where a real fixture says
  `"invalid"`, and `run.js` then checks registers it would otherwise skip.
  (`tests/nasm/gen_fixtures_qemu.sh` works around the worst of this, see
  below, but can't recover which registers are *empty*.)

**Are the other 15,491 reliable?** As emulated fixtures for *this* generated
set, they match real hardware under `run.js`'s rules. Still, don't treat them
as ground truth:

- That list is a property of these specific random inputs. New tests or seeds
  can hit the categories above.
- The tag-word defect can produce failures that aren't v86 bugs.
- How v86_64 itself fares against emulated versus real fixtures hasn't been
  measured.

The trustworthy areas by category: integer ALU and flags (except `F6`/`F7 /1`),
shifts, string ops, MMX and SSE results, and x87 arithmetic, loads and stores
that don't overflow or underflow the register stack.

And you don't need emulated fixtures, because real ones are cheap.

## Recommended workflow

1. **Real fixtures come from x86 CI.** The `Build and test` job in
   `.github/workflows/ci.yml` generates them on `ubuntu-latest` (x86) and
   uploads them as the artifact `nasmtests-fixtures-<hash>`, where `<hash>`
   covers every input listed above, so the name changes exactly when the
   fixtures would. The artifact holds `fixtures.tar.gz` (the `.fixture` files),
   `images.sha256` (hashes of the `.img` files) and `toolchain.txt` (nasm/ld
   versions). Running
   `make nasmtests` on any x86 machine, Intel or AMD, gives the same files.
2. **Run locally against them:**
   - Run `tests/nasm/create_tests.js`, then check your `.img` hashes against
     the artifact's: `cd tests/nasm/build && sha256sum -c images.sha256`. If
     they differ, your tests changed and those fixtures are stale; stop.
   - Unpack the fixtures into `tests/nasm/build/`
     (`tar xzf fixtures.tar.gz -C tests/nasm/build`) and run `tests/nasm/run.js` (and
     `tests/nasm/run.js --force-jit`). This runs all 16,529 tests here, against
     real-hardware truth. Don't run `make nasmtests` after this: it would run
     `gen_fixtures.js`, which regenerates any fixture that is missing or older
     than its `.img`. Call `create_tests.js` first and copy the fixtures in
     afterwards (or run `run.js` directly).
3. **When you add or change nasm tests:** the real fixtures for them must come
   from x86. **Ask the human to re-run the test generator on an x86 machine**
   (any Intel or AMD host), or to push and let CI regenerate them. For a fast
   local loop before that, you may generate *provisional* fixtures for just the
   new tests under qemu (next section). Then:
   - Never commit them.
   - Treat any failure in the categories above as unknown, not as a bug.
   - When local and CI disagree, CI wins.
4. **Upstream v86's `qemutests` and differential testing against qemu are
   fine** wherever the known-bad list above doesn't apply. Don't use qemu as
   the arbiter for x87 exception and stack behaviour, the `FCOMI` family,
   transcendentals, or the listed encodings.

## Provisional fixtures under qemu

Running the test binaries under qemu's user-mode emulation with its gdb stub,
and extracting the state with the same `gdb-extract-def`, works on aarch64.

gdb can't run x86 binaries itself on aarch64, and **gdb doesn't work inside a
`linux/amd64` or `linux/386` container** (`ptrace: Function not implemented`).
Use a *native* aarch64 container (or host) with `qemu-user-static` and
`gdb-multiarch` (Ubuntu 24.04: qemu 8.2.2, gdb 15.1):

```sh
# one-time: an image with the tools. The Docker daemon may see only the
# repository, so bind-mount it at the same path when running.
docker build -t v86-qemu-gdb - <<'EOF'
FROM ubuntu:24.04
RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    nasm binutils qemu-user-static gdb-multiarch ca-certificates && rm -rf /var/lib/apt/lists/*
EOF

# create the .img files (on the host, with the ld shim from above) ...
tests/nasm/create_tests.js

# ... and generate provisional fixtures for the tests matching a regex, which
# is matched against the name without ".img" (default: every .img without an
# up-to-date .fixture)
docker run --rm -v "$PWD":"$PWD" -w "$PWD" v86-qemu-gdb \
    tests/nasm/gen_fixtures_qemu.sh '^(fptan|cmpxchg8b)$'

# then, as usual
TEST_NAME='^(fptan|cmpxchg8b)\.img' tests/nasm/run.js
```

`tests/nasm/gen_fixtures_qemu.sh` starts `qemu-i386 -g <port> test.img`,
attaches `gdb-multiarch`, continues to the `loop` breakpoint (or the first
SIGILL/SIGFPE/SIGSEGV/SIGBUS) and runs `gdb-extract-def`'s `extract-state`,
rewritten on the fly so that `run` becomes `target remote` + `continue`.
`gdb-qemu-x87.py` classifies x87 registers from their raw bits, because
qemu reports the tag word as 0 and gdb aborts printing the unsupported 80-bit
values that MMX initialisation leaves in st0-st7 (nearly every generated test
does that). It writes `tests/nasm/build/PROVISIONAL-QEMU-FIXTURES` as a marker;
delete it and the fixtures before switching to real ones.

### Single-stepping by hand

To see what a test does step by step (single-stepping this way was found to
match real hardware step for step; the check in the verification log below
confirms it runs, not that it matches hardware):

```sh
# in a native ubuntu:24.04 container: apt-get install qemu-user-static gdb-multiarch
qemu-i386-static -g 1234 ./t.img &
cat > step.gdb <<'EOF'
set architecture i386
file ./t.img
target remote :1234
set $n = 0
while $pc != loop
  printf "step %2d  eip=0x%08x  eax=0x%08x ecx=0x%x eflags=0x%03x\n", $n, $pc, $eax, $ecx, $eflags
  stepi
  set $n = $n + 1
end
kill
EOF
gdb-multiarch -batch -x step.gdb
```

`loop:` is the label `footer.inc` puts after the test code. Any test image
from `tests/nasm/build/` works as `t.img`, as does one you assemble by hand
the way `create_tests.js` does (`nasm -felf32`, then
`ld -m elf_i386 --section-start=.bss=0x100000 --section-start=.text=0x80000 --section-start=.multiboot=0x20000`).

**Reading x87 registers:** aarch64 gdb converts them through a double when
printing, so read the exact bits from `info registers st0`'s `(raw 0x…)`. The
register-stack tag word and `fop` always read 0 there.

## 64-bit

The measurement covered 32-bit code through `qemu-i386`. `qemu-x86_64` uses the
same TCG front end, so expect the same kinds of problems (x87 stack faults,
transcendental precision, decode corner cases), but it wasn't measured. When
long mode needs new expected values, get them from real x86. For example, a CI
job that runs a 64-bit nasm test under gdb, the same way `gen_fixtures.js` does
for 32-bit. **Don't take them from `qemu-x86_64`.**

## Verification log

What was run to check the qemu + gdb approach, on an aarch64 devcontainer
(Debian trixie, Node 24, rustc 1.99, nasm 2.16.03) with `ubuntu:24.04` as the
qemu/gdb container (qemu-i386 8.2.2, gdb 15.1), v86_64 at `99e5057`:

- **Single-stepping works.** `qemu-i386-static -g` plus `gdb-multiarch`
  `stepi` walked a 21-instruction test from `main` to `loop`, with correct
  register and flag values at every step (for example `add eax, 1` on
  `0x7fffffff` gave `eax=0x80000000, eflags=0xa96`, OF set).
- **Fixture extraction works.** `create_tests.js` produced all 16,529 `.img`
  files, and `gen_fixtures_qemu.sh` produced a fixture for each of them (about
  11 minutes with 8 jobs). `gdb-extract-def` is used unchanged apart from the
  rewrites described above. Without `gdb-qemu-x87.py`, about 90% of the
  generated tests failed to produce a fixture at all.
- **`run.js` against those provisional fixtures:** 15,462 of 16,529 pass, and
  the same 1,067 fail with and without `--force-jit`. They break down as:

  | Failures | Cause |
  |---|---|
  | 976 (`gen_d8`..`gen_df`) | x87 stack overflow/underflow (920) and `FCOMI` family (56): the qemu defects listed above |
  | 33 (`gen_f6`, `gen_f7`, `gen_0f18`) | hint NOPs that qemu raises #UD for |
  | 20 (`gen_0fe7`, `gen_660fe7`, `gen_660f12`, `gen_660f16`) | register forms of `MOVNTQ`/`MOVNTDQ`/`MOVLPD`/`MOVHPD` |
  | 9 (`gen_c8`, `fyl2xp1`, `fsave_frstor`, `fprem`, `fprem1`) | 16-bit `ENTER`, `FSAVE` reserved bits, `fprem` C3, `fyl2xp1` |
  | 20 (`gen_0f51`, `gen_f30f51`, `gen_660f51`, `gen_f20f51`) | **not a qemu defect:** `sqrt` of a negative number. A real x86 CPU, and qemu, return the default NaN `0xffc00000`; v86 compiles to a wasm `f32.sqrt`/`f64.sqrt`, whose NaN sign is up to the host, and gives `0x7fc00000` on aarch64. Expect these 20 to fail against *real* fixtures on an aarch64 host too |
  | 9 (`comiss`, `ucomiss`, `psrl*`, `pcmpeqw`, `pcmpgtb`, `gen_0fd2`) | MMX state that `gdb-qemu-x87.py` can't tell from x87 state (all registers zero or small), so the tag word is wrong |

  The first four rows add up to exactly the 1,038 tests the original
  measurement found to disagree between qemu and real hardware (920 + 56 + 33 +
  20 + 9): the same set, found independently. This is the expected outcome, and
  it is why these fixtures are provisional only. It does **not** show that v86
  passes against real hardware.
- **Not verified:** a run of `run.js` against real-hardware fixtures (no x86
  machine was available), and the CI artifact step in `ci.yml` (it can only run
  on GitHub). `qemu-x86_64` for 64-bit tests was not tried.

Other things found while doing this on aarch64:

- `tests/nasm/run.js` passed `fs.readFileSync(...).buffer` to `load_multiboot`.
  On this host Node returned a pooled Buffer, whose `.buffer` is a larger
  ArrayBuffer with the file at a non-zero offset, so the multiboot header was
  never found and every test panicked or timed out. `run.js` now copies out
  exactly the file's bytes.
- `make longmode-tests` needs **nasm 3.x**. nasm 2.16.03 (Debian trixie)
  assembles `o32 jmp 0x08:longmode` in 16-bit code with a 16-bit offset
  (`66 ea 10 01 08 00` instead of `66 ea 12 01 0f 00 08 00`), so the guest
  jumps to a garbage selector and v86 triple-faults. With nasm 3.01 (run in a
  `linux/amd64` container) all of `make longmode-tests` passes: 151 tests in
  four configurations, multiboot64 and selfipi. `nasm_3.01-1_amd64.deb` can be
  unpacked with `dpkg-deb -x` and run with `docker run` as a shim on `PATH`.
