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
- **Against real-hardware fixtures, v86 on aarch64 passes 16,509 of 16,529
  nasmtests**, interpreter and `--force-jit` alike. The 20 failures are a real
  v86 bug on non-x86 hosts (`sqrt` NaN sign, see the
  [verification log](#verification-log)).
- **For 64-bit code, `make fuzz-hw-tests`** compares v86 with a CPU on the
  64-bit fuzz cases: natively on x86 (CI does this), provisionally under
  qemu-user here. See [64-bit](#64-bit).

Sources: the measurements below come from
[isolated-devcontainers-suede PR #53](https://github.com/pmalacho-mit/isolated-devcontainers-suede/pull/53)
([x86-emulation.md](https://github.com/pmalacho-mit/isolated-devcontainers-suede/blob/main/docs/x86-emulation.md),
[x86-equivalence-evidence.md](https://github.com/pmalacho-mit/isolated-devcontainers-suede/blob/main/docs/x86-equivalence-evidence.md)).
Its `release/emulation-equivalence.sh gen ref` produces real fixtures for any
v86_64 commit on any x86 machine with Docker, and the PR has the AMD and Intel
fixture sets for `c394574` attached.

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
  `.img` files, in `ubuntu:24.04` with the repository at `/src`. The path
  matters: each `.img` embeds the absolute path of its `.asm` (a FILE symbol),
  so the same tests built elsewhere hash differently. `tests/nasm/img_digest.js`
  hashes only what is loaded (entry point and allocated sections), which is the
  same across paths and across nasm 2.16.01/ld 2.42 and nasm 2.16.03/ld 2.44.
- **Real x86 faulted in 920 of the 3,345 generated x87 tests** (register-stack
  overflow/underflow), which is why the x87 stack-fault row below is so large.
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
- **Below `run.js`'s masks (raw register bits only):** qemu doesn't set the
  x87 exponent field to all ones on MMX writes (real CPUs do, and so does v86),
  leaves RF clear in fault frames, misses FPU C1 ("rounded up") and DE,
  `MXCSR.DE`, and doesn't normalise reserved bits of the control word on
  `FLDCW`. Intel and AMD agreed on all of it, and their single-step traces
  differ only in one C1 bit midway through `f2xm1`.
- **Not observable through qemu's gdb server:** `ftag`, `fiseg`, `fioff`,
  `foseg`, `fooff` and `fop` always read 0.
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
- v86 against real fixtures: 16,509 of 16,529 (measured here, see the
  verification log). Against emulated ones: 15,462.

The trustworthy areas by category: integer ALU and flags (except `F6`/`F7 /1`),
shifts, string ops, MMX and SSE results, and x87 arithmetic, loads and stores
that don't overflow or underflow the register stack.

And you don't need emulated fixtures, because real ones are cheap.

## Recommended workflow

1. **Real fixtures come from x86 CI.** The `Build and test` job in
   `.github/workflows/ci.yml` generates them on `ubuntu-latest` (x86) and
   uploads them as the artifact `nasmtests-fixtures-<hash>`, where `<hash>`
   is a hash of the image digests and of the fixture extraction scripts, so
   the name changes exactly when the fixtures would. The artifact holds
   `fixtures.tar.gz` (the `.fixture` files), `images.digest`
   (`tests/nasm/img_digest.js` output) and `toolchain.txt` (nasm, ld, gdb, CPU).
   Running `make nasmtests` on any x86 machine, Intel or AMD, gives the same
   fixtures.
2. **Run locally against them:**
   - Run `tests/nasm/create_tests.js`, then check your images against the
     artifact's: `tests/nasm/img_digest.js --check images.digest`. If any
     differ, your tests changed (or your nasm encodes differently) and those
     fixtures are stale for them; stop.
   - With the fixture sets attached to suede PR #53 (`IMG_SHA256`, plain
     sha256 of images built at `/src`), build the same way to compare:
     `docker run --rm -u "$(id -u):$(id -g)" -v "$PWD":/src -w /src/tests/nasm -e PATH=/opt/x86ld:/usr/bin:/bin v86-qemu-gdb node create_tests.js`
     (the `v86-qemu-gdb` image from the next section),
     then `cd tests/nasm/build && sha256sum -c IMG_SHA256`.
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
    nasm binutils binutils-x86-64-linux-gnu qemu-user-static gdb-multiarch nodejs ca-certificates \
 && rm -rf /var/lib/apt/lists/* && mkdir -p /opt/x86ld && ln -s /usr/bin/x86_64-linux-gnu-ld /opt/x86ld/ld
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
Each test gets its own port (a qemu that outlives its test must never be
attached to as the next one's), and tests whose gdb couldn't connect are
retried one at a time, up to three times; the script fails if any fixture is
still missing.
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

The nasmtests are 32-bit only, and their measurement covered `qemu-i386`.
For long mode there is `make fuzz-hw-tests` (`tests/fuzz/hw.js`):

- It takes the 64-bit cases of the instruction fuzzer (`tests/fuzz/gen_case.js`:
  random encodings with REX, r8-r15, xmm8-15, RIP-relative and 0x67
  addressing, random GPR/x87/MMX/XMM/flags/MXCSR state), 27,429 of them by
  default, and runs each one twice: in v86, and as a ring-3 Linux process on
  an x86-64 CPU (`tests/fuzz/hw/runner.c`, a freestanding static binary that
  reads the final state from the signal context, so no gdb is involved).
  The final registers, flags, x87/MMX/XMM state, MXCSR, scratch memory and
  exception vector are compared.
- `tests/fuzz/run.js` only compares v86's interpreter with its JIT, so it
  can't see bugs both share. This can.
- **On x86-64 Linux the CPU is the reference.** CI runs it on GitHub's x86
  runners (report only for now, see the `fuzz-hw-log` artifact). On aarch64,
  `make fuzz-hw-tests` runs the runner under qemu-user
  (`docker run --platform linux/amd64`), which is **provisional**: the script
  detects it (qemu doesn't report `trapno`) and says so.
- Differences that come from the environment rather than the CPU (ring 3 vs
  ring 0, the first 64 KiB that Linux can't map, environment-dependent
  instructions such as `cpuid`, `pushf`, segment selectors and FPU
  environment stores) are excluded or reported as *inconclusive*; documented
  v86 limitations are reported as *known*. Only the rest are mismatches.

**qemu-x86_64 8.2.2 defects found this way** (in addition to the 32-bit
list above, which applies too; `hw.js` filters these when the reference is
qemu):

| Defect | How it shows |
|---|---|
| `trapno` is -1 in the signal context | the vector has to be inferred from the signal |
| A non-canonical address faults like an unmapped one (`SEGV_MAPERR` with the address) | looks like #PF; real CPUs raise #GP |
| REX.R/REX.B are applied to MMX register operands of some instructions | `41 0F 38 04 C0` (`pmaddubsw mm0, mm0`) yields 0; without the REX byte it's correct |
| MMX writes leave the x87 exponent at 0 | real CPUs and v86 set it to `0xffff` |

When long mode needs new hard-coded expected values (`tests/longmode/`), take
them from real x86, never from `qemu-x86_64`.

## Verification log

What was run to check the qemu + gdb approach, on an aarch64 devcontainer
(Debian trixie, Node 24, rustc 1.99, nasm 2.16.03) with `ubuntu:24.04` as the
qemu/gdb container (qemu-i386 8.2.2, gdb 15.1), v86_64 at `99e5057` plus this
branch:

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
- **`run.js` against real-hardware fixtures:** the AMD EPYC 9V74 and Intel
  Xeon Gold 6140 sets attached to suede PR #53 (`c394574`; the nasm test
  inputs are unchanged since). All 16,529 images, built in `ubuntu:24.04` at
  `/src`, matched their `IMG_SHA256`. **16,509 of 16,529 pass**, with both
  fixture sets, with and without `--force-jit`. The 20 failures are exactly
  the `sqrt` NaN-sign row above.
- **`make fuzz-hw-tests` under qemu-user** (`qemu-x86_64-static` 8.2.2 from the
  VM's binfmt registration), 27,429 cases, interpreter: 20,635 match, 1,596
  mismatch, 2,487 known v86 limitations (2,433 misaligned SSE operands without
  #GP, 54 NaN sign), 2,711 inconclusive (2,461 of them the qemu REX/MMX
  defect). The runner takes about 20 s, the comparison about 50 s.
  Mismatches that are certain v86 bugs, because the SDM is unambiguous: in
  64-bit mode v86 executes `06`, `0E`, `16`, `1E` (push segment register),
  `60`, `61`, `82`, `D4` and `D5` instead of raising #UD. The rest (SSE
  rounding modes, NaN propagation, some MMX and flag results) need the real
  x86 run in CI to decide.
- **`--jit`:** 148 of the cases crash the JIT with an invalid wasm module
  (`i64.store[0] expected type i32, found i64.load`). `tests/fuzz/run.js`
  reproduces it (`TEST_NAME='^fuzz64_33_r_32r_2\+2$'`), so it's a JIT bug
  at this commit, independent of this tooling.
- **Not verified:** the CI steps in `ci.yml` (fixture artifact, fuzz-hw-tests),
  which only run on GitHub.

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
