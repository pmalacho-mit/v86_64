Use the corresponding `make` target in the root directory to run a test. The
following list is roughtly sorted from most interesting/useful to least.

- [nasm](nasm/): Small unit tests written in assembly, which are run using gdb
  on the host.
- [fuzz](fuzz/): Instruction set fuzzer. Generates random test cases (operand
  encodings and initial CPU state) for every implemented instruction from
  gen/x86_table.js and compares the resulting CPU state between the
  interpreter and the jit. Covers 32-bit protected mode (with and without
  paging) and 64-bit long mode: REX prefixes, r8-r15/xmm8-15, RIP-relative
  and address-size-override (0x67) addressing, the 16-bit modrm decoder,
  non-canonical addresses, imm64, misaligned SSE operands, randomized MXCSR
  and fpu control word, special float values (NaN/denormal/inf), chained
  instruction sequences (lazy-flag producer/consumer pairs), lock-prefixed
  read-modify-write ops and conditional jumps over marker writes. Seeded and
  reproducible; parallelized with cluster workers; survives (and reports)
  emulator crashes. See the header of fuzz/run.js for the
  FUZZ_SEED/FUZZ_CASES knobs. `make fuzz-hw-tests` (fuzz/hw.js) runs the
  64-bit cases on the host's x86-64 CPU too and compares v86 with it, which
  also finds bugs the interpreter and the JIT share.
- [qemu](qemu/): Based on tests from qemu. Builds a Linux binary, which tests
  many CPU features, which are then compared to a run on qemu.
- [kvm-unit-test](kvm-unit-test/): Based on tests from the KVM project, tests
  various CPU features.
- [full](full/): Starts several OSes and checks if they boot correctly.
- [jit-paging](jit-paging/): Tests jit and paging interaction.
- [api](api/): Tests for several API functions of v86.
- [devices](devices/): Device tests.
- [rust](rust/): Rust unit test helpers.
- [expect](expect/): Expect tests for the jit output. Contains a set of
  asm+wasm files, where the jit is expected to produce the wasm file given the
  asm file.

The following environmental variables are respected by most tests if applicable:

- `TEST_RELEASE_BUILD=1`: Test the release build (libv86.js, v86.wasm) instead of the
  debug build (source files with v86-debug.wasm)
- `MAX_PARALLEL_TESTS=n`: Maximum number of tests to run in parallel. Defaults
  to the number of cores in your system or less.
- `TEST_NAME="…"`: Run only the specified test (only expect, full, nasm)
