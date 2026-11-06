# About

The tests in this folder are not comprehensive by any means at the
moment.

# Automated Testing

- Run `make nasmtests` in the root of the project

# Fixtures need a real x86 CPU

`gen_fixtures.js` records the expected state by running each test binary under
gdb on the host CPU, so `make nasmtests` only produces trustworthy fixtures on
an x86 host (Intel and AMD give bit-identical results). On aarch64, run
`run.js` against fixtures generated on x86 (the CI artifact
`nasmtests-fixtures-<hash>`), and ask the human to re-run the generator on an
x86 machine when you add or change tests. `gen_fixtures_qemu.sh` produces
*provisional* fixtures under qemu for a quick local check; they must never be
committed or trusted for x87 stack faults, the `FCOMI` family, transcendentals
and a few other encodings. See [docs/testing-aarch64.md](../../docs/testing-aarch64.md).
