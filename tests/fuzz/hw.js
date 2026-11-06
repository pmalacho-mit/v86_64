#!/usr/bin/env node

// 64-bit fuzz cases against a real (or emulated) x86-64 CPU.
//
// tests/fuzz/run.js compares v86's interpreter with its JIT: it finds
// divergences between the two, not bugs they share. This script uses the same
// generated 64-bit cases (tests/fuzz/gen_case.js) and compares v86 with the
// host CPU instead:
//
//   1. export: write the cases to a batch file
//        tests/fuzz/hw.js export cases.bin
//   2. run the batch natively on x86-64 Linux (tests/fuzz/hw/runner.c, a
//      freestanding static binary; see the Makefile target fuzz-hw-runner):
//        build/fuzz-hw/runner < cases.bin > results.bin
//      On real x86 hardware this is the reference. On an aarch64 host the same
//      binary runs under qemu-x86_64 (`docker run --platform linux/amd64`),
//      which is only a *provisional* reference: see docs/testing-aarch64.md
//      for what qemu gets wrong.
//   3. compare: run each case in v86 and compare the final state
//        tests/fuzz/hw.js compare results.bin [--jit]
//
// The runner executes each case's 64-bit body as a ring-3 Linux process, v86
// executes it in ring 0 with an identity map of the first 2 MiB. Where those
// environments legitimately differ the case is reported as "inconclusive"
// instead of as a mismatch (see classify()), and instructions whose result
// depends on the environment (segment selectors, pushf, cpuid, FPU
// environment stores, ...) are not exported at all.
//
// Environment variables (as in run.js; must be the same for export and compare):
//   FUZZ_SEED, FUZZ_CASES, TEST_NAME, FUZZ_MAX_CASES
//   V86_WASM_PATH   override the wasm build
//   HW_VERBOSE=1    print every mismatch in full (default: the first 3 per encoding)

import fs from "node:fs";
import url from "node:url";
import process from "node:process";

const __dirname = url.fileURLToPath(new URL(".", import.meta.url));

const gen = await import("./gen_case.js");
const table = (await import("../../gen/x86_table.js")).default;

const FUZZ_SEED = (process.env.FUZZ_SEED === undefined ? 1 : +process.env.FUZZ_SEED) >>> 0;
const FUZZ_CASES = +process.env.FUZZ_CASES || 10;
const FUZZ_MAX_CASES = +process.env.FUZZ_MAX_CASES || Infinity;
const TEST_NAME = new RegExp(process.env.TEST_NAME || "", "i");
const SINGLE_CASE_TIMEOUT = 10000;

const CF = 1, PF = 1 << 2, AF = 1 << 4, ZF = 1 << 6, SF = 1 << 7, DF = 1 << 10, OF = 1 << 11;
const MASK_ARITH = CF | PF | AF | ZF | SF | OF;
const ALL_ARITH = MASK_ARITH;

const FPU_STATUS_MASK = 0xFFFF & ~(1 << 9 | 1 << 5 | 1 << 3 | 1 << 1); // as in tests/nasm/run.js
const MXCSR_FLAGS = 0x3F;

const LOW_START = 0x10000; // Linux can't map below mmap_min_addr
const LOW_END = 0x200000;  // v86: one 2 MiB identity-mapped page

// Instructions whose result depends on the environment rather than on the
// CPU (ring 3 under Linux vs ring 0 in v86, different GDT/selectors, the
// emulator's cpuid, ...). Cases containing any of them aren't exported.
const ENV_DEPENDENT = new Set([
    0x8C, 0x8E,                         // mov r/m, sreg / mov sreg, r/m: selectors differ
    0x9C,                               // pushf: IF and IOPL differ by ring
    0xE4, 0xE5, 0xE6, 0xE7, 0xEC, 0xED, 0xEE, 0xEF, // in/out
    0x6C, 0x6D, 0x6E, 0x6F,             // ins/outs
    0xFA, 0xFB,                         // cli/sti
    0x0F00, 0x0F01,                     // sldt/str/lldt/ltr/verr/verw, sgdt/sidt/lgdt/smsw/...
    0x0F02, 0x0F03,                     // lar/lsl: descriptor tables differ
    0x0F06, 0x0F08, 0x0F09,             // clts/invd/wbinvd
    0x0F20, 0x0F21, 0x0F22, 0x0F23,     // mov cr/dr
    0x0F30, 0x0F32, 0x0F33,             // wrmsr/rdmsr/rdpmc
    0x0FA0, 0x0FA1, 0x0FA8, 0x0FA9,     // push/pop fs/gs
    0x0FA2,                             // cpuid
    0x0FB2, 0x0FB4, 0x0FB5,             // lss/lfs/lgs
    0x0FC7,                             // cmpxchg16b/rdrand/rdseed/rdpid (group): mixed, skipped whole
]);

function env_dependent(op)
{
    if(ENV_DEPENDENT.has(op.opcode)) return true;
    // FPU environment stores record FPU IP/DP and selectors
    if(op.opcode === 0xD9 && (op.fixed_g === 4 || op.fixed_g === 6)) return true; // fldenv/fnstenv
    if(op.opcode === 0xDD && (op.fixed_g === 4 || op.fixed_g === 6)) return true; // frstor/fnsave
    if(op.opcode === 0x0FAE && op.fixed_g !== undefined && op.fixed_g <= 1) return true; // fxsave/fxrstor
    if(op.opcode === 0x0FAE && (op.fixed_g === 4 || op.fixed_g === 5)) return true; // xsave/xrstor
    return false;
}

// Flags the SDM leaves undefined after an instruction. Real CPUs set them
// deterministically, but not necessarily as v86 (or another CPU) does.
function undefined_flags(op)
{
    const o = op.opcode, g = op.fixed_g;
    let f = op.mask_flags || 0;
    // shifts and rotates: OF for counts other than 1, AF for shifts
    if([0xC0, 0xC1, 0xD0, 0xD1, 0xD2, 0xD3].includes(o)) f |= OF | AF;
    if([0x0FA4, 0x0FA5, 0x0FAC, 0x0FAD].includes(o)) f |= ALL_ARITH; // shld/shrd (also: all flags if count > size)
    // logic ops: AF
    if([0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25,
        0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x84, 0x85, 0xA8, 0xA9].includes(o)) f |= AF;
    if([0x80, 0x81, 0x83].includes(o) && [1, 4, 6].includes(g)) f |= AF;
    if([0xF6, 0xF7].includes(o) && (g === 0 || g === 1)) f |= AF;
    // mul/imul: SF ZF AF PF; div/idiv: everything
    if([0xF6, 0xF7].includes(o) && (g === 4 || g === 5)) f |= SF | ZF | AF | PF;
    if([0xF6, 0xF7].includes(o) && (g === 6 || g === 7)) f |= ALL_ARITH;
    if([0x0FAF, 0x69, 0x6B].includes(o)) f |= SF | ZF | AF | PF;
    // bsf/bsr: all but ZF; bt*: OF SF AF PF
    if([0x0FBC, 0x0FBD].includes(o)) f |= CF | OF | SF | AF | PF;
    if([0x0FA3, 0x0FAB, 0x0FB3, 0x0FBB, 0x0FBA].includes(o)) f |= OF | SF | AF | PF;
    return f;
}

// ---------------------------------------------------------------------------
// case enumeration (the 64-bit subset of run.js's)
// ---------------------------------------------------------------------------

function* cases()
{
    let n = 0;
    for(const op of table)
    {
        if(!gen.is_fuzzable(op)) continue;
        for(const config of gen.fuzz_configs(op))
        {
            if(config.mode !== 64) continue;
            for(let nth = 0; nth < FUZZ_CASES; nth++)
            {
                const seed = gen.case_seed(FUZZ_SEED, op, config, nth);
                const test_case = gen.generate_case(op, config, seed, nth, { user_mode: true });
                if(!TEST_NAME.test(test_case.name)) continue;
                if(test_case.chain_ops.some(env_dependent)) continue;
                test_case.seed = seed;
                yield test_case;
                if(++n >= FUZZ_MAX_CASES) return;
            }
        }
    }
}

// the page table words v86's fuzz driver writes (tests/fuzz/run.js setup_long_mode_paging)
function page_table_regions()
{
    const word = v => { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, v, true); return b; };
    return [
        [gen.PML4_ADDR, word(gen.PDPT_ADDR | 3)],
        [gen.PDPT_ADDR, word(gen.PD_ADDR | 3)],
        [gen.PD_ADDR, word(0x83)],
    ];
}

function do_export(file)
{
    const chunks = [];
    let count = 0;
    const u16 = v => { const b = Buffer.alloc(2); b.writeUInt16LE(v); return b; };
    const u32 = v => { const b = Buffer.alloc(4); b.writeUInt32LE(v >>> 0); return b; };
    const u64 = v => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(v)); return b; };
    for(const c of cases())
    {
        const regions = [[gen.LOAD_ADDR, c.image], [gen.SCRATCH_ADDR, c.scratch], ...page_table_regions()];
        const name = Buffer.from(c.name);
        chunks.push(u16(name.length), name, u64(c.entry64 + 1), u64(c.done_addr), u32(regions.length));
        for(const [addr, bytes] of regions)
        {
            chunks.push(u32(addr), u32(bytes.length), Buffer.from(bytes));
        }
        count++;
    }
    fs.writeFileSync(file, Buffer.concat([Buffer.from("V86FUZZ1"), u32(count), ...chunks]));
    console.log("[+] exported %d 64-bit cases to %s (seed=%d, cases per config=%d)", count, file, FUZZ_SEED, FUZZ_CASES);
}

// ---------------------------------------------------------------------------
// runner results
// ---------------------------------------------------------------------------

// glibc REG_* order
const GREG = { r8: 0, r9: 1, r10: 2, r11: 3, r12: 4, r13: 5, r14: 6, r15: 7, rdi: 8, rsi: 9, rbp: 10,
    rbx: 11, rdx: 12, rax: 13, rcx: 14, rsp: 15, rip: 16, efl: 17 };
// v86 register order: rax rcx rdx rbx rsp rbp rsi rdi r8..r15
const V86_TO_GREG = [GREG.rax, GREG.rcx, GREG.rdx, GREG.rbx, GREG.rsp, GREG.rbp, GREG.rsi, GREG.rdi,
    0, 1, 2, 3, 4, 5, 6, 7];
const GPR_NAMES = ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
    "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"];

const RESULT_SIZE = 4 * 4 + 8 * 3 + 8 * 23 + 512 + gen.SCRATCH_SIZE;

// The exception vector. Linux reports it exactly (trapno); qemu-user leaves
// trapno at -1, so infer the vector from the signal there, which can't tell
// some vectors apart: "#GP" also stands for #SS, and "#MF/#XM" for both.
function signal_vector(r)
{
    if(r.trapno >= 0)
    {
        r.exact_vector = true;
        return r.trapno;
    }
    r.exact_vector = false;
    switch(r.signo)
    {
        case 11:
            // SIGSEGV: #PF, or #GP/#SS. qemu reports a non-canonical address
            // like an unmapped one (SEGV_MAPERR with the address); a real CPU
            // raises #GP for it.
            if(r.cr2 === 0n) return 13;
            return BigInt.asIntN(48, r.cr2) === BigInt.asIntN(64, r.cr2) ? 14 : 13;
        case 4: return 6;                        // SIGILL: #UD
        case 8: return r.si_code === 1 || r.si_code === 2 ? 0 : 16; // SIGFPE: FPE_INTDIV/INTOVF: #DE; else x87/SSE
        case 7: return 17;                       // SIGBUS: #AC (or #SS)
        case 5: return 1;                        // SIGTRAP
        default: return -1;
    }
}

function read_results(file)
{
    const buf = fs.readFileSync(file);
    const results = new Map();
    let at = 0;
    while(at < buf.length)
    {
        if(buf.toString("latin1", at, at + 4) !== "CASE") throw new Error("bad results file at offset " + at);
        const name_len = buf.readUInt16LE(at + 4);
        const name = buf.toString("utf8", at + 6, at + 6 + name_len);
        const r = buf.subarray(at + 6 + name_len, at + 6 + name_len + RESULT_SIZE);
        at += 6 + name_len + RESULT_SIZE;
        const gregs = [];
        for(let i = 0; i < 23; i++) gregs.push(r.readBigUInt64LE(40 + 8 * i));
        const fx = r.subarray(224, 736);
        const result = {
            status: r.readUInt32LE(0),
            signo: r.readUInt32LE(4),
            si_code: r.readInt32LE(8),
            trapno: Number(r.readBigInt64LE(16)),
            err: r.readBigUInt64LE(24),
            cr2: r.readBigUInt64LE(32),
            gregs,
            fcw: fx.readUInt16LE(0),
            fsw: fx.readUInt16LE(2),
            ftw: fx[4], // abridged: bit i set = physical register i non-empty
            mxcsr: fx.readUInt32LE(24),
            st: [0, 1, 2, 3, 4, 5, 6, 7].map(i => fx.subarray(32 + 16 * i, 32 + 16 * i + 10)),
            xmm: [...Array(16).keys()].map(i => fx.subarray(160 + 16 * i, 176 + 16 * i)),
            scratch: r.subarray(736, 736 + gen.SCRATCH_SIZE),
        };
        result.vector = signal_vector(result);
        results.set(name, result);
    }
    return results;
}

// ---------------------------------------------------------------------------
// v86 side (as in run.js, interpreter or forced JIT)
// ---------------------------------------------------------------------------

const { V86 } = await import("../../src/main.js");
let emulator, active = null;

function create_emulator()
{
    const e = new V86({
        autostart: false,
        memory_size: 2 * 1024 * 1024,
        log_level: 0,
        wasm_path: process.env.V86_WASM_PATH,
    });
    e.bus.register("cpu-event-halt", () => active && active.on_halt());
    e.cpu_exception_hook = n => active ? active.on_exception(n) : true;
    return e;
}

function run_until_halt(recorded)
{
    return new Promise((resolve, reject) =>
    {
        const cpu = emulator.v86.cpu;
        const finish = error =>
        {
            clearTimeout(timeout);
            active = null;
            if(error) reject(error);
            else { emulator.stop(); resolve(); }
        };
        const timeout = setTimeout(() => finish(new Error("case timed out")), SINGLE_CASE_TIMEOUT);
        active = {
            on_crash: error => finish(error instanceof Error ? error : new Error(String(error))),
            on_exception(n)
            {
                cpu.instruction_counter[0] += 100000;
                if(recorded.exception) return true;
                const eip = cpu.instruction_pointer[0];
                recorded.exception = { vector: n, eip };
                cpu.write32(cpu.translate_address_system_read(eip), 0xF4F4F4F4);
                return true;
            },
            on_halt: () => finish(null),
        };
        emulator.run();
    });
}

process.on("uncaughtException", e => { if(active) active.on_crash(e); else throw e; });

function force_generate(entry)
{
    const cpu = emulator.v86.cpu;
    return new Promise(resolve =>
    {
        cpu.test_hook_did_finalize_wasm = () => { cpu.test_hook_did_finalize_wasm = null; setTimeout(resolve, 0); };
        cpu.jit_force_generate(entry);
    });
}

async function run_v86(test_case, use_jit)
{
    const cpu = emulator.v86.cpu;
    cpu.reboot_internal();
    cpu.reset_memory();
    const vga = cpu.devices.vga;
    if(vga) { vga.vga_memory.fill(0); vga.svga_memory.fill(0); vga.latch_dword = 0; }
    cpu.load_multiboot(test_case.image.buffer);
    for(const [addr, bytes] of page_table_regions()) cpu.mem8.set(bytes, addr);
    cpu.mem8.set(test_case.scratch, gen.SCRATCH_ADDR);
    cpu.set_jit_config(0, use_jit ? 0 : 1);

    const recorded = { exception: null };
    await run_until_halt(recorded); // the 32-bit mode switch stub
    if(recorded.exception) throw new Error("mode switch stub faulted: " + JSON.stringify(recorded.exception));
    cpu.in_hlt[0] = 0;
    if(use_jit) await force_generate(cpu.instruction_pointer[0]);
    await run_until_halt(recorded);

    const regs = [];
    for(let i = 0; i < 16; i++)
    {
        regs.push(BigInt(cpu.reg32s[2 * i] >>> 0) | BigInt(cpu.reg32s[2 * i + 1] >>> 0) << 32n);
    }
    const top = cpu.fpu_stack_ptr[0] & 7;
    const st = [];
    for(let i = 0; i < 8; i++)
    {
        const p = (top + i) & 7;
        const b = new Uint8Array(10);
        const v = new DataView(b.buffer);
        v.setInt32(0, cpu.fpu_st[4 * p], true);
        v.setInt32(4, cpu.fpu_st[4 * p + 1], true);
        v.setUint16(8, cpu.fpu_st[4 * p + 2] & 0xFFFF, true);
        st.push(b);
    }
    const xmm = [];
    for(let i = 0; i < 16; i++)
    {
        const src = i < 8 ? cpu.reg_xmm32s.subarray(4 * i, 4 * i + 4) : cpu.reg_xmm32s_high.subarray(4 * (i - 8), 4 * (i - 8) + 4);
        xmm.push(new Uint8Array(Int32Array.from(src).buffer));
    }
    return {
        exception: recorded.exception,
        rip: cpu.instruction_pointer[0] >>> 0,
        regs,
        eflags: cpu.get_eflags() >>> 0,
        fsw: cpu.fpu_load_status_word() & 0xFFFF,
        fcw: cpu.fpu_control_word[0],
        // abridged tag word, physical registers
        ftw: ~cpu.fpu_stack_empty[0] & 0xFF,
        mxcsr: cpu.mxcsr[0] >>> 0,
        st,
        xmm,
        scratch: new Uint8Array(cpu.mem8.subarray(gen.SCRATCH_ADDR, gen.SCRATCH_ADDR + gen.SCRATCH_SIZE)),
    };
}

// ---------------------------------------------------------------------------
// comparison
// ---------------------------------------------------------------------------

const hex = (v, n = 16) => "0x" + BigInt.asUintN(64, BigInt(v)).toString(16).padStart(n, "0");
const bytes_hex = b => Buffer.from(b).toString("hex");

// x86 exception vectors as seen in the signal context (trapno)
const VECTOR_NAMES = { 0: "#DE", 1: "#DB", 3: "#BP", 5: "#BR", 6: "#UD", 7: "#NM", 12: "#SS", 13: "#GP", 14: "#PF",
    16: "#MF", 17: "#AC", 19: "#XM" };

// The instruction at a guest address of the case image: legacy prefixes, REX,
// opcode (0x0Fxx, 0x0F38xx, 0x0F3Axx for the escapes) and the modrm byte.
function decode_at(c, addr)
{
    const bytes = c.image;
    let i = addr - gen.LOAD_ADDR;
    const prefixes = new Set();
    while(i < bytes.length && [0x66, 0x67, 0xF0, 0xF2, 0xF3, 0x26, 0x2E, 0x36, 0x3E, 0x64, 0x65].includes(bytes[i]))
    {
        prefixes.add(bytes[i++]);
    }
    let rex = 0;
    if((bytes[i] & 0xF0) === 0x40) rex = bytes[i++];
    let opcode = bytes[i++];
    if(opcode === 0x0F)
    {
        opcode = 0x0F00 | bytes[i++];
        if(opcode === 0x0F38 || opcode === 0x0F3A) opcode = opcode << 8 | bytes[i++];
    }
    return { prefixes, rex, opcode, modrm: bytes[i] };
}

// Differences that tests/fuzz's qemu reference is known to get wrong
// (docs/testing-aarch64.md; measured on 32-bit code, the same TCG front end).
// Only applied when the results come from qemu (no exact trapno).
function qemu_known_defect(c, hw_vector, v_vector, hw_rip, v_eip)
{
    // hint NOPs (0F 18-1F) and the undocumented F6/F7 /1 TEST alias: qemu #UD, real CPUs execute them
    if(hw_vector === 6 && v_vector === null)
    {
        const d = decode_at(c, hw_rip);
        if(d.opcode >= 0x0F18 && d.opcode <= 0x0F1F) return "qemu: #UD on a hint NOP";
        if((d.opcode === 0xF6 || d.opcode === 0xF7) && (d.modrm >> 3 & 7) === 1) return "qemu: #UD on F6/F7 /1";
    }
    // register forms of memory-only moves: qemu executes them, real CPUs raise #UD
    if(hw_vector === null && v_vector === 6)
    {
        const d = decode_at(c, v_eip);
        const reg_form = (d.modrm >> 6) === 3;
        if(reg_form && (d.opcode === 0x0FE7 || (d.prefixes.has(0x66) && (d.opcode === 0x0F12 || d.opcode === 0x0F16 || d.opcode === 0x0F13 || d.opcode === 0x0F17))))
        {
            return "qemu: executes the register form of a memory-only move";
        }
    }
    return null;
}

// MMX instructions (no mandatory 66/F2/F3 prefix) with REX.R or REX.B: real
// CPUs ignore those bits for mm operands; qemu 8.2 doesn't always (e.g.
// `41 0F 38 04 C0`, pmaddubsw mm0, mm0, yields 0 there).
function qemu_rex_mmx(c)
{
    return c.instr_starts.some(addr =>
    {
        const d = decode_at(c, addr);
        if(!(d.rex & 5) || d.prefixes.has(0x66) || d.prefixes.has(0xF2) || d.prefixes.has(0xF3)) return false;
        const o = d.opcode;
        return o >= 0x0F60 && o <= 0x0F7F || o >= 0x0FD0 && o <= 0x0FFF || o >= 0x0F3800 && o <= 0x0F381F ||
            o === 0x0F3A0F || o === 0x0F2A || o === 0x0F2C || o === 0x0F2D || o === 0x0FC4 || o === 0x0FC5;
    });
}

// two little-endian hex strings of xmm registers that only differ in the sign
// of default NaNs (0xffc00000 vs 0x7fc00000, 0xfff8000000000000 vs 0x7ff8000000000000)
function nan_sign_only(a, b)
{
    if(a.length !== b.length || a === b) return false;
    for(let i = 0; i < a.length; i += 8)
    {
        const x = a.slice(i, i + 8), y = b.slice(i, i + 8);
        if(x === y) continue;
        const pair = new Set([x, y]);
        const f32 = pair.has("0000c0ff") && pair.has("0000c07f");
        const f64_hi = pair.has("0000f8ff") && pair.has("0000f87f");
        if(!f32 && !f64_hi) return false;
    }
    return true;
}

// Returns { verdict: "match" | "mismatch" | "known" | "inconclusive", why, diffs }
function classify(c, hw, v)
{
    if(hw.status !== 0)
    {
        return { verdict: "inconclusive", why: "runner: no signal context (killed by signal " + hw.signo + ")" };
    }
    if(!hw.exact_vector && qemu_rex_mmx(c))
    {
        return { verdict: "inconclusive", why: "qemu: REX.R/B on an MMX instruction" };
    }
    const hw_done = hw.vector === 13 && Number(hw.gregs[GREG.rip]) === c.done_addr;
    const hw_vector = hw_done ? null : hw.vector;
    let v_vector = v.exception ? v.exception.vector : null;
    if(!hw.exact_vector && v_vector !== null && hw_vector !== null)
    {
        // collapse what the signal can't distinguish
        if(v_vector === 12 && hw_vector === 13) v_vector = 13;
        if(v_vector === 19 && hw_vector === 16) v_vector = 16;
    }

    if(hw_vector !== v_vector)
    {
        const qemu_why = !hw.exact_vector &&
            qemu_known_defect(c, hw_vector, v_vector, Number(hw.gregs[GREG.rip]), v.exception && v.exception.eip);
        if(qemu_why) return { verdict: "inconclusive", why: qemu_why };
        // environment differences, not CPU differences
        if(hw_vector === 14 && hw.cr2 < BigInt(LOW_START))
        {
            return { verdict: "inconclusive", why: "access below 0x10000 (mapped in v86, not mappable in Linux)" };
        }
        if(hw_vector === 13 && v_vector === null)
        {
            // A legacy SSE instruction with a 128-bit memory operand: almost
            // certainly misaligned (#GP on real CPUs; v86 doesn't check).
            // Imprecise (a missed non-canonical #GP on such an instruction
            // would land here too), hence reported in its own category.
            const d = decode_at(c, Number(hw.gregs[GREG.rip]));
            const sse_ps = [0x0F28, 0x0F29, 0x0F2B, 0x0F14, 0x0F15, 0x0FC2, 0x0FC6].includes(d.opcode) ||
                d.opcode >= 0x0F51 && d.opcode <= 0x0F5F;
            const sse128 = d.prefixes.has(0x66) && d.opcode > 0x0F00 && !d.prefixes.has(0xF2) && !d.prefixes.has(0xF3) ||
                sse_ps && !d.prefixes.size;
            if(sse128 && d.modrm >> 6 !== 3)
            {
                return { verdict: "known", why: "v86 doesn't raise #GP for misaligned SSE memory operands (most likely)" };
            }
            return { verdict: "inconclusive", why: "#GP in ring 3 only (privileged in user mode?)" };
        }
        // #SS/#GP/#PF are all "bad address" to Linux, but the vector is exact (trapno)
        const where = (vec, rip) => vec === null ? "none" :
            (VECTOR_NAMES[vec] || String(vec)) + " at " + hex(rip, 5);
        return { verdict: "mismatch", diffs: [["exception",
            where(hw_vector, Number(hw.gregs[GREG.rip])), where(v_vector, v.exception && v.exception.eip)]] };
    }

    const diffs = [];
    if(hw_vector !== null)
    {
        const hw_rip = Number(hw.gregs[GREG.rip]);
        if(hw_rip !== v.exception.eip) diffs.push(["exception rip", hex(hw_rip, 8), hex(v.exception.eip, 8)]);
        // after a fault, the architectural state is that of before the
        // faulting instruction; compare it too
    }

    const undef = c.chain_ops.reduce((f, op) => f | undefined_flags(op), 0);
    const flag_mask = (MASK_ARITH | DF) & ~undef;
    const hw_flags = Number(hw.gregs[GREG.efl]) & flag_mask;
    if(hw_flags !== (v.eflags & flag_mask)) diffs.push(["rflags", hex(hw_flags, 4), hex(v.eflags & flag_mask, 4)]);

    for(let i = 0; i < 16; i++)
    {
        const a = hw.gregs[V86_TO_GREG[i]], b = v.regs[i];
        if(a !== b) diffs.push([GPR_NAMES[i], hex(a), hex(b)]);
    }
    for(let i = 0; i < 16; i++)
    {
        if(bytes_hex(hw.xmm[i]) !== bytes_hex(v.xmm[i])) diffs.push(["xmm" + i, bytes_hex(hw.xmm[i]), bytes_hex(v.xmm[i])]);
    }
    // v86 doesn't set the MXCSR exception flags (bits 0-5)
    if((hw.mxcsr & ~MXCSR_FLAGS) !== (v.mxcsr & ~MXCSR_FLAGS))
    {
        diffs.push(["mxcsr", hex(hw.mxcsr, 8), hex(v.mxcsr, 8)]);
    }
    if(hw.fcw !== v.fcw) diffs.push(["fpu control word", hex(hw.fcw, 4), hex(v.fcw, 4)]);
    // qemu's x87 isn't a reference for stack faults, transcendentals and
    // several status bits; with a qemu reference, x87 instructions are only
    // checked for their effect on everything else
    const check_x87 = hw.exact_vector || !c.chain_ops.some(op => op.is_fpu);
    if(check_x87 && (hw.fsw & FPU_STATUS_MASK) !== (v.fsw & FPU_STATUS_MASK))
    {
        diffs.push(["fpu status word", hex(hw.fsw, 4), hex(v.fsw, 4)]);
    }
    if(check_x87 && hw.ftw !== v.ftw) diffs.push(["fpu tag (abridged)", hex(hw.ftw, 2), hex(v.ftw, 2)]);
    const hw_top = hw.fsw >> 11 & 7;
    for(let i = 0; i < 8 && check_x87; i++)
    {
        // empty registers hold stale values; compare the non-empty ones
        if(!(hw.ftw >> ((hw_top + i) & 7) & 1)) continue;
        if(bytes_hex(hw.st[i]) === bytes_hex(v.st[i])) continue;
        if(!hw.exact_vector && bytes_hex(hw.st[i].subarray(0, 8)) === bytes_hex(v.st[i].subarray(0, 8)) &&
            hw.st[i][8] === 0 && hw.st[i][9] === 0 && v.st[i][8] === 0xFF && v.st[i][9] === 0xFF)
        {
            // qemu doesn't set the exponent to all ones on MMX writes; real CPUs (and v86) do
            continue;
        }
        diffs.push(["st" + i, bytes_hex(hw.st[i]), bytes_hex(v.st[i])]);
    }
    let mem_diffs = 0;
    for(let i = 0; i < gen.SCRATCH_SIZE; i += 4)
    {
        const a = Buffer.from(hw.scratch.subarray(i, i + 4)).readUInt32LE(0);
        const b = Buffer.from(v.scratch.subarray(i, i + 4)).readUInt32LE(0);
        if(a !== b)
        {
            if(mem_diffs++ < 8) diffs.push(["mem[" + hex(gen.SCRATCH_ADDR + i, 6) + "]", hex(a, 8), hex(b, 8)]);
        }
    }
    if(mem_diffs > 8) diffs.push(["mem", (mem_diffs - 8) + " more dwords differ", ""]);

    if(!diffs.length) return { verdict: "match" };
    if(process.arch !== "x64" && diffs.every(([n, a, b]) => n.startsWith("xmm") && nan_sign_only(a, b)))
    {
        // wasm leaves the sign of a NaN produced by sqrt (and other ops) to
        // the host; x86 hosts give x86's default NaN, aarch64 hosts don't
        return { verdict: "known", why: "default NaN sign on a non-x86 host (wasm NaN sign is host dependent)" };
    }
    if(c.has_branch && undef && diffs.every(([n]) => n.startsWith("mem")))
    {
        return { verdict: "inconclusive", why: "branch on undefined flags" };
    }
    return { verdict: "mismatch", diffs };
}

async function do_compare(file, use_jit)
{
    const results = read_results(file);
    const any = results.values().next().value;
    const reference = !any ? "none" : any.exact_vector ? "native x86 (trapno reported)" :
        "qemu-user (no trapno): provisional, see docs/testing-aarch64.md";
    emulator = create_emulator();
    await new Promise(r => emulator.add_listener("emulator-loaded", r));

    const counts = { match: 0, mismatch: 0, inconclusive: 0, known: 0, missing: 0, crash: 0 };
    const inconclusive_why = new Map();
    const mismatches_by_encoding = new Map();

    for(const c of cases())
    {
        const hw = results.get(c.name);
        if(!hw) { counts.missing++; continue; }
        let v;
        try
        {
            v = await run_v86(c, use_jit);
        }
        catch(e)
        {
            counts.crash++;
            console.error("[-] CRASH %s: %s", c.name, String(e.message || e).split("\n")[0]);
            emulator.v86.stop();
            emulator = create_emulator();
            await new Promise(r => emulator.add_listener("emulator-loaded", r));
            continue;
        }
        const r = classify(c, hw, v);
        counts[r.verdict]++;
        if(r.verdict === "inconclusive" || r.verdict === "known")
        {
            inconclusive_why.set(r.why, (inconclusive_why.get(r.why) || 0) + 1);
        }
        else if(r.verdict === "mismatch")
        {
            const encoding = c.name.replace(/^fuzz64_/, "").replace(/_\d+(\+\d)?j?$/, "");
            const list = mismatches_by_encoding.get(encoding) || [];
            list.push({ c, r });
            mismatches_by_encoding.set(encoding, list);
        }
    }

    for(const [encoding, list] of [...mismatches_by_encoding].sort())
    {
        console.log("[-] %s: %d mismatching case(s)", encoding, list.length);
        for(const { c, r } of process.env.HW_VERBOSE ? list : list.slice(0, 3))
        {
            console.log("    %s (replay: FUZZ_SEED=%d TEST_NAME=^%s$) instr: %s",
                c.name, FUZZ_SEED, c.name.replace(/\+/g, "\\+"), bytes_hex(c.instr_bytes));
            for(const [what, a, b] of r.diffs.slice(0, 8))
            {
                console.log("        %s: hw=%s v86=%s", what, a, b);
            }
        }
    }
    console.log("\n[+] %d cases (%s, reference: %s): %d match, %d mismatch, %d known v86 limitation, %d inconclusive, %d crashed, %d without a result",
        counts.match + counts.mismatch + counts.inconclusive + counts.known + counts.crash, use_jit ? "jit" : "interpreter",
        reference, counts.match, counts.mismatch, counts.known, counts.inconclusive, counts.crash, counts.missing);
    for(const [why, n] of [...inconclusive_why].sort((a, b) => b[1] - a[1])) console.log("    %d: %s", n, why);
    console.log("    mismatching encodings: %d", mismatches_by_encoding.size);
    await emulator.destroy();
    process.exit(counts.mismatch || counts.crash ? 1 : 0);
}

const [cmd, file] = process.argv.slice(2);
if(cmd === "export" && file) do_export(file);
else if(cmd === "compare" && file) await do_compare(file, process.argv.includes("--jit"));
else
{
    console.error("usage: tests/fuzz/hw.js export <cases.bin> | compare <results.bin> [--jit]");
    process.exit(2);
}
