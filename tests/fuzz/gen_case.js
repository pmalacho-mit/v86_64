// Instruction fuzz case generator.
//
// Generates small self-contained multiboot images that:
//   1. (64-bit cases) enter long mode via a 32-bit stub (page tables, PAE,
//      EFER.LME, paging, far jump into a 64-bit code segment);
//      (32p cases) enable classic 32-bit paging (2-level walks)
//   2. initialise GPRs, x87/MMX, XMM and eflags to pseudo-random values,
//      biased towards edge cases (special floats, sign bits, byte patterns);
//      sometimes randomise the fpu control word and MXCSR (rounding modes)
//   3. execute one to three fuzzed instructions with random
//      operand/encodings (chaining exercises flag producer/consumer pairs
//      across jit block boundaries), occasionally a lock prefix on
//      read-modify-write memory ops, and occasionally a conditional jump
//      over a marker write (the compared state reveals the branch outcome)
//   4. hlt
//
// The instruction set under test is enumerated from gen/x86_table.js, the
// same table that drives the interpreter/jit generators, so every
// implemented instruction is covered.
//
// 64-bit edge cases covered: REX.W/R/X/B combinations (r8-r15, xmm8-xmm15,
// sil/dil/bpl/spl vs ah/ch/dh/bh), 16/32/64-bit operand sizes, RIP-relative
// addressing with computed displacements, address-size override (0x67) with
// garbage upper address bits, non-canonical addresses (expect #GP), wild
// displacements (expect #PF/#GP), imm64 moves, 64-bit moffs, misaligned SSE
// operands, 64-bit shift counts and 32-bit result zero-extension.
//
// 32-bit edge cases covered: 16-bit addressing forms (0x67: the bx/bp/si/di
// modrm16 decoder), execution with paging enabled (page walks, TLB fills,
// #PF delivery) and 16/32-bit operand sizes.
//
// Used by run.js, which executes each case in the interpreter and in the
// JIT and compares the resulting CPU state.

import encodings from "../../gen/x86_table.js";
import Rand from "../nasm/rand.js";

export const LOAD_ADDR = 0x80000;
export const SCRATCH_ADDR = 0x100000;
export const SCRATCH_SIZE = 0x2000;

// 16-bit addressing forms (0x67 in 32-bit mode) can only reach the first
// 64 KiB; that region is additionally compared for such cases
export const LOWMEM_SIZE = 0x10000;

// page tables, written by the driver (identity maps of low memory):
// long mode (PAE): PML4 -> PDPT -> PD (2 MiB page)
export const PML4_ADDR = 0x10000;
export const PDPT_ADDR = 0x11000;
export const PD_ADDR = 0x12000;
// 32-bit paging: PD (4 MiB page)
export const PD32_ADDR = 0x13000;

const STACK_TOP = SCRATCH_ADDR + SCRATCH_SIZE;
const HEADER_SIZE = 0x20;

const MULTIBOOT_HEADER_MAGIC = 0x1BADB002;
const MULTIBOOT_HEADER_ADDRESS = 0x10000;

// bits that must not be set by the random eflags value pushed in the prologue
const FLAGS_MASK_OUT = (1 << 8) | (1 << 9) | (1 << 14) | (1 << 16) | (1 << 17); // TF IF NT RF VM

const REG_EBX = 3, REG_ESP = 4, REG_EBP = 5, REG_ESI = 6, REG_EDI = 7;

// opcodes where the lock prefix (F0) is legal with a memory operand
// (read-modify-write instructions); used to fuzz lock-prefix decode
function is_lockable_mem_op(op)
{
    const o = op.opcode;
    // alu r/m, r forms (write-back dest is the r/m operand)
    if([0x00, 0x01, 0x08, 0x09, 0x10, 0x11, 0x18, 0x19, 0x20, 0x21, 0x28, 0x29, 0x30, 0x31]
        .includes(o)) return true;
    // group1 r/m, imm (all but cmp /7)
    if((o === 0x80 || o === 0x81 || o === 0x83) && op.fixed_g !== 7) return true;
    // inc/dec r/m
    if(o === 0xFE && (op.fixed_g === 0 || op.fixed_g === 1)) return true;
    if(o === 0xFF && (op.fixed_g === 0 || op.fixed_g === 1)) return true;
    // xchg r/m, r (implicitly locked, F0 tolerated)
    if(o === 0x86 || o === 0x87) return true;
    // bt*/cmpxchg/xadd/cmpxchg8b with memory destination
    if([0x0FA3, 0x0FAB, 0x0FB3, 0x0FBB, 0x0FB1, 0x0FC0, 0x0FC1].includes(o)) return true;
    if(o === 0x0FBA && op.fixed_g !== undefined && op.fixed_g >= 4) return true;
    if(o === 0x0FC7 && op.fixed_g === 1) return true;
    return false;
}

// all regs except rsp (indices 0-15; 0-7 are eax..edi, 8-15 are r8-r15)
function all_gprs(mode)
{
    const n = mode === 64 ? 16 : 8;
    const r = [];
    for(let i = 0; i < n; i++)
    {
        if(i !== REG_ESP) r.push(i);
    }
    return r;
}

// Instructions the harness cannot support, even if the table doesn't mark
// them as skip: anything that redirects control flow or interacts with the
// (asynchronous, non-deterministic) environment.
const DENY_OPCODES = new Set([
    0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, // jcc rel8
    0x78, 0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E, 0x7F,
    0xE0, 0xE1, 0xE2, 0xE3,                         // loopne/loope/loop/jcxz
    0xE8, 0xE9, 0xEA, 0xEB,                         // call/jmp rel/far
    0x9A,                                           // callf
    0xC2, 0xC3, 0xCA, 0xCB,                         // ret
    0xCC, 0xCD, 0xCE, 0xCF,                         // int/into/iret
    0xF1, 0xF4,                                     // int1/hlt
    0x0F05, 0x0F07, 0x0F34, 0x0F35,                 // syscall/sysret/sysenter/sysexit
    0x0F31,                                         // rdtsc (non-deterministic)
]);
for(let i = 0; i < 0x10; i++) DENY_OPCODES.add(0x0F80 | i); // jcc rel16/32

// opcodes whose low 3 bits encode a register (no modrm); the register can be
// extended to r8-r15 with REX.B in 64-bit mode
const OPCODE_REG_BASES = new Set([0x50, 0x58, 0x90, 0xB0, 0xB8]);

function has_opcode_reg(op)
{
    return !op.e && op.fixed_g === undefined &&
        OPCODE_REG_BASES.has(op.opcode & ~7) && op.opcode < 0x100;
}

// classic edge patterns for integers/vectors
const U32_EDGE = [
    0, 1, 2, 3, 0x7F, 0x80, 0xFF, 0x100, 0x7FFF, 0x8000, 0xFFFF, 0x10000,
    0x3F, 0x40, 0x1F, 0x20,
    0x7FFFFFFF | 0, 0x80000000 | 0, 0xFFFFFFFF | 0, 0xFFFFFFFE | 0,
    0x55555555, 0xAAAAAAAA | 0, 0x33333333, 0xCCCCCCCC | 0,
    0x01010101, 0x80808080 | 0, 0x0000FFFF, 0xFFFF0000 | 0,
];
// double-precision edge values as [lo, hi] int32 pairs
const F64_EDGE = [
    [0, 0],                        // +0.0
    [0, 0x80000000 | 0],           // -0.0
    [0, 0x3FF00000],               // 1.0
    [0, 0xBFF00000 | 0],           // -1.0
    [0, 0x40000000],               // 2.0
    [0, 0x3FE00000],               // 0.5
    [0, 0x7FF00000],               // +inf
    [0, 0xFFF00000 | 0],           // -inf
    [0, 0x7FF80000],               // quiet nan
    [1, 0x7FF80000],               // nan with payload
    [1, 0],                        // smallest denormal
    [0xFFFFFFFF | 0, 0x000FFFFF],  // largest denormal
    [0xFFFFFFFF | 0, 0x7FEFFFFF],  // largest finite
    [0x54442D18, 0x400921FB],      // pi
    [0, 0x3E700000],               // tiny value (near f32 underflow)
];

function interesting_immediate(rng)
{
    const r = rng.uint32();
    if(r % 4 === 0)
    {
        return U32_EDGE[rng.uint32() % U32_EDGE.length] | 0;
    }
    if(r & 1)
    {
        return rng.int32();
    }
    // small/shifted values: more likely to hit edge cases
    return rng.int32() << (rng.int32() & 31) >> (rng.int32() & 31);
}

// 64-bit immediate as [lo, hi] int32 pair, biased towards edge cases
function interesting_immediate64(rng)
{
    if(rng.uint32() & 1)
    {
        const e = F64_EDGE[rng.uint32() % F64_EDGE.length];
        return [e[0] | 0, e[1] | 0];
    }
    if(rng.uint32() & 1)
    {
        return [interesting_immediate(rng), 0];
    }
    return [interesting_immediate(rng), interesting_immediate(rng)];
}

// a 32-bit value for the fpu init data (interpreted as half of a double)
function interesting_fpu_half(rng, high)
{
    if(rng.uint32() & 1)
    {
        const e = F64_EDGE[rng.uint32() % F64_EDGE.length];
        return e[high ? 1 : 0] | 0;
    }
    return rng.int32();
}

/**
 * Is this table entry fuzzable by the harness?
 */
export function is_fuzzable(op)
{
    if(op.prefix || op.skip)
    {
        return false;
    }
    if(DENY_OPCODES.has(op.opcode))
    {
        return false;
    }
    if(op.opcode === 0xFF && op.fixed_g !== undefined && op.fixed_g >= 2 && op.fixed_g <= 5)
    {
        // call/jmp r/m
        return false;
    }
    return true;
}

const CHAINABLE_OPS = encodings.filter(is_fuzzable);

function fuzz_sizes(op, mode)
{
    const sized = op.os || op.opcode % 2 === 1;
    if(!sized)
    {
        return [8];
    }
    if(mode === 32)
    {
        return [16, 32];
    }
    return op.d64 ? [16, 64] : [16, 32, 64];
}

/**
 * The applicable { mode, paging, size, mem } configurations for an
 * instruction. mode: 32 or 64 (long mode); paging: 32-bit paging enabled;
 * size: operand size; mem: use the memory operand form.
 */
export function fuzz_configs(op)
{
    const configs = [];
    for(const mode of [32, 64])
    {
        if(mode === 64 && op.opcode >= 0x40 && op.opcode <= 0x4F && op.opcode < 0x100)
        {
            // single-byte inc/dec are REX prefixes in long mode
            continue;
        }
        for(const mem of [0, 1])
        {
            if(mem ? op.skip_mem : op.skip_reg)
            {
                continue;
            }
            if(!op.e && mem)
            {
                // doesn't use memory, don't test both
                continue;
            }

            for(const size of fuzz_sizes(op, mode))
            {
                if(mode === 32)
                {
                    configs.push({ mode, paging: false, size, mem });
                    configs.push({ mode, paging: true, size, mem });
                }
                else
                {
                    configs.push({ mode, paging: true, size, mem });
                }
            }
        }
    }
    return configs;
}

export function format_opcode(n)
{
    let x = n.toString(16);
    return (x.length === 1 || x.length === 3) ? "0" + x : x;
}

// three-byte maps: e.g. 660f3800 for 66 0F 38 00
export function format_encoding(op)
{
    if(!op.map)
    {
        return format_opcode(op.opcode);
    }
    return format_opcode(op.opcode >> 8) + op.map.toString(16) + format_opcode(op.opcode & 0xFF).padStart(2, "0");
}

export function case_name(op, config, nth)
{
    const mode_name = config.mode === 64 ? "64" : config.paging ? "32p" : "32";
    return "fuzz" + mode_name + "_" + format_encoding(op) + "_" +
        (op.fixed_g === undefined ? "r" : op.fixed_g) + "_" +
        config.size + (config.mem ? "m" : "r") + "_" + nth;
}

/**
 * Deterministic seed for a case.
 */
export function case_seed(global_seed, op, config, nth)
{
    let h = global_seed >>> 0;
    h = Math.imul(h ^ (op.opcode >>> 0), 0x01000193) >>> 0;
    h = Math.imul(h ^ ((op.fixed_g || 0) * 31 + config.size * 3 + config.mem + config.mode * 7 + (config.paging ? 13 : 0)), 0x01000193) >>> 0;
    h = Math.imul(h ^ nth, 0x01000193) >>> 0;
    h = Math.imul(h ^ (op.map || 0), 0x01000193) >>> 0;
    return h >>> 0;
}

class Emitter
{
    constructor()
    {
        this.bytes = [];
    }
    u8(v) { this.bytes.push(v & 0xFF); }
    u16(v) { this.u8(v); this.u8(v >> 8); }
    u32(v) { this.u16(v); this.u16(v >>> 16); }
    u64(lo, hi) { this.u32(lo); this.u32(hi); }
    at()
    {
        // current absolute guest address
        return LOAD_ADDR + HEADER_SIZE + this.bytes.length;
    }
    align(n)
    {
        while((this.at() & (n - 1)) !== 0) this.u8(0x90);
    }
    patch_u32(byte_offset, v)
    {
        this.bytes[byte_offset] = v & 0xFF;
        this.bytes[byte_offset + 1] = v >> 8 & 0xFF;
        this.bytes[byte_offset + 2] = v >> 16 & 0xFF;
        this.bytes[byte_offset + 3] = v >>> 24 & 0xFF;
    }
    patch_u16(byte_offset, v)
    {
        this.bytes[byte_offset] = v & 0xFF;
        this.bytes[byte_offset + 1] = v >> 8 & 0xFF;
    }
}

// mov reg, imm32 (32-bit mode prologue)
function emit_mov_reg_imm32(e, reg, value)
{
    e.u8(0xB8 + reg);
    e.u32(value);
}

// mov r64, imm64 (64-bit mode prologue)
function emit_mov_reg_imm64(e, reg, lo, hi)
{
    e.u8(reg >= 8 ? 0x49 : 0x48); // REX.W(+B)
    e.u8(0xB8 + (reg & 7));
    e.u64(lo, hi);
}

/**
 * Generate a random effective address in 16-bit addressing form (0x67 in
 * 32-bit mode): the bx/bp/si/di modrm16 decoder. Addresses wrap at 64 KiB,
 * so accesses land in low memory (which run.js additionally compares).
 */
function gen_ea_mem16(rng, g)
{
    // rm: 0=bx+si 1=bx+di 2=bp+si 3=bp+di 4=si 5=di 6=(mod0?disp16:bp) 7=bx
    let mod = rng.uint32() % 3;
    let rm = rng.uint32() & 7;
    if(mod === 0 && rm === 6)
    {
        // disp16-only form
        return {
            form: "abs16",
            modrm: [(g << 3) | 6],
            disp16: rng.uint32() % 0x8000,
            regs16: [],
        };
    }
    const regs16 = [];
    if(rm <= 3)
    {
        regs16.push(rm & 2 ? REG_EBP : REG_EBX, rm & 1 ? REG_EDI : REG_ESI);
    }
    else if(rm === 4)
    {
        regs16.push(REG_ESI);
    }
    else if(rm === 5)
    {
        regs16.push(REG_EDI);
    }
    else
    {
        regs16.push(rm === 6 ? REG_EBP : REG_EBX);
    }

    const disp = [];
    if(mod === 1)
    {
        disp.push(rng.uint32() & 0xFF);
    }
    else if(mod === 2)
    {
        const d = rng.uint32() & 0xFFFF;
        disp.push(d & 0xFF, d >> 8);
    }
    return {
        form: "mem16",
        modrm: [(mod << 6) | (g << 3) | rm],
        disp,
        regs16,
    };
}

/**
 * Generate a random effective address (32-bit or 64-bit).
 *
 * Forms: absolute [disp32], [base], [base + disp], [base + index*scale +
 * disp]. Addresses are biased into the scratch region so most accesses hit
 * mapped memory; occasional wild accesses fault deterministically in both
 * engines.
 *
 * 64-bit additionally: RIP-relative (disp patched by the caller once the
 * instruction length is known), SIB-absolute, registers r8-r15 as
 * base/index, address-size override (0x67) with garbage upper base bits,
 * non-canonical base addresses (#GP) and wild displacements (#PF/#GP).
 */
function gen_ea_mem(rng, g, mode)
{
    if(mode === 64 && (rng.uint32() & 3) === 0)
    {
        // RIP-relative [rip + disp32] (disp patched by the caller once the
        // instruction length is known)
        return {
            form: "rip",
            modrm: [(g & 7) << 3 | 5],
            disp_placeholder: true,
            base_reg: undefined,
            index_reg: undefined,
            a32: false,
        };
    }

    if((mode === 64 ? rng.uint32() % 5 : rng.uint32() % 4) === 0)
    {
        // absolute [disp32]; in 64-bit encoded via SIB with base=5
        // (mod=0 rm=5 is RIP-relative there)
        const wild = mode === 64 && rng.uint32() % 8 === 0;
        const d = wild ? rng.uint32() : SCRATCH_ADDR + 0x200 + (rng.uint32() % 0x1800);
        const disp = [d & 0xFF, d >> 8 & 0xFF, d >> 16 & 0xFF, d >>> 24 & 0xFF];
        if(mode === 64)
        {
            return {
                form: "abs",
                modrm: [(g & 7) << 3 | 4, 0x25], // sib: scale 0, no index, base 5
                disp,
                base_reg: undefined,
                index_reg: undefined,
                a32: false,
            };
        }
        return {
            form: "abs",
            modrm: [(g & 7) << 3 | 5],
            disp,
            base_reg: undefined,
            index_reg: undefined,
            a32: false,
        };
    }

    const gprs = all_gprs(mode);
    let base_reg = gprs[rng.uint32() % gprs.length];
    let index_reg;
    let use_sib = (rng.uint32() & 1) === 0;

    if((base_reg & 7) === REG_ESP)
    {
        // esp/r12 as base require a SIB
        use_sib = true;
    }

    let mod = rng.uint32() % 3; // 0, 1 or 2
    if((base_reg & 7) === 5 && mod === 0)
    {
        // [ebp/rbp] and [r13] with mod=0 are the disp32-only form;
        // use mod=1 with disp8=0
        mod = 1;
    }

    if(use_sib)
    {
        do { index_reg = rng.uint32() % (mode === 64 ? 16 : 8); } while((index_reg & 7) === REG_ESP);
    }

    // occasionally exercise the address-size override (0x67) in long mode:
    // EA is truncated to 32 bits, so garbage in the upper half of the base
    // register must be ignored
    const a32 = mode === 64 && (rng.uint32() & 3) === 0;

    // occasionally a non-canonical base (64-bit): must #GP identically in
    // both engines
    const noncanonical = mode === 64 && !a32 && rng.uint32() % 10 === 0;

    const disp = [];
    if(mod === 1)
    {
        disp.push(rng.uint32() & 0xFF);
    }
    else if(mod === 2)
    {
        const d = rng.uint32() % 0x200; // keep close to the scratch region
        disp.push(d & 0xFF, d >> 8 & 0xFF, 0, 0);
    }

    const ea = {
        form: use_sib ? "sib" : "base",
        modrm: [],
        disp,
        base_reg,
        index_reg,
        a32,
        noncanonical,
        scale: rng.uint32() & 3,
        mod,
    };

    if(use_sib)
    {
        ea.modrm.push((mod << 6) | ((g & 7) << 3) | 4);
        ea.modrm.push((ea.scale << 6) | ((index_reg & 7) << 3) | (base_reg & 7));
    }
    else
    {
        ea.modrm.push((mod << 6) | ((g & 7) << 3) | (base_reg & 7));
    }
    return ea;
}

/**
 * The value a prologue register should get, given its role in the effective
 * addresses of the case's instructions (if any). Returns [lo, hi] int32
 * pair (hi unused in 32-bit mode).
 */
function ea_reg_value(rng, reg, eas, mode)
{
    // first ea that uses this reg as base wins
    for(const ea of eas)
    {
        if(ea && reg === ea.base_reg)
        {
            let offset = 0x200 + (rng.uint32() % 0x1600);
            if(rng.uint32() & 1) offset &= ~0xF;
            const ptr = SCRATCH_ADDR + offset;

            if(ea.noncanonical)
            {
                // bit 47 set with zero upper bits, or canonical-looking high
                // address with bit 47 clear: both non-canonical
                if(rng.uint32() & 1)
                {
                    return [(rng.uint32() % 0x1000) | 0, 0x8000];
                }
                return [-(rng.uint32() % 0x1000) | 0, 0xFFFF7FFF | 0];
            }
            if(ea.a32)
            {
                // upper bits must be truncated away by the 0x67 prefix
                return [ptr | 0, rng.int32()];
            }
            return [ptr | 0, 0];
        }
    }
    for(const ea of eas)
    {
        if(ea && reg === ea.index_reg)
        {
            return [(rng.uint32() % 0x100) | 0, ea.a32 ? rng.int32() : 0];
        }
    }
    // 16-bit addressing uses bx/bp/si/di; give them random values whose low
    // halves are interesting offsets (accesses wrap at 64 KiB)
    for(const ea of eas)
    {
        if(ea && ea.regs16 && ea.regs16.includes(reg))
        {
            return [interesting_immediate(rng) & 0xFFFC | 0, 0];
        }
    }
    if(mode === 64)
    {
        return interesting_immediate64(rng);
    }
    return [interesting_immediate(rng), 0];
}

/**
 * Emit one fuzzed instruction. `c` is { op, size, mem, g, ea }.
 * Returns the emitted bytes.
 */
function emit_instruction(e, rng, c, mode)
{
    const { op, size, ea } = c;

    if(op.is_string)
    {
        // constrain string ops to the scratch region
        const ecx = 1 + rng.uint32() % 4;
        const edi = SCRATCH_ADDR + 0x1000 + (rng.uint32() % 0x100);
        const esi = SCRATCH_ADDR + 0x800 + (rng.uint32() % 0x100);
        if(mode === 64)
        {
            emit_mov_reg_imm64(e, 1, ecx, 0);
            emit_mov_reg_imm64(e, 7, edi, 0);
            emit_mov_reg_imm64(e, 6, esi, 0);
        }
        else
        {
            emit_mov_reg_imm32(e, 1, ecx);
            emit_mov_reg_imm32(e, 7, edi);
            emit_mov_reg_imm32(e, 6, esi);
        }
    }

    if([0x0FA5, 0x0FAD].includes(op.opcode) && size === 16)
    {
        // shld/shrd cl: shift counts larger than opsize are undefined, but
        // the count is anded with opsize-1, so only bit 4 needs clearing
        e.u8(0x80); e.u8(0xE1); e.u8(~16 & 0xFF); // and cl, ~16
    }

    const instr_start = e.bytes.length;

    // occasionally add a lock prefix to lockable read-modify-write memory ops
    // (v86 is single-processor, but the prefix must decode identically in both
    // engines)
    if(c.mem && is_lockable_mem_op(op) && (rng.uint32() & 7) === 0)
    {
        e.u8(0xF0); // lock
    }

    const a16 = ea && ea.form !== undefined && (ea.form === "mem16" || ea.form === "abs16");
    if(a16 || (ea && ea.a32))
    {
        e.u8(0x67); // address-size override
    }
    if(size === 16)
    {
        e.u8(0x66);
    }

    let opcode = op.opcode;
    if(opcode >= 0x10000)
    {
        const legacy_prefix = opcode >> 16;
        console.assert(legacy_prefix === 0x66 || legacy_prefix === 0xF3 || legacy_prefix === 0xF2,
            "opcode prefix");
        e.u8(legacy_prefix);
        opcode &= ~0xFF0000;
    }

    if(mode === 64)
    {
        // REX prefix: after legacy prefixes, before a possible 0F escape
        let rex = 0x40;
        if(size === 64) rex |= 8;                    // W
        if(c.g >= 8) rex |= 4;                      // R
        if(ea && ea.index_reg >= 8) rex |= 2;       // X
        if(ea && (ea.form === "reg" ? ea.rm_reg >= 8 : ea.base_reg >= 8)) rex |= 1; // B
        if(c.opcode_reg_ext) rex |= 1;              // B (opcode-embedded reg)
        if(rex !== 0x40 || (size === 8 && op.e && (rng.uint32() & 3) === 0))
        {
            // bare REX is emitted occasionally for byte ops: reg/rm 4-7
            // encode spl/bpl/sil/dil instead of ah/ch/dh/bh
            e.u8(rex);
        }
    }

    if(opcode >= 0x100)
    {
        e.u8(opcode >> 8); // 0x0F (or F2/F3 for the rep string entries)
        opcode &= 0xFF;
        if(op.map)
        {
            e.u8(op.map);
        }
    }
    e.u8(opcode);

    let rip_disp_patch = -1;
    if(ea)
    {
        for(const b of ea.modrm) e.u8(b);
        if(ea.disp_placeholder)
        {
            rip_disp_patch = e.bytes.length;
            e.u32(0);
        }
        else if(ea.disp16 !== undefined)
        {
            e.u16(ea.disp16);
        }
        else
        {
            for(const b of ea.disp) e.u8(b);
        }
    }

    if(op.opcode === 0xC8) // enter: small frame, nesting level 0
    {
        e.u16(rng.uint32() % 0x400);
        e.u8(0);
    }
    else if(op.imm8 || op.imm8s || op.imm16 || op.imm1632 || op.imm32 || op.immaddr)
    {
        if(op.imm8 || op.imm8s)
        {
            if([0x0FA4, 0x0FAC].includes(op.opcode))
            {
                // shld/shrd imm: counts larger than opsize are undefined
                e.u8(rng.uint32() & (size === 16 ? 15 : size === 64 ? 63 : 31));
            }
            else
            {
                e.u8(rng.uint32());
            }
        }
        else if(op.immaddr)
        {
            // moffs: 64-bit absolute address in long mode
            const addr = SCRATCH_ADDR + 0x200 + (rng.uint32() % 0x1800);
            if(mode === 64)
            {
                e.u64(addr, 0);
            }
            else
            {
                e.u32(addr);
            }
        }
        else if(op.imm3264 && size === 64)
        {
            // mov r64, imm64
            const [lo, hi] = interesting_immediate64(rng);
            e.u64(lo, hi);
        }
        else if(op.imm16 || (op.imm1632 && size === 16))
        {
            e.u16(rng.uint32() & 0xFFFF);
        }
        else
        {
            e.u32(rng.int32());
        }
        if(op.extra_imm8)
        {
            e.u8(rng.uint32());
        }
        else if(op.extra_imm16)
        {
            e.u16(rng.uint32() & 0xFFFF);
        }
    }

    if(rip_disp_patch >= 0)
    {
        // RIP-relative: disp32 is relative to the next instruction
        const target = SCRATCH_ADDR + 0x200 + (rng.uint32() % 0x1800);
        const next_rip = LOAD_ADDR + HEADER_SIZE + e.bytes.length;
        e.patch_u32(rip_disp_patch, target - next_rip | 0);
    }

    return e.bytes.slice(instr_start);
}

/**
 * Generate a fuzz case.
 *
 * op: entry from gen/x86_table.js
 * config: { mode: 32|64, paging: bool, size: 8|16|32|64, mem: 0|1 }
 * seed: 32-bit seed
 *
 * options.user_mode: the case will also be run as a ring-3 Linux process
 *   (tests/fuzz/hw.js), so keep EFLAGS.AC clear: Linux sets CR0.AM, and AC=1
 *   would turn misaligned accesses into #AC there but not in v86's ring 0.
 *   Masking doesn't change the random number sequence.
 *
 * Returns { name, mode, paging, image, scratch, instr_bytes, low_mem_compare,
 *           entry64, done_addr, chain_ops, instr_starts, has_branch }:
 *   image: Uint8Array, a multiboot kernel image
 *   scratch: Uint8Array(SCRATCH_SIZE), initial scratch memory contents
 *   instr_bytes: the encodings of the fuzzed instructions (for logging)
 *   low_mem_compare: true if the case may write below 64 KiB (16-bit
 *                    addressing); run.js then also compares low memory
 *   entry64: (64-bit) address of the hlt ending the mode-switch stub; the
 *            64-bit body starts right after it
 *   done_addr: address of the hlt after the fuzzed instructions
 *   chain_ops: the table entries of the fuzzed instructions
 *   instr_starts: their addresses
 *   has_branch: a conditional jump over a marker write follows them
 */
export function generate_case(op, config, seed, nth, options = {})
{
    const mode = config.mode;
    const rng = new Rand(seed);
    const e = new Emitter();

    // === mode-setup stub ===
    let gdt_ptr_patch = null;
    let entry64_patch = null;
    let entry64 = 0;
    if(mode === 64)
    {
        emit_mov_reg_imm32(e, 0, PML4_ADDR);       // mov eax, PML4
        e.u8(0x0F); e.u8(0x22); e.u8(0xD8);        // mov cr3, eax
        e.u8(0x0F); e.u8(0x20); e.u8(0xE0);        // mov eax, cr4
        e.u8(0x0D); e.u32(0x620);                  // or eax, PAE|OSFXSR|OSXMMEXCPT
        e.u8(0x0F); e.u8(0x22); e.u8(0xE0);        // mov cr4, eax
        emit_mov_reg_imm32(e, 1, 0xC0000080);      // mov ecx, EFER
        e.u8(0x0F); e.u8(0x32);                    // rdmsr
        e.u8(0x0D); e.u32(0x100);                  // or eax, LME
        e.u8(0x0F); e.u8(0x30);                    // wrmsr
        e.u8(0x0F); e.u8(0x20); e.u8(0xC0);        // mov eax, cr0
        e.u8(0x0D); e.u32(0x80000001);             // or eax, PG|PE
        e.u8(0x0F); e.u8(0x22); e.u8(0xC0);        // mov cr0, eax
        e.u8(0x0F); e.u8(0x01); e.u8(0x15);        // lgdt [gdt_ptr]
        gdt_ptr_patch = e.bytes.length;
        e.u32(0);
        e.u8(0xEA);                                // jmp far 0x08:entry64
        entry64_patch = e.bytes.length;
        e.u32(0);
        e.u16(0x08);
        entry64 = e.at();
        e.u8(0xF4);                                // hlt: driver resumes after it
    }
    else if(config.paging)
    {
        // classic 32-bit paging (2-level walk): identity-map low memory
        // with a 4 MiB page
        emit_mov_reg_imm32(e, 0, PD32_ADDR);       // mov eax, PD32
        e.u8(0x0F); e.u8(0x22); e.u8(0xD8);        // mov cr3, eax
        e.u8(0x0F); e.u8(0x20); e.u8(0xE0);        // mov eax, cr4
        e.u8(0x0D); e.u32(0x10);                   // or eax, PSE
        e.u8(0x0F); e.u8(0x22); e.u8(0xE0);        // mov cr4, eax
        e.u8(0x0F); e.u8(0x20); e.u8(0xC0);        // mov eax, cr0
        e.u8(0x0D); e.u32(0x80000001);             // or eax, PG|PE
        e.u8(0x0F); e.u8(0x22); e.u8(0xC0);        // mov cr0, eax
    }

    // === choose the instruction chain ===
    // The primary instruction uses the systematically enumerated config;
    // optional extra instructions are fully random (reg or mem form).
    const chain = [{ op, size: (op.os || op.opcode % 2 === 1) ? config.size : 8, mem: config.mem }];
    while(chain.length < 3 && rng.uint32() % 3 === 0)
    {
        const op2 = CHAINABLE_OPS[rng.uint32() % CHAINABLE_OPS.length];
        if(mode === 64 && op2.opcode >= 0x40 && op2.opcode <= 0x4F && op2.opcode < 0x100)
        {
            continue; // REX in long mode; retry would bias the distribution
        }
        const sizes2 = fuzz_sizes(op2, mode);
        chain.push({
            op: op2,
            size: sizes2[rng.uint32() % sizes2.length],
            mem: op2.e ? (rng.uint32() & 1) : 0,
        });
    }

    const n_regs = mode === 64 ? 16 : 8;
    for(const c of chain)
    {
        c.g = c.op.fixed_g !== undefined ? c.op.fixed_g : (rng.uint32() % n_regs);
        c.opcode_reg_ext = mode === 64 && has_opcode_reg(c.op) && (rng.uint32() & 1) ? 1 : 0;
        const is_modrm = c.op.e || c.op.fixed_g !== undefined;
        if(!is_modrm)
        {
            c.ea = null;
        }
        else if(!c.mem)
        {
            const rm_reg = rng.uint32() % n_regs;
            c.ea = {
                form: "reg",
                modrm: [0xC0 | ((c.g & 7) << 3) | (rm_reg & 7)],
                disp: [],
                base_reg: undefined,
                index_reg: undefined,
                rm_reg,
                a32: false,
            };
        }
        else if(mode === 32 && (rng.uint32() & 3) === 0)
        {
            // 16-bit addressing (0x67): exercises the modrm16 decoder
            c.ea = gen_ea_mem16(rng, c.g);
        }
        else
        {
            c.ea = gen_ea_mem(rng, c.g, mode);
        }
    }

    const eas = chain.map(c => c.ea);
    const low_mem_compare = eas.some(ea => ea && (ea.form === "mem16" || ea.form === "abs16"));

    // === prologue: random machine state ===

    // stack pointer: near the top of the scratch region, headroom for pushes
    const sp = STACK_TOP - 0x100 - (rng.uint32() % 0x40);
    if(mode === 64)
    {
        emit_mov_reg_imm64(e, REG_ESP, sp | 0, 0);
    }
    else
    {
        emit_mov_reg_imm32(e, REG_ESP, sp);
    }

    for(const reg of all_gprs(mode))
    {
        const [lo, hi] = ea_reg_value(rng, reg, eas, mode);
        if(mode === 64)
        {
            emit_mov_reg_imm64(e, reg, lo, hi);
        }
        else
        {
            emit_mov_reg_imm32(e, reg, lo);
        }
    }

    // fpu/mmx and xmm state; initialised from a data area embedded in the image
    const fpu_data_refs = []; // byte offsets of disp32 fields
    const xmm_data_refs = [];
    const misc_data_refs = [];
    const emit_abs_ref = (refs, g_field) =>
    {
        if(mode === 64)
        {
            // SIB absolute form (mod=0 rm=5 is RIP-relative in long mode)
            refs.push(e.bytes.length + 2);
            e.u8((g_field & 7) << 3 | 4); // modrm
            e.u8(0x25);                   // sib: no index, base 5
            e.u32(0);                     // disp32 placeholder
        }
        else
        {
            refs.push(e.bytes.length + 1);
            e.u8((g_field & 7) << 3 | 5); // modrm: [disp32]
            e.u32(0);
        }
    };

    if(chain.some(c => c.op.is_fpu))
    {
        e.u8(0xDB); e.u8(0xE3); // finit
        for(let i = 0; i < 8; i++)
        {
            e.u8(0xDD); // fld qword [data]
            emit_abs_ref(fpu_data_refs, 0);
        }
        for(let i = 0; i < 4; i++)
        {
            // fstp qword [scratch]: half-empty fpu stack
            e.u8(0xDD);
            if(mode === 64)
            {
                e.u8(0x1C); e.u8(0x25);
            }
            else
            {
                e.u8(0x1D);
            }
            e.u32(SCRATCH_ADDR + 0x1F00);
        }

        if(rng.uint32() & 1)
        {
            // randomise the fpu control word (rounding/precision control)
            e.u8(0xD9); // fldcw [data]
            misc_data_refs.push({ at: null, kind: "cw" });
            if(mode === 64)
            {
                misc_data_refs[misc_data_refs.length - 1].at = e.bytes.length + 2;
                e.u8(0x2C); e.u8(0x25); e.u32(0);
            }
            else
            {
                misc_data_refs[misc_data_refs.length - 1].at = e.bytes.length + 1;
                e.u8(0x2D); e.u32(0);
            }
        }
    }
    else
    {
        for(let i = 0; i < 8; i++)
        {
            e.u8(0x0F); e.u8(0x6F); // movq mm<i>, [data]
            emit_abs_ref(fpu_data_refs, i);
        }
    }

    const n_xmm = mode === 64 ? 16 : 8;
    for(let i = 0; i < n_xmm; i++)
    {
        // movdqu xmm<i>, [data]
        e.u8(0xF3);
        if(i >= 8) e.u8(0x44); // REX.R
        e.u8(0x0F); e.u8(0x6F);
        emit_abs_ref(xmm_data_refs, i & 7);
    }

    if(rng.uint32() & 1)
    {
        // randomise MXCSR (sse rounding mode, denormals-are-zero,
        // flush-to-zero, exception masks)
        e.u8(0x0F); e.u8(0xAE); // ldmxcsr [data]
        misc_data_refs.push({ at: null, kind: "mxcsr" });
        if(mode === 64)
        {
            misc_data_refs[misc_data_refs.length - 1].at = e.bytes.length + 2;
            e.u8(0x14); e.u8(0x25); e.u32(0);
        }
        else
        {
            misc_data_refs[misc_data_refs.length - 1].at = e.bytes.length + 1;
            e.u8(0x15); e.u32(0);
        }
    }

    // random eflags (TF/IF/NT/RF/VM masked out); push imm32; popf also works
    // in long mode (sign-extended imm32, popfq)
    e.u8(0x68);
    e.u32(rng.int32() & ~FLAGS_MASK_OUT & ~(options.user_mode ? 1 << 18 : 0));
    e.u8(0x9D);

    if(rng.int32() & 1)
    {
        // set flags via an arithmetic instruction: not well-distributed,
        // but can trigger bugs in lazy flag calculation
        if(rng.int32() & 1)
        {
            e.u8(0x02); e.u8(0xC4); // add al, ah
        }
        else
        {
            e.u8(0x2A); e.u8(0xC0); // sub al, al
        }
    }

    // === the fuzzed instructions ===
    const instr_bytes = [];
    const instr_starts = [];
    for(const c of chain)
    {
        instr_starts.push(e.at());
        instr_bytes.push(...emit_instruction(e, rng, c, mode));
    }

    // occasionally: a conditional jump over a marker write. The marker lands
    // in the (compared) scratch region only when the branch is not taken, so
    // the final state reveals whether both engines evaluated the condition
    // identically (exercises the lazy-flag state and the jit flag fusion).
    let has_branch = false;
    if(rng.uint32() % 5 === 0)
    {
        has_branch = true;
        const cond = rng.uint32() & 15;
        const marker_addr = SCRATCH_ADDR + 4 * (rng.uint32() % 0x600);
        const marker_val = rng.int32();
        const near = (rng.uint32() & 1) === 0;

        // marker instruction: mov dword [marker_addr], marker_val
        // (absolute addressing; SIB form in 64-bit where rm=5 is RIP-relative)
        const marker_len = mode === 64 ? 11 : 10;

        if(near)
        {
            e.u8(0x0F); e.u8(0x80 | cond); e.u32(marker_len); // jcc near
        }
        else
        {
            e.u8(0x70 | cond); e.u8(marker_len); // jcc short
        }
        if(mode === 64)
        {
            e.u8(0xC7); e.u8(0x04); e.u8(0x25); // mov dword [sib disp32], imm32
        }
        else
        {
            e.u8(0xC7); e.u8(0x05); // mov dword [disp32], imm32
        }
        e.u32(marker_addr);
        e.u32(marker_val);
    }

    // hlt (+ padding: a RIP-relative write may land here, keep it harmless)
    const done_addr = e.at();
    e.u8(0xF4);
    for(let i = 0; i < 64; i++) e.u8(0xF4);

    // === data area ===
    e.align(16);
    const fpu_data_addr = e.at();
    const fpu_data = [];
    for(let i = 0; i < 8; i++)
    {
        fpu_data.push(interesting_fpu_half(rng, false), interesting_fpu_half(rng, true));
    }
    const xmm_data_addr = fpu_data_addr + 64;
    const xmm_data = [];
    for(let i = 0; i < n_xmm * 4; i++)
    {
        xmm_data.push(rng.uint32() & 1 ? (U32_EDGE[rng.uint32() % U32_EDGE.length] | 0) : rng.int32());
    }
    let cw_addr = 0, mxcsr_addr = 0;
    for(const v of fpu_data) e.u32(v);
    for(const v of xmm_data) e.u32(v);
    for(const ref of misc_data_refs)
    {
        if(ref.kind === "cw")
        {
            e.align(2);
            cw_addr = e.at();
            // rounding control (bits 10-11) and precision control (8-9)
            e.u16(0x007F | (rng.uint32() & 0x0F00));
            e.patch_u32(ref.at, cw_addr);
        }
        else
        {
            e.align(4);
            mxcsr_addr = e.at();
            e.u32(rng.uint32() & 0xFFFF); // only the defined low 16 bits
            e.patch_u32(ref.at, mxcsr_addr);
        }
    }

    for(const [i, off] of fpu_data_refs.entries())
    {
        e.patch_u32(off, fpu_data_addr + i * 8);
    }
    for(const [i, off] of xmm_data_refs.entries())
    {
        e.patch_u32(off, xmm_data_addr + i * 16);
    }

    if(mode === 64)
    {
        // gdt: null, 64-bit code (0x08), data (0x10)
        e.align(16);
        const gdt_addr = e.at();
        e.u64(0, 0);
        e.u64(0x0000FFFF | 0, 0x00AF9A00); // code64: L=1, D=0, P, type=exec/read
        e.u64(0x0000FFFF | 0, 0x00CF9200); // data: G=1, D/B=1, P, type=read/write
        const gdt_ptr_addr = e.at();
        e.patch_u32(gdt_ptr_patch, gdt_ptr_addr);
        e.u16(3 * 8 - 1);      // limit
        e.u32(gdt_addr);       // base (32-bit lgdt)
        e.patch_u32(entry64_patch, entry64);
    }

    // random initial scratch memory contents, biased towards edge values
    const scratch = new Uint8Array(SCRATCH_SIZE);
    for(let i = 0; i < SCRATCH_SIZE; i += 4)
    {
        const v = rng.uint32() & 1 ? (U32_EDGE[rng.uint32() % U32_EDGE.length] >>> 0) : rng.uint32();
        scratch[i] = v & 0xFF;
        scratch[i + 1] = v >> 8 & 0xFF;
        scratch[i + 2] = v >> 16 & 0xFF;
        scratch[i + 3] = v >>> 24 & 0xFF;
    }

    // === multiboot image ===
    const image = new Uint8Array(HEADER_SIZE + e.bytes.length);
    const view = new DataView(image.buffer);
    image.set(e.bytes, HEADER_SIZE);
    const load_end = LOAD_ADDR + image.length;
    view.setUint32(0, MULTIBOOT_HEADER_MAGIC, true);
    view.setUint32(4, MULTIBOOT_HEADER_ADDRESS, true);
    view.setUint32(8, -(MULTIBOOT_HEADER_MAGIC + MULTIBOOT_HEADER_ADDRESS) | 0, true);
    view.setUint32(12, LOAD_ADDR, true);            // header_addr
    view.setUint32(16, LOAD_ADDR, true);            // load_addr
    view.setUint32(20, load_end, true);             // load_end_addr
    view.setUint32(24, load_end, true);             // bss_end_addr
    view.setUint32(28, LOAD_ADDR + HEADER_SIZE, true); // entry_addr

    return {
        name: case_name(op, config, nth === undefined ? 0 : nth) +
            (chain.length > 1 ? "+" + chain.length : "") + (has_branch ? "j" : ""),
        mode,
        paging: config.paging,
        image,
        scratch,
        instr_bytes: new Uint8Array(instr_bytes),
        low_mem_compare,
        entry64,
        done_addr,
        chain_ops: chain.map(c => c.op),
        instr_starts,
        has_branch,
    };
}
