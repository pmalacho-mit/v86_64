#![allow(non_snake_case)]

//! Code generation for 64-bit (long mode) code.
//!
//! The 64-bit jit shares the block finder, control flow structuring and module generation with
//! the 32-bit jit (see jit.rs), but has its own instruction table (gen/generate_jit64.js). In this
//! first version the guest registers stay in memory, and instructions are compiled to calls of
//! generated wrappers (gen/jit64_wrappers.rs) around the interpreter's instruction handlers, with
//! the instruction already decoded: operands are passed as constants and memory addresses are
//! computed inline. This removes the interpreter's fetch/decode/dispatch overhead and lets
//! control flow (jumps, loops, calls within a page) stay inside the compiled module.
//!
//! The wrappers set up the instruction pointer and prefixes like the interpreter would and
//! return whether the compiled code has to be left: when the instruction raised an exception
//! (the interpreter delivers it immediately, see JIT64_EXIT), or wrote to a page containing
//! compiled code.

use crate::codegen;
use crate::cpu::cpu;
use crate::cpu::global_pointers;
use crate::cpu_context::CpuContext;
use crate::gen;
use crate::jit::{is_near_end_of_page, JitContext, JIT_INSTR_BLOCK_BOUNDARY_FLAG};
use crate::prefix::{
    PREFIX_66, PREFIX_67, PREFIX_F2, PREFIX_F3, PREFIX_MASK_REX, PREFIX_MASK_SEGMENT,
    PREFIX_REX_PRESENT, PREFIX_REX_W,
};
use crate::regs::{CS, DS, ES, FS, GS, SS};
use crate::wasmgen::wasm_builder::WasmBuilder;
use crate::wasmgen::wasm_opcodes as op;

/// Set when the interpreter delivers an exception or interrupt, or when a page with compiled
/// code is written to: the 64-bit jit must not continue with the rest of the compiled block
pub static mut JIT64_EXIT: bool = false;

/// An argument for a generic instruction call
pub enum A {
    I32(i32),
    I64(u64),
}

/// A decoded memory operand in 64-bit mode (register numbers include the REX extensions)
#[derive(Copy, Clone, PartialEq)]
pub struct Modrm64 {
    base: Option<u32>,
    index: Option<u32>,
    shift: u8,
    disp: i32,
    rip_relative: bool,
}

pub fn decode_modrm(cpu: &mut CpuContext, modrm_byte: u8) -> Modrm64 {
    dbg_assert!(modrm_byte < 0xC0);
    let md = modrm_byte >> 6;
    let rm = modrm_byte & 7;

    let mut m = Modrm64 {
        base: None,
        index: None,
        shift: 0,
        disp: 0,
        rip_relative: false,
    };

    if rm == 4 {
        let sib = cpu.read_imm8();
        let base = sib & 7;
        let index = (sib >> 3 & 7) as u32 | cpu.rex_x();
        m.shift = sib >> 6;
        // index 4 (rsp) means no index, but r12 (4 with REX.X) is a valid index
        m.index = if index == 4 { None } else { Some(index) };
        if base == 5 && md == 0 {
            m.disp = cpu.read_imm32() as i32;
            return m;
        }
        m.base = Some(base as u32 | cpu.rex_b());
    }
    else if rm == 5 && md == 0 {
        m.rip_relative = true;
        m.disp = cpu.read_imm32() as i32;
        return m;
    }
    else {
        m.base = Some(rm as u32 | cpu.rex_b());
    }

    m.disp = match md {
        0 => 0,
        1 => cpu.read_imm8s() as i32,
        _ => cpu.read_imm32() as i32,
    };
    m
}

pub fn skip_modrm(cpu: &mut CpuContext, modrm_byte: u8) {
    if modrm_byte < 0xC0 {
        let _ = decode_modrm(cpu, modrm_byte);
    }
}

/// Generate the effective address (without segment base) of a memory operand as an i64.
/// Must be called after all of the instruction's bytes have been read (rip-relative operands
/// are relative to the end of the instruction).
pub fn gen_modrm_offset(ctx: &mut JitContext, m: &Modrm64) {
    if m.rip_relative {
        // the page of the instruction is the page of instruction_pointer
        codegen::gen_get_eip64(ctx.builder);
        ctx.builder.const_i64(!0xFFF);
        ctx.builder.and_i64();
        ctx.builder
            .const_i64((ctx.cpu.eip & 0xFFF) as i64 + m.disp as i64);
        ctx.builder.add_i64();
        return;
    }

    let mut have_something_on_stack = false;
    if let Some(base) = m.base {
        gen_get_reg(ctx, 64, base);
        have_something_on_stack = true;
    }
    if let Some(index) = m.index {
        gen_get_reg(ctx, 64, index);
        if m.shift != 0 {
            ctx.builder.const_i64(m.shift as i64);
            ctx.builder.shl_i64();
        }
        if have_something_on_stack {
            ctx.builder.add_i64();
        }
        have_something_on_stack = true;
    }
    if m.disp != 0 || !have_something_on_stack {
        ctx.builder.const_i64(m.disp as i64);
        if have_something_on_stack {
            ctx.builder.add_i64();
        }
    }
}

/// Generate the linear address of a memory operand as an i64: in 64-bit mode only fs and gs
/// have a segment base
pub fn gen_modrm_address(ctx: &mut JitContext, m: &Modrm64) {
    gen_modrm_offset(ctx, m);
    let segment_prefix = ctx.cpu.prefixes & PREFIX_MASK_SEGMENT;
    if segment_prefix == FS as u16 + 1 {
        ctx.builder.load_fixed_i64(global_pointers::fs_base as u32);
        ctx.builder.add_i64();
    }
    else if segment_prefix == GS as u16 + 1 {
        ctx.builder.load_fixed_i64(global_pointers::gs_base as u32);
        ctx.builder.add_i64();
    }
}

// ---------------------------------------------------------------------------------------------
// Native code generation for common integer SSE instructions (66 prefix). These instructions
// set no flags and don't touch general-purpose registers, so unlike gen_call_wrapper they
// don't commit the deferred lazy flags, don't spill registers and don't reset the data tlb
// cache; the only exceptions they can raise are #NM (task switch) and page faults on the
// memory operand.

/// #NM check as the interpreter's SSE handlers do it (task_switch_test_mmx): the cold branch
/// commits the deferred lazy flags before the exception is delivered.
pub fn gen_sse_task_switch_check(ctx: &mut JitContext) {
    let cr0_offset = global_pointers::get_creg_offset(0);
    ctx.builder.load_fixed_u8(cr0_offset);
    ctx.builder
        .const_i32((crate::regs::CR0_EM | crate::regs::CR0_TS) as i32);
    ctx.builder.and_i32();
    // raises #NM: a cold exception exit from the module
    ctx.builder.if_void_unlikely();
    {
        gen_commit_deferred_flags_cold(ctx);
        codegen::gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
        codegen::gen_fn1_const(
            ctx.builder,
            "task_switch_test_mmx_jit",
            (ctx.start_of_current_instruction & 0xFFF) as u32,
        );
        ctx.builder.br(ctx.exit_with_fault_label);
    }
    ctx.builder.block_end();
}

/// cli: clear IF when IOPL allows it. In 64-bit mode protected mode is always on and the
/// vm86 flag is never set, so the interpreter's check reduces to iopl >= cpl. The (rare)
/// #GP path raises the fault through exit_with_fault_label. It must not spill the registers
/// itself: gen_spill_dirty_registers clears the compile-time dirty set, which the hot path
/// would then inherit (and later register writes before a wrapper call would be lost).
pub fn gen_cli(ctx: &mut JitContext) {
    // if iopl >= cpl: flags &= ~FLAG_INTERRUPT
    ctx.builder.load_fixed_i32(global_pointers::flags as u32);
    ctx.builder.const_i32(12);
    ctx.builder.shr_u_i32();
    ctx.builder.const_i32(3);
    ctx.builder.and_i32();
    ctx.builder.load_fixed_u8(global_pointers::cpl as u32);
    ctx.builder.ge_i32();

    ctx.builder.if_void_likely();
    {
        ctx.builder.const_i32(global_pointers::flags as i32);
        ctx.builder.load_fixed_i32(global_pointers::flags as u32);
        ctx.builder.const_i32(!crate::cpu::cpu::FLAG_INTERRUPT);
        ctx.builder.and_i32();
        ctx.builder.store_aligned_i32(0);
    }
    ctx.builder.else_();
    {
        gen_commit_deferred_flags_cold(ctx);
        codegen::gen_trigger_gp(ctx, 0);
    }
    ctx.builder.block_end();
}

/// Read the low and high 64-bit halves of the memory operand of an SSE instruction into
/// locals. Two independent 64-bit reads, so a page-crossing access faults like
/// safe_read128s (and the second read hits the data tlb cache).
fn gen_sse_read_mem_halves(ctx: &mut JitContext, m: &Modrm64) -> (WasmLocalI64, WasmLocalI64) {
    gen_modrm_address(ctx, m);
    let addr = ctx.builder.set_new_local_i64();
    gen_safe_read(ctx, 64, &addr);
    let lo = ctx.builder.set_new_local_i64();
    ctx.builder.get_local_i64(&addr);
    ctx.builder.const_i64(8);
    ctx.builder.add_i64();
    let addr2 = ctx.builder.set_new_local_i64();
    gen_safe_read(ctx, 64, &addr2);
    let hi = ctx.builder.set_new_local_i64();
    ctx.builder.free_local_i64(addr);
    ctx.builder.free_local_i64(addr2);
    (lo, hi)
}

/// dst = dst op src, one lane_op per 64-bit half (pxor/por/pand/paddq)
pub fn gen_sse_binary64(ctx: &mut JitContext, op: u32, modrm_byte: u8) {
    gen_sse_task_switch_check(ctx);
    let r = (modrm_byte >> 3 & 7) as u32 | ctx.cpu.rex_r();
    let lane_op: fn(&mut WasmBuilder) = match op {
        0x660FEF => WasmBuilder::xor_i64,
        0x660FEB => WasmBuilder::or_i64,
        0x660FDB => WasmBuilder::and_i64,
        0x660FD4 => WasmBuilder::add_i64,
        _ => unreachable!(),
    };
    let (lo, hi) = if modrm_byte >= 0xC0 {
        let rm = (modrm_byte & 7) as u32 | ctx.cpu.rex_b();
        let off = global_pointers::get_reg_xmm_offset(rm);
        ctx.builder.load_fixed_i64(off);
        let lo = ctx.builder.set_new_local_i64();
        ctx.builder.load_fixed_i64(off + 8);
        let hi = ctx.builder.set_new_local_i64();
        (lo, hi)
    }
    else {
        let m = decode_modrm(ctx.cpu, modrm_byte);
        gen_sse_read_mem_halves(ctx, &m)
    };
    apply_sse_binary64(ctx, r, &lo, &hi, lane_op);
    ctx.builder.free_local_i64(lo);
    ctx.builder.free_local_i64(hi);
}

fn apply_sse_binary64(
    ctx: &mut JitContext,
    dst: u32,
    lo: &WasmLocalI64,
    hi: &WasmLocalI64,
    lane_op: fn(&mut WasmBuilder),
) {
    let dst_off = global_pointers::get_reg_xmm_offset(dst);
    ctx.builder.const_i32(dst_off as i32);
    ctx.builder.load_fixed_i64(dst_off);
    ctx.builder.get_local_i64(lo);
    lane_op(&mut ctx.builder);
    ctx.builder.store_aligned_i64(0);
    ctx.builder.const_i32(dst_off as i32 + 8);
    ctx.builder.load_fixed_i64(dst_off + 8);
    ctx.builder.get_local_i64(hi);
    lane_op(&mut ctx.builder);
    ctx.builder.store_aligned_i64(0);
}

/// movdqa/movdqu (66/F3 0F 6F): xmm = xmm/m128
pub fn gen_sse_mov_load(ctx: &mut JitContext, modrm_byte: u8) {
    gen_sse_task_switch_check(ctx);
    let r = (modrm_byte >> 3 & 7) as u32 | ctx.cpu.rex_r();
    let (lo, hi) = if modrm_byte >= 0xC0 {
        let rm = (modrm_byte & 7) as u32 | ctx.cpu.rex_b();
        let off = global_pointers::get_reg_xmm_offset(rm);
        ctx.builder.load_fixed_i64(off);
        let lo = ctx.builder.set_new_local_i64();
        ctx.builder.load_fixed_i64(off + 8);
        let hi = ctx.builder.set_new_local_i64();
        (lo, hi)
    }
    else {
        let m = decode_modrm(ctx.cpu, modrm_byte);
        gen_sse_read_mem_halves(ctx, &m)
    };
    let dst_off = global_pointers::get_reg_xmm_offset(r);
    ctx.builder.const_i32(dst_off as i32);
    ctx.builder.get_local_i64(&lo);
    ctx.builder.store_aligned_i64(0);
    ctx.builder.const_i32(dst_off as i32 + 8);
    ctx.builder.get_local_i64(&hi);
    ctx.builder.store_aligned_i64(0);
    ctx.builder.free_local_i64(lo);
    ctx.builder.free_local_i64(hi);
}

/// movdqa/movdqu store (66/F3 0F 7F): xmm/m128 = xmm
pub fn gen_sse_mov_store(ctx: &mut JitContext, modrm_byte: u8) {
    gen_sse_task_switch_check(ctx);
    let r = (modrm_byte >> 3 & 7) as u32 | ctx.cpu.rex_r();
    let src_off = global_pointers::get_reg_xmm_offset(r);
    if modrm_byte >= 0xC0 {
        // register to register (xmm_rm = xmm_r): copy both halves. A wasm store takes the
        // address below the value, so the destination is pushed before the load
        let rm = (modrm_byte & 7) as u32 | ctx.cpu.rex_b();
        let dst_off = global_pointers::get_reg_xmm_offset(rm);
        ctx.builder.const_i32(dst_off as i32);
        ctx.builder.load_fixed_i64(src_off);
        ctx.builder.store_aligned_i64(0);
        ctx.builder.const_i32(dst_off as i32 + 8);
        ctx.builder.load_fixed_i64(src_off + 8);
        ctx.builder.store_aligned_i64(0);
        return;
    }
    let m = decode_modrm(ctx.cpu, modrm_byte);
    gen_modrm_address(ctx, &m);
    let addr = ctx.builder.set_new_local_i64();
    ctx.builder.load_fixed_i64(src_off);
    let lo = ctx.builder.set_new_local_i64();
    gen_safe_write(ctx, 64, &addr, &Val::I64(lo.unsafe_clone()));
    ctx.builder.get_local_i64(&addr);
    ctx.builder.const_i64(8);
    ctx.builder.add_i64();
    let addr2 = ctx.builder.set_new_local_i64();
    ctx.builder.load_fixed_i64(src_off + 8);
    let hi = ctx.builder.set_new_local_i64();
    gen_safe_write(ctx, 64, &addr2, &Val::I64(hi.unsafe_clone()));
    ctx.builder.free_local_i64(addr);
    ctx.builder.free_local_i64(addr2);
    ctx.builder.free_local_i64(lo);
    ctx.builder.free_local_i64(hi);
}

/// Load the full xmm register as v128 (for the SIMD-native instructions below)
fn gen_xmm_load_v128(ctx: &mut JitContext, r: u32) {
    ctx.builder
        .const_i32(global_pointers::get_reg_xmm_offset(r) as i32);
    ctx.builder.load_v128(0);
}

/// Store the v128 on the stack into the xmm register; the caller must have pushed the
/// register's address first (wasm stores take the address below the value)
fn gen_xmm_store_v128(ctx: &mut JitContext, _r: u32) { ctx.builder.store_v128(0); }

/// Push the address of an xmm register (for gen_xmm_store_v128)
fn gen_xmm_addr(ctx: &mut JitContext, r: u32) {
    ctx.builder
        .const_i32(global_pointers::get_reg_xmm_offset(r) as i32);
}

/// 66 0F FC/FD/FE (paddb/paddw/paddd) with a register source: one v128 add
/// 66 0F FC/FD/FE/F8/F9/FA (padd/psub b/w/d) with a register source: one v128 op
pub fn gen_sse_padd(ctx: &mut JitContext, op: u32, modrm_byte: u8) {
    gen_sse_task_switch_check(ctx);
    dbg_assert!(modrm_byte >= 0xC0);
    let add_op = match op {
        0x660FFC => op::SIMD_I8X16_ADD,
        0x660FFD => op::SIMD_I16X8_ADD,
        0x660FFE => op::SIMD_I32X4_ADD,
        0x660FF8 => op::SIMD_I8X16_SUB,
        0x660FF9 => op::SIMD_I16X8_SUB,
        0x660FFA => op::SIMD_I32X4_SUB,
        _ => unreachable!(),
    };
    let r = (modrm_byte >> 3 & 7) as u32 | ctx.cpu.rex_r();
    let rm = (modrm_byte & 7) as u32 | ctx.cpu.rex_b();
    gen_xmm_addr(ctx, r);
    gen_xmm_load_v128(ctx, r);
    gen_xmm_load_v128(ctx, rm);
    ctx.builder.simd_binop(add_op);
    gen_xmm_store_v128(ctx, r);
}

/// 66 0F 71/72/73 group shifts by an immediate (imm already read as a constant):
/// 71: psrlw/psraw/psllw (/2 /4 /6), 72: psrld/psrad/pslld, 73: psrlq/psllq and the
/// byte shifts psrldq (/3) / pslldq (/7). Memory forms are #UD (handled by the caller).
pub fn gen_sse_shift_imm(ctx: &mut JitContext, op: u32, kind: u32, imm: i32, modrm_byte: u8) {
    gen_sse_task_switch_check(ctx);
    dbg_assert!(modrm_byte >= 0xC0);
    let r = (modrm_byte & 7) as u32 | ctx.cpu.rex_b();
    // psrldq/psrlq-family byte shifts: shuffle the lanes (zero as the second operand)
    if op == 0x660F73 && (kind == 3 || kind == 7) {
        let s = (imm as u32).min(16) as usize;
        let mut lanes = [16u8; 16]; // default: second operand (zero)
        for i in 0..16 {
            if kind == 3 {
                // psrldq: out[i] = in[i + s]
                if i + s < 16 {
                    lanes[i] = (i + s) as u8;
                }
            }
            else {
                // pslldq: out[i] = in[i - s]
                if i >= s {
                    lanes[i] = (i - s) as u8;
                }
            }
        }
        gen_xmm_addr(ctx, r);
        gen_xmm_load_v128(ctx, r);
        ctx.builder.const_v128(&[0; 16]);
        ctx.builder.i8x16_shuffle(&lanes);
        gen_xmm_store_v128(ctx, r);
        return;
    }
    // plain lane shifts: (wasm op, lane width, signed?)
    let (shift_op, width, signed): (u32, i32, bool) = match (op, kind) {
        (0x660F71, 2) => (op::SIMD_I16X8_SHR_U, 16, false),
        (0x660F71, 4) => (op::SIMD_I16X8_SHR_S, 16, true),
        (0x660F71, 6) => (op::SIMD_I16X8_SHL, 16, false),
        (0x660F72, 2) => (op::SIMD_I32X4_SHR_U, 32, false),
        (0x660F72, 4) => (op::SIMD_I32X4_SHR_S, 32, true),
        (0x660F72, 6) => (op::SIMD_I32X4_SHL, 32, false),
        (0x660F73, 2) => (op::SIMD_I64X2_SHR_U, 64, false),
        (0x660F73, 6) => (op::SIMD_I64X2_SHL, 64, false),
        _ => unreachable!(),
    };
    // signed shifts clamp at width-1 (shifting by the lane width or more gives the
    // sign-fill, which the shift by width-1 already produces); logical/left shifts
    // give zero. Wasm shifts take the count modulo the lane width, so the count must
    // be masked here either way.
    let imm = if signed { imm.min(width - 1) } else { imm };
    if !signed && imm >= width {
        gen_xmm_store_zero(ctx, r);
        return;
    }
    gen_xmm_addr(ctx, r);
    gen_xmm_load_v128(ctx, r);
    ctx.builder.const_i32(imm);
    ctx.builder.simd_shift(shift_op);
    gen_xmm_store_v128(ctx, r);
}

fn gen_xmm_store_zero(ctx: &mut JitContext, r: u32) {
    gen_xmm_addr(ctx, r);
    ctx.builder.const_v128(&[0; 16]);
    gen_xmm_store_v128(ctx, r);
}

/// 66 0F 70 pshufd (register source): select four dwords by the immediate
pub fn gen_sse_pshufd(ctx: &mut JitContext, imm: i32, modrm_byte: u8) {
    gen_sse_task_switch_check(ctx);
    dbg_assert!(modrm_byte >= 0xC0);
    let r = (modrm_byte >> 3 & 7) as u32 | ctx.cpu.rex_r();
    let rm = (modrm_byte & 7) as u32 | ctx.cpu.rex_b();
    let mut lanes = [0u8; 16];
    for k in 0..4u32 {
        let sel = ((imm as u32) >> (2 * k)) & 3;
        for j in 0..4u32 {
            lanes[(k * 4 + j) as usize] = (sel * 4 + j) as u8;
        }
    }
    gen_xmm_addr(ctx, r);
    gen_xmm_load_v128(ctx, rm);
    gen_xmm_load_v128(ctx, rm);
    ctx.builder.i8x16_shuffle(&lanes);
    gen_xmm_store_v128(ctx, r);
}

/// 66 0F 6C/6D punpcklqdq/punpckhqdq (register source)
pub fn gen_sse_punpckqdq(ctx: &mut JitContext, op: u32, modrm_byte: u8) {
    gen_sse_task_switch_check(ctx);
    dbg_assert!(modrm_byte >= 0xC0);
    let r = (modrm_byte >> 3 & 7) as u32 | ctx.cpu.rex_r();
    let rm = (modrm_byte & 7) as u32 | ctx.cpu.rex_b();
    let lo = op == 0x660F6C;
    let mut lanes = [0u8; 16];
    for i in 0..8u32 {
        lanes[i as usize] = if lo { i as u8 } else { (i + 8) as u8 };
        lanes[(i + 8) as usize] = if lo { (i + 16) as u8 } else { (i + 24) as u8 };
    }
    gen_xmm_addr(ctx, r);
    gen_xmm_load_v128(ctx, r);
    gen_xmm_load_v128(ctx, rm);
    ctx.builder.i8x16_shuffle(&lanes);
    gen_xmm_store_v128(ctx, r);
}

/// 66 0F 38 00 pshufb (register source): result[i] = (src[i] & 0x80) == 0 ? dst[src[i] & 15] : 0.
/// i8x16.swizzle zero-fills out-of-range indices, so the index only needs the 0x0f mask;
/// the sign test becomes a per-byte select between the swizzled value and zero.
pub fn gen_sse_pshufb(ctx: &mut JitContext, modrm_byte: u8) {
    gen_sse_task_switch_check(ctx);
    dbg_assert!(modrm_byte >= 0xC0);
    let r = (modrm_byte >> 3 & 7) as u32 | ctx.cpu.rex_r();
    let rm = (modrm_byte & 7) as u32 | ctx.cpu.rex_b();
    gen_xmm_addr(ctx, r);
    gen_xmm_load_v128(ctx, r); // data
    gen_xmm_load_v128(ctx, rm); // selector
    ctx.builder.const_v128(&[0x0f; 16]);
    ctx.builder.simd_binop(op::SIMD_V128_AND);
    ctx.builder.simd_op_swizzle();
    // per-byte mask: 0xff where (selector & 0x80) == 0
    ctx.builder.const_v128(&[0; 16]);
    gen_xmm_load_v128(ctx, rm);
    ctx.builder.const_v128(&[0x80; 16]);
    ctx.builder.simd_binop(op::SIMD_V128_AND);
    ctx.builder.const_v128(&[0; 16]);
    ctx.builder.simd_op_i8x16_eq();
    ctx.builder.simd_op_bitselect();
    gen_xmm_store_v128(ctx, r);
}

/// 66 0F 3A 0F palignr (register source): result[i] = byte (shift + i) of the
/// concatenation src:dst (src in the low 16 bytes), zero above 32 bytes.
/// A shuffle reads byte j of the first operand for j < 16 and byte j - 16 of the
/// second otherwise, so for a shift below 16 the lane index is shift + i for both
/// halves, and for a shift of 16..31 the result comes entirely from dst.
pub fn gen_sse_palignr(ctx: &mut JitContext, imm: i32, modrm_byte: u8) {
    gen_sse_task_switch_check(ctx);
    dbg_assert!(modrm_byte >= 0xC0);
    let r = (modrm_byte >> 3 & 7) as u32 | ctx.cpu.rex_r();
    let rm = (modrm_byte & 7) as u32 | ctx.cpu.rex_b();
    let shift = (imm & 0xFF) as u32;
    gen_xmm_addr(ctx, r);
    if shift == 0 {
        // dst = src
        gen_xmm_load_v128(ctx, rm);
    }
    else if shift < 16 {
        let mut lanes = [0u8; 16];
        for (i, lane) in lanes.iter_mut().enumerate() {
            *lane = (shift as usize + i) as u8;
        }
        gen_xmm_load_v128(ctx, rm);
        gen_xmm_load_v128(ctx, r);
        ctx.builder.i8x16_shuffle(&lanes);
    }
    else if shift < 32 {
        let mut lanes = [0u8; 16];
        for (i, lane) in lanes.iter_mut().enumerate() {
            *lane = (shift as usize + i - 16) as u8;
        }
        gen_xmm_load_v128(ctx, r);
        gen_xmm_load_v128(ctx, r);
        ctx.builder.i8x16_shuffle(&lanes);
    }
    else {
        ctx.builder.const_v128(&[0; 16]);
    }
    gen_xmm_store_v128(ctx, r);
}

/// The direction increment (runtime, from the direction flag) in a local
fn gen_string_increment(ctx: &mut JitContext, size_bytes: i32) -> WasmLocal {
    let inc = ctx.builder.new_local();
    // select takes the condition last: DF set ? -size : size
    ctx.builder.const_i32(-size_bytes);
    ctx.builder.const_i32(size_bytes);
    ctx.builder.load_fixed_i32(global_pointers::flags as u32);
    ctx.builder.const_i32(crate::cpu::cpu::FLAG_DIRECTION);
    ctx.builder.and_i32();
    ctx.builder.select();
    ctx.builder.set_local(&inc);
    inc
}

/// add the increment to a register (64-bit; the asize-32 form never reaches here)
fn gen_string_advance(ctx: &mut JitContext, r: u32, inc: &WasmLocal) {
    gen_get_reg(ctx, 64, r);
    ctx.builder.get_local(inc);
    ctx.builder.extend_signed_i32_to_i64();
    ctx.builder.add_i64();
    gen_set_reg_from_stack(ctx, 64, r);
}

/// movs/stos/lods without rep: one flagless access plus rsi/rdi updates. The deferred lazy
/// flags survive (the instructions set no flags), and only rsi/rdi become dirty. In 64-bit
/// mode es and ds have a zero base, so fs/gs overrides fall back to the interpreter's
/// wrapper (which reads the segment prefix from the cpu state).
fn gen_string_segment_is_zero_base(ctx: &JitContext) -> bool {
    // 0 (no prefix), es, cs, ss, ds all have a zero base in long mode
    matches!(ctx.cpu.prefixes & PREFIX_MASK_SEGMENT, 0 | 1 | 2 | 3 | 4)
}

pub fn gen_string_movs(ctx: &mut JitContext, bits: u32, fallback: &'static str) {
    if !gen_string_segment_is_zero_base(ctx) {
        return gen_generic(ctx, fallback, &[]);
    }
    let size_bytes = (bits / 8) as i32;
    gen_get_reg(ctx, 64, crate::regs::ESI);
    let src = ctx.builder.set_new_local_i64();
    gen_safe_read(ctx, bits, &src);
    let value = match bits {
        64 => Val::I64(ctx.builder.set_new_local_i64()),
        _ => Val::I32(ctx.builder.set_new_local()),
    };
    gen_get_reg(ctx, 64, crate::regs::EDI);
    let dst = ctx.builder.set_new_local_i64();
    gen_safe_write(ctx, bits, &dst, &value);
    match value {
        Val::I32(l) => ctx.builder.free_local(l),
        Val::I64(l) => ctx.builder.free_local_i64(l),
    }
    ctx.builder.free_local_i64(dst);
    ctx.builder.free_local_i64(src);
    let inc = gen_string_increment(ctx, size_bytes);
    gen_string_advance(ctx, crate::regs::ESI, &inc);
    gen_string_advance(ctx, crate::regs::EDI, &inc);
    ctx.builder.free_local(inc);
}

pub fn gen_string_stos(ctx: &mut JitContext, bits: u32, fallback: &'static str) {
    if !gen_string_segment_is_zero_base(ctx) {
        return gen_generic(ctx, fallback, &[]);
    }
    let size_bytes = (bits / 8) as i32;
    gen_get_reg(ctx, bits, crate::regs::EAX);
    let value = set_new_val(ctx, bits);
    gen_get_reg(ctx, 64, crate::regs::EDI);
    let dst = ctx.builder.set_new_local_i64();
    gen_safe_write(ctx, bits, &dst, &value);
    value.free(ctx);
    ctx.builder.free_local_i64(dst);
    let inc = gen_string_increment(ctx, size_bytes);
    gen_string_advance(ctx, crate::regs::EDI, &inc);
    ctx.builder.free_local(inc);
}

pub fn gen_string_lods(ctx: &mut JitContext, bits: u32, fallback: &'static str) {
    if !gen_string_segment_is_zero_base(ctx) {
        return gen_generic(ctx, fallback, &[]);
    }
    let size_bytes = (bits / 8) as i32;
    gen_get_reg(ctx, 64, crate::regs::ESI);
    let src = ctx.builder.set_new_local_i64();
    gen_safe_read(ctx, bits, &src);
    ctx.builder.free_local_i64(src);
    gen_set_reg_from_stack(ctx, bits, crate::regs::EAX);
    let inc = gen_string_increment(ctx, size_bytes);
    gen_string_advance(ctx, crate::regs::ESI, &inc);
    ctx.builder.free_local(inc);
}

/// Start and end of the current instruction within its page, packed for the wrappers
fn instruction_ips(ctx: &JitContext) -> i32 {
    (ctx.start_of_current_instruction & 0xFFF | (ctx.cpu.eip & 0xFFF) << 12) as i32
}

/// Call a generated wrapper (jit64_*) for an instruction that is run by the interpreter's
/// handler, and leave the compiled code if it raised an exception.
/// mem: the memory operand, passed as the first argument
// Optional profiling of the instructions that are run through the interpreter's handlers
// (set_jit_config(6, 1)): counts per wrapper name, printed by jit64_print_profile
pub static mut PROFILE_GENERIC: bool = false;
static mut PROFILE_NAMES: Vec<String> = Vec::new();
static mut PROFILE_COUNTS: [u32; 4096] = [0; 4096];

fn gen_profile_count(ctx: &mut JitContext, name: &str) {
    #[allow(static_mut_refs)]
    unsafe {
        if !PROFILE_GENERIC {
            return;
        }
        let id = match PROFILE_NAMES.iter().position(|n| n == name) {
            Some(id) => id,
            None => {
                PROFILE_NAMES.push(name.to_string());
                PROFILE_NAMES.len() - 1
            },
        };
        if id >= PROFILE_COUNTS.len() {
            return;
        }
        let addr = &raw mut PROFILE_COUNTS[id] as u32;
        ctx.builder.const_i32(addr as i32);
        ctx.builder.load_fixed_i32(addr);
        ctx.builder.const_i32(1);
        ctx.builder.add_i32();
        ctx.builder.store_aligned_i32(0);
    }
}

#[no_mangle]
pub unsafe fn jit64_print_profile() {
    #[allow(static_mut_refs)]
    let mut v: Vec<(u32, &String)> = PROFILE_NAMES
        .iter()
        .enumerate()
        .map(|(i, n)| (PROFILE_COUNTS[i], n))
        .collect();
    v.sort_by(|a, b| b.0.cmp(&a.0));
    for (count, name) in v.iter().take(40) {
        #[cfg(target_arch = "wasm32")]
        crate::dbg::console_log_to_js_console(format!("{:>12} {}", count, name));
        #[cfg(not(target_arch = "wasm32"))]
        let _ = (count, name);
    }
}

fn gen_call_wrapper(ctx: &mut JitContext, name: &str, mem: Option<&Modrm64>, args: &[A]) {
    gen_profile_count(ctx, name);
    gen_commit_deferred_flags(ctx);
    // the interpreter handler may change page mappings (invlpg, mov cr3, ...): the cached
    // tlb entry doesn't survive the call
    gen_tlb_cache_reset(ctx);
    ctx.flags64 = Flags64::Unknown;
    gen_spill_dirty_registers(ctx);
    ctx.builder.const_i32(instruction_ips(ctx));
    ctx.builder.const_i32(ctx.cpu.prefixes as i32);

    let mut first_is_i64 = false;
    if let Some(m) = mem {
        gen_modrm_address(ctx, m);
        first_is_i64 = true;
    }
    let mut i32_args = 0;
    for (i, arg) in args.iter().enumerate() {
        match *arg {
            A::I32(x) => {
                ctx.builder.const_i32(x);
                i32_args += 1;
            },
            A::I64(x) => {
                dbg_assert!(
                    i == 0 && mem.is_none(),
                    "only the first argument can be 64-bit"
                );
                ctx.builder.const_i64(x as i64);
                first_is_i64 = true;
            },
        }
    }

    match (first_is_i64, i32_args) {
        (false, 0) => ctx.builder.call_fn2_ret(name),
        (false, 1) => ctx.builder.call_fn3_ret(name),
        (false, 2) => ctx.builder.call_fn4_ret(name),
        (false, 3) => ctx.builder.call_fn5_ret(name),
        (true, 0) => ctx.builder.call_fn3_i32_i32_i64_ret(name),
        (true, 1) => ctx.builder.call_fn4_i32_i32_i64_i32_ret(name),
        (true, 2) => ctx.builder.call_fn5_i32_i32_i64_i32_i32_ret(name),
        _ => {
            dbg_assert!(false, "unsupported wrapper signature");
        },
    }
    let written = gpr_writes_of_current_instruction(ctx);
    gen_reload_registers_and_exit_if(ctx, written);
}

/// The general purpose registers that the interpreter's handler for the current instruction may
/// write, unless it raises an exception. Conservative: all registers for anything not listed.
fn gpr_writes_of_current_instruction(ctx: &JitContext) -> u16 {
    const RAX: u16 = 1 << 0;
    const RCX: u16 = 1 << 1;
    const RSI: u16 = 1 << 6;
    const RDI: u16 = 1 << 7;
    let mut cpu = ctx.cpu.clone();
    cpu.eip = ctx.start_of_current_instruction;
    let mut opcode;
    loop {
        opcode = cpu.read_imm8() as u32;
        match opcode {
            0x26 | 0x2E | 0x36 | 0x3E | 0x64 | 0x65 | 0x66 | 0x67 | 0xF0 | 0xF2 | 0xF3 => {},
            0x40..=0x4F => {},
            _ => break,
        }
    }
    if opcode == 0x0F {
        opcode = 0x0F00 | cpu.read_imm8() as u32;
    }
    match opcode {
        // x87; only fnstsw ax (DF E0) writes a general purpose register
        0xD8..=0xDE => 0,
        0xDF => RAX,
        // string instructions (also with rep)
        0xA4..=0xA7 => RCX | RSI | RDI,
        0xAA | 0xAB | 0xAE | 0xAF => RCX | RDI,
        0xAC | 0xAD => RAX | RCX | RSI,
        // sse/mmx with an xmm/mm register or memory as the destination. Not included because
        // they can write general purpose registers: 0F 2C/2D (cvt to integer), 50 (movmsk),
        // 78/79, 7E (movd r/m, xmm), AE (rdfsbase etc.), C5 (pextrw), D7 (pmovmskb)
        0x0F10..=0x0F17
        | 0x0F28..=0x0F2B
        | 0x0F2E
        | 0x0F2F
        | 0x0F51..=0x0F77
        | 0x0F7C
        | 0x0F7D
        | 0x0F7F
        | 0x0FC2
        | 0x0FC4
        | 0x0FC6
        | 0x0FD0..=0x0FD6
        | 0x0FD8..=0x0FFF => 0,
        _ => 0xFFFF,
    }
}

/// Write the registers that may have been changed since the last sync back to memory
fn gen_spill_dirty_registers(ctx: &mut JitContext) {
    for i in 0..16 {
        if ctx.dirty_registers64 & 1 << i != 0 {
            ctx.builder
                .const_i32(global_pointers::get_reg64_offset(i) as i32);
            ctx.builder.get_local_i64(&reg_local(ctx, i));
            ctx.builder.store_aligned_i64(0);
        }
    }
    ctx.dirty_registers64 = 0;
}

/// After a call into the interpreter (with all registers synced to memory): leave the module if
/// the call returned non-zero, reloading all registers first since an exception may have changed
/// any of them (the exit path spills the locals). Otherwise reload the registers in `written`.
fn gen_reload_registers_and_exit_if(ctx: &mut JitContext, written: u16) {
    dbg_assert!(ctx.dirty_registers64 == 0);
    // the wrapper only returns non-zero when it raised an exception or the cpu halted:
    // both leave the compiled code
    ctx.builder.if_void_unlikely();
    codegen::gen_move_registers_from_memory_to_locals(ctx);
    ctx.builder.br(ctx.exit_label);
    ctx.builder.block_end();
    for i in 0..16 {
        if written & 1 << i != 0 {
            ctx.builder
                .load_fixed_i64(global_pointers::get_reg64_offset(i));
            ctx.builder.set_local_i64(&reg_local(ctx, i));
        }
    }
}

pub fn gen_generic(ctx: &mut JitContext, name: &str, args: &[A]) {
    gen_call_wrapper(ctx, name, None, args)
}
pub fn gen_generic_mem(ctx: &mut JitContext, name: &str, m: &Modrm64, args: &[A]) {
    gen_call_wrapper(ctx, name, Some(m), args)
}

/// Run the current instruction in the interpreter (for encodings that the 64-bit jit doesn't
/// decode itself), then leave the compiled code. Its bytes must have been consumed by the caller.
pub fn gen_interpret_one(ctx: &mut JitContext, instr_flags: &mut u32) {
    gen_profile_count(ctx, "interpret_one");
    gen_commit_deferred_flags(ctx);
    gen_tlb_cache_reset(ctx);
    ctx.flags64 = Flags64::Unknown;
    gen_spill_dirty_registers(ctx);
    ctx.builder
        .const_i32((ctx.start_of_current_instruction & 0xFFF) as i32);
    ctx.builder.call_fn1_ret("jit64_interpret_one");
    ctx.builder.drop_();
    codegen::gen_move_registers_from_memory_to_locals(ctx);
    ctx.dirty_registers64 = 0;
    ctx.builder.br(ctx.exit_label);
    *instr_flags |= JIT_INSTR_BLOCK_BOUNDARY_FLAG;
}

// ---------------------------------------------------------------------------------------------
// Instruction decoding entry points and prefixes

pub fn jit_instruction(ctx: &mut JitContext, instr_flags: &mut u32) {
    ctx.cpu.prefixes = 0;
    ctx.start_of_current_instruction = ctx.cpu.eip;
    jit_opcode(ctx, instr_flags);
}

/// Read and compile an opcode (after any legacy prefixes): REX prefixes and operand size tier
/// selection as in analysis::analyze_opcode64
fn jit_opcode(ctx: &mut JitContext, instr_flags: &mut u32) {
    let opcode = ctx.cpu.read_imm8() as u32;
    if opcode & 0xF0 == 0x40 {
        ctx.cpu.prefixes =
            ctx.cpu.prefixes & !PREFIX_MASK_REX | PREFIX_REX_PRESENT | ((opcode as u16 & 0xF) << 8);
        return jit_opcode(ctx, instr_flags);
    }
    let tier = if ctx.cpu.prefixes & PREFIX_REX_W != 0
        || ctx.cpu.osize_32() && gen::interpreter::is_default_64_operand_size(opcode)
    {
        0x200
    }
    else {
        (ctx.cpu.osize_32() as u32) << 8
    };
    gen::jit64::jit(opcode | tier, ctx, instr_flags)
}

fn jit_handle_prefix(ctx: &mut JitContext, instr_flags: &mut u32) {
    if ctx.cpu.prefixes & PREFIX_67 != 0 {
        // 32-bit addressing in 64-bit mode: rare, leave it to the interpreter. The analyzer has
        // made this instruction a block boundary (see analysis::fixup_analysis64)
        let mut analysis_cpu = ctx.cpu.clone();
        let mut analysis = crate::analysis::Analysis {
            no_next_instruction: false,
            absolute_jump: false,
            ty: crate::analysis::AnalysisType::Normal,
        };
        crate::analysis::analyze_opcode64(&mut analysis_cpu, &mut analysis);
        ctx.cpu.eip = analysis_cpu.eip;
        return gen_interpret_one(ctx, instr_flags);
    }
    jit_opcode(ctx, instr_flags)
}

fn jit_handle_segment_prefix(segment: u32, ctx: &mut JitContext, instr_flags: &mut u32) {
    dbg_assert!(segment <= 5);
    ctx.cpu.prefixes =
        ctx.cpu.prefixes & !(PREFIX_MASK_SEGMENT | PREFIX_MASK_REX) | (segment as u16 + 1);
    jit_handle_prefix(ctx, instr_flags)
}

pub fn instr_26_jit64(ctx: &mut JitContext, f: &mut u32) { jit_handle_segment_prefix(ES, ctx, f) }
pub fn instr_2E_jit64(ctx: &mut JitContext, f: &mut u32) { jit_handle_segment_prefix(CS, ctx, f) }
pub fn instr_36_jit64(ctx: &mut JitContext, f: &mut u32) { jit_handle_segment_prefix(SS, ctx, f) }
pub fn instr_3E_jit64(ctx: &mut JitContext, f: &mut u32) { jit_handle_segment_prefix(DS, ctx, f) }
pub fn instr_64_jit64(ctx: &mut JitContext, f: &mut u32) { jit_handle_segment_prefix(FS, ctx, f) }
pub fn instr_65_jit64(ctx: &mut JitContext, f: &mut u32) { jit_handle_segment_prefix(GS, ctx, f) }

fn jit_handle_legacy_prefix(prefix: u16, ctx: &mut JitContext, f: &mut u32) {
    // a legacy prefix after a REX prefix annuls the REX prefix
    ctx.cpu.prefixes = ctx.cpu.prefixes & !PREFIX_MASK_REX | prefix;
    jit_handle_prefix(ctx, f)
}
pub fn instr_66_jit64(ctx: &mut JitContext, f: &mut u32) {
    jit_handle_legacy_prefix(PREFIX_66, ctx, f)
}
pub fn instr_67_jit64(ctx: &mut JitContext, f: &mut u32) {
    jit_handle_legacy_prefix(PREFIX_67, ctx, f)
}
pub fn instr_F0_jit64(ctx: &mut JitContext, f: &mut u32) { jit_handle_legacy_prefix(0, ctx, f) }
pub fn instr_F2_jit64(ctx: &mut JitContext, f: &mut u32) {
    jit_handle_legacy_prefix(PREFIX_F2, ctx, f)
}
pub fn instr_F3_jit64(ctx: &mut JitContext, f: &mut u32) {
    jit_handle_legacy_prefix(PREFIX_F3, ctx, f)
}

pub fn instr16_0F_jit64(ctx: &mut JitContext, f: &mut u32) {
    let opcode = ctx.cpu.read_imm8() as u32;
    gen::jit64_0f::jit(opcode, ctx, f)
}
pub fn instr32_0F_jit64(ctx: &mut JitContext, f: &mut u32) {
    let opcode = ctx.cpu.read_imm8() as u32;
    if gen::interpreter0f::is_default_64_operand_size(opcode) {
        gen::jit64_0f::jit(opcode | 0x200, ctx, f)
    }
    else {
        gen::jit64_0f::jit(opcode | 0x100, ctx, f)
    }
}
pub fn instr64_0F_jit64(ctx: &mut JitContext, f: &mut u32) {
    let opcode = ctx.cpu.read_imm8() as u32;
    gen::jit64_0f::jit(opcode | 0x200, ctx, f)
}
pub fn instr16_0F38_jit64(ctx: &mut JitContext, f: &mut u32) {
    gen::jit64_0f38::jit(ctx.cpu.read_imm8() as u32, ctx, f)
}
pub fn instr32_0F38_jit64(ctx: &mut JitContext, f: &mut u32) {
    gen::jit64_0f38::jit(ctx.cpu.read_imm8() as u32 | 0x100, ctx, f)
}
pub fn instr64_0F38_jit64(ctx: &mut JitContext, f: &mut u32) {
    gen::jit64_0f38::jit(ctx.cpu.read_imm8() as u32 | 0x200, ctx, f)
}
pub fn instr16_0F3A_jit64(ctx: &mut JitContext, f: &mut u32) {
    gen::jit64_0f3a::jit(ctx.cpu.read_imm8() as u32, ctx, f)
}
pub fn instr32_0F3A_jit64(ctx: &mut JitContext, f: &mut u32) {
    gen::jit64_0f3a::jit(ctx.cpu.read_imm8() as u32 | 0x100, ctx, f)
}
pub fn instr64_0F3A_jit64(ctx: &mut JitContext, f: &mut u32) {
    gen::jit64_0f3a::jit(ctx.cpu.read_imm8() as u32 | 0x200, ctx, f)
}

// ---------------------------------------------------------------------------------------------
// Instructions with custom code generation

/// lea: the effective address without segment base, truncated to the operand size
pub fn instr_8D_jit64(ctx: &mut JitContext, modrm_byte: u8, size: u32) {
    let r = (modrm_byte >> 3 & 7) as u32 | ctx.cpu.rex_r();
    if modrm_byte >= 0xC0 {
        gen_generic(ctx, "jit64_trigger_ud", &[]);
        return;
    }
    let m = decode_modrm(ctx.cpu, modrm_byte);
    gen_modrm_offset(ctx, &m);
    if size != 64 {
        ctx.builder.wrap_i64_to_i32();
        if size == 16 {
            ctx.builder.const_i32(0xFFFF);
            ctx.builder.and_i32();
        }
    }
    gen_set_reg_from_stack(ctx, size, r);
}

/// call rel32: push the return address. The jump itself is generated by the block glue (the
/// analyzer reports a jump)
pub fn instr64_E8_jit64(ctx: &mut JitContext, _imm: i32) {
    // return address: the end of this instruction
    codegen::gen_get_eip64(ctx.builder);
    ctx.builder.const_i64(!0xFFF);
    ctx.builder.and_i64();
    ctx.builder.const_i64((ctx.cpu.eip & 0xFFF) as i64);
    ctx.builder.or_i64();
    let value = set_new_val(ctx, 64);
    gen_push64_value(ctx, &value);
    value.free(ctx);
}

/// sti: handle_irqs is called by the block glue one instruction later (interrupt shadow)
pub fn instr_FB_jit64(ctx: &mut JitContext) {
    // may raise #gp, which changes registers and pushes an exception frame (reads the flags)
    gen_commit_deferred_flags(ctx);
    gen_spill_dirty_registers(ctx);
    ctx.builder.const_i32(instruction_ips(ctx));
    ctx.builder.call_fn1_ret("jit64_sti");
    // registers only change if it raises an exception
    gen_reload_registers_and_exit_if(ctx, 0);
}

/// pop r/m: the address is computed after rsp has been incremented, leave it to the interpreter
pub fn instr_8F_jit64(ctx: &mut JitContext, modrm_byte: u8, instr_flags: &mut u32) {
    skip_modrm(ctx.cpu, modrm_byte);
    gen_interpret_one(ctx, instr_flags);
}

/// What last set the lazy flags (within the current basic block), which allows conditions to
/// be computed inline
#[derive(Copy, Clone, PartialEq)]
pub enum Flags64 {
    Unknown,
    /// sub/cmp with the operand size: last_op1 and last_result hold the operands
    Sub(u32),
    Add(u32),
    /// and/or/xor/test: cf=of=0
    Logic(u32),
    /// inc/dec with the operand size: like add/sub of 1, but cf comes from the previous flags
    /// (the carry is kept in the cf flag local while the state is deferred)
    IncDec {
        bits: u32,
        is_dec: bool,
    },
}

/// Whether the 64-bit jit keeps lazy-flag values in wasm locals within a basic block instead
/// of storing them to the cpu state after every instruction (see gen_defer_flags). The locals
/// are committed to memory at observation points: block ends, calls into the interpreter and
/// the memory slow paths. set_jit_config(8, 0) restores the eager stores.
pub static mut JIT64_DEFER_FLAGS: bool = true;

/// Compile-time part of a deferred lazy-flags state (the values live in FlagLocals64)
#[derive(Copy, Clone)]
pub struct DeferredFlags64 {
    pub bits: u32,
    /// the value last_op_size/flags_changed are committed with (as in the eager code)
    pub flags_changed: i32,
    /// whether last_op1 holds a meaningful value (not needed for logical operations)
    pub has_op1: bool,
    /// flags-word bits that still have to be cleared when committing (logic operations clear
    /// cf/of/af in the flags word eagerly otherwise)
    pub pending_clear: i32,
    /// inc/dec: the cf local holds the preserved carry flag
    pub cf_valid: bool,
}

/// The function-wide wasm locals that hold the deferred lazy-flag values
pub struct FlagLocals64 {
    pub op1: WasmLocalI64,
    pub result: WasmLocalI64,
    pub cf: WasmLocal,
}
impl FlagLocals64 {
    fn clone(&self) -> FlagLocals64 {
        FlagLocals64 {
            op1: self.op1.unsafe_clone(),
            result: self.result.unsafe_clone(),
            cf: self.cf.unsafe_clone(),
        }
    }
}

fn flag_locals(ctx: &JitContext) -> FlagLocals64 { ctx.jit64_flag_locals.as_ref().unwrap().clone() }

fn defer_flags_enabled(ctx: &JitContext) -> bool {
    ctx.jit64_flag_locals.is_some() && unsafe { JIT64_DEFER_FLAGS }
}

/// Whether the 64-bit jit caches the last used data tlb entry in wasm locals, checked inline
/// before the full (hashed) lookup. set_jit_config(9, 0) disables it.
pub static mut JIT64_TLB_CACHE: bool = true;

/// The function-wide wasm locals of the data tlb cache: the virtual page base and the tlb
/// entry of the last fast-path-validated page. The entry is zero (invalid) at module entry
/// and after calls into the interpreter (the only points where mappings can change during a
/// module's execution: page walks only fill tlb entries, tlb_set_has_code runs at compile
/// time, and the cpl of a module is fixed).
pub struct TlbCacheLocals {
    pub vbase: WasmLocalI64,
    #[cfg(feature = "mem64")]
    pub entry: WasmLocalI64,
    #[cfg(not(feature = "mem64"))]
    pub entry: WasmLocal,
}
impl TlbCacheLocals {
    fn clone(&self) -> TlbCacheLocals {
        TlbCacheLocals {
            vbase: self.vbase.unsafe_clone(),
            entry: self.entry.unsafe_clone(),
        }
    }
}

fn tlb_cache(ctx: &JitContext) -> Option<TlbCacheLocals> {
    if unsafe { JIT64_TLB_CACHE } {
        ctx.jit64_tlb_cache.as_ref().map(|c| c.clone())
    }
    else {
        None
    }
}

/// Invalidate the data tlb cache (at calls into the interpreter, which may change mappings)
fn gen_tlb_cache_reset(ctx: &mut JitContext) {
    if let Some(c) = tlb_cache(ctx) {
        #[cfg(feature = "mem64")]
        {
            ctx.builder.const_i64(0);
            ctx.builder.set_local_i64(&c.entry);
        }
        #[cfg(not(feature = "mem64"))]
        {
            ctx.builder.const_i32(0);
            ctx.builder.set_local(&c.entry);
        }
    }
}

/// Push the tlb-cache hit condition for a data access: the address is in the cached page
/// and the cached entry allows the access. (address ^ vbase) < 0x1001 - bytes covers both
/// the same-page check (the xor stays below 0x1000 exactly within the cached page) and the
/// crossing check (it equals the offset). The hit condition is checked entirely at runtime,
/// so the cache stays valid across block boundaries and loop back edges within a module's
/// execution.
fn gen_tlb_cache_hit_condition(
    ctx: &mut JitContext,
    bits: u32,
    address: &WasmLocalI64,
    for_writing: bool,
) {
    dbg_assert!(bits <= 64);
    let c = tlb_cache(ctx).unwrap();
    ctx.builder.get_local_i64(address);
    ctx.builder.get_local_i64(&c.vbase);
    ctx.builder.xor_i64();
    ctx.builder.const_i64(0x1001 - (bits / 8) as i64);
    ctx.builder.ltu_i64();
    // the cached entry is fast-path validated; re-check the access-type specific bits
    // (and validity: the entry is zeroed on invalidation)
    #[cfg(feature = "mem64")]
    {
        ctx.builder.get_local_i64(&c.entry);
        if for_writing {
            ctx.builder
                .const_i64((TLB_VALID | TLB_READONLY | TLB_HAS_CODE) as i64);
            ctx.builder.and_i64();
            ctx.builder.const_i64(TLB_VALID as i64);
            ctx.builder.eq_i64();
        }
        else {
            ctx.builder.const_i64(TLB_VALID as i64);
            ctx.builder.and_i64();
            ctx.builder.wrap_i64_to_i32();
        }
    }
    #[cfg(not(feature = "mem64"))]
    {
        ctx.builder.get_local(&c.entry);
        if for_writing {
            ctx.builder
                .const_i32(TLB_VALID | TLB_READONLY | TLB_HAS_CODE);
            ctx.builder.and_i32();
            ctx.builder.const_i32(TLB_VALID);
            ctx.builder.eq_i32();
        }
        else {
            ctx.builder.const_i32(TLB_VALID);
            ctx.builder.and_i32();
        }
    }
    ctx.builder.and_i32();
}

#[cfg(feature = "mem64")]
fn cached_entry_val(c: &TlbCacheLocals) -> Val { Val::I64(c.entry.unsafe_clone()) }
#[cfg(not(feature = "mem64"))]
fn cached_entry_val(c: &TlbCacheLocals) -> Val { Val::I32(c.entry.unsafe_clone()) }

/// A new local for a host pointer (i64 under mem64)
fn new_pointer_local(ctx: &mut JitContext) -> Val {
    if cfg!(feature = "mem64") {
        Val::I64(ctx.builder.new_local_i64())
    }
    else {
        Val::I32(ctx.builder.new_local())
    }
}

fn set_pointer_local(ctx: &mut JitContext, pointer: &Val) {
    match pointer {
        Val::I32(l) => ctx.builder.set_local(l),
        Val::I64(l) => ctx.builder.set_local_i64(l),
    }
}

/// Populate the data tlb cache after a fast-path validated lookup (entry in `entry`)
fn gen_tlb_cache_populate(ctx: &mut JitContext, address: &WasmLocalI64, entry: &Val) {
    let c = tlb_cache(ctx).unwrap();
    ctx.builder.get_local_i64(address);
    ctx.builder.const_i64(!0xFFF);
    ctx.builder.and_i64();
    ctx.builder.set_local_i64(&c.vbase);
    #[cfg(feature = "mem64")]
    {
        let l = match entry {
            Val::I64(l) => l,
            Val::I32(_) => unreachable!(),
        };
        ctx.builder.get_local_i64(l);
        ctx.builder.set_local_i64(&c.entry);
    }
    #[cfg(not(feature = "mem64"))]
    {
        let l = match entry {
            Val::I32(l) => l,
            Val::I64(_) => unreachable!(),
        };
        ctx.builder.get_local(l);
        ctx.builder.set_local(&c.entry);
    }
}

/// Defer the lazy-flag state of an instruction: keep last_op1/last_result in the flag locals
/// and record how to commit them. flags_changed covers the flags-word bits the new state
/// computes lazily, making that part of a pending clear dead. apply_clear adds flags-word
/// bits that must be cleared at commit (logic operations).
fn gen_defer_flags(
    ctx: &mut JitContext,
    bits: u32,
    op1: Option<&Val>,
    result: &Val,
    flags_changed: i32,
    apply_clear: i32,
) {
    let fl = flag_locals(ctx);
    let pending_clear = ctx
        .jit64_deferred
        .map_or(0, |d| d.pending_clear & !flags_changed)
        | apply_clear;
    if let Some(op1) = op1 {
        op1.get(ctx);
        if bits != 64 {
            ctx.builder.extend_unsigned_i32_to_i64();
        }
        ctx.builder.set_local_i64(&fl.op1);
    }
    result.get(ctx);
    if bits != 64 {
        ctx.builder.extend_unsigned_i32_to_i64();
    }
    ctx.builder.set_local_i64(&fl.result);
    ctx.jit64_deferred = Some(DeferredFlags64 {
        bits,
        flags_changed,
        has_op1: op1.is_some(),
        pending_clear,
        cf_valid: false,
    });
}

/// Emit the stores that write a deferred lazy-flag state back to the cpu state in memory.
/// Stack-neutral and idempotent (only reads the flag locals).
fn gen_emit_deferred_flags(ctx: &mut JitContext, d: &DeferredFlags64) {
    let fl = flag_locals(ctx);
    // last_op_size and flags_changed are adjacent: write both with one store
    ctx.builder.const_i32(global_pointers::last_op_size as i32);
    ctx.builder
        .const_i64((d.flags_changed as u32 as i64) << 32 | opsize(d.bits) as u32 as i64);
    ctx.builder.store_aligned_i64(0);
    if d.has_op1 {
        if d.bits == 64 {
            ctx.builder.const_i32(global_pointers::last_op1_64 as i32);
            ctx.builder.get_local_i64(&fl.op1);
            ctx.builder.store_aligned_i64(0);
        }
        else {
            ctx.builder.const_i32(global_pointers::last_op1 as i32);
            ctx.builder.get_local_i64(&fl.op1);
            ctx.builder.wrap_i64_to_i32();
            ctx.builder.store_aligned_i32(0);
        }
    }
    if d.bits == 64 {
        ctx.builder
            .const_i32(global_pointers::last_result_64 as i32);
        ctx.builder.get_local_i64(&fl.result);
        ctx.builder.store_aligned_i64(0);
    }
    else {
        ctx.builder.const_i32(global_pointers::last_result as i32);
        ctx.builder.get_local_i64(&fl.result);
        ctx.builder.wrap_i64_to_i32();
        ctx.builder.store_aligned_i32(0);
    }
    let clear = d.pending_clear | if d.cf_valid { FLAG_CARRY } else { 0 };
    if clear != 0 || d.cf_valid {
        ctx.builder.const_i32(global_pointers::flags as i32);
        ctx.builder.load_fixed_i32(global_pointers::flags as u32);
        ctx.builder.const_i32(!clear);
        ctx.builder.and_i32();
        if d.cf_valid {
            ctx.builder.get_local(&fl.cf);
            ctx.builder.or_i32();
        }
        ctx.builder.store_aligned_i32(0);
    }
}

/// Commit the deferred lazy-flag state on a linear path: afterwards the values are read from
/// memory again. A compile-time no-op when there is no deferred state (always the case in the
/// 32-bit jit).
pub fn gen_commit_deferred_flags(ctx: &mut JitContext) {
    match ctx.jit64_deferred.take() {
        Some(d) => gen_emit_deferred_flags(ctx, &d),
        None => {},
    }
}

/// Commit the deferred lazy-flag state inside a cold branch (a memory slow path): the branch
/// may not be taken at runtime, so the compile-time state stays deferred — the stores are
/// idempotent and the flag locals remain authoritative on both paths.
pub fn gen_commit_deferred_flags_cold(ctx: &mut JitContext) {
    match ctx.jit64_deferred {
        Some(d) => gen_emit_deferred_flags(ctx, &d),
        None => {},
    }
}

/// Push the lazy flag operand slots for an operation of the given size
fn gen_get_last_op1(ctx: &mut JitContext, bits: u32) {
    if let Some(d) = ctx.jit64_deferred {
        // the value is zero-extended in the flag local (logic operations don't reach here:
        // their conditions never read last_op1)
        dbg_assert!(d.has_op1 && d.bits == bits);
        let fl = flag_locals(ctx);
        ctx.builder.get_local_i64(&fl.op1);
        if bits != 64 {
            ctx.builder.wrap_i64_to_i32();
        }
        return;
    }
    if bits == 64 {
        ctx.builder
            .load_fixed_i64(global_pointers::last_op1_64 as u32)
    }
    else {
        ctx.builder.load_fixed_i32(global_pointers::last_op1 as u32)
    }
}
fn gen_get_last_result(ctx: &mut JitContext, bits: u32) {
    if let Some(d) = ctx.jit64_deferred {
        dbg_assert!(d.bits == bits);
        let fl = flag_locals(ctx);
        ctx.builder.get_local_i64(&fl.result);
        if bits != 64 {
            ctx.builder.wrap_i64_to_i32();
        }
        return;
    }
    if bits == 64 {
        ctx.builder
            .load_fixed_i64(global_pointers::last_result_64 as u32)
    }
    else {
        ctx.builder
            .load_fixed_i32(global_pointers::last_result as u32)
    }
}

/// Test a (non-negated) condition from a fully committed flags word (flags_changed == 0),
/// pushing 0/1. Only the conditions that don't need the parity table are handled.
fn gen_condition_from_flags_word(ctx: &mut JitContext, cc: u8) {
    dbg_assert!(cc & 1 == 0);
    ctx.builder.load_fixed_i32(global_pointers::flags as u32);
    match cc {
        0 => {
            // o: bit 11
            ctx.builder.const_i32(11);
            ctx.builder.shr_u_i32();
            ctx.builder.const_i32(1);
            ctx.builder.and_i32();
        },
        2 => {
            // b: bit 0
            ctx.builder.const_i32(1);
            ctx.builder.and_i32();
        },
        4 => {
            // z: bit 6
            ctx.builder.const_i32(6);
            ctx.builder.shr_u_i32();
            ctx.builder.const_i32(1);
            ctx.builder.and_i32();
        },
        6 => {
            // be: carry or zero
            ctx.builder.const_i32(cpu::FLAG_CARRY | cpu::FLAG_ZERO);
            ctx.builder.and_i32();
            ctx.builder.eqz_i32();
            ctx.builder.eqz_i32();
        },
        8 => {
            // s: bit 7
            ctx.builder.const_i32(7);
            ctx.builder.shr_u_i32();
            ctx.builder.const_i32(1);
            ctx.builder.and_i32();
        },
        12 => {
            // l: sf != of
            ctx.builder.const_i32(7);
            ctx.builder.shr_u_i32();
            ctx.builder.load_fixed_i32(global_pointers::flags as u32);
            ctx.builder.const_i32(11);
            ctx.builder.shr_u_i32();
            ctx.builder.xor_i32();
            ctx.builder.const_i32(1);
            ctx.builder.and_i32();
        },
        14 => {
            // le: zero or sf != of
            ctx.builder.const_i32(6);
            ctx.builder.shr_u_i32();
            ctx.builder.const_i32(1);
            ctx.builder.and_i32();
            ctx.builder.load_fixed_i32(global_pointers::flags as u32);
            ctx.builder.const_i32(7);
            ctx.builder.shr_u_i32();
            ctx.builder.load_fixed_i32(global_pointers::flags as u32);
            ctx.builder.const_i32(11);
            ctx.builder.shr_u_i32();
            ctx.builder.xor_i32();
            ctx.builder.const_i32(1);
            ctx.builder.and_i32();
            ctx.builder.or_i32();
        },
        _ => unreachable!(),
    }
}

pub fn gen_condition_fn(ctx: &mut JitContext, condition: u8) {
    dbg_assert!(condition & 0xF0 == 0x70 || condition & 0xF0 == 0x80);
    let cc = condition & 0xF;
    if gen_condition_inline(ctx, cc & !1) {
        if cc & 1 != 0 {
            ctx.builder.eqz_i32();
        }
        // jcc ends the block (the setcc/cmovcc callers pay an occasional extra commit):
        // the following blocks read the lazy flags from memory. The commit is stack-neutral,
        // the computed condition stays on the stack.
        gen_commit_deferred_flags(ctx);
        return;
    }
    gen_commit_deferred_flags(ctx);
    if cc & !1 != 10 {
        // runtime fast path: with no lazy flags pending (fully committed state, e.g.
        // after shifts, mul, popf or at a block boundary following them), the condition
        // is a bit test of the flags word instead of a call to jit64_test_cc
        ctx.builder
            .load_fixed_i32(global_pointers::flags_changed as u32);
        ctx.builder.eqz_i32();
        ctx.builder.if_i32();
        {
            gen_condition_from_flags_word(ctx, cc & !1);
            if cc & 1 != 0 {
                ctx.builder.eqz_i32();
            }
        }
        ctx.builder.else_();
        {
            // jit64_test_cc handles the negated conditions itself
            ctx.builder.const_i32(cc as i32);
            ctx.builder.call_fn1_ret("jit64_test_cc");
        }
        ctx.builder.block_end();
    }
    else {
        ctx.builder.const_i32(cc as i32);
        ctx.builder.call_fn1_ret("jit64_test_cc");
    }
}

/// Generate a (non-negated) condition after inc/dec from the deferred lazy flags state,
/// false if not possible. z/s/o/l/le come from the result; cf is the preserved carry.
fn gen_condition_incdec(ctx: &mut JitContext, cc: u8, bits: u32, is_dec: bool) -> bool {
    if bits < 32 {
        return false;
    }
    let w = bits == 64;
    macro_rules! op {
        ($i32:ident, $i64:ident) => {
            if w {
                ctx.builder.$i64()
            }
            else {
                ctx.builder.$i32()
            }
        };
    }
    // of: inc overflows at result == msb, dec at result == msb - 1
    macro_rules! of {
        () => {{
            gen_get_last_result(ctx, bits);
            let msb = 1i64 << (bits - 1);
            let v = if is_dec { msb - 1 } else { msb };
            if w {
                ctx.builder.const_i64(v);
            }
            else {
                ctx.builder.const_i32(v as i32);
            }
            op!(eq_i32, eq_i64);
        }};
    }
    macro_rules! sf {
        () => {{
            gen_get_last_result(ctx, bits);
            if w {
                ctx.builder.const_i64(0);
            }
            else {
                ctx.builder.const_i32(0);
            }
            op!(lt_i32, lt_i64);
        }};
    }
    macro_rules! zf {
        () => {{
            gen_get_last_result(ctx, bits);
            op!(eqz_i32, eqz_i64);
        }};
    }
    match cc {
        4 => zf!(),
        8 => sf!(),
        0 => of!(),
        12 => {
            // l: sf != of
            sf!();
            of!();
            ctx.builder.xor_i32();
        },
        14 => {
            // le: zf || sf != of
            sf!();
            of!();
            ctx.builder.xor_i32();
            zf!();
            ctx.builder.or_i32();
        },
        2 | 6 => {
            // b: cf; be: cf || zf — cf is the carry preserved by inc/dec
            match ctx.jit64_deferred {
                Some(d) if d.cf_valid => {
                    let fl = flag_locals(ctx);
                    ctx.builder.get_local(&fl.cf);
                },
                Some(_) => {
                    // unreachable: the carry is saved unless the following instructions
                    // overwrite all flags before anyone reads them (gen_defer_save_cf)
                    gen_commit_deferred_flags(ctx);
                    gen_getcf_generic(ctx);
                },
                None => {
                    // committed: cf is in the flags word
                    ctx.builder.load_fixed_i32(global_pointers::flags as u32);
                    ctx.builder.const_i32(1);
                    ctx.builder.and_i32();
                },
            }
            if cc == 6 {
                zf!();
                ctx.builder.or_i32();
            }
        },
        _ => return false,
    }
    true
}

/// Generate a (non-negated) condition from the known lazy flags state, false if not possible.
/// cc: o=0, b=2, z=4, be=6, s=8, p=10, l=12, le=14
fn gen_condition_inline(ctx: &mut JitContext, cc: u8) -> bool {
    match ctx.flags64 {
        Flags64::IncDec { bits, is_dec } => {
            return gen_condition_incdec(ctx, cc, bits, is_dec);
        },
        _ => {},
    }
    let (bits, is_logic, is_add) = match ctx.flags64 {
        Flags64::Sub(b) if b >= 32 => (b, false, false),
        Flags64::Add(b) if b >= 32 => (b, false, true),
        Flags64::Logic(b) if b >= 32 => (b, true, false),
        _ => return false,
    };
    let w = bits == 64;
    macro_rules! op {
        ($i32:ident, $i64:ident) => {
            if w {
                ctx.builder.$i64()
            }
            else {
                ctx.builder.$i32()
            }
        };
    }
    macro_rules! zero {
        () => {
            if w {
                ctx.builder.const_i64(0)
            }
            else {
                ctx.builder.const_i32(0)
            }
        };
    }

    // the zero and sign flags only depend on the result
    match cc {
        4 => {
            gen_get_last_result(ctx, bits);
            op!(eqz_i32, eqz_i64);
            return true;
        },
        8 => {
            gen_get_last_result(ctx, bits);
            zero!();
            op!(lt_i32, lt_i64);
            return true;
        },
        _ => {},
    }

    if is_logic {
        match cc {
            // of=cf=0
            0 | 2 => ctx.builder.const_i32(0),
            // be: zf
            6 => {
                gen_get_last_result(ctx, bits);
                op!(eqz_i32, eqz_i64);
            },
            // l: sf != of = sf
            12 => {
                gen_get_last_result(ctx, bits);
                zero!();
                op!(lt_i32, lt_i64);
            },
            // le: zf || sf
            14 => {
                gen_get_last_result(ctx, bits);
                zero!();
                op!(le_i32, le_i64);
            },
            _ => return false,
        }
        return true;
    }

    if is_add {
        match cc {
            // b: result <u op1
            2 => {
                gen_get_last_result(ctx, bits);
                gen_get_last_op1(ctx, bits);
                op!(ltu_i32, ltu_i64);
            },
            // be: result <u op1 || result == 0
            6 => {
                gen_get_last_result(ctx, bits);
                gen_get_last_op1(ctx, bits);
                op!(ltu_i32, ltu_i64);
                gen_get_last_result(ctx, bits);
                op!(eqz_i32, eqz_i64);
                ctx.builder.or_i32();
            },
            _ => return false,
        }
        return true;
    }

    // sub/cmp: op2 = op1 - result
    let gen_op1_op2 = |ctx: &mut JitContext| {
        gen_get_last_op1(ctx, bits);
        gen_get_last_op1(ctx, bits);
        gen_get_last_result(ctx, bits);
        if w {
            ctx.builder.sub_i64()
        }
        else {
            ctx.builder.sub_i32()
        }
    };
    match cc {
        // o: ((op1 ^ op2) & (op1 ^ result)) < 0
        0 => {
            gen_get_last_op1(ctx, bits);
            gen_get_last_op1(ctx, bits);
            gen_get_last_result(ctx, bits);
            op!(sub_i32, sub_i64);
            op!(xor_i32, xor_i64);
            gen_get_last_op1(ctx, bits);
            gen_get_last_result(ctx, bits);
            op!(xor_i32, xor_i64);
            op!(and_i32, and_i64);
            zero!();
            op!(lt_i32, lt_i64);
        },
        2 => {
            gen_op1_op2(ctx);
            op!(ltu_i32, ltu_i64);
        },
        6 => {
            gen_op1_op2(ctx);
            op!(leu_i32, leu_i64);
        },
        12 => {
            gen_op1_op2(ctx);
            op!(lt_i32, lt_i64);
        },
        14 => {
            gen_op1_op2(ctx);
            op!(le_i32, le_i64);
        },
        _ => return false,
    }
    true
}

// ---------------------------------------------------------------------------------------------
// Runtime support, called from compiled code

/// Set up the cpu state for running an instruction's interpreter handler from compiled code:
/// ips packs the instruction's start (bits 0-11) and end (bits 12-23) within the current page
#[inline]
pub unsafe fn jit64_enter(ips: i32, prefixes: i32) {
    let page = *global_pointers::instruction_pointer & !0xFFF;
    *global_pointers::previous_ip = page | (ips & 0xFFF) as u64;
    *global_pointers::instruction_pointer = page | (ips >> 12 & 0xFFF) as u64;
    *global_pointers::prefixes = prefixes as u16;
    JIT64_EXIT = false;
    #[cfg(debug_assertions)]
    {
        cpu::in_jit = false;
    }
}

#[inline]
pub unsafe fn jit64_leave() -> i32 {
    *global_pointers::prefixes = 0;
    #[cfg(debug_assertions)]
    {
        cpu::in_jit = true;
    }
    JIT64_EXIT as i32
}

#[no_mangle]
pub unsafe fn jit64_push_return_address(ips: i32) -> i32 {
    jit64_enter(ips, 0);
    let _ = crate::cpu::misc_instr::push64(*global_pointers::instruction_pointer);
    jit64_leave()
}

#[no_mangle]
pub unsafe fn jit64_sti(ips: i32) -> i32 {
    jit64_enter(ips, 0);
    if !crate::cpu::instructions::instr_FB_without_fault() {
        cpu::trigger_gp(0);
    }
    jit64_leave()
}

#[no_mangle]
pub unsafe fn jit64_interpret_one(start: i32) -> i32 {
    let page = *global_pointers::instruction_pointer & !0xFFF;
    *global_pointers::instruction_pointer = page | (start & 0xFFF) as u64;
    *global_pointers::previous_ip = *global_pointers::instruction_pointer;
    *global_pointers::prefixes = 0;
    #[cfg(debug_assertions)]
    {
        cpu::in_jit = false;
    }
    cpu::run_shadow_instruction();
    jit64_leave();
    1
}

#[no_mangle]
pub unsafe fn jit64_test_cc(condition: i32) -> i32 {
    use crate::cpu::misc_instr::*;
    let r = match condition & 0xF {
        0x0 => test_o(),
        0x1 => !test_o(),
        0x2 => test_b(),
        0x3 => !test_b(),
        0x4 => test_z(),
        0x5 => !test_z(),
        0x6 => test_be(),
        0x7 => !test_be(),
        0x8 => test_s(),
        0x9 => !test_s(),
        0xA => test_p(),
        0xB => !test_p(),
        0xC => test_l(),
        0xD => !test_l(),
        0xE => test_le(),
        _ => !test_le(),
    };
    r as i32
}

/// At the (normal) exit of a module: continue directly with the entry point for the instruction
/// pointer, like cycle_internal would: by branching to reenter_label (which dispatches on the
/// initial state) if it's in this module, otherwise through a tail call of the other module that
/// passes the registers. Falls through if there is none, the cpu halted or the instruction budget
/// of the current do_many_cycles is used up (so that timers and interrupts are handled). The
/// instruction counter must have been updated.
pub fn gen_chain_to_next_module(ctx: &mut JitContext, reenter_label: Label) {
    use crate::cpu::cpu::{Code, WASM_TABLE_OFFSET};
    use std::mem::offset_of;

    // !in_hlt && (int)(jit_chain_instruction_limit - instruction_counter) > 0
    //     && !debug_mode_active()
    ctx.builder.load_fixed_u8(global_pointers::in_hlt as u32);
    ctx.builder.eqz_i32();
    ctx.builder
        .load_fixed_i32(&raw const cpu::jit_chain_instruction_limit as u32);
    ctx.builder
        .load_fixed_i32(global_pointers::instruction_counter as u32);
    ctx.builder.sub_i32();
    ctx.builder.const_i32(0);
    ctx.builder.gt_i32();
    ctx.builder.and_i32();
    // Breakpoints and single-stepping are only checked by the interpreter: cycle_internal
    // doesn't run compiled code while they are armed (see cpu::debug_mode_active), so chaining
    // must not either. They are armed by instructions run through the interpreter's handlers
    // (popf, iret, sysret, mov dr7, ...), which set JIT64_EXIT to leave the module.
    ctx.builder.load_fixed_i32(global_pointers::flags as u32);
    ctx.builder
        .const_i32(crate::cpu::cpu::FLAG_TRAP | crate::cpu::cpu::FLAG_RF);
    ctx.builder.and_i32();
    ctx.builder
        .load_fixed_u8(&raw const cpu::debug_bp_armed as u32);
    ctx.builder.or_i32();
    ctx.builder.eqz_i32();
    ctx.builder.and_i32();
    // the steady state of a busy guest: chain into the next module (or reenter this
    // one) instead of falling back to the interpreter loop, which only happens when
    // the instruction budget is used up, halted or a debug trap is armed
    ctx.builder.if_void_likely();
    {
        codegen::gen_get_eip64(ctx.builder);
        let eip = ctx.builder.set_new_local_i64();

        // code = tlb_code_get(canonicalize(eip) >> 12), a pointer to the page's Code or 0
        ctx.builder.get_local_i64(&eip);
        ctx.builder.const_i64(32);
        ctx.builder.shr_u_i64();
        ctx.builder.eqz_i64();
        ctx.builder.if_i32();
        {
            ctx.builder.get_local_i64(&eip);
            ctx.builder.wrap_i64_to_i32();
            ctx.builder.const_i32(12);
            ctx.builder.shr_u_i32();
            ctx.builder.const_i32(2);
            ctx.builder.shl_i32();
            ctx.builder
                .load_aligned_i32(&raw const cpu::tlb_code as u32);
        }
        ctx.builder.else_();
        {
            // eip is canonical, so its arithmetic page number is the tag (cpu::tlb_high_tag)
            ctx.builder.get_local_i64(&eip);
            ctx.builder.const_i64(12);
            ctx.builder.shr_s_i64();
            let page = ctx.builder.tee_new_local_i64();
            // slot index: tlb_high_hash(tag)
            ctx.builder.const_i64(0x9E37_79B9_7F4A_7C15u64 as i64);
            ctx.builder.mul_i64();
            ctx.builder.const_i64(25);
            ctx.builder.shr_u_i64();
            ctx.builder.wrap_i64_to_i32();
            ctx.builder.const_i32((TLB_HIGH_SIZE - 1) as i32);
            ctx.builder.and_i32();
            let slot = ctx.builder.set_new_local();

            ctx.builder.get_local(&slot);
            ctx.builder.const_i32(2);
            ctx.builder.shl_i32();
            ctx.builder
                .load_aligned_i32(&raw const cpu::tlb_code_high as u32);
            ctx.builder.const_i32(0);
            ctx.builder.get_local(&slot);
            ctx.builder.const_i32(3);
            ctx.builder.shl_i32();
            ctx.builder
                .load_aligned_i64(&raw const cpu::tlb_high_page as u32);
            ctx.builder.get_local_i64(&page);
            ctx.builder.eq_i64();
            ctx.builder.select();
            ctx.builder.free_local(slot);
            ctx.builder.free_local_i64(page);
        }
        ctx.builder.block_end();
        let code = ctx.builder.tee_new_local();

        // code != 0 && code.state_flags == *state_flags && code.state_flags.is_64
        //
        // The is_64 test is not redundant with the equality check: the cpu
        // mode may have changed to compat mode *inside* this module (e.g. an
        // iret to a 32-bit user segment), in which case the target entry with
        // matching (compat) state flags holds a 32-bit module — the tail call
        // below uses the jit64 signature and would trap with a signature
        // mismatch. Without a chain the main loop dispatches it correctly.
        ctx.builder.if_void();
        {
            ctx.builder.get_local(&code);
            ctx.builder.load_u8(offset_of!(Code, state_flags) as u32);
            let code_flags = ctx.builder.tee_new_local();
            ctx.builder
                .load_fixed_u8(global_pointers::state_flags as u32);
            ctx.builder.eq_i32();
            // code must be a jit64 module (compat mode compiles to 32-bit modules)
            ctx.builder.get_local(&code_flags);
            ctx.builder.const_i32(1 << 4); // CachedStateFlags is_64 bit
            ctx.builder.and_i32();
            ctx.builder.eqz_i32();
            ctx.builder.eqz_i32();
            ctx.builder.and_i32();
            ctx.builder.free_local(code_flags);
            ctx.builder.if_void();
            {
                // state = code.state_table[eip & 0xFFF]
                ctx.builder.get_local(&code);
                ctx.builder.get_local_i64(&eip);
                ctx.builder.wrap_i64_to_i32();
                ctx.builder.const_i32(0xFFF);
                ctx.builder.and_i32();
                ctx.builder.const_i32(1);
                ctx.builder.shl_i32();
                ctx.builder.add_i32();
                ctx.builder
                    .load_aligned_u16(offset_of!(Code, state_table) as u32);
                let state = ctx.builder.tee_new_local();
                ctx.builder.const_i32(u16::MAX as i32);
                ctx.builder.ne_i32();
                // the target module almost always has an entry for the target address
                // (u16::MAX marks "not present"; the interpreter trains the table)
                ctx.builder.if_void_likely();
                {
                    // in this module?
                    ctx.builder.get_local(&code);
                    ctx.builder
                        .load_aligned_u16(offset_of!(Code, wasm_table_index) as u32);
                    ctx.builder.const_i32(ctx.wasm_table_index.to_u16() as i32);
                    ctx.builder.eq_i32();
                    ctx.builder.if_void();
                    {
                        ctx.builder.get_local(&state);
                        ctx.builder
                            .set_local(&ctx.builder.arg_local_initial_state.unsafe_clone());
                        // counted in the instruction counter already
                        ctx.builder.const_i32(0);
                        ctx.builder.set_local(&ctx.instruction_counter);
                        ctx.builder.br(reenter_label);
                    }
                    ctx.builder.block_end();

                    codegen::gen_profiler_stat_increment(
                        ctx.builder,
                        crate::profiler::stat::RUN_FROM_CACHE,
                    );
                    ctx.builder.get_local(&state);
                    for r in 0..16 {
                        ctx.builder.get_local_i64(&reg_local(ctx, r));
                    }
                    ctx.builder.get_local(&code);
                    ctx.builder
                        .load_aligned_u16(offset_of!(Code, wasm_table_index) as u32);
                    ctx.builder.const_i32(WASM_TABLE_OFFSET as i32);
                    ctx.builder.add_i32();
                    ctx.builder.return_call_indirect_jit64();
                }
                ctx.builder.block_end();
                ctx.builder.free_local(state);
            }
            ctx.builder.block_end();
        }
        ctx.builder.block_end();
        ctx.builder.free_local(code);
        ctx.builder.free_local_i64(eip);
    }
    ctx.builder.block_end();
}

/// After jumping to another page within a module: leave the module unless the instruction
/// pointer still maps to next_block_phys. Checked inline if the page is in the tlb, by
/// jit_page_switch_check64 otherwise.
pub fn gen_page_switch_check64(ctx: &mut JitContext, next_block_phys: u64) {
    let cont = ctx.builder.block_void();
    codegen::gen_get_eip64(ctx.builder);
    let address = ctx.builder.set_new_local_i64();
    gen_tlb_entry(ctx, &address);
    // the stack copy of the entry continues into the flag checks; tee only saves it
    let entry = if cfg!(feature = "mem64") {
        let l = ctx.builder.tee_new_local_i64();
        ctx.builder.wrap_i64_to_i32();
        Val::I64(l)
    }
    else {
        Val::I32(ctx.builder.tee_new_local())
    };
    let user = if ctx.cpu.cpl3() { TLB_NO_USER } else { 0 };
    ctx.builder
        .const_i32(TLB_VALID | TLB_NOT_EXECUTABLE | user | crate::cpu::cpu::TLB_IN_MAPPED_RANGE);
    ctx.builder.and_i32();
    ctx.builder.const_i32(TLB_VALID);
    ctx.builder.eq_i32();
    gen_pointer_from_entry(ctx, &address, &entry);
    entry.free(ctx);
    ctx.builder.free_local_i64(address);
    if cfg!(feature = "mem64") {
        // the pointer is i64 (gen_pointer_from_entry on the i64 entry)
        ctx.builder.const_i64(!0xFFF);
        ctx.builder.and_i64();
        ctx.builder
            .const_i64(crate::cpu::memory::tlb_host_base(next_block_phys & !0xFFF) as i64);
        ctx.builder.eq_i64();
    }
    else {
        ctx.builder.const_i32(!0xFFF);
        ctx.builder.and_i32();
        ctx.builder
            .const_i32(crate::cpu::memory::tlb_host_base(next_block_phys & !0xFFF) as i32);
        ctx.builder.eq_i32();
    }
    // flag check && mapping check
    ctx.builder.and_i32();
    ctx.builder.br_if(cont);

    ctx.builder.const_i32(next_block_phys as i32);
    ctx.builder.const_i32((next_block_phys >> 32) as i32);
    ctx.builder.call_fn2_ret("jit_page_switch_check64");
    ctx.builder.br_if(ctx.exit_label);
    ctx.builder.block_end();
}

/// After jumping to another page within compiled code: returns 1 if the new page isn't mapped
/// to the physical page the code was compiled for, or can't be executed (NX, SMEP; the
/// interpreter's instruction fetch then raises the page fault)
#[no_mangle]
pub unsafe fn jit_page_switch_check64(next_block_phys_low: u32, next_block_phys_high: u32) -> i32 {
    let next_block_phys = next_block_phys_low as u64 | (next_block_phys_high as u64) << 32;
    match cpu::translate_address_read_code_no_side_effects(*global_pointers::instruction_pointer) {
        Ok(phys) => (phys != next_block_phys) as i32,
        Err(()) => 1,
    }
}

// ---------------------------------------------------------------------------------------------
// Native code generation: registers (in memory), memory accesses (inline tlb lookup), lazy flags

use crate::cpu::cpu::{
    FLAGS_ALL, FLAG_ADJUST, FLAG_CARRY, FLAG_OVERFLOW, FLAG_SUB, OPSIZE_16, OPSIZE_32, OPSIZE_64,
    OPSIZE_8, TLB_GLOBAL, TLB_HAS_CODE, TLB_HIGH_SIZE, TLB_NOT_EXECUTABLE, TLB_NO_USER,
    TLB_READONLY, TLB_VALID,
};
use crate::prefix::PREFIX_REX_PRESENT as REX_PRESENT;
use crate::wasmgen::wasm_builder::{Label, WasmLocal, WasmLocalI64};

/// A value in a wasm local: i32 for operand sizes up to 32 bits, i64 for 64 bits
pub enum Val {
    I32(WasmLocal),
    I64(WasmLocalI64),
}
impl Val {
    pub fn get(&self, ctx: &mut JitContext) {
        match self {
            Val::I32(l) => ctx.builder.get_local(l),
            Val::I64(l) => ctx.builder.get_local_i64(l),
        }
    }
    pub fn free(self, ctx: &mut JitContext) {
        match self {
            Val::I32(l) => ctx.builder.free_local(l),
            Val::I64(l) => ctx.builder.free_local_i64(l),
        }
    }
}

/// Pop a value of the given operand size from the stack into a new local
pub fn set_new_val(ctx: &mut JitContext, bits: u32) -> Val {
    if bits == 64 {
        Val::I64(ctx.builder.set_new_local_i64())
    }
    else {
        Val::I32(ctx.builder.set_new_local())
    }
}

fn reg_local(ctx: &JitContext, r: u32) -> WasmLocalI64 {
    ctx.register_locals64[r as usize].unsafe_clone()
}

/// 8-bit registers: with a REX prefix, registers 4-7 are spl/bpl/sil/dil, without they are
/// ah/ch/dh/bh. Returns the register and whether it's the high byte
fn reg8(ctx: &JitContext, r: u32) -> (u32, bool) {
    if r >= 4 && r < 8 && ctx.cpu.prefixes & REX_PRESENT == 0 {
        (r - 4, true)
    }
    else {
        (r, false)
    }
}

/// Push a register (zero-extended to i32 for 8 and 16 bits, i64 for 64 bits)
pub fn gen_get_reg(ctx: &mut JitContext, bits: u32, r: u32) {
    match bits {
        8 => {
            let (r, high) = reg8(ctx, r);
            ctx.builder.get_local_i64(&reg_local(ctx, r));
            ctx.builder.wrap_i64_to_i32();
            if high {
                ctx.builder.const_i32(8);
                ctx.builder.shr_u_i32();
            }
            ctx.builder.const_i32(0xFF);
            ctx.builder.and_i32();
        },
        16 => {
            ctx.builder.get_local_i64(&reg_local(ctx, r));
            ctx.builder.wrap_i64_to_i32();
            ctx.builder.const_i32(0xFFFF);
            ctx.builder.and_i32();
        },
        32 => {
            ctx.builder.get_local_i64(&reg_local(ctx, r));
            ctx.builder.wrap_i64_to_i32();
        },
        _ => ctx.builder.get_local_i64(&reg_local(ctx, r)),
    }
}

/// Write a value to a register. 32-bit writes zero-extend into the full 64-bit register,
/// 8- and 16-bit writes leave the other bits unchanged
pub fn gen_set_reg(ctx: &mut JitContext, bits: u32, r: u32, value: &Val) {
    ctx.dirty_registers64 |= 1 << (if bits == 8 { reg8(ctx, r).0 } else { r });
    match bits {
        8 | 16 => {
            // merge into the register: reg = reg & ~(mask << shift) | (value & mask) << shift
            let (r, shift) = if bits == 8 {
                let (r, high) = reg8(ctx, r);
                (r, if high { 8 } else { 0 })
            }
            else {
                (r, 0)
            };
            let mask: i64 = if bits == 8 { 0xFF } else { 0xFFFF };
            let local = reg_local(ctx, r);
            ctx.builder.get_local_i64(&local);
            ctx.builder.const_i64(!(mask << shift));
            ctx.builder.and_i64();
            value.get(ctx);
            ctx.builder.const_i32(mask as i32);
            ctx.builder.and_i32();
            ctx.builder.extend_unsigned_i32_to_i64();
            if shift != 0 {
                ctx.builder.const_i64(shift);
                ctx.builder.shl_i64();
            }
            ctx.builder.or_i64();
            ctx.builder.set_local_i64(&local);
        },
        32 => {
            value.get(ctx);
            ctx.builder.extend_unsigned_i32_to_i64();
            ctx.builder.set_local_i64(&reg_local(ctx, r));
        },
        _ => {
            value.get(ctx);
            ctx.builder.set_local_i64(&reg_local(ctx, r));
        },
    }
}

/// Pops a value and writes it to a register
pub fn gen_set_reg_from_stack(ctx: &mut JitContext, bits: u32, r: u32) {
    let value = set_new_val(ctx, bits);
    gen_set_reg(ctx, bits, r, &value);
    value.free(ctx);
}

fn eip_and_wasm_table_index(ctx: &JitContext) -> i32 {
    (ctx.start_of_current_instruction & 0xFFF) as i32 | (ctx.wasm_table_index.to_u16() as i32) << 16
}

/// Generate the fast path tlb check for a 64-bit address: returns a local holding the
/// tlb entry (i32 in the default build, i64 under mem64) and leaves 1 on the stack if the
/// fast path can be used. The stack copy of the entry continues into the flag checks.
fn gen_tlb_fast_path_check(
    ctx: &mut JitContext,
    bits: u32,
    address: &WasmLocalI64,
    for_writing: bool,
) -> Val {
    gen_tlb_entry(ctx, address);
    let entry = if cfg!(feature = "mem64") {
        let l = ctx.builder.tee_new_local_i64();
        ctx.builder.wrap_i64_to_i32();
        Val::I64(l)
    }
    else {
        Val::I32(ctx.builder.tee_new_local())
    };

    // flags that must be clear (or set) for the fast path
    let user = if ctx.cpu.cpl3() { TLB_NO_USER } else { 0 };
    let mask = if for_writing {
        TLB_VALID | TLB_READONLY | TLB_HAS_CODE | user | crate::cpu::cpu::TLB_IN_MAPPED_RANGE
    }
    else {
        TLB_VALID | user | crate::cpu::cpu::TLB_IN_MAPPED_RANGE
    };
    dbg_assert!(mask & (TLB_GLOBAL | TLB_NOT_EXECUTABLE) == 0);
    ctx.builder.const_i32(mask);
    ctx.builder.and_i32();
    ctx.builder.const_i32(TLB_VALID);
    ctx.builder.eq_i32();

    if bits != 8 {
        ctx.builder.get_local_i64(address);
        ctx.builder.wrap_i64_to_i32();
        ctx.builder.const_i32(0xFFF);
        ctx.builder.and_i32();
        ctx.builder.const_i32(0x1000 - (bits / 8) as i32);
        ctx.builder.le_i32();
        ctx.builder.and_i32();
    }

    entry
}

/// Push the low 32 bits of the tlb entry for the page of a 64-bit address (0 if there is none).
/// Addresses below 4 GiB use the flat tlb (tlb_data), higher ones the hashed tlb (tlb_high_*);
/// both have the same entry format.
fn gen_tlb_entry(ctx: &mut JitContext, address: &WasmLocalI64) {
    ctx.builder.get_local_i64(address);
    ctx.builder.const_i64(32);
    ctx.builder.shr_u_i64();
    ctx.builder.eqz_i64();

    if cfg!(feature = "mem64") {
        ctx.builder.if_i64();
        {
            // entry = tlb_data[address >> 12]
            ctx.builder.get_local_i64(address);
            ctx.builder.wrap_i64_to_i32();
            ctx.builder.const_i32(12);
            ctx.builder.shr_u_i32();
            ctx.builder.const_i32(3);
            ctx.builder.shl_i32();
            ctx.builder
                .load_aligned_i64(&raw const cpu::tlb_data as u32);
        }
        ctx.builder.else_();
        gen_tlb_high_entry(ctx, address);
        ctx.builder.block_end();
    }
    else {
        ctx.builder.if_i32();
        {
            // entry = tlb_data[address >> 12]
            ctx.builder.get_local_i64(address);
            ctx.builder.wrap_i64_to_i32();
            ctx.builder.const_i32(12);
            ctx.builder.shr_u_i32();
            ctx.builder.const_i32(2);
            ctx.builder.shl_i32();
            ctx.builder
                .load_aligned_i32(&raw const cpu::tlb_data as u32);
        }
        ctx.builder.else_();
        gen_tlb_high_entry(ctx, address);
        ctx.builder.block_end();
    }
}

/// Push the low 32 bits of the hashed tlb's entry for the page of a 64-bit address, or 0 (not
/// valid) if the slot holds a different page
fn gen_tlb_high_entry(ctx: &mut JitContext, address: &WasmLocalI64) {
    // tag = address >> 12 (arithmetic): equals cpu::tlb_high_tag of the page for canonical
    // addresses, and no tag for non-canonical ones, which miss and #GP in the slow path
    ctx.builder.get_local_i64(address);
    ctx.builder.const_i64(12);
    ctx.builder.shr_s_i64();
    let page = ctx.builder.tee_new_local_i64();

    // byte offset of the slot: tlb_high_hash(tag) * 8
    ctx.builder.const_i64(0x9E37_79B9_7F4A_7C15u64 as i64);
    ctx.builder.mul_i64();
    ctx.builder.const_i64(25 - 3);
    ctx.builder.shr_u_i64();
    ctx.builder.wrap_i64_to_i32();
    ctx.builder.const_i32(((TLB_HIGH_SIZE - 1) << 3) as i32);
    ctx.builder.and_i32();
    let slot = ctx.builder.set_new_local();

    ctx.builder.get_local(&slot);
    ctx.builder
        .load_aligned_i64(&raw const cpu::tlb_high_entry as u32);
    if cfg!(feature = "mem64") {
        ctx.builder.const_i64(0);
    }
    else {
        ctx.builder.wrap_i64_to_i32();
        ctx.builder.const_i32(0);
    }
    ctx.builder.get_local(&slot);
    ctx.builder
        .load_aligned_i64(&raw const cpu::tlb_high_page as u32);
    ctx.builder.get_local_i64(&page);
    ctx.builder.eq_i64();
    ctx.builder.select();
    ctx.builder.free_local(slot);
    ctx.builder.free_local_i64(page);
}

/// pointer = (entry & ~0xFFF) ^ address
fn gen_pointer_from_entry(ctx: &mut JitContext, address: &WasmLocalI64, entry: &Val) {
    match entry {
        Val::I32(entry) => {
            ctx.builder.get_local(entry);
            ctx.builder.const_i32(!0xFFF);
            ctx.builder.and_i32();
            ctx.builder.get_local_i64(address);
            ctx.builder.wrap_i64_to_i32();
            ctx.builder.xor_i32();
        },
        Val::I64(entry) => {
            ctx.builder.get_local_i64(entry);
            ctx.builder.const_i64(!0xFFF);
            ctx.builder.and_i64();
            ctx.builder.get_local_i64(address);
            ctx.builder.xor_i64();
            // strip the address' sign bits, which cancel against the entry's above only for
            // canonical forms (see TODOS.md §4.1, §4.3)
            ctx.builder.const_i64(0xFFFF_FFFF_FFFF);
            ctx.builder.and_i64();
        },
    }
}

fn gen_load(ctx: &mut JitContext, bits: u32) {
    if cfg!(feature = "mem64") {
        // the pointer is an i64 under mem64 and the access targets the guest memory
        match bits {
            8 => return ctx.builder.load_guest_u8(0),
            16 => return ctx.builder.load_guest_u16(0),
            32 => return ctx.builder.load_guest_i32(0),
            _ => return ctx.builder.load_guest_i64(0),
        }
    }
    match bits {
        8 => ctx.builder.load_u8(0),
        16 => ctx.builder.load_unaligned_u16(0),
        32 => ctx.builder.load_unaligned_i32(0),
        _ => ctx.builder.load_unaligned_i64(0),
    }
}
fn gen_store(ctx: &mut JitContext, bits: u32) {
    if cfg!(feature = "mem64") {
        match bits {
            8 => return ctx.builder.store_guest_u8(0),
            16 => return ctx.builder.store_guest_u16(0),
            32 => return ctx.builder.store_guest_i32(0),
            _ => return ctx.builder.store_guest_i64(0),
        }
    }
    match bits {
        8 => ctx.builder.store_u8(0),
        16 => ctx.builder.store_unaligned_u16(0),
        32 => ctx.builder.store_unaligned_i32(0),
        _ => ctx.builder.store_unaligned_i64(0),
    }
}

/// Read from memory, push the value (i32 zero-extended for up to 32 bits, i64 for 64 bits).
/// Exceptions leave the compiled code through exit_with_fault_label.
pub fn gen_safe_read(ctx: &mut JitContext, bits: u32, address: &WasmLocalI64) {
    let cache = tlb_cache(ctx);
    let done = cache.as_ref().map(|_| ctx.builder.block_void());
    let pointer = cache.as_ref().map(|_| new_pointer_local(ctx));
    if let Some(c) = &cache {
        gen_tlb_cache_hit_condition(ctx, bits, address, false);
        ctx.builder.if_void_likely();
        gen_pointer_from_entry(ctx, address, &cached_entry_val(c));
        set_pointer_local(ctx, pointer.as_ref().unwrap());
        ctx.builder.br(done.unwrap());
        ctx.builder.block_end();
    }
    let cont = ctx.builder.block_void();
    let fastpop = cache.as_ref().map(|_| ctx.builder.block_void());
    let entry = gen_tlb_fast_path_check(ctx, bits, address, false);
    ctx.builder.br_if_likely(fastpop.unwrap_or(cont));

    // the slow path may deliver a page fault, which reads the flags: commit the deferred
    // lazy flags of the previous instructions (this instruction hasn't run yet)
    gen_commit_deferred_flags_cold(ctx);
    ctx.builder.get_local_i64(address);
    ctx.builder
        .const_i32((ctx.start_of_current_instruction & 0xFFF) as i32);
    ctx.builder.call_fn2_i64_i32_ret(match bits {
        8 => "safe_read8_slow_jit64",
        16 => "safe_read16_slow_jit64",
        32 => "safe_read32s_slow_jit64",
        _ => "safe_read64s_slow_jit64",
    });
    // default: the return value is the entry; mem64: it's a status and the entry is stashed
    if cfg!(feature = "mem64") {
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();
        ctx.builder.br_if_unlikely(ctx.exit_with_fault_label);
        codegen::gen_load_slow_entry(ctx);
        ctx.builder.set_local_i64(match &entry {
            Val::I64(l) => l,
            Val::I32(_) => unreachable!(),
        });
    }
    else {
        ctx.builder.tee_local(match &entry {
            Val::I32(l) => l,
            Val::I64(_) => unreachable!(),
        });
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();
        ctx.builder.br_if_unlikely(ctx.exit_with_fault_label);
    }
    if fastpop.is_some() {
        // the slow path's entry may point into mapped memory or scratch: only the fast
        // path's entries are cached
        ctx.builder.br(cont);
        ctx.builder.block_end(); // fastpop
        gen_tlb_cache_populate(ctx, address, &entry);
    }
    ctx.builder.block_end(); // cont

    gen_pointer_from_entry(ctx, address, &entry);
    entry.free(ctx);
    if let Some(pointer) = &pointer {
        set_pointer_local(ctx, pointer);
        ctx.builder.block_end(); // done
        pointer.get(ctx);
        match pointer {
            Val::I32(l) => ctx.builder.free_local(l.unsafe_clone()),
            Val::I64(l) => ctx.builder.free_local_i64(l.unsafe_clone()),
        }
    }
    gen_load(ctx, bits);
}

/// Write a value to memory
pub fn gen_safe_write(ctx: &mut JitContext, bits: u32, address: &WasmLocalI64, value: &Val) {
    let cache = tlb_cache(ctx);
    let done = cache.as_ref().map(|_| ctx.builder.block_void());
    let pointer = cache.as_ref().map(|_| new_pointer_local(ctx));
    if let Some(c) = &cache {
        gen_tlb_cache_hit_condition(ctx, bits, address, true);
        ctx.builder.if_void_likely();
        gen_pointer_from_entry(ctx, address, &cached_entry_val(c));
        set_pointer_local(ctx, pointer.as_ref().unwrap());
        ctx.builder.br(done.unwrap());
        ctx.builder.block_end();
    }
    let cont = ctx.builder.block_void();
    let fastpop = cache.as_ref().map(|_| ctx.builder.block_void());
    let entry = gen_tlb_fast_path_check(ctx, bits, address, true);
    ctx.builder.br_if_likely(fastpop.unwrap_or(cont));

    gen_write_slow_path(ctx, bits, address, value);
    // default: the return value is the entry; mem64: it's a status and the entry is stashed
    if cfg!(feature = "mem64") {
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();
        ctx.builder.br_if_unlikely(ctx.exit_with_fault_label);
        codegen::gen_load_slow_entry(ctx);
        ctx.builder.set_local_i64(match &entry {
            Val::I64(l) => l,
            Val::I32(_) => unreachable!(),
        });
    }
    else {
        ctx.builder.tee_local(match &entry {
            Val::I32(l) => l,
            Val::I64(_) => unreachable!(),
        });
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();
        ctx.builder.br_if_unlikely(ctx.exit_with_fault_label);
    }
    if fastpop.is_some() {
        // the slow path's entry may point into mapped memory or scratch: only the fast
        // path's entries are cached
        ctx.builder.br(cont);
        ctx.builder.block_end(); // fastpop
        gen_tlb_cache_populate(ctx, address, &entry);
    }
    ctx.builder.block_end(); // cont

    gen_pointer_from_entry(ctx, address, &entry);
    entry.free(ctx);
    if let Some(pointer) = &pointer {
        set_pointer_local(ctx, pointer);
        ctx.builder.block_end(); // done
        pointer.get(ctx);
        match pointer {
            Val::I32(l) => ctx.builder.free_local(l.unsafe_clone()),
            Val::I64(l) => ctx.builder.free_local_i64(l.unsafe_clone()),
        }
    }
    value.get(ctx);
    gen_store(ctx, bits);
}

fn gen_write_slow_path(ctx: &mut JitContext, bits: u32, address: &WasmLocalI64, value: &Val) {
    // may deliver a page fault, which reads the flags: commit the deferred lazy flags
    gen_commit_deferred_flags_cold(ctx);
    ctx.builder.get_local_i64(address);
    value.get(ctx);
    ctx.builder.const_i32(eip_and_wasm_table_index(ctx));
    match bits {
        8 => ctx
            .builder
            .call_fn3_i64_i32_i32_ret("safe_write8_slow_jit64"),
        16 => ctx
            .builder
            .call_fn3_i64_i32_i32_ret("safe_write16_slow_jit64"),
        32 => ctx
            .builder
            .call_fn3_i64_i32_i32_ret("safe_write32_slow_jit64"),
        _ => ctx
            .builder
            .call_fn3_i64_i64_i32_ret("safe_write64_slow_jit64"),
    }
}

/// Read-modify-write: f is called with the old value on the stack and must leave the new one
pub fn gen_safe_read_write(
    ctx: &mut JitContext,
    bits: u32,
    address: &WasmLocalI64,
    f: &dyn Fn(&mut JitContext),
) {
    let cache = tlb_cache(ctx);
    let can_use_fast_path = ctx.builder.new_local();
    let done = cache.as_ref().map(|_| ctx.builder.block_void());
    let hit_pointer = cache.as_ref().map(|_| new_pointer_local(ctx));
    if let Some(c) = &cache {
        gen_tlb_cache_hit_condition(ctx, bits, address, true);
        ctx.builder.if_void_likely();
        ctx.builder.const_i32(1);
        ctx.builder.set_local(&can_use_fast_path);
        gen_pointer_from_entry(ctx, address, &cached_entry_val(c));
        set_pointer_local(ctx, hit_pointer.as_ref().unwrap());
        ctx.builder.br(done.unwrap());
        ctx.builder.block_end();
    }
    let cont = ctx.builder.block_void();
    let fastpop = cache.as_ref().map(|_| ctx.builder.block_void());
    let entry = gen_tlb_fast_path_check(ctx, bits, address, true);
    ctx.builder.tee_local(&can_use_fast_path);
    ctx.builder.br_if_likely(fastpop.unwrap_or(cont));

    // may deliver a page fault, which reads the flags: commit the deferred lazy flags
    // (still those of the previous instructions: f runs after this at runtime)
    gen_commit_deferred_flags_cold(ctx);
    ctx.builder.get_local_i64(address);
    ctx.builder.const_i32(eip_and_wasm_table_index(ctx));
    ctx.builder.call_fn2_i64_i32_ret(match bits {
        8 => "safe_read_write8_slow_jit64",
        16 => "safe_read_write16_slow_jit64",
        32 => "safe_read_write32s_slow_jit64",
        _ => "safe_read_write64_slow_jit64",
    });
    // default: the return value is the entry; mem64: it's a status and the entry is stashed
    if cfg!(feature = "mem64") {
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();
        ctx.builder.br_if_unlikely(ctx.exit_with_fault_label);
        codegen::gen_load_slow_entry(ctx);
        ctx.builder.set_local_i64(match &entry {
            Val::I64(l) => l,
            Val::I32(_) => unreachable!(),
        });
    }
    else {
        ctx.builder.tee_local(match &entry {
            Val::I32(l) => l,
            Val::I64(_) => unreachable!(),
        });
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();
        ctx.builder.br_if_unlikely(ctx.exit_with_fault_label);
    }
    if fastpop.is_some() {
        // the slow path's entry may point into mapped memory or scratch: only the fast
        // path's entries are cached
        ctx.builder.br(cont);
        ctx.builder.block_end(); // fastpop
        gen_tlb_cache_populate(ctx, address, &entry);
    }
    ctx.builder.block_end(); // cont

    gen_pointer_from_entry(ctx, address, &entry);
    entry.free(ctx);
    let pointer = if let Some(hit_pointer) = &hit_pointer {
        set_pointer_local(ctx, hit_pointer);
        ctx.builder.block_end(); // done
        hit_pointer.get(ctx);
        if cfg!(feature = "mem64") {
            Val::I64(ctx.builder.tee_new_local_i64())
        }
        else {
            Val::I32(ctx.builder.tee_new_local())
        }
    }
    else if cfg!(feature = "mem64") {
        Val::I64(ctx.builder.tee_new_local_i64())
    }
    else {
        Val::I32(ctx.builder.tee_new_local())
    };
    gen_load(ctx, bits);

    f(ctx);

    let value = set_new_val(ctx, bits);

    // the slow path for reading has done all checks, but writes across pages or to mapped
    // memory have to be done by the slow path for writing
    ctx.builder.get_local(&can_use_fast_path);
    ctx.builder.eqz_i32();
    ctx.builder.if_void_unlikely();
    {
        gen_write_slow_path(ctx, bits, address, &value);
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();
        ctx.builder.br_if_unlikely(ctx.exit_with_fault_label);
    }
    ctx.builder.block_end();
    ctx.builder.free_local(can_use_fast_path);

    pointer.get(ctx);
    value.get(ctx);
    gen_store(ctx, bits);
    pointer.free(ctx);
    value.free(ctx);
    if let Some(hit_pointer) = &hit_pointer {
        match hit_pointer {
            Val::I32(l) => ctx.builder.free_local(l.unsafe_clone()),
            Val::I64(l) => ctx.builder.free_local_i64(l.unsafe_clone()),
        }
    }
}

/// Compute the address of a memory operand into a new local
pub fn gen_modrm_address_local(ctx: &mut JitContext, m: &Modrm64) -> WasmLocalI64 {
    gen_modrm_address(ctx, m);
    ctx.builder.set_new_local_i64()
}

// Lazy flags (see cpu/arith.rs: add/sub use last_op1 and last_result, logical operations only
// last_result; 64-bit operations use the separate 64-bit slots)

pub fn opsize(bits: u32) -> i32 {
    match bits {
        8 => OPSIZE_8,
        16 => OPSIZE_16,
        32 => OPSIZE_32,
        _ => OPSIZE_64,
    }
}

fn gen_set_last_op1(ctx: &mut JitContext, bits: u32, value: &Val) {
    if bits == 64 {
        ctx.builder.const_i32(global_pointers::last_op1_64 as i32);
        value.get(ctx);
        ctx.builder.store_aligned_i64(0);
    }
    else {
        ctx.builder.const_i32(global_pointers::last_op1 as i32);
        value.get(ctx);
        ctx.builder.store_aligned_i32(0);
    }
}
fn gen_set_last_result(ctx: &mut JitContext, bits: u32, value: &Val) {
    if bits == 64 {
        ctx.builder
            .const_i32(global_pointers::last_result_64 as i32);
        value.get(ctx);
        ctx.builder.store_aligned_i64(0);
    }
    else {
        ctx.builder.const_i32(global_pointers::last_result as i32);
        value.get(ctx);
        ctx.builder.store_aligned_i32(0);
    }
}
fn gen_set_op_size_and_flags_changed(ctx: &mut JitContext, bits: u32, flags_changed: i32) {
    // last_op_size and flags_changed are adjacent: write both with one store
    dbg_assert!(global_pointers::flags_changed as u32 == global_pointers::last_op_size as u32 + 4);
    ctx.builder.const_i32(global_pointers::last_op_size as i32);
    ctx.builder
        .const_i64((flags_changed as u32 as i64) << 32 | opsize(bits) as u32 as i64);
    ctx.builder.store_aligned_i64(0);
}
fn gen_clear_flags(ctx: &mut JitContext, clear: i32) {
    ctx.builder.const_i32(global_pointers::flags as i32);
    ctx.builder.load_fixed_i32(global_pointers::flags as u32);
    ctx.builder.const_i32(!clear);
    ctx.builder.and_i32();
    ctx.builder.store_aligned_i32(0);
}

/// flags for add/sub/cmp (op1 and result have the operand size)
pub fn gen_flags_arith(ctx: &mut JitContext, bits: u32, op1: &Val, result: &Val, is_sub: bool) {
    let flags_changed = FLAGS_ALL | if is_sub { FLAG_SUB } else { 0 };
    if defer_flags_enabled(ctx) {
        gen_defer_flags(ctx, bits, Some(op1), result, flags_changed, 0);
        return;
    }
    gen_set_last_op1(ctx, bits, op1);
    gen_set_last_result(ctx, bits, result);
    gen_set_op_size_and_flags_changed(ctx, bits, flags_changed);
}

/// flags for and/or/xor/test
pub fn gen_flags_logic(ctx: &mut JitContext, bits: u32, result: &Val) {
    let flags_changed = FLAGS_ALL & !FLAG_CARRY & !FLAG_OVERFLOW & !FLAG_ADJUST;
    if defer_flags_enabled(ctx) {
        gen_defer_flags(
            ctx,
            bits,
            None,
            result,
            flags_changed,
            FLAG_CARRY | FLAG_OVERFLOW | FLAG_ADJUST,
        );
        return;
    }
    gen_set_last_result(ctx, bits, result);
    gen_set_op_size_and_flags_changed(ctx, bits, flags_changed);
    gen_clear_flags(ctx, FLAG_CARRY | FLAG_OVERFLOW | FLAG_ADJUST);
}

// ---------------------------------------------------------------------------------------------
// Native instructions

/// An operand of a natively compiled instruction
#[derive(Copy, Clone, PartialEq)]
pub enum Opnd {
    Reg(u32),
    Mem(Modrm64),
    Imm(i64),
}

pub const OP_ADD: u32 = 0;
pub const OP_OR: u32 = 1;
pub const OP_AND: u32 = 4;
pub const OP_SUB: u32 = 5;
pub const OP_XOR: u32 = 6;
pub const OP_CMP: u32 = 7;
pub const OP_TEST: u32 = 8;
pub const OP_MOV: u32 = 9;

fn mask_for(bits: u32) -> i32 {
    match bits {
        8 => 0xFF,
        16 => 0xFFFF,
        _ => -1,
    }
}

fn gen_const(ctx: &mut JitContext, bits: u32, value: i64) {
    if bits == 64 {
        ctx.builder.const_i64(value);
    }
    else {
        ctx.builder.const_i32(value as i32 & mask_for(bits));
    }
}

/// Push the value of a source operand
fn gen_get_operand(ctx: &mut JitContext, bits: u32, src: &Opnd) {
    match src {
        Opnd::Reg(r) => gen_get_reg(ctx, bits, *r),
        Opnd::Imm(i) => gen_const(ctx, bits, *i),
        Opnd::Mem(m) => {
            let address = gen_modrm_address_local(ctx, m);
            gen_safe_read(ctx, bits, &address);
            ctx.builder.free_local_i64(address);
        },
    }
}

/// result = op1 <op> op2 (both on the stack), masked to the operand size
fn gen_binop(ctx: &mut JitContext, op: u32, bits: u32) {
    if bits == 64 {
        match op {
            OP_ADD => ctx.builder.add_i64(),
            OP_OR => ctx.builder.or_i64(),
            OP_AND | OP_TEST => ctx.builder.and_i64(),
            OP_SUB | OP_CMP => ctx.builder.sub_i64(),
            OP_XOR => ctx.builder.xor_i64(),
            _ => dbg_assert!(false),
        }
    }
    else {
        match op {
            OP_ADD => ctx.builder.add_i32(),
            OP_OR => ctx.builder.or_i32(),
            OP_AND | OP_TEST => ctx.builder.and_i32(),
            OP_SUB | OP_CMP => ctx.builder.sub_i32(),
            OP_XOR => ctx.builder.xor_i32(),
            _ => dbg_assert!(false),
        }
        if bits < 32 && (op == OP_ADD || op == OP_SUB || op == OP_CMP) {
            ctx.builder.const_i32(mask_for(bits));
            ctx.builder.and_i32();
        }
    }
}

/// Compute op1 <op> src, set the lazy flags, leave the result in a new local
fn gen_alu_value(ctx: &mut JitContext, op: u32, bits: u32, op1: &Val, src: &Val) -> Val {
    op1.get(ctx);
    src.get(ctx);
    gen_binop(ctx, op, bits);
    let result = set_new_val(ctx, bits);
    match op {
        OP_ADD => {
            gen_flags_arith(ctx, bits, op1, &result, false);
            ctx.flags64 = Flags64::Add(bits);
        },
        OP_SUB | OP_CMP => {
            gen_flags_arith(ctx, bits, op1, &result, true);
            ctx.flags64 = Flags64::Sub(bits);
        },
        _ => {
            gen_flags_logic(ctx, bits, &result);
            ctx.flags64 = Flags64::Logic(bits);
        },
    }
    result
}

/// add/or/and/sub/xor/cmp/test/mov dst, src
pub fn gen_alu(ctx: &mut JitContext, op: u32, bits: u32, dst: Opnd, src: Opnd) {
    // read the source first (a memory source may fault before anything is modified)
    let src_is_mem = match src {
        Opnd::Mem(_) => true,
        _ => false,
    };
    dbg_assert!(!src_is_mem || !matches!(dst, Opnd::Mem(_)));
    gen_get_operand(ctx, bits, &src);
    let src = set_new_val(ctx, bits);

    match dst {
        Opnd::Reg(r) => {
            if op == OP_MOV {
                gen_set_reg(ctx, bits, r, &src);
            }
            else {
                gen_get_reg(ctx, bits, r);
                let op1 = set_new_val(ctx, bits);
                let result = gen_alu_value(ctx, op, bits, &op1, &src);
                if op != OP_CMP && op != OP_TEST {
                    gen_set_reg(ctx, bits, r, &result);
                }
                result.free(ctx);
                op1.free(ctx);
            }
        },
        Opnd::Mem(m) => {
            let address = gen_modrm_address_local(ctx, &m);
            if op == OP_MOV {
                gen_safe_write(ctx, bits, &address, &src);
            }
            else if op == OP_CMP || op == OP_TEST {
                gen_safe_read(ctx, bits, &address);
                let op1 = set_new_val(ctx, bits);
                let result = gen_alu_value(ctx, op, bits, &op1, &src);
                result.free(ctx);
                op1.free(ctx);
            }
            else {
                let src_ref = &src;
                gen_safe_read_write(ctx, bits, &address, &|ctx| {
                    let op1 = set_new_val(ctx, bits);
                    let result = gen_alu_value(ctx, op, bits, &op1, src_ref);
                    result.get(ctx);
                    result.free(ctx);
                    op1.free(ctx);
                });
            }
            ctx.builder.free_local_i64(address);
        },
        Opnd::Imm(_) => dbg_assert!(false),
    }
    src.free(ctx);
}

/// movzx/movsx/movsxd r, r/m: the source has src_bits, the destination dst_bits
pub fn gen_movx(
    ctx: &mut JitContext,
    signed: bool,
    src_bits: u32,
    dst_bits: u32,
    r: u32,
    src: Opnd,
) {
    gen_get_operand(ctx, src_bits, &src);
    if signed && src_bits < 32 {
        let shift = 32 - src_bits as i32;
        ctx.builder.const_i32(shift);
        ctx.builder.shl_i32();
        ctx.builder.const_i32(shift);
        ctx.builder.shr_s_i32();
    }
    if dst_bits == 64 {
        if signed {
            ctx.builder.extend_signed_i32_to_i64();
        }
        else {
            ctx.builder.extend_unsigned_i32_to_i64();
        }
    }
    else if dst_bits == 16 {
        ctx.builder.const_i32(0xFFFF);
        ctx.builder.and_i32();
    }
    gen_set_reg_from_stack(ctx, dst_bits, r);
}

/// Compute the carry flag into the cf flag local before inc/dec overwrites the lazy flags
/// state (returns false when the following instructions make it dead). The carry is then
/// carried in the local and written to the flags word when the inc/dec state is committed.
fn gen_defer_save_cf(ctx: &mut JitContext) -> bool {
    if next_instructions_overwrite_flags(ctx) {
        // the saved cf would be dead
        return false;
    }
    let fl = flag_locals(ctx);
    match ctx.flags64 {
        Flags64::Logic(b) if b >= 32 => {
            // logic operations clear cf
            ctx.builder.const_i32(0);
            ctx.builder.set_local(&fl.cf);
        },
        Flags64::Add(b) | Flags64::Sub(b) if b >= 32 => {
            // cf is condition "b" (2) of the deferred state
            let ok = gen_condition_inline(ctx, 2);
            dbg_assert!(ok);
            ctx.builder.set_local(&fl.cf);
        },
        Flags64::IncDec { bits: b, .. } if b >= 32 => match ctx.jit64_deferred {
            Some(d) if d.cf_valid => {
                // the local already holds the carry
            },
            Some(_) => {
                gen_commit_deferred_flags(ctx);
                gen_getcf_generic(ctx);
                ctx.builder.set_local(&fl.cf);
            },
            None => {
                ctx.builder.load_fixed_i32(global_pointers::flags as u32);
                ctx.builder.const_i32(1);
                ctx.builder.and_i32();
                ctx.builder.set_local(&fl.cf);
            },
        },
        _ => {
            gen_commit_deferred_flags(ctx);
            gen_getcf_generic(ctx);
            ctx.builder.set_local(&fl.cf);
        },
    }
    true
}

/// inc/dec r/m: cf is preserved
pub fn gen_incdec(ctx: &mut JitContext, is_dec: bool, bits: u32, dst: Opnd) {
    let defer = defer_flags_enabled(ctx);
    let mut cf_valid = false;
    if defer {
        // move the carry into the cf local before the lazy state is overwritten
        cf_valid = gen_defer_save_cf(ctx);
    }
    else {
        // materialise cf into flags before the lazy state is overwritten
        gen_save_cf(ctx);
        ctx.flags64 = Flags64::Unknown;
    }
    let one = set_new_val_const(ctx, bits, 1);
    let flags_changed = FLAGS_ALL & !FLAG_CARRY | if is_dec { FLAG_SUB } else { 0 };
    let op = if is_dec { OP_SUB } else { OP_ADD };
    let compute = |ctx: &mut JitContext, op1: &Val| -> Val {
        op1.get(ctx);
        one.get(ctx);
        gen_binop(ctx, op, bits);
        let result = set_new_val(ctx, bits);
        if defer {
            gen_defer_flags(ctx, bits, Some(op1), &result, flags_changed, 0);
            if let Some(d) = &mut ctx.jit64_deferred {
                d.cf_valid = cf_valid;
            }
            ctx.flags64 = Flags64::IncDec { bits, is_dec };
        }
        else {
            gen_set_last_op1(ctx, bits, op1);
            gen_set_last_result(ctx, bits, &result);
            gen_set_op_size_and_flags_changed(ctx, bits, flags_changed);
        }
        result
    };
    match dst {
        Opnd::Reg(r) => {
            gen_get_reg(ctx, bits, r);
            let op1 = set_new_val(ctx, bits);
            let result = compute(ctx, &op1);
            gen_set_reg(ctx, bits, r, &result);
            result.free(ctx);
            op1.free(ctx);
        },
        Opnd::Mem(m) => {
            let address = gen_modrm_address_local(ctx, &m);
            gen_safe_read_write(ctx, bits, &address, &|ctx| {
                let op1 = set_new_val(ctx, bits);
                let result = compute(ctx, &op1);
                result.get(ctx);
                result.free(ctx);
                op1.free(ctx);
            });
            ctx.builder.free_local_i64(address);
        },
        Opnd::Imm(_) => dbg_assert!(false),
    }
    one.free(ctx);
}

fn set_new_val_const(ctx: &mut JitContext, bits: u32, value: i64) -> Val {
    gen_const(ctx, bits, value);
    set_new_val(ctx, bits)
}

/// push (64-bit operand size): the store happens before rsp is updated
pub fn gen_push64(ctx: &mut JitContext, src: Opnd) {
    gen_get_operand(ctx, 64, &src);
    let value = set_new_val(ctx, 64);
    gen_push64_value(ctx, &value);
    value.free(ctx);
}

pub fn gen_push64_value(ctx: &mut JitContext, value: &Val) {
    gen_get_reg(ctx, 64, crate::regs::ESP);
    ctx.builder.const_i64(8);
    ctx.builder.sub_i64();
    let new_rsp = ctx.builder.set_new_local_i64();
    gen_safe_write(ctx, 64, &new_rsp, value);
    let new_rsp = Val::I64(new_rsp);
    gen_set_reg(ctx, 64, crate::regs::ESP, &new_rsp);
    new_rsp.free(ctx);
}

/// pop r64
pub fn gen_pop64(ctx: &mut JitContext, r: u32) {
    gen_get_reg(ctx, 64, crate::regs::ESP);
    let rsp = ctx.builder.set_new_local_i64();
    gen_safe_read(ctx, 64, &rsp);
    let value = set_new_val(ctx, 64);
    ctx.builder.get_local_i64(&rsp);
    ctx.builder.const_i64(8);
    ctx.builder.add_i64();
    ctx.builder.free_local_i64(rsp);
    gen_set_reg_from_stack(ctx, 64, crate::regs::ESP);
    gen_set_reg(ctx, 64, r, &value);
    value.free(ctx);
}

/// ret (near, 64-bit): the block glue continues at the new instruction pointer
pub fn gen_ret64(ctx: &mut JitContext, imm16: u32) {
    gen_get_reg(ctx, 64, crate::regs::ESP);
    let rsp = ctx.builder.set_new_local_i64();
    gen_safe_read(ctx, 64, &rsp);
    let target = ctx.builder.set_new_local_i64();
    ctx.builder.get_local_i64(&rsp);
    ctx.builder.const_i64(8 + imm16 as i64);
    ctx.builder.add_i64();
    ctx.builder.free_local_i64(rsp);
    gen_set_reg_from_stack(ctx, 64, crate::regs::ESP);
    ctx.builder
        .const_i32(global_pointers::instruction_pointer as i32);
    ctx.builder.get_local_i64(&target);
    ctx.builder.store_aligned_i64(0);
    ctx.builder.free_local_i64(target);
}

#[no_mangle]
pub unsafe fn jit64_save_cf() {
    let cf = crate::cpu::misc_instr::getcf();
    *global_pointers::flags = *global_pointers::flags & !FLAG_CARRY | cf as i32;
}

pub enum ShiftCount {
    /// already masked, non-zero
    Imm(u32),
    /// cl, masked at runtime
    Cl,
}

/// rol (0), ror (1), shl (4), shr (5), sar (7) r/m by a constant or cl. Rotates are only
/// compiled for 32/64-bit operands; shifts for all sizes. See cpu/arith.rs: shifts set the
/// lazy result, rotates only cf and of. With a count of zero the flags are unchanged (the
/// register is still written, i.e. zero-extended for 32 bits; an 8/16-bit register write of
/// the unchanged value is a no-op and the memory operand is still read and written back).
/// Counts are masked to 5 bits (6 for 64-bit operands), so an 8/16-bit shift by a count
/// larger than the operand produces 0 (sar: the sign fill), exactly like arith::shl8 and
/// friends.
pub fn gen_shift(ctx: &mut JitContext, kind: u32, bits: u32, dst: Opnd, count: ShiftCount) {
    dbg_assert!(bits == 32 || bits == 64 || kind == 4 || kind == 5 || kind == 7);
    dbg_assert!(kind == 0 || kind == 1 || kind == 4 || kind == 5 || kind == 7);
    let w = bits == 64;
    // sets the flags eagerly (and rmw's the flags word): commit the deferred state first
    gen_commit_deferred_flags(ctx);
    // skip the flag update when the following instructions overwrite all flags before
    // anything reads them (same argument as the dead cf save in gen_save_cf)
    let flags_dead = next_instructions_overwrite_flags(ctx);

    // count (i32, masked)
    let count_local = ctx.builder.new_local();
    match count {
        ShiftCount::Imm(c) => {
            dbg_assert!(c != 0 && c < if bits == 64 { 64 } else { 32 });
            ctx.builder.const_i32(c as i32);
        },
        ShiftCount::Cl => {
            gen_get_reg(ctx, 8, crate::regs::ECX);
            ctx.builder.const_i32(if bits == 64 { 63 } else { 31 });
            ctx.builder.and_i32();
        },
    }
    ctx.builder.set_local(&count_local);
    let count_is_constant = matches!(count, ShiftCount::Imm(_));

    // push the msb or a given bit of a value as i32 0/1
    fn gen_bit(ctx: &mut JitContext, w: bool, value: &Val, shift: &dyn Fn(&mut JitContext)) {
        value.get(ctx);
        shift(ctx);
        if w {
            ctx.builder.shr_u_i64();
            ctx.builder.wrap_i64_to_i32();
        }
        else {
            ctx.builder.shr_u_i32();
        }
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();
    }

    let compute = |ctx: &mut JitContext, x: &Val| -> Val {
        // result
        x.get(ctx);
        ctx.builder.get_local(&count_local);
        if w {
            ctx.builder.extend_unsigned_i32_to_i64();
            match kind {
                0 => ctx.builder.rotl_i64(),
                1 => ctx.builder.rotr_i64(),
                4 => ctx.builder.shl_i64(),
                5 => ctx.builder.shr_u_i64(),
                _ => ctx.builder.shr_s_i64(),
            }
        }
        else {
            match kind {
                0 => ctx.builder.rotl_i32(),
                1 => ctx.builder.rotr_i32(),
                4 => ctx.builder.shl_i32(),
                5 => ctx.builder.shr_u_i32(),
                7 if bits == 32 => ctx.builder.shr_s_i32(),
                // sar with an 8/16-bit operand: the count is already on the stack (the
                // shared code pushed it), so drop it, sign-extend the value and shift by
                // the masked count — a count larger than the operand then produces the
                // sign fill (like arith::sar8)
                _ => {
                    dbg_assert!(kind == 7);
                    ctx.builder.drop_();
                    ctx.builder.const_i32(32 - bits as i32);
                    ctx.builder.shl_i32();
                    ctx.builder.const_i32(32 - bits as i32);
                    ctx.builder.shr_s_i32();
                    ctx.builder.get_local(&count_local);
                    ctx.builder.shr_s_i32();
                },
            }
        }
        let result = set_new_val(ctx, bits);

        if flags_dead {
            // the flag update is dead: skip it (with a variable count the register write
            // still happens, only the flags are skipped)
            return result;
        }

        if !count_is_constant {
            ctx.builder.get_local(&count_local);
            ctx.builder.if_void();
        }

        let shift_const = |n: i32| {
            move |ctx: &mut JitContext| {
                if w {
                    ctx.builder.const_i64(n as i64)
                }
                else {
                    ctx.builder.const_i32(n)
                }
            }
        };
        let msb = shift_const(bits as i32 - 1);

        if kind == 0 || kind == 1 {
            ctx.builder.const_i32(global_pointers::flags_changed as i32);
            ctx.builder
                .load_fixed_i32(global_pointers::flags_changed as u32);
            ctx.builder.const_i32(!(FLAG_CARRY | FLAG_OVERFLOW));
            ctx.builder.and_i32();
            ctx.builder.store_aligned_i32(0);
        }
        else {
            gen_set_last_result(ctx, bits, &result);
            gen_set_op_size_and_flags_changed(ctx, bits, FLAGS_ALL & !FLAG_CARRY & !FLAG_OVERFLOW);
        }

        // flags = flags & ~(cf|of) | cf | of << 11
        ctx.builder.const_i32(global_pointers::flags as i32);
        ctx.builder.load_fixed_i32(global_pointers::flags as u32);
        ctx.builder.const_i32(!(FLAG_CARRY | FLAG_OVERFLOW));
        ctx.builder.and_i32();
        // cf
        match kind {
            // rol: lsb of the result
            0 => gen_bit(ctx, w, &result, &shift_const(0)),
            // ror: msb of the result
            1 => gen_bit(ctx, w, &result, &msb),
            // shl: bit (bits - count) of x. 8/16-bit counts can exceed the operand
            // width (they are masked to 5 bits), where x's bit (bits - count) doesn't
            // exist: bit `bits` of the result is 0 there, like arith::shl8's
            // (x << count) >> 8 & 1
            4 if bits < 32 => gen_bit(ctx, false, &result, &shift_const(bits as i32)),
            4 => gen_bit(ctx, w, x, &|ctx: &mut JitContext| {
                ctx.builder.const_i32(bits as i32);
                ctx.builder.get_local(&count_local);
                ctx.builder.sub_i32();
                if w {
                    ctx.builder.extend_unsigned_i32_to_i64();
                }
            }),
            // sar: bit (count - 1) of the sign-extended value; for 8/16-bit operands a
            // count larger than the width makes every shifted-out bit the sign (like
            // arith::sar8's count >= bits branch)
            7 if bits < 32 => {
                x.get(ctx);
                ctx.builder.const_i32(32 - bits as i32);
                ctx.builder.shl_i32();
                ctx.builder.const_i32(32 - bits as i32);
                ctx.builder.shr_s_i32();
                ctx.builder.get_local(&count_local);
                ctx.builder.const_i32(1);
                ctx.builder.sub_i32();
                ctx.builder.shr_s_i32();
                ctx.builder.const_i32(1);
                ctx.builder.and_i32();
            },
            // shr/sar: bit (count - 1) of x
            _ => gen_bit(ctx, w, x, &|ctx: &mut JitContext| {
                ctx.builder.get_local(&count_local);
                ctx.builder.const_i32(1);
                ctx.builder.sub_i32();
                if w {
                    ctx.builder.extend_unsigned_i32_to_i64();
                }
            }),
        }
        let cf = ctx.builder.tee_new_local();
        ctx.builder.or_i32();
        // of
        let has_of = kind != 7;
        if has_of {
            match kind {
                // rol: cf ^ msb(result)
                0 | 4 => {
                    ctx.builder.get_local(&cf);
                    gen_bit(ctx, w, &result, &msb);
                    ctx.builder.xor_i32();
                },
                // ror: msb(result) ^ msb-1(result)
                1 => {
                    gen_bit(ctx, w, &result, &msb);
                    gen_bit(ctx, w, &result, &shift_const(bits as i32 - 2));
                    ctx.builder.xor_i32();
                },
                // shr: msb(x)
                _ => gen_bit(ctx, w, x, &msb),
            }
            ctx.builder.const_i32(11);
            ctx.builder.shl_i32();
            ctx.builder.or_i32();
        }
        ctx.builder.free_local(cf);
        ctx.builder.store_aligned_i32(0);

        if !count_is_constant {
            ctx.builder.block_end();
        }
        result
    };
    ctx.flags64 = Flags64::Unknown;
    match dst {
        Opnd::Reg(r) => {
            gen_get_reg(ctx, bits, r);
            let x = set_new_val(ctx, bits);
            let result = compute(ctx, &x);
            gen_set_reg(ctx, bits, r, &result);
            result.free(ctx);
            x.free(ctx);
        },
        Opnd::Mem(m) => {
            let address = gen_modrm_address_local(ctx, &m);
            gen_safe_read_write(ctx, bits, &address, &|ctx| {
                let x = set_new_val(ctx, bits);
                let result = compute(ctx, &x);
                result.get(ctx);
                result.free(ctx);
                x.free(ctx);
            });
            ctx.builder.free_local_i64(address);
        },
        Opnd::Imm(_) => dbg_assert!(false),
    }
    ctx.builder.free_local(count_local);
}

/// imul r, r/m[, imm]. cf and of are set if the full product doesn't fit into the destination.
/// 32-bit: the product of two sign-extended 32-bit values always fits into an i64, so it's
/// computed inline. 64-bit: inline if both operands fit into 32 bits (then there's no
/// overflow), otherwise a helper computes the 128-bit product.
pub fn gen_imul(ctx: &mut JitContext, bits: u32, r: u32, src: Opnd, imm: Option<i64>) {
    dbg_assert!(bits == 32 || bits == 64);
    // sets the flags eagerly (and the wide multiply runs in the interpreter): commit first
    gen_commit_deferred_flags(ctx);
    // skip the flag update when the following instructions overwrite all flags before
    // anything reads them (same argument as the dead cf save in gen_save_cf). Only for the
    // inline 32-bit path: the 64-bit path's interpreter call sets the flags regardless.
    let flags_dead = bits == 32 && next_instructions_overwrite_flags(ctx);
    gen_get_operand(ctx, bits, &src);
    let a = set_new_val(ctx, bits);
    match imm {
        Some(i) => gen_const(ctx, bits, i),
        None => gen_get_reg(ctx, bits, r),
    }
    let b = set_new_val(ctx, bits);

    if bits == 32 {
        a.get(ctx);
        ctx.builder.extend_signed_i32_to_i64();
        b.get(ctx);
        ctx.builder.extend_signed_i32_to_i64();
        ctx.builder.mul_i64();
        let product = ctx.builder.tee_new_local_i64();
        ctx.builder.wrap_i64_to_i32();
        let result = set_new_val(ctx, 32);

        if !flags_dead {
            gen_set_last_result(ctx, 32, &result);
            // flags = flags & ~(cf | of) | (product != sign_extend(result) ? cf | of : 0)
            ctx.builder.const_i32(global_pointers::flags as i32);
            ctx.builder.load_fixed_i32(global_pointers::flags as u32);
            ctx.builder.const_i32(!(FLAG_CARRY | FLAG_OVERFLOW));
            ctx.builder.and_i32();
            ctx.builder.const_i32(FLAG_CARRY | FLAG_OVERFLOW);
            ctx.builder.const_i32(0);
            ctx.builder.get_local_i64(&product);
            result.get(ctx);
            ctx.builder.extend_signed_i32_to_i64();
            ctx.builder.ne_i64();
            ctx.builder.select();
            ctx.builder.or_i32();
            ctx.builder.store_aligned_i32(0);
            gen_set_op_size_and_flags_changed(ctx, 32, FLAGS_ALL & !FLAG_CARRY & !FLAG_OVERFLOW);
        }
        ctx.builder.free_local_i64(product);

        gen_set_reg(ctx, 32, r, &result);
        result.free(ctx);
    }
    else {
        let fits_in_32 = |ctx: &mut JitContext, x: &Val| {
            x.get(ctx);
            x.get(ctx);
            ctx.builder.wrap_i64_to_i32();
            ctx.builder.extend_signed_i32_to_i64();
            ctx.builder.eq_i64();
        };
        fits_in_32(ctx, &a);
        if imm.map_or(true, |i| i as i32 as i64 != i) {
            fits_in_32(ctx, &b);
            ctx.builder.and_i32();
        }
        ctx.builder.if_i64();
        {
            a.get(ctx);
            b.get(ctx);
            ctx.builder.mul_i64();
            let result = set_new_val(ctx, 64);
            gen_set_last_result(ctx, 64, &result);
            gen_clear_flags(ctx, FLAG_CARRY | FLAG_OVERFLOW);
            gen_set_op_size_and_flags_changed(ctx, 64, FLAGS_ALL & !FLAG_CARRY & !FLAG_OVERFLOW);
            result.get(ctx);
            result.free(ctx);
        }
        ctx.builder.else_();
        {
            a.get(ctx);
            b.get(ctx);
            ctx.builder.call_fn2_i64_i64_ret_i64("jit64_imul64");
        }
        ctx.builder.block_end();
        gen_set_reg_from_stack(ctx, 64, r);
    }
    b.free(ctx);
    a.free(ctx);
    ctx.flags64 = Flags64::Unknown;
}

#[no_mangle]
pub unsafe fn jit64_imul64(a: u64, b: u64) -> u64 { crate::cpu::arith::imul_reg64(a, b) }

/// setcc r/m8
pub fn gen_setcc(ctx: &mut JitContext, condition: u8, dst: Opnd) {
    gen_condition_fn(ctx, 0x80 | condition);
    let value = set_new_val(ctx, 8);
    match dst {
        Opnd::Reg(r) => gen_set_reg(ctx, 8, r, &value),
        Opnd::Mem(m) => {
            let address = gen_modrm_address_local(ctx, &m);
            gen_safe_write(ctx, 8, &address, &value);
            ctx.builder.free_local_i64(address);
        },
        Opnd::Imm(_) => dbg_assert!(false),
    }
    value.free(ctx);
}

/// cmovcc r, r/m: the source is always read; a 32-bit destination is zero-extended even if the
/// condition is false
pub fn gen_cmovcc(ctx: &mut JitContext, condition: u8, bits: u32, r: u32, src: Opnd) {
    gen_get_operand(ctx, bits, &src);
    let value = set_new_val(ctx, bits);
    gen_get_reg(ctx, bits, r);
    let old = set_new_val(ctx, bits);
    value.get(ctx);
    old.get(ctx);
    gen_condition_fn(ctx, 0x80 | condition);
    ctx.builder.select();
    let result = set_new_val(ctx, bits);
    gen_set_reg(ctx, bits, r, &result);
    result.free(ctx);
    old.free(ctx);
    value.free(ctx);
}

/// shift/rotate by a constant 0: no flags change, but a 32-bit register is written (and so
/// zero-extended), like the interpreter does
pub fn gen_shift_by_zero(ctx: &mut JitContext, bits: u32, dst: Opnd) {
    match dst {
        Opnd::Reg(r) => {
            if bits == 32 {
                gen_alu(ctx, OP_MOV, 32, Opnd::Reg(r), Opnd::Reg(r));
            }
        },
        Opnd::Mem(m) => {
            // A shift by zero doesn't change the value, but the operand is
            // still read and written back (like the interpreter's
            // safe_read_write*): it faults on unmapped/read-only memory and
            // dirties the page
            let address = gen_modrm_address_local(ctx, &m);
            gen_safe_read_write(ctx, bits, &address, &|ctx| {
                let x = set_new_val(ctx, bits);
                x.get(ctx);
                x.free(ctx);
            });
            ctx.builder.free_local_i64(address);
        },
        Opnd::Imm(_) => dbg_assert!(false),
    }
}

/// flags = flags & ~cf | getcf(), inline when the last flags operation is known
fn gen_save_cf(ctx: &mut JitContext) {
    let known = match ctx.flags64 {
        Flags64::Sub(b) | Flags64::Add(b) | Flags64::Logic(b) => b >= 32,
        Flags64::Unknown => false,
        // inc/dec only defer (gen_incdec); gen_save_cf is the eager path
        Flags64::IncDec { .. } => {
            dbg_assert!(false);
            false
        },
    };
    if next_instructions_overwrite_flags(ctx) {
        // the saved cf would be dead
        return;
    }
    if !known {
        gen_profile_count(ctx, "save_cf (inc/dec)");
        ctx.builder.const_i32(global_pointers::flags as i32);
        ctx.builder.load_fixed_i32(global_pointers::flags as u32);
        ctx.builder.const_i32(!FLAG_CARRY);
        ctx.builder.and_i32();
        gen_getcf_generic(ctx);
        ctx.builder.or_i32();
        ctx.builder.store_aligned_i32(0);
        return;
    }
    ctx.builder.const_i32(global_pointers::flags as i32);
    ctx.builder.load_fixed_i32(global_pointers::flags as u32);
    ctx.builder.const_i32(!FLAG_CARRY);
    ctx.builder.and_i32();
    if !matches!(ctx.flags64, Flags64::Logic(_)) {
        // cf is condition "b" (2)
        let ok = gen_condition_inline(ctx, 2);
        dbg_assert!(ok);
        ctx.builder.or_i32();
    }
    ctx.builder.store_aligned_i32(0);
}

/// not (2) and neg (3) r/m
pub fn gen_not_neg(ctx: &mut JitContext, is_neg: bool, bits: u32, dst: Opnd) {
    let compute = |ctx: &mut JitContext, x: &Val| -> Val {
        if is_neg {
            gen_const(ctx, bits, 0);
            let zero = set_new_val(ctx, bits);
            let result = gen_alu_value(ctx, OP_SUB, bits, &zero, x);
            zero.free(ctx);
            result
        }
        else {
            x.get(ctx);
            gen_const(ctx, bits, -1);
            if bits == 64 {
                ctx.builder.xor_i64();
            }
            else {
                ctx.builder.xor_i32();
            }
            set_new_val(ctx, bits)
        }
    };
    match dst {
        Opnd::Reg(r) => {
            gen_get_reg(ctx, bits, r);
            let x = set_new_val(ctx, bits);
            let result = compute(ctx, &x);
            gen_set_reg(ctx, bits, r, &result);
            result.free(ctx);
            x.free(ctx);
        },
        Opnd::Mem(m) => {
            let address = gen_modrm_address_local(ctx, &m);
            gen_safe_read_write(ctx, bits, &address, &|ctx| {
                let x = set_new_val(ctx, bits);
                let result = compute(ctx, &x);
                result.get(ctx);
                result.free(ctx);
                x.free(ctx);
            });
            ctx.builder.free_local_i64(address);
        },
        Opnd::Imm(_) => dbg_assert!(false),
    }
}

/// cbw/cwde/cdqe (0x98): sign-extend the lower half of rax into the operand size
pub fn gen_cbw(ctx: &mut JitContext, bits: u32) {
    match bits {
        16 => {
            gen_get_reg(ctx, 8, crate::regs::EAX);
            ctx.builder.const_i32(24);
            ctx.builder.shl_i32();
            ctx.builder.const_i32(24);
            ctx.builder.shr_s_i32();
        },
        32 => {
            gen_get_reg(ctx, 16, crate::regs::EAX);
            ctx.builder.const_i32(16);
            ctx.builder.shl_i32();
            ctx.builder.const_i32(16);
            ctx.builder.shr_s_i32();
        },
        _ => {
            gen_get_reg(ctx, 32, crate::regs::EAX);
            ctx.builder.extend_signed_i32_to_i64();
        },
    }
    gen_set_reg_from_stack(ctx, bits, crate::regs::EAX);
}

/// cwd/cdq/cqo (0x99): rdx = sign of rax
pub fn gen_cwd(ctx: &mut JitContext, bits: u32) {
    gen_get_reg(ctx, bits, crate::regs::EAX);
    if bits == 64 {
        ctx.builder.const_i64(63);
        ctx.builder.shr_s_i64();
    }
    else {
        if bits == 16 {
            ctx.builder.const_i32(16);
            ctx.builder.shl_i32();
        }
        ctx.builder.const_i32(31);
        ctx.builder.shr_s_i32();
    }
    gen_set_reg_from_stack(ctx, bits, crate::regs::EDX);
}

fn gen_bswap32_on_stack(ctx: &mut JitContext) {
    let x = ctx.builder.set_new_local();
    ctx.builder.get_local(&x);
    ctx.builder.const_i32(8);
    ctx.builder.rotl_i32();
    ctx.builder.const_i32(0x00FF00FF);
    ctx.builder.and_i32();
    ctx.builder.get_local(&x);
    ctx.builder.const_i32(8);
    ctx.builder.rotr_i32();
    ctx.builder.const_i32(0xFF00FF00u32 as i32);
    ctx.builder.and_i32();
    ctx.builder.or_i32();
    ctx.builder.free_local(x);
}

/// bswap r32/r64
pub fn gen_bswap(ctx: &mut JitContext, bits: u32, r: u32) {
    if bits == 64 {
        // swap the halves and bswap each
        gen_get_reg(ctx, 64, r);
        ctx.builder.wrap_i64_to_i32();
        gen_bswap32_on_stack(ctx);
        ctx.builder.extend_unsigned_i32_to_i64();
        ctx.builder.const_i64(32);
        ctx.builder.shl_i64();
        gen_get_reg(ctx, 64, r);
        ctx.builder.const_i64(32);
        ctx.builder.shr_u_i64();
        ctx.builder.wrap_i64_to_i32();
        gen_bswap32_on_stack(ctx);
        ctx.builder.extend_unsigned_i32_to_i64();
        ctx.builder.or_i64();
    }
    else {
        gen_get_reg(ctx, 32, r);
        gen_bswap32_on_stack(ctx);
    }
    gen_set_reg_from_stack(ctx, bits, r);
}

/// call r/m64 (is_call) and jmp r/m64: the block glue continues at the new instruction pointer
pub fn gen_indirect_jump64(ctx: &mut JitContext, is_call: bool, src: Opnd) {
    gen_get_operand(ctx, 64, &src);
    let target = set_new_val(ctx, 64);
    if is_call {
        codegen::gen_get_eip64(ctx.builder);
        ctx.builder.const_i64(!0xFFF);
        ctx.builder.and_i64();
        ctx.builder.const_i64((ctx.cpu.eip & 0xFFF) as i64);
        ctx.builder.or_i64();
        let ret = set_new_val(ctx, 64);
        gen_push64_value(ctx, &ret);
        ret.free(ctx);
    }
    ctx.builder
        .const_i32(global_pointers::instruction_pointer as i32);
    target.get(ctx);
    ctx.builder.store_aligned_i64(0);
    target.free(ctx);
}

/// xchg r, r/m (the memory form is atomic on hardware, which doesn't matter here)
pub fn gen_xchg(ctx: &mut JitContext, bits: u32, r: u32, other: Opnd) {
    gen_get_reg(ctx, bits, r);
    let reg_value = set_new_val(ctx, bits);
    match other {
        Opnd::Reg(r2) => {
            gen_get_reg(ctx, bits, r2);
            let v2 = set_new_val(ctx, bits);
            gen_set_reg(ctx, bits, r2, &reg_value);
            gen_set_reg(ctx, bits, r, &v2);
            v2.free(ctx);
        },
        Opnd::Mem(m) => {
            let address = gen_modrm_address_local(ctx, &m);
            let old = ctx.builder.new_local();
            let old64 = ctx.builder.new_local_i64();
            let rv = &reg_value;
            let (old_ref, old64_ref) = (&old, &old64);
            gen_safe_read_write(ctx, bits, &address, &|ctx| {
                if bits == 64 {
                    ctx.builder.set_local_i64(old64_ref);
                }
                else {
                    ctx.builder.set_local(old_ref);
                }
                rv.get(ctx);
            });
            ctx.builder.free_local_i64(address);
            if bits == 64 {
                ctx.builder.get_local_i64(&old64);
            }
            else {
                ctx.builder.get_local(&old);
            }
            gen_set_reg_from_stack(ctx, bits, r);
            ctx.builder.free_local(old);
            ctx.builder.free_local_i64(old64);
        },
        Opnd::Imm(_) => dbg_assert!(false),
    }
    reg_value.free(ctx);
}

/// Whether the following instructions (within the current block) overwrite all arithmetic
/// flags before anything reads them. Looks at a few instructions, skipping those that neither
/// read nor write flags.
fn next_instructions_overwrite_flags(ctx: &JitContext) -> bool {
    let mut cpu = ctx.cpu.clone();
    for _ in 0..4 {
        if cpu.eip >= ctx.block_end || is_near_end_of_page(cpu.eip) {
            return false;
        }
        let start = cpu.eip;
        // prefixes
        let mut opcode;
        loop {
            opcode = cpu.read_imm8() as u32;
            match opcode {
                0x26 | 0x2E | 0x36 | 0x3E | 0x64 | 0x65 | 0x66 | 0x67 | 0xF0 | 0xF2 | 0xF3 => {},
                0x40..=0x4F => {},
                _ => break,
            }
        }
        if opcode == 0x0F {
            opcode = 0x0F00 | cpu.read_imm8() as u32;
        }
        // only forms without memory operands: a fault in a memory access would deliver the
        // exception with the (unsaved) carry flag of the inc/dec
        let writes_all = match opcode {
            0x00..=0x3F => {
                let op = opcode >> 3;
                op != 2
                    && op != 3
                    && (opcode & 7 == 4
                        || opcode & 7 == 5
                        || opcode & 7 < 4 && cpu.read_imm8() >= 0xC0)
            },
            0x80 | 0x81 | 0x83 => {
                let modrm = cpu.read_imm8();
                let g = modrm >> 3 & 7;
                modrm >= 0xC0 && g != 2 && g != 3
            },
            0x84 | 0x85 => cpu.read_imm8() >= 0xC0,
            0xA8 | 0xA9 => true,
            0xF6 | 0xF7 => {
                let modrm = cpu.read_imm8();
                modrm >= 0xC0 && modrm >> 3 & 7 == 0
            },
            _ => false,
        };
        if writes_all {
            return true;
        }
        // instructions that don't use the flags and can't fault (no memory or stack access)
        let neutral = match opcode {
            0x88..=0x8B | 0x63 | 0x0FB6 | 0x0FB7 | 0x0FBE | 0x0FBF => cpu.read_imm8() >= 0xC0,
            0x8D | 0xB0..=0xBF | 0x90 => true,
            _ => false,
        };
        if !neutral {
            return false;
        }
        // skip the rest of the instruction
        let mut analysis_cpu = ctx.cpu.clone();
        analysis_cpu.eip = start;
        analysis_cpu.prefixes = 0;
        let _ = crate::analysis::analyze_step(&mut analysis_cpu);
        cpu.eip = analysis_cpu.eip;
        cpu.prefixes = 0;
    }
    false
}

/// Push cf (i32 0/1) for the current lazy flags state, without changing it
fn gen_getcf(ctx: &mut JitContext) {
    if !gen_condition_inline(ctx, 2) {
        gen_getcf_generic(ctx);
    }
}

/// Push cf (i32 0/1) from the lazy flags state at runtime (see misc_instr::getcf)
fn gen_getcf_generic(ctx: &mut JitContext) {
    gen_commit_deferred_flags(ctx);
    ctx.builder
        .load_fixed_i32(global_pointers::flags_changed as u32);
    ctx.builder.const_i32(1);
    ctx.builder.and_i32();
    ctx.builder.if_i32();
    {
        // sub: last_op1 < last_result, add: last_result < last_op1 (unsigned), via a mask
        ctx.builder
            .load_fixed_i32(global_pointers::last_op_size as u32);
        ctx.builder.const_i32(OPSIZE_64);
        ctx.builder.eq_i32();
        ctx.builder.if_i32();
        {
            let mask = ctx.builder.new_local_i64();
            ctx.builder
                .load_fixed_i32(global_pointers::flags_changed as u32);
            ctx.builder.const_i32(31);
            ctx.builder.shr_s_i32();
            ctx.builder.extend_signed_i32_to_i64();
            ctx.builder.tee_local_i64(&mask);
            ctx.builder
                .load_fixed_i64(global_pointers::last_result_64 as u32);
            ctx.builder.xor_i64();
            ctx.builder.get_local_i64(&mask);
            ctx.builder
                .load_fixed_i64(global_pointers::last_op1_64 as u32);
            ctx.builder.xor_i64();
            ctx.builder.ltu_i64();
            ctx.builder.free_local_i64(mask);
        }
        ctx.builder.else_();
        {
            let mask = ctx.builder.new_local();
            ctx.builder
                .load_fixed_i32(global_pointers::flags_changed as u32);
            ctx.builder.const_i32(31);
            ctx.builder.shr_s_i32();
            ctx.builder.tee_local(&mask);
            ctx.builder
                .load_fixed_i32(global_pointers::last_result as u32);
            ctx.builder.xor_i32();
            ctx.builder.get_local(&mask);
            ctx.builder.load_fixed_i32(global_pointers::last_op1 as u32);
            ctx.builder.xor_i32();
            ctx.builder.ltu_i32();
            ctx.builder.free_local(mask);
        }
        ctx.builder.block_end();
    }
    ctx.builder.else_();
    {
        ctx.builder.load_fixed_i32(global_pointers::flags as u32);
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();
    }
    ctx.builder.block_end();
}
#[no_mangle]
pub unsafe fn jit64_getcf() -> i32 { crate::cpu::misc_instr::getcf() as i32 }

/// Push bit (bits - 1) of a value as i32 0/1
fn gen_msb_of(ctx: &mut JitContext, bits: u32) {
    if bits == 64 {
        ctx.builder.const_i64(63);
        ctx.builder.shr_u_i64();
        ctx.builder.wrap_i64_to_i32();
    }
    else {
        ctx.builder.const_i32(31);
        ctx.builder.shr_u_i32();
    }
}

/// adc/sbb dst, src (32 and 64 bits), see cpu/arith.rs adc/sbb
pub fn gen_adc_sbb(ctx: &mut JitContext, is_sbb: bool, bits: u32, dst: Opnd, src: Opnd) {
    dbg_assert!(bits == 32 || bits == 64);
    let w = bits == 64;
    // reads cf and sets the flags eagerly: commit the deferred state first
    gen_commit_deferred_flags(ctx);
    gen_get_operand(ctx, bits, &src);
    let y = set_new_val(ctx, bits);
    gen_getcf(ctx);
    let cf = ctx.builder.set_new_local();

    let compute = |ctx: &mut JitContext, x: &Val| -> Val {
        macro_rules! op {
            ($a:ident, $b:ident) => {
                if w {
                    ctx.builder.$b()
                }
                else {
                    ctx.builder.$a()
                }
            };
        }
        // res = x +- y +- cf
        x.get(ctx);
        y.get(ctx);
        if is_sbb {
            op!(sub_i32, sub_i64);
        }
        else {
            op!(add_i32, add_i64);
        }
        ctx.builder.get_local(&cf);
        if w {
            ctx.builder.extend_unsigned_i32_to_i64();
        }
        if is_sbb {
            op!(sub_i32, sub_i64);
        }
        else {
            op!(add_i32, add_i64);
        }
        let res = set_new_val(ctx, bits);

        gen_set_last_op1(ctx, bits, x);
        gen_set_last_result(ctx, bits, &res);
        gen_set_op_size_and_flags_changed(
            ctx,
            bits,
            FLAGS_ALL & !FLAG_CARRY & !FLAG_ADJUST & !FLAG_OVERFLOW
                | if is_sbb { FLAG_SUB } else { 0 },
        );

        ctx.builder.const_i32(global_pointers::flags as i32);
        ctx.builder.load_fixed_i32(global_pointers::flags as u32);
        ctx.builder
            .const_i32(!(FLAG_CARRY | FLAG_ADJUST | FLAG_OVERFLOW));
        ctx.builder.and_i32();
        // cf: adc: msb(x ^ ((x ^ y) & (y ^ res))), sbb: msb(res ^ ((res ^ y) & (y ^ x)))
        let (a, b) = if is_sbb { (&res, x) } else { (x, &res) };
        a.get(ctx);
        a.get(ctx);
        y.get(ctx);
        op!(xor_i32, xor_i64);
        y.get(ctx);
        b.get(ctx);
        op!(xor_i32, xor_i64);
        op!(and_i32, and_i64);
        op!(xor_i32, xor_i64);
        gen_msb_of(ctx, bits);
        ctx.builder.or_i32();
        // af: (x ^ y ^ res) & 0x10
        x.get(ctx);
        y.get(ctx);
        op!(xor_i32, xor_i64);
        res.get(ctx);
        op!(xor_i32, xor_i64);
        if w {
            ctx.builder.wrap_i64_to_i32();
        }
        ctx.builder.const_i32(FLAG_ADJUST);
        ctx.builder.and_i32();
        ctx.builder.or_i32();
        // of: adc: msb((y ^ res) & (x ^ res)), sbb: msb((y ^ x) & (res ^ x))
        let (p, q) = if is_sbb { (x, &res) } else { (&res, x) };
        y.get(ctx);
        p.get(ctx);
        op!(xor_i32, xor_i64);
        q.get(ctx);
        p.get(ctx);
        op!(xor_i32, xor_i64);
        op!(and_i32, and_i64);
        gen_msb_of(ctx, bits);
        ctx.builder.const_i32(11);
        ctx.builder.shl_i32();
        ctx.builder.or_i32();
        ctx.builder.store_aligned_i32(0);
        res
    };
    gen_rmw(ctx, bits, dst, &compute);
    ctx.builder.free_local(cf);
    y.free(ctx);
    ctx.flags64 = Flags64::Unknown;
}

/// Apply compute to a register or memory operand and write the result back
fn gen_rmw(
    ctx: &mut JitContext,
    bits: u32,
    dst: Opnd,
    compute: &dyn Fn(&mut JitContext, &Val) -> Val,
) {
    match dst {
        Opnd::Reg(r) => {
            gen_get_reg(ctx, bits, r);
            let x = set_new_val(ctx, bits);
            let result = compute(ctx, &x);
            gen_set_reg(ctx, bits, r, &result);
            result.free(ctx);
            x.free(ctx);
        },
        Opnd::Mem(m) => {
            let address = gen_modrm_address_local(ctx, &m);
            gen_safe_read_write(ctx, bits, &address, &|ctx| {
                let x = set_new_val(ctx, bits);
                let result = compute(ctx, &x);
                result.get(ctx);
                result.free(ctx);
                x.free(ctx);
            });
            ctx.builder.free_local_i64(address);
        },
        Opnd::Imm(_) => dbg_assert!(false),
    }
}

/// bt (4), bts (5), btr (6), btc (7) with a constant bit offset, or a register offset with a
/// register destination
pub fn gen_bt(ctx: &mut JitContext, kind: u32, bits: u32, dst: Opnd, offset: Opnd) {
    dbg_assert!(bits == 32 || bits == 64);
    dbg_assert!(!(matches!(dst, Opnd::Mem(_)) && matches!(offset, Opnd::Reg(_))));
    let w = bits == 64;
    // rmw's the flags word eagerly: commit the deferred state first
    gen_commit_deferred_flags(ctx);
    // offset & (bits - 1), as i32. An immediate offset is masked to the operand size for
    // memory operands too (only register offsets address a bit string, and those run in
    // the interpreter)
    let off = ctx.builder.new_local();
    match offset {
        Opnd::Imm(i) => ctx.builder.const_i32((i as i32) & (bits as i32 - 1)),
        Opnd::Reg(r) => {
            gen_get_reg(ctx, 32, r);
            ctx.builder.const_i32(bits as i32 - 1);
            ctx.builder.and_i32();
        },
        Opnd::Mem(_) => dbg_assert!(false),
    }
    ctx.builder.set_local(&off);

    let compute = |ctx: &mut JitContext, x: &Val| -> Val {
        // cf = x >> off & 1
        ctx.builder.const_i32(global_pointers::flags as i32);
        ctx.builder.load_fixed_i32(global_pointers::flags as u32);
        ctx.builder.const_i32(!FLAG_CARRY);
        ctx.builder.and_i32();
        x.get(ctx);
        ctx.builder.get_local(&off);
        if w {
            ctx.builder.extend_unsigned_i32_to_i64();
            ctx.builder.shr_u_i64();
            ctx.builder.wrap_i64_to_i32();
        }
        else {
            ctx.builder.shr_u_i32();
        }
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();
        ctx.builder.or_i32();
        ctx.builder.store_aligned_i32(0);
        ctx.builder.const_i32(global_pointers::flags_changed as i32);
        ctx.builder
            .load_fixed_i32(global_pointers::flags_changed as u32);
        ctx.builder.const_i32(!FLAG_CARRY);
        ctx.builder.and_i32();
        ctx.builder.store_aligned_i32(0);

        // the new value: x |/&~/^ (1 << off)
        x.get(ctx);
        if kind != 4 {
            if w {
                ctx.builder.const_i64(1);
                ctx.builder.get_local(&off);
                ctx.builder.extend_unsigned_i32_to_i64();
                ctx.builder.shl_i64();
                if kind == 6 {
                    ctx.builder.const_i64(-1);
                    ctx.builder.xor_i64();
                    ctx.builder.and_i64();
                }
                else if kind == 5 {
                    ctx.builder.or_i64();
                }
                else {
                    ctx.builder.xor_i64();
                }
            }
            else {
                ctx.builder.const_i32(1);
                ctx.builder.get_local(&off);
                ctx.builder.shl_i32();
                if kind == 6 {
                    ctx.builder.const_i32(-1);
                    ctx.builder.xor_i32();
                    ctx.builder.and_i32();
                }
                else if kind == 5 {
                    ctx.builder.or_i32();
                }
                else {
                    ctx.builder.xor_i32();
                }
            }
        }
        set_new_val(ctx, bits)
    };
    if kind == 4 {
        // bt doesn't write
        gen_get_operand(ctx, bits, &dst);
        let x = set_new_val(ctx, bits);
        let v = compute(ctx, &x);
        v.free(ctx);
        x.free(ctx);
    }
    else {
        gen_rmw(ctx, bits, dst, &compute);
    }
    ctx.builder.free_local(off);
    ctx.flags64 = Flags64::Unknown;
}

/// cmpxchg r/m, r: compare rax with the destination; equal: dst = r, else rax = dst. The
/// destination is written in both cases (like the interpreter)
pub fn gen_cmpxchg(ctx: &mut JitContext, bits: u32, dst: Opnd, r: u32) {
    gen_get_reg(ctx, bits, crate::regs::EAX);
    let rax = set_new_val(ctx, bits);
    gen_get_reg(ctx, bits, r);
    let src = set_new_val(ctx, bits);
    let old = ctx.builder.new_local();
    let old64 = ctx.builder.new_local_i64();
    let equal = ctx.builder.new_local();
    {
        let (rax, src, old, old64, equal) = (&rax, &src, &old, &old64, &equal);
        let compute = |ctx: &mut JitContext, x: &Val| -> Val {
            match x {
                Val::I32(_) => {
                    x.get(ctx);
                    ctx.builder.set_local(old);
                },
                Val::I64(_) => {
                    x.get(ctx);
                    ctx.builder.set_local_i64(old64);
                },
            }
            let result = gen_alu_value(ctx, OP_CMP, bits, rax, x);
            result.free(ctx);
            rax.get(ctx);
            x.get(ctx);
            if bits == 64 {
                ctx.builder.eq_i64();
            }
            else {
                ctx.builder.eq_i32();
            }
            ctx.builder.set_local(equal);
            src.get(ctx);
            x.get(ctx);
            ctx.builder.get_local(equal);
            ctx.builder.select();
            set_new_val(ctx, bits)
        };
        gen_rmw(ctx, bits, dst, &compute);
    }
    // not equal: rax = old value
    ctx.builder.get_local(&equal);
    ctx.builder.eqz_i32();
    ctx.builder.if_void();
    if bits == 64 {
        ctx.builder.get_local_i64(&old64);
    }
    else {
        ctx.builder.get_local(&old);
    }
    gen_set_reg_from_stack(ctx, bits, crate::regs::EAX);
    ctx.builder.block_end();
    ctx.builder.free_local(old);
    ctx.builder.free_local_i64(old64);
    ctx.builder.free_local(equal);
    src.free(ctx);
    rax.free(ctx);
    ctx.flags64 = Flags64::Sub(bits);
}

/// xadd r/m, r: dst = dst + r, r = old dst
pub fn gen_xadd(ctx: &mut JitContext, bits: u32, dst: Opnd, r: u32) {
    gen_get_reg(ctx, bits, r);
    let reg_value = set_new_val(ctx, bits);
    let old = ctx.builder.new_local();
    let old64 = ctx.builder.new_local_i64();
    // when dst and r alias (xadd r, r) the destination write happens last:
    // the register ends up with the sum, not the old value
    let aliased = matches!(dst, Opnd::Reg(dr) if dr == r);
    {
        let (reg_value, old, old64) = (&reg_value, &old, &old64);
        let compute = |ctx: &mut JitContext, x: &Val| -> Val {
            x.get(ctx);
            if bits == 64 {
                ctx.builder.set_local_i64(old64);
            }
            else {
                ctx.builder.set_local(old);
            }
            gen_alu_value(ctx, OP_ADD, bits, x, reg_value)
        };
        gen_rmw(ctx, bits, dst, &compute);
    }
    if !aliased {
        if bits == 64 {
            ctx.builder.get_local_i64(&old64);
        }
        else {
            ctx.builder.get_local(&old);
        }
        gen_set_reg_from_stack(ctx, bits, r);
    }
    ctx.builder.free_local(old);
    ctx.builder.free_local_i64(old64);
    reg_value.free(ctx);
    ctx.flags64 = Flags64::Add(bits);
}

/// pushf (64-bit)
pub fn gen_pushf64(ctx: &mut JitContext) {
    // no #gp in 64-bit mode (vm86 is impossible); get_eflags reads the lazy flags from memory
    gen_commit_deferred_flags(ctx);
    ctx.builder.call_fn0_ret("jit64_get_eflags");
    ctx.builder.const_i32(0xFCFFFF);
    ctx.builder.and_i32();
    ctx.builder.extend_unsigned_i32_to_i64();
    let value = set_new_val(ctx, 64);
    gen_push64_value(ctx, &value);
    value.free(ctx);
}
#[no_mangle]
pub unsafe fn jit64_get_eflags() -> i32 { cpu::get_eflags() }
