// 64-bit instruction implementations (long mode).
// Filled in incrementally; unimplemented ones panic (only reachable in 64-bit mode).
#![allow(unused_variables)]
#![allow(non_snake_case)]
#![allow(clippy::all)]

use crate::cpu::arith::*;
use crate::cpu::cpu::*;
use crate::cpu::global_pointers::*;
use crate::cpu::misc_instr::*;
use crate::cpu::string::{
    cmpsq_no_rep, cmpsq_repnz, cmpsq_repz, insd_no_rep, lodsq_no_rep, lodsq_rep, movsq_no_rep,
    movsq_rep, outsd_no_rep, scasq_no_rep, scasq_repnz, scasq_repz, stosq_no_rep, stosq_rep,
};
use crate::prefix;

pub unsafe fn instr64_01_mem(addr: u64, r: i32) {
    safe_read_write64(addr, &|x| add64(x, read_reg64(r)))
}
pub unsafe fn instr64_01_reg(r1: i32, r: i32) {
    write_reg64(r1, add64(read_reg64(r1), read_reg64(r)));
}
pub unsafe fn instr64_03_mem(addr: u64, r: i32) {
    write_reg64(
        r,
        add64(read_reg64(r), return_on_pagefault!(safe_read64s(addr))),
    );
}
pub unsafe fn instr64_03_reg(r1: i32, r: i32) {
    write_reg64(r, add64(read_reg64(r), read_reg64(r1)));
}
pub unsafe fn instr64_05(a0: i32) { write_reg64(EAX, add64(read_reg64(EAX), a0 as i64 as u64)); }
pub unsafe fn instr64_06() { trigger_ud(); }
pub unsafe fn instr64_07() { trigger_ud(); }
pub unsafe fn instr64_09_mem(addr: u64, r: i32) {
    safe_read_write64(addr, &|x| or64(x, read_reg64(r)))
}
pub unsafe fn instr64_09_reg(r1: i32, r: i32) {
    write_reg64(r1, or64(read_reg64(r1), read_reg64(r)));
}
pub unsafe fn instr64_0B_mem(addr: u64, r: i32) {
    write_reg64(
        r,
        or64(read_reg64(r), return_on_pagefault!(safe_read64s(addr))),
    );
}
pub unsafe fn instr64_0B_reg(r1: i32, r: i32) {
    write_reg64(r, or64(read_reg64(r), read_reg64(r1)));
}
pub unsafe fn instr64_0D(a0: i32) { write_reg64(EAX, or64(read_reg64(EAX), a0 as i64 as u64)); }
pub unsafe fn instr64_0E() { trigger_ud(); }
pub unsafe fn instr64_0F() {
    // 0F escape with REX.W: dispatch into the 64-bit tier of the 0F table
    let opcode = return_on_pagefault!(read_imm8());
    crate::gen::interpreter0f::run(opcode as u32 | 0x200);
}
pub unsafe fn instr64_0F00_0_mem(addr: u64) {
    crate::cpu::instructions_0f::instr16_0F00_0_mem(addr)
}
pub unsafe fn instr64_0F00_0_reg(r: i32) { crate::cpu::instructions_0f::instr16_0F00_0_reg(r) }
pub unsafe fn instr64_0F00_1_mem(addr: u64) {
    crate::cpu::instructions_0f::instr16_0F00_1_mem(addr)
}
pub unsafe fn instr64_0F00_1_reg(r: i32) { crate::cpu::instructions_0f::instr16_0F00_1_reg(r) }
pub unsafe fn instr64_0F00_2_mem(addr: u64) {
    crate::cpu::instructions_0f::instr16_0F00_2_mem(addr)
}
pub unsafe fn instr64_0F00_2_reg(r: i32) { crate::cpu::instructions_0f::instr16_0F00_2_reg(r) }
pub unsafe fn instr64_0F00_3_mem(addr: u64) {
    crate::cpu::instructions_0f::instr16_0F00_3_mem(addr)
}
pub unsafe fn instr64_0F00_3_reg(r: i32) { crate::cpu::instructions_0f::instr16_0F00_3_reg(r) }
pub unsafe fn instr64_0F00_4_mem(addr: u64) {
    crate::cpu::instructions_0f::instr16_0F00_4_mem(addr)
}
pub unsafe fn instr64_0F00_4_reg(r: i32) { crate::cpu::instructions_0f::instr16_0F00_4_reg(r) }
pub unsafe fn instr64_0F00_5_mem(addr: u64) {
    crate::cpu::instructions_0f::instr16_0F00_5_mem(addr)
}
pub unsafe fn instr64_0F00_5_reg(r: i32) { crate::cpu::instructions_0f::instr16_0F00_5_reg(r) }
pub unsafe fn instr64_0F01_0_mem(addr: u64) {
    // sgdt: 10-byte pseudo descriptor in 64-bit mode (8-byte offset, 2-byte limit)
    if 0 != *cpl {
        trigger_gp(0);
        return;
    }
    return_on_pagefault!(safe_write64(addr, *gdtr_offset64));
    return_on_pagefault!(safe_write16(addr + 8, *gdtr_size));
}
pub unsafe fn instr64_0F01_0_reg(_a0: i32) { trigger_ud(); }
pub unsafe fn instr64_0F01_1_mem(addr: u64) {
    // sidt
    if 0 != *cpl {
        trigger_gp(0);
        return;
    }
    return_on_pagefault!(safe_write64(addr, *idtr_offset64));
    return_on_pagefault!(safe_write16(addr + 8, *idtr_size));
}
pub unsafe fn instr64_0F01_1_reg(_a0: i32) { trigger_ud(); }
pub unsafe fn instr64_0F01_2_mem(addr: u64) {
    // lgdt: 10-byte pseudo descriptor in 64-bit mode
    if 0 != *cpl {
        trigger_gp(0);
        return;
    }
    let limit = return_on_pagefault!(safe_read16(addr));
    let offset = return_on_pagefault!(safe_read64s(addr + 2));
    *gdtr_size = limit;
    *gdtr_offset = offset as u32 as i32;
    *gdtr_offset64 = offset;
}
pub unsafe fn instr64_0F01_2_reg(_a0: i32) { trigger_ud(); }
pub unsafe fn instr64_0F01_3_mem(addr: u64) {
    // lidt
    if 0 != *cpl {
        trigger_gp(0);
        return;
    }
    let limit = return_on_pagefault!(safe_read16(addr));
    let offset = return_on_pagefault!(safe_read64s(addr + 2));
    *idtr_size = limit;
    *idtr_offset = offset as u32 as i32;
    *idtr_offset64 = offset;
}
pub unsafe fn instr64_0F01_3_reg(_a0: i32) { trigger_ud(); }
pub unsafe fn instr64_0F01_4_mem(addr: u64) {
    crate::cpu::instructions_0f::instr32_0F01_4_mem(addr)
}
pub unsafe fn instr64_0F01_4_reg(_a0: i32) { trigger_ud(); }
pub unsafe fn instr64_0F01_6_mem(addr: u64) {
    crate::cpu::instructions_0f::instr32_0F01_6_mem(addr)
}
pub unsafe fn instr64_0F01_6_reg(_a0: i32) { trigger_ud(); }
pub unsafe fn instr64_0F01_7_mem(addr: u64) {
    // invlpg
    if 0 != *cpl {
        trigger_gp(0);
        return;
    }
    invlpg(addr);
}
pub unsafe fn instr64_0F01_7_reg(r: i32) {
    if r == 0 {
        // swapgs (0F 01 F8): exchange gs_base and kernel_gs_base
        if 0 != *cpl {
            trigger_gp(0);
            return;
        }
        let base = *gs_base;
        *gs_base = *kernel_gs_base;
        *kernel_gs_base = base;
    }
    else {
        trigger_ud();
    }
}
pub unsafe fn instr64_0F02_mem(addr: u64, r: i32) {
    write_reg64(
        r,
        lar(return_on_pagefault!(safe_read16(addr)), read_reg32(r)) as u32 as u64,
    );
}
pub unsafe fn instr64_0F02_reg(r1: i32, r: i32) {
    write_reg64(r, lar(read_reg32(r1), read_reg32(r)) as u32 as u64);
}
pub unsafe fn instr64_0F03_mem(addr: u64, r: i32) {
    write_reg64(
        r,
        lsl(return_on_pagefault!(safe_read16(addr)), read_reg32(r)) as u32 as u64,
    );
}
pub unsafe fn instr64_0F03_reg(r1: i32, r: i32) {
    write_reg64(r, lsl(read_reg32(r1), read_reg32(r)) as u32 as u64);
}
pub unsafe fn instr64_0F40_mem(addr: u64, r: i32) {
    cmovcc64(test_o(), return_on_pagefault!(safe_read64s(addr)), r);
}
pub unsafe fn instr64_0F40_reg(r1: i32, r: i32) { cmovcc64(test_o(), read_reg64(r1), r); }
pub unsafe fn instr64_0F41_mem(addr: u64, r: i32) {
    cmovcc64(!test_o(), return_on_pagefault!(safe_read64s(addr)), r);
}
pub unsafe fn instr64_0F41_reg(r1: i32, r: i32) { cmovcc64(!test_o(), read_reg64(r1), r); }
pub unsafe fn instr64_0F42_mem(addr: u64, r: i32) {
    cmovcc64(test_b(), return_on_pagefault!(safe_read64s(addr)), r);
}
pub unsafe fn instr64_0F42_reg(r1: i32, r: i32) { cmovcc64(test_b(), read_reg64(r1), r); }
pub unsafe fn instr64_0F43_mem(addr: u64, r: i32) {
    cmovcc64(!test_b(), return_on_pagefault!(safe_read64s(addr)), r);
}
pub unsafe fn instr64_0F43_reg(r1: i32, r: i32) { cmovcc64(!test_b(), read_reg64(r1), r); }
pub unsafe fn instr64_0F44_mem(addr: u64, r: i32) {
    cmovcc64(test_z(), return_on_pagefault!(safe_read64s(addr)), r);
}
pub unsafe fn instr64_0F44_reg(r1: i32, r: i32) { cmovcc64(test_z(), read_reg64(r1), r); }
pub unsafe fn instr64_0F45_mem(addr: u64, r: i32) {
    cmovcc64(!test_z(), return_on_pagefault!(safe_read64s(addr)), r);
}
pub unsafe fn instr64_0F45_reg(r1: i32, r: i32) { cmovcc64(!test_z(), read_reg64(r1), r); }
pub unsafe fn instr64_0F46_mem(addr: u64, r: i32) {
    cmovcc64(test_be(), return_on_pagefault!(safe_read64s(addr)), r);
}
pub unsafe fn instr64_0F46_reg(r1: i32, r: i32) { cmovcc64(test_be(), read_reg64(r1), r); }
pub unsafe fn instr64_0F47_mem(addr: u64, r: i32) {
    cmovcc64(!test_be(), return_on_pagefault!(safe_read64s(addr)), r);
}
pub unsafe fn instr64_0F47_reg(r1: i32, r: i32) { cmovcc64(!test_be(), read_reg64(r1), r); }
pub unsafe fn instr64_0F48_mem(addr: u64, r: i32) {
    cmovcc64(test_s(), return_on_pagefault!(safe_read64s(addr)), r);
}
pub unsafe fn instr64_0F48_reg(r1: i32, r: i32) { cmovcc64(test_s(), read_reg64(r1), r); }
pub unsafe fn instr64_0F49_mem(addr: u64, r: i32) {
    cmovcc64(!test_s(), return_on_pagefault!(safe_read64s(addr)), r);
}
pub unsafe fn instr64_0F49_reg(r1: i32, r: i32) { cmovcc64(!test_s(), read_reg64(r1), r); }
pub unsafe fn instr64_0F4A_mem(addr: u64, r: i32) {
    cmovcc64(test_p(), return_on_pagefault!(safe_read64s(addr)), r);
}
pub unsafe fn instr64_0F4A_reg(r1: i32, r: i32) { cmovcc64(test_p(), read_reg64(r1), r); }
pub unsafe fn instr64_0F4B_mem(addr: u64, r: i32) {
    cmovcc64(!test_p(), return_on_pagefault!(safe_read64s(addr)), r);
}
pub unsafe fn instr64_0F4B_reg(r1: i32, r: i32) { cmovcc64(!test_p(), read_reg64(r1), r); }
pub unsafe fn instr64_0F4C_mem(addr: u64, r: i32) {
    cmovcc64(test_l(), return_on_pagefault!(safe_read64s(addr)), r);
}
pub unsafe fn instr64_0F4C_reg(r1: i32, r: i32) { cmovcc64(test_l(), read_reg64(r1), r); }
pub unsafe fn instr64_0F4D_mem(addr: u64, r: i32) {
    cmovcc64(!test_l(), return_on_pagefault!(safe_read64s(addr)), r);
}
pub unsafe fn instr64_0F4D_reg(r1: i32, r: i32) { cmovcc64(!test_l(), read_reg64(r1), r); }
pub unsafe fn instr64_0F4E_mem(addr: u64, r: i32) {
    cmovcc64(test_le(), return_on_pagefault!(safe_read64s(addr)), r);
}
pub unsafe fn instr64_0F4E_reg(r1: i32, r: i32) { cmovcc64(test_le(), read_reg64(r1), r); }
pub unsafe fn instr64_0F4F_mem(addr: u64, r: i32) {
    cmovcc64(!test_le(), return_on_pagefault!(safe_read64s(addr)), r);
}
pub unsafe fn instr64_0F4F_reg(r1: i32, r: i32) { cmovcc64(!test_le(), read_reg64(r1), r); }
pub unsafe fn instr64_0F80(a0: i32) { jmpcc64(test_o(), a0); }
pub unsafe fn instr64_0F81(a0: i32) { jmpcc64(!test_o(), a0); }
pub unsafe fn instr64_0F82(a0: i32) { jmpcc64(test_b(), a0); }
pub unsafe fn instr64_0F83(a0: i32) { jmpcc64(!test_b(), a0); }
pub unsafe fn instr64_0F84(a0: i32) { jmpcc64(test_z(), a0); }
pub unsafe fn instr64_0F85(a0: i32) { jmpcc64(!test_z(), a0); }
pub unsafe fn instr64_0F86(a0: i32) { jmpcc64(test_be(), a0); }
pub unsafe fn instr64_0F87(a0: i32) { jmpcc64(!test_be(), a0); }
pub unsafe fn instr64_0F88(a0: i32) { jmpcc64(test_s(), a0); }
pub unsafe fn instr64_0F89(a0: i32) { jmpcc64(!test_s(), a0); }
pub unsafe fn instr64_0F8A(a0: i32) { jmpcc64(test_p(), a0); }
pub unsafe fn instr64_0F8B(a0: i32) { jmpcc64(!test_p(), a0); }
pub unsafe fn instr64_0F8C(a0: i32) { jmpcc64(test_l(), a0); }
pub unsafe fn instr64_0F8D(a0: i32) { jmpcc64(!test_l(), a0); }
pub unsafe fn instr64_0F8E(a0: i32) { jmpcc64(test_le(), a0); }
pub unsafe fn instr64_0F8F(a0: i32) { jmpcc64(!test_le(), a0); }
pub unsafe fn instr64_0FA0() {
    return_on_pagefault!(push64(*sreg.offset(FS as isize) as i32 as u64));
}
pub unsafe fn instr64_0FA1() {
    let v = return_on_pagefault!(pop64());
    switch_seg(FS, v as i32 & 0xFFFF);
}
pub unsafe fn instr64_0FA3_mem(addr: u64, r: i32) { bt_mem(addr, read_reg32(r)); }
pub unsafe fn instr64_0FA3_reg(r1: i32, r: i32) { bt_reg64(read_reg64(r1), read_reg32(r) & 63); }
pub unsafe fn instr64_0FA4_mem(addr: u64, r: i32, imm: i32) {
    safe_read_write64(addr, &|x| shld64(x, read_reg64(r), imm & 63))
}
pub unsafe fn instr64_0FA4_reg(r1: i32, r: i32, imm: i32) {
    write_reg64(r1, shld64(read_reg64(r1), read_reg64(r), imm & 63));
}
pub unsafe fn instr64_0FA5_mem(addr: u64, r: i32) {
    safe_read_write64(addr, &|x| shld64(x, read_reg64(r), read_reg8(ECX) & 63))
}
pub unsafe fn instr64_0FA5_reg(r1: i32, r: i32) {
    write_reg64(
        r1,
        shld64(read_reg64(r1), read_reg64(r), read_reg8(ECX) & 63),
    );
}
pub unsafe fn instr64_0FA8() {
    return_on_pagefault!(push64(*sreg.offset(GS as isize) as i32 as u64));
}
pub unsafe fn instr64_0FA9() {
    let v = return_on_pagefault!(pop64());
    switch_seg(GS, v as i32 & 0xFFFF);
}
pub unsafe fn instr64_0FAB_mem(addr: u64, r: i32) { bts_mem(addr, read_reg32(r)); }
pub unsafe fn instr64_0FAB_reg(r1: i32, r: i32) {
    write_reg64(r1, bts_reg64(read_reg64(r1), read_reg32(r) & 63));
}
pub unsafe fn instr64_0FAC_mem(addr: u64, r: i32, imm: i32) {
    safe_read_write64(addr, &|x| shrd64(x, read_reg64(r), imm & 63))
}
pub unsafe fn instr64_0FAC_reg(r1: i32, r: i32, imm: i32) {
    write_reg64(r1, shrd64(read_reg64(r1), read_reg64(r), imm & 63));
}
pub unsafe fn instr64_0FAD_mem(addr: u64, r: i32) {
    safe_read_write64(addr, &|x| shrd64(x, read_reg64(r), read_reg8(ECX) & 63))
}
pub unsafe fn instr64_0FAD_reg(r1: i32, r: i32) {
    write_reg64(
        r1,
        shrd64(read_reg64(r1), read_reg64(r), read_reg8(ECX) & 63),
    );
}
pub unsafe fn instr64_0FAF_mem(addr: u64, r: i32) {
    write_reg64(
        r,
        imul_reg64(read_reg64(r), return_on_pagefault!(safe_read64s(addr))),
    );
}
pub unsafe fn instr64_0FAF_reg(r1: i32, r: i32) {
    write_reg64(r, imul_reg64(read_reg64(r), read_reg64(r1)));
}
pub unsafe fn instr64_0FB1_mem(addr: u64, r: i32) {
    safe_read_write64(addr, &|x| cmpxchg64(x, r))
}
pub unsafe fn instr64_0FB1_reg(r1: i32, r: i32) {
    let v = cmpxchg64(read_reg64(r1), r);
    write_reg64(r1, v);
}
pub unsafe fn instr64_0FB2_mem(addr: u64, r: i32) {
    let new_reg = return_on_pagefault!(safe_read64s(addr));
    let new_seg = return_on_pagefault!(safe_read16(addr + 8));
    if !switch_seg(SS, new_seg as i32) {
        return;
    }
    write_reg64(r, new_reg);
}
pub unsafe fn instr64_0FB2_reg(r1: i32, r: i32) { trigger_ud(); }
pub unsafe fn instr64_0FB3_mem(addr: u64, r: i32) { btr_mem(addr, read_reg32(r)); }
pub unsafe fn instr64_0FB3_reg(r1: i32, r: i32) {
    write_reg64(r1, btr_reg64(read_reg64(r1), read_reg32(r) & 63));
}
pub unsafe fn instr64_0FB4_mem(addr: u64, r: i32) {
    let new_reg = return_on_pagefault!(safe_read64s(addr));
    let new_seg = return_on_pagefault!(safe_read16(addr + 8));
    if !switch_seg(FS, new_seg as i32) {
        return;
    }
    write_reg64(r, new_reg);
}
pub unsafe fn instr64_0FB4_reg(a0: i32, a1: i32) { trigger_ud(); }
pub unsafe fn instr64_0FB5_mem(addr: u64, r: i32) {
    let new_reg = return_on_pagefault!(safe_read64s(addr));
    let new_seg = return_on_pagefault!(safe_read16(addr + 8));
    if !switch_seg(GS, new_seg as i32) {
        return;
    }
    write_reg64(r, new_reg);
}
pub unsafe fn instr64_0FB5_reg(a0: i32, a1: i32) { trigger_ud(); }
pub unsafe fn instr64_0FB6_mem(addr: u64, r: i32) {
    write_reg64(r, return_on_pagefault!(safe_read8(addr)) as u64);
}
pub unsafe fn instr64_0FB6_reg(r1: i32, r: i32) { write_reg64(r, read_reg8(r1) as u64); }
pub unsafe fn instr64_0FB7_mem(addr: u64, r: i32) {
    write_reg64(r, return_on_pagefault!(safe_read16(addr)) as u64);
}
pub unsafe fn instr64_0FB7_reg(r1: i32, r: i32) { write_reg64(r, read_reg16(r1) as u64); }
pub unsafe fn instr64_0FB8_mem(_a0: u64, _a1: i32) { trigger_ud(); }
pub unsafe fn instr64_0FB8_reg(_a0: i32, _a1: i32) { trigger_ud(); }
pub unsafe fn instr64_0FBA_4_mem(addr: u64, imm: i32) { bt_mem(addr, imm & 63); }
pub unsafe fn instr64_0FBA_4_reg(r1: i32, imm: i32) { bt_reg64(read_reg64(r1), imm & 63); }
pub unsafe fn instr64_0FBA_5_mem(addr: u64, imm: i32) { bts_mem(addr, imm & 63); }
pub unsafe fn instr64_0FBA_5_reg(r1: i32, imm: i32) {
    write_reg64(r1, bts_reg64(read_reg64(r1), imm & 63));
}
pub unsafe fn instr64_0FBA_6_mem(addr: u64, imm: i32) { btr_mem(addr, imm & 63); }
pub unsafe fn instr64_0FBA_6_reg(r1: i32, imm: i32) {
    write_reg64(r1, btr_reg64(read_reg64(r1), imm & 63));
}
pub unsafe fn instr64_0FBA_7_mem(addr: u64, imm: i32) { btc_mem(addr, imm & 63); }
pub unsafe fn instr64_0FBA_7_reg(r1: i32, imm: i32) {
    write_reg64(r1, btc_reg64(read_reg64(r1), imm & 63));
}
pub unsafe fn instr64_0FBB_mem(addr: u64, r: i32) { btc_mem(addr, read_reg32(r)); }
pub unsafe fn instr64_0FBB_reg(r1: i32, r: i32) {
    write_reg64(r1, btc_reg64(read_reg64(r1), read_reg32(r) & 63));
}
pub unsafe fn instr64_0FBC_mem(addr: u64, r: i32) {
    write_reg64(
        r,
        bsf64(read_reg64(r), return_on_pagefault!(safe_read64s(addr))),
    );
}
pub unsafe fn instr64_0FBC_reg(r1: i32, r: i32) {
    write_reg64(r, bsf64(read_reg64(r), read_reg64(r1)));
}
pub unsafe fn instr64_0FBD_mem(addr: u64, r: i32) {
    write_reg64(
        r,
        bsr64(read_reg64(r), return_on_pagefault!(safe_read64s(addr))),
    );
}
pub unsafe fn instr64_0FBD_reg(r1: i32, r: i32) {
    write_reg64(r, bsr64(read_reg64(r), read_reg64(r1)));
}
pub unsafe fn instr64_0FBE_mem(addr: u64, r: i32) {
    write_reg64(
        r,
        return_on_pagefault!(safe_read8(addr)) as i8 as i64 as u64,
    );
}
pub unsafe fn instr64_0FBE_reg(r1: i32, r: i32) {
    write_reg64(r, read_reg8(r1) as i8 as i64 as u64);
}
pub unsafe fn instr64_0FBF_mem(addr: u64, r: i32) {
    write_reg64(
        r,
        return_on_pagefault!(safe_read16(addr)) as i16 as i64 as u64,
    );
}
pub unsafe fn instr64_0FBF_reg(r1: i32, r: i32) {
    write_reg64(r, read_reg16(r1) as i16 as i64 as u64);
}
pub unsafe fn instr64_0FC1_mem(addr: u64, r: i32) {
    safe_read_write64(addr, &|x| xadd64(x, r))
}
pub unsafe fn instr64_0FC1_reg(r1: i32, r: i32) {
    let source = read_reg64(r);
    write_reg64(r, read_reg64(r1));
    write_reg64(r1, add64(read_reg64(r1), source));
}
pub unsafe fn instr64_0FC7_1_reg(_r: i32) { trigger_ud(); }
pub unsafe fn instr64_0FC7_1_mem(addr: u64) {
    // cmpxchg16b
    return_on_pagefault!(writable_or_pagefault(addr, 16));
    let m128 = return_on_pagefault!(safe_read128s(addr));
    let m_low = m128.u64[0];
    let m_high = m128.u64[1];
    if read_reg64(EAX) == m_low && read_reg64(EDX) == m_high {
        *flags |= FLAG_ZERO;
        return_on_pagefault!(safe_write128(
            addr,
            reg128 {
                u64: [read_reg64(EBX), read_reg64(ECX)],
            },
        ));
    }
    else {
        *flags &= !FLAG_ZERO;
        write_reg64(EAX, m_low);
        write_reg64(EDX, m_high);
    }
    *flags_changed &= !FLAG_ZERO;
}
pub unsafe fn instr64_0FC7_6_reg(r: i32) {
    // rdrand r64
    let rand = (js::get_rand_int() as u32 as u64) | (js::get_rand_int() as u32 as u64) << 32;
    write_reg64(r, rand);
    *flags &= !FLAGS_ALL;
    *flags |= 1;
    *flags_changed = 0;
}
pub unsafe fn instr64_0FC7_6_mem(_a0: u64) { trigger_ud(); }
pub unsafe fn instr64_11_mem(addr: u64, r: i32) {
    safe_read_write64(addr, &|x| adc64(x, read_reg64(r)))
}
pub unsafe fn instr64_11_reg(r1: i32, r: i32) {
    write_reg64(r1, adc64(read_reg64(r1), read_reg64(r)));
}
pub unsafe fn instr64_13_mem(addr: u64, r: i32) {
    write_reg64(
        r,
        adc64(read_reg64(r), return_on_pagefault!(safe_read64s(addr))),
    );
}
pub unsafe fn instr64_13_reg(r1: i32, r: i32) {
    write_reg64(r, adc64(read_reg64(r), read_reg64(r1)));
}
pub unsafe fn instr64_15(a0: i32) { write_reg64(EAX, adc64(read_reg64(EAX), a0 as i64 as u64)); }
pub unsafe fn instr64_16() { trigger_ud(); }
pub unsafe fn instr64_17() { trigger_ud(); }
pub unsafe fn instr64_19_mem(addr: u64, r: i32) {
    safe_read_write64(addr, &|x| sbb64(x, read_reg64(r)))
}
pub unsafe fn instr64_19_reg(r1: i32, r: i32) {
    write_reg64(r1, sbb64(read_reg64(r1), read_reg64(r)));
}
pub unsafe fn instr64_1B_mem(addr: u64, r: i32) {
    write_reg64(
        r,
        sbb64(read_reg64(r), return_on_pagefault!(safe_read64s(addr))),
    );
}
pub unsafe fn instr64_1B_reg(r1: i32, r: i32) {
    write_reg64(r, sbb64(read_reg64(r), read_reg64(r1)));
}
pub unsafe fn instr64_1D(a0: i32) { write_reg64(EAX, sbb64(read_reg64(EAX), a0 as i64 as u64)); }
pub unsafe fn instr64_1E() { trigger_ud(); }
pub unsafe fn instr64_1F() { trigger_ud(); }
pub unsafe fn instr64_21_mem(addr: u64, r: i32) {
    safe_read_write64(addr, &|x| and64(x, read_reg64(r)))
}
pub unsafe fn instr64_21_reg(r1: i32, r: i32) {
    write_reg64(r1, and64(read_reg64(r1), read_reg64(r)));
}
pub unsafe fn instr64_23_mem(addr: u64, r: i32) {
    write_reg64(
        r,
        and64(read_reg64(r), return_on_pagefault!(safe_read64s(addr))),
    );
}
pub unsafe fn instr64_23_reg(r1: i32, r: i32) {
    write_reg64(r, and64(read_reg64(r), read_reg64(r1)));
}
pub unsafe fn instr64_25(a0: i32) { write_reg64(EAX, and64(read_reg64(EAX), a0 as i64 as u64)); }
pub unsafe fn instr64_29_mem(addr: u64, r: i32) {
    safe_read_write64(addr, &|x| sub64(x, read_reg64(r)))
}
pub unsafe fn instr64_29_reg(r1: i32, r: i32) {
    write_reg64(r1, sub64(read_reg64(r1), read_reg64(r)));
}
pub unsafe fn instr64_2B_mem(addr: u64, r: i32) {
    write_reg64(
        r,
        sub64(read_reg64(r), return_on_pagefault!(safe_read64s(addr))),
    );
}
pub unsafe fn instr64_2B_reg(r1: i32, r: i32) {
    write_reg64(r, sub64(read_reg64(r), read_reg64(r1)));
}
pub unsafe fn instr64_2D(a0: i32) { write_reg64(EAX, sub64(read_reg64(EAX), a0 as i64 as u64)); }
pub unsafe fn instr64_31_mem(addr: u64, r: i32) {
    safe_read_write64(addr, &|x| xor64(x, read_reg64(r)))
}
pub unsafe fn instr64_31_reg(r1: i32, r: i32) {
    write_reg64(r1, xor64(read_reg64(r1), read_reg64(r)));
}
pub unsafe fn instr64_33_mem(addr: u64, r: i32) {
    write_reg64(
        r,
        xor64(read_reg64(r), return_on_pagefault!(safe_read64s(addr))),
    );
}
pub unsafe fn instr64_33_reg(r1: i32, r: i32) {
    write_reg64(r, xor64(read_reg64(r), read_reg64(r1)));
}
pub unsafe fn instr64_35(a0: i32) { write_reg64(EAX, xor64(read_reg64(EAX), a0 as i64 as u64)); }
pub unsafe fn instr64_39_mem(addr: u64, r: i32) {
    cmp64(return_on_pagefault!(safe_read64s(addr)), read_reg64(r));
}
pub unsafe fn instr64_39_reg(r1: i32, r: i32) { cmp64(read_reg64(r1), read_reg64(r)); }
pub unsafe fn instr64_3B_mem(addr: u64, r: i32) {
    cmp64(read_reg64(r), return_on_pagefault!(safe_read64s(addr)));
}
pub unsafe fn instr64_3B_reg(r1: i32, r: i32) { cmp64(read_reg64(r), read_reg64(r1)); }
pub unsafe fn instr64_3D(a0: i32) { cmp64(read_reg64(EAX), a0 as i64 as u64); }
pub unsafe fn instr64_40() { trigger_ud(); }
pub unsafe fn instr64_41() { trigger_ud(); }
pub unsafe fn instr64_42() { trigger_ud(); }
pub unsafe fn instr64_43() { trigger_ud(); }
pub unsafe fn instr64_44() { trigger_ud(); }
pub unsafe fn instr64_45() { trigger_ud(); }
pub unsafe fn instr64_46() { trigger_ud(); }
pub unsafe fn instr64_47() { trigger_ud(); }
pub unsafe fn instr64_48() { trigger_ud(); }
pub unsafe fn instr64_49() { trigger_ud(); }
pub unsafe fn instr64_4A() { trigger_ud(); }
pub unsafe fn instr64_4B() { trigger_ud(); }
pub unsafe fn instr64_4C() { trigger_ud(); }
pub unsafe fn instr64_4D() { trigger_ud(); }
pub unsafe fn instr64_4E() { trigger_ud(); }
pub unsafe fn instr64_4F() { trigger_ud(); }
pub unsafe fn instr64_50() {
    return_on_pagefault!(push64(read_reg64(0 + rex_b())));
}
pub unsafe fn instr64_51() {
    return_on_pagefault!(push64(read_reg64(1 + rex_b())));
}
pub unsafe fn instr64_52() {
    return_on_pagefault!(push64(read_reg64(2 + rex_b())));
}
pub unsafe fn instr64_53() {
    return_on_pagefault!(push64(read_reg64(3 + rex_b())));
}
pub unsafe fn instr64_54() {
    return_on_pagefault!(push64(read_reg64(4 + rex_b())));
}
pub unsafe fn instr64_55() {
    return_on_pagefault!(push64(read_reg64(5 + rex_b())));
}
pub unsafe fn instr64_56() {
    return_on_pagefault!(push64(read_reg64(6 + rex_b())));
}
pub unsafe fn instr64_57() {
    return_on_pagefault!(push64(read_reg64(7 + rex_b())));
}
pub unsafe fn instr64_58() {
    let v = return_on_pagefault!(pop64());
    write_reg64(0 + rex_b(), v);
}
pub unsafe fn instr64_59() {
    let v = return_on_pagefault!(pop64());
    write_reg64(1 + rex_b(), v);
}
pub unsafe fn instr64_5A() {
    let v = return_on_pagefault!(pop64());
    write_reg64(2 + rex_b(), v);
}
pub unsafe fn instr64_5B() {
    let v = return_on_pagefault!(pop64());
    write_reg64(3 + rex_b(), v);
}
pub unsafe fn instr64_5C() {
    let v = return_on_pagefault!(pop64());
    write_reg64(4 + rex_b(), v);
}
pub unsafe fn instr64_5D() {
    let v = return_on_pagefault!(pop64());
    write_reg64(5 + rex_b(), v);
}
pub unsafe fn instr64_5E() {
    let v = return_on_pagefault!(pop64());
    write_reg64(6 + rex_b(), v);
}
pub unsafe fn instr64_5F() {
    let v = return_on_pagefault!(pop64());
    write_reg64(7 + rex_b(), v);
}
pub unsafe fn instr64_60() { trigger_ud(); }
pub unsafe fn instr64_61() { trigger_ud(); }
pub unsafe fn instr64_68(a0: i32) {
    return_on_pagefault!(push64(a0 as i64 as u64));
}
pub unsafe fn instr64_69_mem(addr: u64, r: i32, imm: i32) {
    write_reg64(
        r,
        imul_reg64(return_on_pagefault!(safe_read64s(addr)), imm as i64 as u64),
    );
}
pub unsafe fn instr64_69_reg(r1: i32, r: i32, imm: i32) {
    write_reg64(r, imul_reg64(read_reg64(r1), imm as i64 as u64));
}
pub unsafe fn instr64_6A(a0: i32) {
    return_on_pagefault!(push64(a0 as i64 as u64));
}
pub unsafe fn instr64_6B_mem(addr: u64, r: i32, imm: i32) {
    write_reg64(
        r,
        imul_reg64(return_on_pagefault!(safe_read64s(addr)), imm as i64 as u64),
    );
}
pub unsafe fn instr64_6B_reg(r1: i32, r: i32, imm: i32) {
    write_reg64(r, imul_reg64(read_reg64(r1), imm as i64 as u64));
}
pub unsafe fn instr64_6D() { insd_no_rep(is_asize_32()); }
pub unsafe fn instr64_6F() { outsd_no_rep(is_asize_32(), segment_prefix(DS)); }
pub unsafe fn instr64_70(a0: i32) { jmpcc64(test_o(), a0); }
pub unsafe fn instr64_71(a0: i32) { jmpcc64(!test_o(), a0); }
pub unsafe fn instr64_72(a0: i32) { jmpcc64(test_b(), a0); }
pub unsafe fn instr64_73(a0: i32) { jmpcc64(!test_b(), a0); }
pub unsafe fn instr64_74(a0: i32) { jmpcc64(test_z(), a0); }
pub unsafe fn instr64_75(a0: i32) { jmpcc64(!test_z(), a0); }
pub unsafe fn instr64_76(a0: i32) { jmpcc64(test_be(), a0); }
pub unsafe fn instr64_77(a0: i32) { jmpcc64(!test_be(), a0); }
pub unsafe fn instr64_78(a0: i32) { jmpcc64(test_s(), a0); }
pub unsafe fn instr64_79(a0: i32) { jmpcc64(!test_s(), a0); }
pub unsafe fn instr64_7A(a0: i32) { jmpcc64(test_p(), a0); }
pub unsafe fn instr64_7B(a0: i32) { jmpcc64(!test_p(), a0); }
pub unsafe fn instr64_7C(a0: i32) { jmpcc64(test_l(), a0); }
pub unsafe fn instr64_7D(a0: i32) { jmpcc64(!test_l(), a0); }
pub unsafe fn instr64_7E(a0: i32) { jmpcc64(test_le(), a0); }
pub unsafe fn instr64_7F(a0: i32) { jmpcc64(!test_le(), a0); }
pub unsafe fn instr64_81_0_mem(addr: u64, imm: i32) {
    safe_read_write64(addr, &|x| add64(x, imm as i64 as u64))
}
pub unsafe fn instr64_81_0_reg(r1: i32, imm: i32) {
    write_reg64(r1, add64(read_reg64(r1), imm as i64 as u64));
}
pub unsafe fn instr64_81_1_mem(addr: u64, imm: i32) {
    safe_read_write64(addr, &|x| or64(x, imm as i64 as u64))
}
pub unsafe fn instr64_81_1_reg(r1: i32, imm: i32) {
    write_reg64(r1, or64(read_reg64(r1), imm as i64 as u64));
}
pub unsafe fn instr64_81_2_mem(addr: u64, imm: i32) {
    safe_read_write64(addr, &|x| adc64(x, imm as i64 as u64))
}
pub unsafe fn instr64_81_2_reg(r1: i32, imm: i32) {
    write_reg64(r1, adc64(read_reg64(r1), imm as i64 as u64));
}
pub unsafe fn instr64_81_3_mem(addr: u64, imm: i32) {
    safe_read_write64(addr, &|x| sbb64(x, imm as i64 as u64))
}
pub unsafe fn instr64_81_3_reg(r1: i32, imm: i32) {
    write_reg64(r1, sbb64(read_reg64(r1), imm as i64 as u64));
}
pub unsafe fn instr64_81_4_mem(addr: u64, imm: i32) {
    safe_read_write64(addr, &|x| and64(x, imm as i64 as u64))
}
pub unsafe fn instr64_81_4_reg(r1: i32, imm: i32) {
    write_reg64(r1, and64(read_reg64(r1), imm as i64 as u64));
}
pub unsafe fn instr64_81_5_mem(addr: u64, imm: i32) {
    safe_read_write64(addr, &|x| sub64(x, imm as i64 as u64))
}
pub unsafe fn instr64_81_5_reg(r1: i32, imm: i32) {
    write_reg64(r1, sub64(read_reg64(r1), imm as i64 as u64));
}
pub unsafe fn instr64_81_6_mem(addr: u64, imm: i32) {
    safe_read_write64(addr, &|x| xor64(x, imm as i64 as u64))
}
pub unsafe fn instr64_81_6_reg(r1: i32, imm: i32) {
    write_reg64(r1, xor64(read_reg64(r1), imm as i64 as u64));
}
pub unsafe fn instr64_81_7_mem(addr: u64, imm: i32) {
    cmp64(return_on_pagefault!(safe_read64s(addr)), imm as i64 as u64);
}
pub unsafe fn instr64_81_7_reg(r1: i32, imm: i32) { cmp64(read_reg64(r1), imm as i64 as u64); }
pub unsafe fn instr64_83_0_mem(addr: u64, imm: i32) {
    safe_read_write64(addr, &|x| add64(x, imm as i64 as u64))
}
pub unsafe fn instr64_83_0_reg(r1: i32, imm: i32) {
    write_reg64(r1, add64(read_reg64(r1), imm as i64 as u64));
}
pub unsafe fn instr64_83_1_mem(addr: u64, imm: i32) {
    safe_read_write64(addr, &|x| or64(x, imm as i64 as u64))
}
pub unsafe fn instr64_83_1_reg(r1: i32, imm: i32) {
    write_reg64(r1, or64(read_reg64(r1), imm as i64 as u64));
}
pub unsafe fn instr64_83_2_mem(addr: u64, imm: i32) {
    safe_read_write64(addr, &|x| adc64(x, imm as i64 as u64))
}
pub unsafe fn instr64_83_2_reg(r1: i32, imm: i32) {
    write_reg64(r1, adc64(read_reg64(r1), imm as i64 as u64));
}
pub unsafe fn instr64_83_3_mem(addr: u64, imm: i32) {
    safe_read_write64(addr, &|x| sbb64(x, imm as i64 as u64))
}
pub unsafe fn instr64_83_3_reg(r1: i32, imm: i32) {
    write_reg64(r1, sbb64(read_reg64(r1), imm as i64 as u64));
}
pub unsafe fn instr64_83_4_mem(addr: u64, imm: i32) {
    safe_read_write64(addr, &|x| and64(x, imm as i64 as u64))
}
pub unsafe fn instr64_83_4_reg(r1: i32, imm: i32) {
    write_reg64(r1, and64(read_reg64(r1), imm as i64 as u64));
}
pub unsafe fn instr64_83_5_mem(addr: u64, imm: i32) {
    safe_read_write64(addr, &|x| sub64(x, imm as i64 as u64))
}
pub unsafe fn instr64_83_5_reg(r1: i32, imm: i32) {
    write_reg64(r1, sub64(read_reg64(r1), imm as i64 as u64));
}
pub unsafe fn instr64_83_6_mem(addr: u64, imm: i32) {
    safe_read_write64(addr, &|x| xor64(x, imm as i64 as u64))
}
pub unsafe fn instr64_83_6_reg(r1: i32, imm: i32) {
    write_reg64(r1, xor64(read_reg64(r1), imm as i64 as u64));
}
pub unsafe fn instr64_83_7_mem(addr: u64, imm: i32) {
    cmp64(return_on_pagefault!(safe_read64s(addr)), imm as i64 as u64);
}
pub unsafe fn instr64_83_7_reg(r1: i32, imm: i32) { cmp64(read_reg64(r1), imm as i64 as u64); }
pub unsafe fn instr64_85_mem(addr: u64, r: i32) {
    test64(return_on_pagefault!(safe_read64s(addr)), read_reg64(r));
}
pub unsafe fn instr64_85_reg(r1: i32, r: i32) { test64(read_reg64(r1), read_reg64(r)); }
pub unsafe fn instr64_87_mem(addr: u64, r: i32) {
    let tmp = return_on_pagefault!(safe_read64s(addr));
    return_on_pagefault!(safe_write64(addr, read_reg64(r)));
    write_reg64(r, tmp);
}
pub unsafe fn instr64_87_reg(r1: i32, r: i32) {
    let tmp = read_reg64(r1);
    write_reg64(r1, read_reg64(r));
    write_reg64(r, tmp);
}
pub unsafe fn instr64_89_mem(addr: u64, r: i32) {
    return_on_pagefault!(safe_write64(addr, read_reg64(r)));
}
pub unsafe fn instr64_89_reg(r1: i32, r: i32) { write_reg64(r1, read_reg64(r)); }
pub unsafe fn instr64_8B_mem(addr: u64, r: i32) {
    write_reg64(r, return_on_pagefault!(safe_read64s(addr)));
}
pub unsafe fn instr64_8B_reg(r1: i32, r: i32) { write_reg64(r, read_reg64(r1)); }
pub unsafe fn instr64_8C_mem(addr: u64, r: i32) {
    return_on_pagefault!(safe_write16(addr, *sreg.offset(r as isize) as i32));
}
pub unsafe fn instr64_8C_reg(r1: i32, r: i32) { write_reg64(r1, *sreg.offset(r as isize) as u64); }
pub unsafe fn instr64_8D_mem(modrm_byte: i32, r: i32) {
    // note: no trailing immediate, so imm_len is 0
    *prefixes |= prefix::SEG_PREFIX_ZERO;
    if let Ok(addr) = modrm_resolve(modrm_byte, 0) {
        write_reg64(r, addr);
    }
    *prefixes = 0;
}
pub unsafe fn instr64_8D_reg(r1: i32, r: i32) {
    dbg_log!("lea #ud");
    trigger_ud();
}
pub unsafe fn instr64_8F_0_mem(modrm_byte: i32) {
    // pop r/m64: update rsp before resolving the address
    write_reg64(ESP, read_reg64(ESP).wrapping_add(8));
    match modrm_resolve(modrm_byte, 0) {
        Err(()) => {
            write_reg64(ESP, read_reg64(ESP).wrapping_sub(8));
        },
        Ok(addr) => {
            // restore rsp while accessing memory, so that the instruction can be restarted if
            // either access page-faults (same as instr32_8F_0_mem)
            write_reg64(ESP, read_reg64(ESP).wrapping_sub(8));
            let stack_value = return_on_pagefault!(safe_read64s(get_stack_pointer64(0)));
            return_on_pagefault!(safe_write64(addr, stack_value));
            write_reg64(ESP, read_reg64(ESP).wrapping_add(8));
        },
    }
}
pub unsafe fn instr64_8F_0_reg(r1: i32) {
    let v = return_on_pagefault!(pop64());
    write_reg64(r1, v);
}
pub unsafe fn instr64_91() {
    let r = 1 + rex_b();
    let t = read_reg64(EAX);
    write_reg64(EAX, read_reg64(r));
    write_reg64(r, t);
}
pub unsafe fn instr64_92() {
    let r = 2 + rex_b();
    let t = read_reg64(EAX);
    write_reg64(EAX, read_reg64(r));
    write_reg64(r, t);
}
pub unsafe fn instr64_93() {
    let r = 3 + rex_b();
    let t = read_reg64(EAX);
    write_reg64(EAX, read_reg64(r));
    write_reg64(r, t);
}
pub unsafe fn instr64_94() {
    let r = 4 + rex_b();
    let t = read_reg64(EAX);
    write_reg64(EAX, read_reg64(r));
    write_reg64(r, t);
}
pub unsafe fn instr64_95() {
    let r = 5 + rex_b();
    let t = read_reg64(EAX);
    write_reg64(EAX, read_reg64(r));
    write_reg64(r, t);
}
pub unsafe fn instr64_96() {
    let r = 6 + rex_b();
    let t = read_reg64(EAX);
    write_reg64(EAX, read_reg64(r));
    write_reg64(r, t);
}
pub unsafe fn instr64_97() {
    let r = 7 + rex_b();
    let t = read_reg64(EAX);
    write_reg64(EAX, read_reg64(r));
    write_reg64(r, t);
}
pub unsafe fn instr64_98() { write_reg64(EAX, read_reg32(EAX) as i64 as u64); }
pub unsafe fn instr64_99() { write_reg64(EDX, (read_reg64(EAX) as i64 >> 63) as u64); }
pub unsafe fn instr64_9A(_a0: i32, _a1: i32) { trigger_ud(); }
pub unsafe fn instr64_9C() {
    // pushf
    if crate::cpu::instructions::instr_pushf_popf_check() {
        dbg_assert!(*protected_mode);
        dbg_log!("pushf #gp");
        trigger_gp(0);
    }
    else {
        // vm and rf flag are cleared in image stored on the stack
        return_on_pagefault!(push64(get_eflags() as u64 & 0xFCFFFF));
    };
}
pub unsafe fn instr64_9D() {
    // popf
    if crate::cpu::instructions::instr_pushf_popf_check() {
        dbg_log!("popf #gp");
        trigger_gp(0);
        return;
    }
    let old_eflags = *flags;
    update_eflags(return_on_pagefault!(pop64()) as u32 as i32);
    if old_eflags & FLAG_INTERRUPT == 0 && *flags & FLAG_INTERRUPT != 0 {
        handle_irqs_or_defer();
    }
}
pub unsafe fn instr64_A1(moffs: u64) {
    // mov rax, [moffs64]
    let addr = return_on_pagefault!(get_seg_prefix64(DS)).wrapping_add(moffs);
    write_reg64(EAX, return_on_pagefault!(safe_read64s(addr)));
}
pub unsafe fn instr64_A3(moffs: u64) {
    // mov [moffs64], rax
    let addr = return_on_pagefault!(get_seg_prefix64(DS)).wrapping_add(moffs);
    return_on_pagefault!(safe_write64(addr, read_reg64(EAX)));
}
pub unsafe fn instr64_A5() { movsq_no_rep(is_asize_32(), segment_prefix(DS)); }
pub unsafe fn instr64_A7() { cmpsq_no_rep(is_asize_32(), segment_prefix(DS)); }
pub unsafe fn instr64_A9(a0: i32) { test64(read_reg64(EAX), a0 as i64 as u64); }
pub unsafe fn instr64_AB() { stosq_no_rep(is_asize_32()); }
pub unsafe fn instr64_AD() { lodsq_no_rep(is_asize_32(), segment_prefix(DS)); }
pub unsafe fn instr64_AF() { scasq_no_rep(is_asize_32()); }
pub unsafe fn instr64_B8(imm: u64) { write_reg64(0 + rex_b(), imm); }
pub unsafe fn instr64_B9(imm: u64) { write_reg64(1 + rex_b(), imm); }
pub unsafe fn instr64_BA(imm: u64) { write_reg64(2 + rex_b(), imm); }
pub unsafe fn instr64_BB(imm: u64) { write_reg64(3 + rex_b(), imm); }
pub unsafe fn instr64_BC(imm: u64) { write_reg64(4 + rex_b(), imm); }
pub unsafe fn instr64_BD(imm: u64) { write_reg64(5 + rex_b(), imm); }
pub unsafe fn instr64_BE(imm: u64) { write_reg64(6 + rex_b(), imm); }
pub unsafe fn instr64_BF(imm: u64) { write_reg64(7 + rex_b(), imm); }
pub unsafe fn instr64_C1_0_mem(addr: u64, imm: i32) {
    safe_read_write64(addr, &|x| rol64(x, imm & 63))
}
pub unsafe fn instr64_C1_0_reg(r1: i32, imm: i32) {
    write_reg64(r1, rol64(read_reg64(r1), imm & 63));
}
pub unsafe fn instr64_C1_1_mem(addr: u64, imm: i32) {
    safe_read_write64(addr, &|x| ror64(x, imm & 63))
}
pub unsafe fn instr64_C1_1_reg(r1: i32, imm: i32) {
    write_reg64(r1, ror64(read_reg64(r1), imm & 63));
}
pub unsafe fn instr64_C1_2_mem(addr: u64, imm: i32) {
    safe_read_write64(addr, &|x| rcl64(x, imm & 63))
}
pub unsafe fn instr64_C1_2_reg(r1: i32, imm: i32) {
    write_reg64(r1, rcl64(read_reg64(r1), imm & 63));
}
pub unsafe fn instr64_C1_3_mem(addr: u64, imm: i32) {
    safe_read_write64(addr, &|x| rcr64(x, imm & 63))
}
pub unsafe fn instr64_C1_3_reg(r1: i32, imm: i32) {
    write_reg64(r1, rcr64(read_reg64(r1), imm & 63));
}
pub unsafe fn instr64_C1_4_mem(addr: u64, imm: i32) {
    safe_read_write64(addr, &|x| shl64(x, imm & 63))
}
pub unsafe fn instr64_C1_4_reg(r1: i32, imm: i32) {
    write_reg64(r1, shl64(read_reg64(r1), imm & 63));
}
pub unsafe fn instr64_C1_5_mem(addr: u64, imm: i32) {
    safe_read_write64(addr, &|x| shr64(x, imm & 63))
}
pub unsafe fn instr64_C1_5_reg(r1: i32, imm: i32) {
    write_reg64(r1, shr64(read_reg64(r1), imm & 63));
}
pub unsafe fn instr64_C1_6_mem(addr: u64, imm: i32) {
    safe_read_write64(addr, &|x| shl64(x, imm & 63))
}
pub unsafe fn instr64_C1_6_reg(r1: i32, imm: i32) {
    write_reg64(r1, shl64(read_reg64(r1), imm & 63));
}
pub unsafe fn instr64_C1_7_mem(addr: u64, imm: i32) {
    safe_read_write64(addr, &|x| sar64(x, imm & 63))
}
pub unsafe fn instr64_C1_7_reg(r1: i32, imm: i32) {
    write_reg64(r1, sar64(read_reg64(r1), imm & 63));
}
pub unsafe fn instr64_C2(a0: i32) {
    let ip = return_on_pagefault!(pop64());
    *instruction_pointer = ip;
    write_reg64(ESP, read_reg64(ESP).wrapping_add(a0 as u64));
}
pub unsafe fn instr64_C3() {
    let ip = return_on_pagefault!(pop64());
    *instruction_pointer = ip;
}
pub unsafe fn instr64_C4_mem(_a0: u64, _a1: i32) { trigger_ud(); }
pub unsafe fn instr64_C4_reg(_a0: i32, _a1: i32) { trigger_ud(); }
pub unsafe fn instr64_C5_mem(_a0: u64, _a1: i32) { trigger_ud(); }
pub unsafe fn instr64_C5_reg(_a0: i32, _a1: i32) { trigger_ud(); }
pub unsafe fn instr64_C7_0_mem(addr: u64, imm: i32) {
    return_on_pagefault!(safe_write64(addr, imm as i64 as u64));
}
pub unsafe fn instr64_C7_0_reg(r1: i32, imm: i32) { write_reg64(r1, imm as i64 as u64); }
pub unsafe fn instr64_C8(size: i32, nesting: i32) {
    let nesting = nesting & 31;
    return_on_pagefault!(push64(read_reg64(EBP)));
    let frame_temp = read_reg64(ESP);
    if nesting > 0 {
        let mut tmp_ebp = frame_temp;
        for _ in 1..nesting {
            tmp_ebp = tmp_ebp.wrapping_sub(8);
            let v = return_on_pagefault!(safe_read64s(tmp_ebp));
            return_on_pagefault!(push64(v));
        }
        return_on_pagefault!(push64(frame_temp));
    }
    write_reg64(EBP, frame_temp);
    write_reg64(ESP, frame_temp.wrapping_sub(size as u64));
}
pub unsafe fn instr64_C9() {
    let rbp = read_reg64(EBP);
    write_reg64(ESP, rbp);
    let v = return_on_pagefault!(pop64());
    write_reg64(EBP, v);
}
pub unsafe fn instr64_CA(a0: i32) {
    // retf imm16
    let rip = return_on_pagefault!(pop64());
    let cs = return_on_pagefault!(pop64()) as u16;
    far_return64(rip, cs as i32, a0 as u64);
}
pub unsafe fn instr64_CB() {
    // retf
    let rip = return_on_pagefault!(pop64());
    let cs = return_on_pagefault!(pop64()) as u16;
    far_return64(rip, cs as i32, 0);
}
pub unsafe fn instr64_CF() { iret64(); }
pub unsafe fn instr64_D1_0_mem(addr: u64) { safe_read_write64(addr, &|x| rol64(x, 1)) }
pub unsafe fn instr64_D1_0_reg(r1: i32) { write_reg64(r1, rol64(read_reg64(r1), 1)); }
pub unsafe fn instr64_D1_1_mem(addr: u64) { safe_read_write64(addr, &|x| ror64(x, 1)) }
pub unsafe fn instr64_D1_1_reg(r1: i32) { write_reg64(r1, ror64(read_reg64(r1), 1)); }
pub unsafe fn instr64_D1_2_mem(addr: u64) { safe_read_write64(addr, &|x| rcl64(x, 1)) }
pub unsafe fn instr64_D1_2_reg(r1: i32) { write_reg64(r1, rcl64(read_reg64(r1), 1)); }
pub unsafe fn instr64_D1_3_mem(addr: u64) { safe_read_write64(addr, &|x| rcr64(x, 1)) }
pub unsafe fn instr64_D1_3_reg(r1: i32) { write_reg64(r1, rcr64(read_reg64(r1), 1)); }
pub unsafe fn instr64_D1_4_mem(addr: u64) { safe_read_write64(addr, &|x| shl64(x, 1)) }
pub unsafe fn instr64_D1_4_reg(r1: i32) { write_reg64(r1, shl64(read_reg64(r1), 1)); }
pub unsafe fn instr64_D1_5_mem(addr: u64) { safe_read_write64(addr, &|x| shr64(x, 1)) }
pub unsafe fn instr64_D1_5_reg(r1: i32) { write_reg64(r1, shr64(read_reg64(r1), 1)); }
pub unsafe fn instr64_D1_6_mem(addr: u64) { safe_read_write64(addr, &|x| shl64(x, 1)) }
pub unsafe fn instr64_D1_6_reg(r1: i32) { write_reg64(r1, shl64(read_reg64(r1), 1)); }
pub unsafe fn instr64_D1_7_mem(addr: u64) { safe_read_write64(addr, &|x| sar64(x, 1)) }
pub unsafe fn instr64_D1_7_reg(r1: i32) { write_reg64(r1, sar64(read_reg64(r1), 1)); }
pub unsafe fn instr64_D3_0_mem(addr: u64) {
    safe_read_write64(addr, &|x| rol64(x, read_reg8(ECX) & 63))
}
pub unsafe fn instr64_D3_0_reg(r1: i32) {
    write_reg64(r1, rol64(read_reg64(r1), read_reg8(ECX) & 63));
}
pub unsafe fn instr64_D3_1_mem(addr: u64) {
    safe_read_write64(addr, &|x| ror64(x, read_reg8(ECX) & 63))
}
pub unsafe fn instr64_D3_1_reg(r1: i32) {
    write_reg64(r1, ror64(read_reg64(r1), read_reg8(ECX) & 63));
}
pub unsafe fn instr64_D3_2_mem(addr: u64) {
    safe_read_write64(addr, &|x| rcl64(x, read_reg8(ECX) & 63))
}
pub unsafe fn instr64_D3_2_reg(r1: i32) {
    write_reg64(r1, rcl64(read_reg64(r1), read_reg8(ECX) & 63));
}
pub unsafe fn instr64_D3_3_mem(addr: u64) {
    safe_read_write64(addr, &|x| rcr64(x, read_reg8(ECX) & 63))
}
pub unsafe fn instr64_D3_3_reg(r1: i32) {
    write_reg64(r1, rcr64(read_reg64(r1), read_reg8(ECX) & 63));
}
pub unsafe fn instr64_D3_4_mem(addr: u64) {
    safe_read_write64(addr, &|x| shl64(x, read_reg8(ECX) & 63))
}
pub unsafe fn instr64_D3_4_reg(r1: i32) {
    write_reg64(r1, shl64(read_reg64(r1), read_reg8(ECX) & 63));
}
pub unsafe fn instr64_D3_5_mem(addr: u64) {
    safe_read_write64(addr, &|x| shr64(x, read_reg8(ECX) & 63))
}
pub unsafe fn instr64_D3_5_reg(r1: i32) {
    write_reg64(r1, shr64(read_reg64(r1), read_reg8(ECX) & 63));
}
pub unsafe fn instr64_D3_6_mem(addr: u64) {
    safe_read_write64(addr, &|x| shl64(x, read_reg8(ECX) & 63))
}
pub unsafe fn instr64_D3_6_reg(r1: i32) {
    write_reg64(r1, shl64(read_reg64(r1), read_reg8(ECX) & 63));
}
pub unsafe fn instr64_D3_7_mem(addr: u64) {
    safe_read_write64(addr, &|x| sar64(x, read_reg8(ECX) & 63))
}
pub unsafe fn instr64_D3_7_reg(r1: i32) {
    write_reg64(r1, sar64(read_reg64(r1), read_reg8(ECX) & 63));
}
pub unsafe fn instr64_D9_0_mem(addr: u64) { crate::cpu::instructions::instr16_D9_0_mem(addr) }
pub unsafe fn instr64_D9_0_reg(r: i32) { crate::cpu::instructions::instr16_D9_0_reg(r) }
pub unsafe fn instr64_D9_1_mem(addr: u64) { crate::cpu::instructions::instr16_D9_1_mem(addr) }
pub unsafe fn instr64_D9_1_reg(r: i32) { crate::cpu::instructions::instr16_D9_1_reg(r) }
pub unsafe fn instr64_D9_2_mem(addr: u64) { crate::cpu::instructions::instr16_D9_2_mem(addr) }
pub unsafe fn instr64_D9_2_reg(r: i32) { crate::cpu::instructions::instr16_D9_2_reg(r) }
pub unsafe fn instr64_D9_3_mem(addr: u64) { crate::cpu::instructions::instr16_D9_3_mem(addr) }
pub unsafe fn instr64_D9_3_reg(r: i32) { crate::cpu::instructions::instr16_D9_3_reg(r) }
pub unsafe fn instr64_D9_4_mem(addr: u64) { crate::cpu::instructions::instr16_D9_4_mem(addr) }
pub unsafe fn instr64_D9_4_reg(r: i32) { crate::cpu::instructions::instr16_D9_4_reg(r) }
pub unsafe fn instr64_D9_5_mem(addr: u64) { crate::cpu::instructions::instr16_D9_5_mem(addr) }
pub unsafe fn instr64_D9_5_reg(r: i32) { crate::cpu::instructions::instr16_D9_5_reg(r) }
pub unsafe fn instr64_D9_6_mem(addr: u64) { crate::cpu::instructions::instr16_D9_6_mem(addr) }
pub unsafe fn instr64_D9_6_reg(r: i32) { crate::cpu::instructions::instr16_D9_6_reg(r) }
pub unsafe fn instr64_D9_7_mem(addr: u64) { crate::cpu::instructions::instr16_D9_7_mem(addr) }
pub unsafe fn instr64_D9_7_reg(r: i32) { crate::cpu::instructions::instr16_D9_7_reg(r) }
pub unsafe fn instr64_DD_0_mem(addr: u64) { crate::cpu::instructions::instr16_DD_0_mem(addr) }
pub unsafe fn instr64_DD_0_reg(r: i32) { crate::cpu::instructions::instr16_DD_0_reg(r) }
pub unsafe fn instr64_DD_1_mem(addr: u64) { crate::cpu::instructions::instr16_DD_1_mem(addr) }
pub unsafe fn instr64_DD_1_reg(r: i32) { crate::cpu::instructions::instr16_DD_1_reg(r) }
pub unsafe fn instr64_DD_2_mem(addr: u64) { crate::cpu::instructions::instr16_DD_2_mem(addr) }
pub unsafe fn instr64_DD_2_reg(r: i32) { crate::cpu::instructions::instr16_DD_2_reg(r) }
pub unsafe fn instr64_DD_3_mem(addr: u64) { crate::cpu::instructions::instr16_DD_3_mem(addr) }
pub unsafe fn instr64_DD_3_reg(r: i32) { crate::cpu::instructions::instr16_DD_3_reg(r) }
pub unsafe fn instr64_DD_4_mem(addr: u64) { crate::cpu::instructions::instr16_DD_4_mem(addr) }
pub unsafe fn instr64_DD_4_reg(r: i32) { crate::cpu::instructions::instr16_DD_4_reg(r) }
pub unsafe fn instr64_DD_5_mem(addr: u64) { crate::cpu::instructions::instr16_DD_5_mem(addr) }
pub unsafe fn instr64_DD_5_reg(r: i32) { crate::cpu::instructions::instr16_DD_5_reg(r) }
pub unsafe fn instr64_DD_6_mem(addr: u64) { crate::cpu::instructions::instr16_DD_6_mem(addr) }
pub unsafe fn instr64_DD_6_reg(r: i32) { crate::cpu::instructions::instr16_DD_6_reg(r) }
pub unsafe fn instr64_DD_7_mem(addr: u64) { crate::cpu::instructions::instr16_DD_7_mem(addr) }
pub unsafe fn instr64_DD_7_reg(r: i32) { crate::cpu::instructions::instr16_DD_7_reg(r) }
pub unsafe fn instr64_E0(a0: i32) {
    if is_asize_32() {
        crate::cpu::misc_instr::loopne32(a0);
        return;
    }
    let c = read_reg64(ECX).wrapping_sub(1);
    write_reg64(ECX, c);
    jmpcc64(c != 0 && !getzf(), a0);
}
pub unsafe fn instr64_E1(a0: i32) {
    if is_asize_32() {
        crate::cpu::misc_instr::loope32(a0);
        return;
    }
    let c = read_reg64(ECX).wrapping_sub(1);
    write_reg64(ECX, c);
    jmpcc64(c != 0 && getzf(), a0);
}
pub unsafe fn instr64_E2(a0: i32) {
    if is_asize_32() {
        crate::cpu::misc_instr::loop32(a0);
        return;
    }
    let c = read_reg64(ECX).wrapping_sub(1);
    write_reg64(ECX, c);
    jmpcc64(c != 0, a0);
}
pub unsafe fn instr64_E3(a0: i32) {
    if is_asize_32() {
        crate::cpu::misc_instr::jcxz32(a0);
        return;
    }
    jmpcc64(read_reg64(ECX) == 0, a0);
}
pub unsafe fn instr64_E5(a0: i32) { write_reg32(EAX, io_port_read32(a0)); }
pub unsafe fn instr64_E7(a0: i32) { io_port_write32(a0, read_reg32(EAX)); }
pub unsafe fn instr64_E8(a0: i32) {
    return_on_pagefault!(push64(get_real_eip64()));
    *instruction_pointer = (*instruction_pointer).wrapping_add(a0 as i64 as u64);
}
pub unsafe fn instr64_E9(a0: i32) {
    *instruction_pointer = (*instruction_pointer).wrapping_add(a0 as i64 as u64);
}
pub unsafe fn instr64_EA(_a0: i32, _a1: i32) { trigger_ud(); }
pub unsafe fn instr64_EB(a0: i32) {
    // jmp rel8
    *instruction_pointer = (*instruction_pointer).wrapping_add(a0 as i64 as u64);
}
pub unsafe fn instr64_ED() {
    let port = read_reg16(DX);
    write_reg32(EAX, io_port_read32(port));
}
pub unsafe fn instr64_EF() {
    let port = read_reg16(DX);
    io_port_write32(port, read_reg32(EAX));
}
pub unsafe fn instr64_F26D() { insd_no_rep(is_asize_32()); }
pub unsafe fn instr64_F26F() { outsd_no_rep(is_asize_32(), segment_prefix(DS)); }
pub unsafe fn instr64_F2A5() { movsq_rep(is_asize_32(), segment_prefix(DS)); }
pub unsafe fn instr64_F2A7() { cmpsq_repnz(is_asize_32(), segment_prefix(DS)); }
pub unsafe fn instr64_F2AB() { stosq_rep(is_asize_32()); }
pub unsafe fn instr64_F2AD() { lodsq_rep(is_asize_32(), segment_prefix(DS)); }
pub unsafe fn instr64_F2AF() { scasq_repnz(is_asize_32()); }
pub unsafe fn instr64_F30FB8_mem(addr: u64, r: i32) {
    write_reg64(r, popcnt64(return_on_pagefault!(safe_read64s(addr))));
}
pub unsafe fn instr64_F30FB8_reg(r1: i32, r: i32) { write_reg64(r, popcnt64(read_reg64(r1))); }
pub unsafe fn instr64_F36D() { insd_no_rep(is_asize_32()); }
pub unsafe fn instr64_F36F() { outsd_no_rep(is_asize_32(), segment_prefix(DS)); }
pub unsafe fn instr64_F3A5() { movsq_rep(is_asize_32(), segment_prefix(DS)); }
pub unsafe fn instr64_F3A7() { cmpsq_repz(is_asize_32(), segment_prefix(DS)); }
pub unsafe fn instr64_F3AB() { stosq_rep(is_asize_32()); }
pub unsafe fn instr64_F3AD() { lodsq_rep(is_asize_32(), segment_prefix(DS)); }
pub unsafe fn instr64_F3AF() { scasq_repz(is_asize_32()); }
pub unsafe fn instr64_F7_0_mem(addr: u64, imm: i32) {
    test64(return_on_pagefault!(safe_read64s(addr)), imm as i64 as u64);
}
pub unsafe fn instr64_F7_0_reg(r1: i32, imm: i32) { test64(read_reg64(r1), imm as i64 as u64); }
pub unsafe fn instr64_F7_1_mem(addr: u64, imm: i32) {
    test64(return_on_pagefault!(safe_read64s(addr)), imm as i64 as u64);
}
pub unsafe fn instr64_F7_1_reg(r1: i32, imm: i32) { test64(read_reg64(r1), imm as i64 as u64); }
pub unsafe fn instr64_F7_2_mem(addr: u64) { safe_read_write64(addr, &|x| !x) }
pub unsafe fn instr64_F7_2_reg(r1: i32) { write_reg64(r1, !read_reg64(r1)); }
pub unsafe fn instr64_F7_3_mem(addr: u64) { safe_read_write64(addr, &|x| neg64(x)) }
pub unsafe fn instr64_F7_3_reg(r1: i32) { write_reg64(r1, neg64(read_reg64(r1))); }
pub unsafe fn instr64_F7_4_mem(addr: u64) { mul64(return_on_pagefault!(safe_read64s(addr))); }
pub unsafe fn instr64_F7_4_reg(r1: i32) { mul64(read_reg64(r1)); }
pub unsafe fn instr64_F7_5_mem(addr: u64) { imul64(return_on_pagefault!(safe_read64s(addr))); }
pub unsafe fn instr64_F7_5_reg(r1: i32) { imul64(read_reg64(r1)); }
pub unsafe fn instr64_F7_6_mem(addr: u64) { div64(return_on_pagefault!(safe_read64s(addr))); }
pub unsafe fn instr64_F7_6_reg(r1: i32) { div64(read_reg64(r1)); }
pub unsafe fn instr64_F7_7_mem(addr: u64) { idiv64(return_on_pagefault!(safe_read64s(addr))); }
pub unsafe fn instr64_F7_7_reg(r1: i32) { idiv64(read_reg64(r1)); }
pub unsafe fn instr64_FF_0_mem(addr: u64) {
    if !rex_w() {
        crate::cpu::instructions::instr32_FF_0_mem(addr);
        return;
    }
    safe_read_write64(addr, &|x| inc64(x))
}
pub unsafe fn instr64_FF_0_reg(r1: i32) {
    if !rex_w() {
        crate::cpu::instructions::instr32_FF_0_reg(r1);
        return;
    }
    write_reg64(r1, inc64(read_reg64(r1)));
}
pub unsafe fn instr64_FF_1_mem(addr: u64) {
    if !rex_w() {
        crate::cpu::instructions::instr32_FF_1_mem(addr);
        return;
    }
    safe_read_write64(addr, &|x| dec64(x))
}
pub unsafe fn instr64_FF_1_reg(r1: i32) {
    if !rex_w() {
        crate::cpu::instructions::instr32_FF_1_reg(r1);
        return;
    }
    write_reg64(r1, dec64(read_reg64(r1)));
}
pub unsafe fn instr64_FF_2_mem(addr: u64) {
    let data = return_on_pagefault!(safe_read64s(addr));
    return_on_pagefault!(push64(get_real_eip64()));
    *instruction_pointer = data;
}
pub unsafe fn instr64_FF_2_reg(r1: i32) {
    let data = read_reg64(r1);
    return_on_pagefault!(push64(get_real_eip64()));
    *instruction_pointer = data;
}
pub unsafe fn instr64_FF_3_mem(addr: u64) {
    // call far m16:64
    let new_rip = return_on_pagefault!(safe_read64s(addr));
    let new_cs = return_on_pagefault!(safe_read16(addr + 8));
    return_on_pagefault!(push64(*sreg.offset(CS as isize) as u64));
    return_on_pagefault!(push64(*instruction_pointer));
    far_jump64(new_rip, new_cs as i32, false);
}
pub unsafe fn instr64_FF_3_reg(_a0: i32) { trigger_ud(); }
pub unsafe fn instr64_FF_4_mem(addr: u64) {
    *instruction_pointer = return_on_pagefault!(safe_read64s(addr));
}
pub unsafe fn instr64_FF_4_reg(r1: i32) { *instruction_pointer = read_reg64(r1); }
pub unsafe fn instr64_FF_5_mem(addr: u64) {
    // jmp far m16:64
    let new_rip = return_on_pagefault!(safe_read64s(addr));
    let new_cs = return_on_pagefault!(safe_read16(addr + 8));
    far_jump64(new_rip, new_cs as i32, false);
}
pub unsafe fn instr64_FF_5_reg(_a0: i32) { trigger_ud(); }
pub unsafe fn instr64_FF_6_mem(addr: u64) {
    return_on_pagefault!(push64(return_on_pagefault!(safe_read64s(addr))));
}
pub unsafe fn instr64_FF_6_reg(r1: i32) {
    return_on_pagefault!(push64(read_reg64(r1)));
}
