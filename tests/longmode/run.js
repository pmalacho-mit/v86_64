#!/usr/bin/env node
// Long mode (64-bit) smoke test: assembles longmode.asm into a BIOS image,
// boots it, and verifies the results written by the 64-bit test code.
//
// The test program enters long mode from the reset vector (16-bit real mode
// -> protected mode -> CR4.PAE -> EFER.LME -> 4-level paging -> far jump into
// a 64-bit code segment) and exercises 64-bit instructions, storing qword
// results at 0x90000 and a 'K' byte on the serial port when done.

import url from "node:url";
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";

const __dirname = url.fileURLToPath(new URL(".", import.meta.url));
const root_path = path.join(__dirname, "..", "..");

process.on("unhandledRejection", exn => { throw exn; });

const asm_file = path.join(__dirname, "longmode.asm");
const bin_file = path.join(__dirname, "longmode.bin");

try {
    execFileSync("nasm", ["-w+error", "-f", "bin", "-o", bin_file, asm_file]);
} catch(e) {
    console.log("nasm not available or failed, test skipped");
    process.exit(0);
}

const TEST_RELEASE_BUILD = +process.env.TEST_RELEASE_BUILD;
const { V86 } = await import(TEST_RELEASE_BUILD ? root_path + "/build/libv86.mjs" : root_path + "/src/main.js");

const M = 0xFFFFFFFFFFFFFFFFn;
const expected = [
    0x1122334455667788n, // 0:  mov r64, imm64
    0x2233445566778899n, // 1:  mov r8, imm64 + add r64
    0x0000000089ABCDEFn, // 2:  32-bit write zero-extends
    0n,                  // 3:  jo taken on signed overflow
    0n,                  // 4:  js taken on negative result
    0x4142434445464700n, // 5:  sil (8-bit REX register)
    0x42n,               // 6:  r8b
    0xCAFEBABEDEADBEEFn, // 7:  RIP-relative load/store
    0x0123456789ABCDEFn, // 8:  SIB with r8 base, r9 index
    0x1234567890ABCDEFn, // 9:  push/pop r64
    0x42n,               // 10: call/ret
    0x8000000000000000n, // 11: shl by 63
    8n,                  // 12: shr by 60
    M,                   // 13: sar by 63
    0x23456789ABCDEF00n, // 14: shl by cl
    0n,                  // 15: imul 2^32 * 2^32 wraps
    1n,                  // 16: mul high qword
    0x123456789ABCDEn,   // 17: div quotient
    0xF0n,               // 18: div remainder
    0xFFFFFFFF80000001n, // 19: movsxd
    1n,                  // 20: movzx
    1n,                  // 21: movsx
    0x2222n,             // 22: xchg r64
    0xEFCDAB8967452301n, // 23: bswap r64
    0n,                  // 24: inc (ZF) + jz
    0n,                  // 25: 32-bit inc in long mode
    0n,                  // 26: lea RIP-relative vs absolute reference
    15n,                 // 27: loop with rcx
    0xAABBCCDDn,         // 28: fs base via wrmsr IA32_FS_BASE
    0x0F000F000F000F00n, // 29: and r64
    0xFF000F000F000F0Fn, // 30: or r64
    M,                   // 31: xor + not
    1n,                  // 32: neg
    1n,                  // 33: setl
    200n,                // 34: cmovl
    0xFFFFFFFF80000000n, // 35: cdqe
    M,                   // 36: cqo
    6n,                  // 37: adc
    0n,                  // 38: sbb with cf
    1n,                  // 39: syscall: rcx = rip after the syscall
    0x400n,              // 40: syscall: r11 = original rflags
    0n,                  // 41: syscall: rflags df bit masked by sfmask
    0xFEEDFACEF00DF00Dn, // 42: swapgs: gs:[0] after swapgs in the handler
    0x1234000n,          // 43: swapgs: ia32_gs_base after swapgs
    0x123456789ABCDEFn,  // 44: demand paging: value after #pf handler mapped the page
    1n,                  // 45: page fault error code: not-present
    1n,                  // 46: nx fault: I/D bit in the page fault error code
    0xAAAA2222BBBB1111n, // 47: cmpxchg16b: rax <- mem on mismatch
    0n,                  // 48: cmpxchg16b: rdx <- mem+8 on mismatch
    1n,                  // 49: cmpxchg16b: ZF set on exchange
    0x5555555555555555n, // 50: cmpxchg16b: mem <- rcx:rbx
    0xDEADn,             // 51: nx page: mov was skipped by the #PF handler
    0xDEC0DE1122334455n, // 52: movq xmm8 <-> r64
    0xDEC0FFFF2233FFFFn, // 53: movdqa/por with xmm9-11
    0xDEC0DE1122334455n, // 54: movdqu store/load low qword
    0x0000FFFF0000FFFFn, // 55: psrldq after movdqu load
    0x123456789ABCn,     // 56: cvtsi2sd/cvttsd2si with r64
    0xFFFFFFFFFFFE1DC0n, // 57: cvtsi2ss/cvttss2si with r64 (-123456)
    0xAAAA5555CCCC3333n, // 58: fxsave/fxrstor round trip with xmm14
    0x10n,               // 59: SIB lea with r12 index via REX.X
    0xFACEn,             // 60: r12 as SIB base via REX.B
    0xA5A5DEADn,         // 61: IA32_GS_BASE write/read-back
    0x1234000n,          // 62: IA32_KERNEL_GS_BASE untouched by the GS write
    0xCCCCCCCCCCCCCCCCn, // 63: rep stosq: poison baseline (64-bit memset of 16 qwords)
    0x1122334455667788n, // 64: rep stosq: last qword of the memset range
    0xCCCCCCCCCCCCCCCCn, // 65: rep stosq: first qword past the range left untouched
    1n,                  // 66: rep stosq: rdi advanced by exactly the written span
    0x123456789ABCDEFn,  // 67: freshly-mapped 4K page reads phys 0x800000 content
    0x0BADC0DEABAD1234n, // 68: cr3 reload flushes stale translation to phys 0x900000
    0x123456789ABCDEFn,  // 69: invlpg picks up the rewritten PTE
    0x3024n,             // 70: cpuid 0x80000008 address sizes (48 linear, 36 physical)
    0x71717171n,         // 71: write through a high alias of a page with jit entry points
    0x12345A78n,         // 72: mov eax, imm32 (zero-extends) + mov ah, imm8
    0x0008000000000123n, // 73: mov r64, imm64
    0xFFFFFFFFFFFFBEEFn, // 74: mov r16, imm16
    0xFFFFFFFFFFFFFF33n, // 75: mov sil, imm8 (REX)
    0xFFFFFFFFFFFFFF44n, // 76: mov r9b, imm8
    8n,                  // 77: bsf (0F BC) in jitted code
    9n,                  // 78: cmpxchg (0F B1) in jitted code
    0x30a348d26a2889an,  // 79: jl/jbe/js/jo/jb/jle after cmp/test/add/and in a hot loop
    0xb6bf6aaa379b26f0n, // 80: imul/shifts/setcc/cmovcc in a hot loop
    0x154690ceac782d0n,  // 81: rotates/shift by cl/inc cf/not/neg/movsxd/cdqe/cqo/bswap/xchg/call r
    50000n,              // 82: inc at a block start preserves cf for adc
    0x4f2ab3636643714n,  // 83: adc/sbb/bt*/cmpxchg/xadd/pushf in a hot loop
    0x200n,              // 84: sysret loads IF from r11 (it was ignored at cpl 3)
    0x31f72987e3c9ac08n, // 85: imul cf/of (32/64-bit, operands in and out of 32-bit range)
    0n,                  // 86: scratch slot of test 85, cleared
    0x941949e38be6f2b0n, // 87: hot loop with blocks on two pages
    0x254113b2en,        // 88: cf around inc/dec after add/sub
    0xe4c1f3ac2n,        // 89: registers written by string/sse/x87 instructions
    0x402000003F000000n, // 90: sse3 addsubps
    0x4012000000000000n, // 91: sse3 lddqu (unaligned) + addsubpd
    0x1122334455667788n, // 92: pinsrq/pextrq
    0x99aabbccn,         // 93: pextrd (zero-extended)
    0x0ae3273fn,         // 94: crc32 r64, r64
    0x2a47983an,         // 95: crc32 r32, sil
    0x88n,               // 96: pshufb xmm10, xmm11
    5n,                  // 97: pcmpestri with REX.W lengths
    0x7a87c5aef2b81b06n, // 98: cmpxchg (64/32-bit, memory/register)
    0x1c8fe0026780e30bn, // 99: xadd, inc/dec/add/sub/or/and (locked)
    0xa798e5ab28bdd1ccn, // 100: xchg, cmpxchg16b, lock bts/btr
    0xfedcba9876543210n, // 101: movhps xmm10, m64
    0xffffffffffffffffn, // 102: movlhps xmm15, xmm2 (high ^ low)
    0x0f1e2d3c4b5a6978n, // 103: movhpd xmm8, m64
    0x5052454654434821n, // 104: prefetch/prefetchw (0F 0D) are nops
    0x4d50582d4e4f5021n, // 105: mpx 0F 1A/0F 1B are nops
    0x1000000001n,       // 106: bt/bts m64, imm8 mask the offset to 63
    0x1234567800000001n, // 107: REX before lock is annulled (32-bit add)
    320n,                // 108: single-step traps armed by iretq in a hot loop (40 * 8)
    120n,                // 109: #GP on non-canonical data addresses (40 * 3)
    0x2828282828n,       // 110: loads through the canonical alias still work (40 * value)
    0n,                  // 111: #GP error codes are 0
    0n,                  // 112: pxor xmm, xmm (native) zeroes
    0xAAAAAAAAAAAAAAAAn, // 113: pxor xmm, m128 (native), low half
    0x9669966996699669n, // 114: pxor high half via movdqa store
    0x1FF8n,             // 115: paddq low half (lane wraps, no cross-lane carry)
    1n,                  // 116: paddq high half (independent lane)
    0x1FF8n,             // 117: movdqa register-to-register copy
    0x99AABBCCDDEEFF00n, // 118: movdqa load crossing a page boundary
    1n,                  // 119: movdqa store crossing a page boundary (high half of xmm13)
    0x0000000300000001n, // 120: paddd with per-lane wraparound
    0x02468ACF13579BDEn, // 121: psrld 3
    0x2468ACF03579BDE0n, // 122: pslld 4 (per-lane, bits don't cross lanes)
    0xDDEEFF0099AABBCCn, // 123: pshufd 0x1B low qword
    0x5566778811223344n, // 124: pshufd 0x1B high qword
    0xAAAAAAAAn,         // 125: punpcklqdq low qword
    0x0000000013121110n, // 126: psrldq 12
    0n,                  // 127: shl r8, 9 (count > operand width)
    0x1F0n,              // 128: shl r8, 4 (value 0xF0, cf 1 in bit 8)
    0x8n,                // 129: shr r16, 12
    0x1FFn,              // 130: sar r8, 12 (sign fill 0xFF, cf 1 in bit 8)
    0x0En,               // 131: sar r8, 3
    0n,                  // 132: shl m16, 20
    0x08n,               // 133: shl r8, cl=35 (masked to 5 bits)
    0x4000n,             // 134: shr r16, cl=1
    0xF8n,               // 135: sar r8, 1
    0n,                  // 136: cli clears IF at cpl 0
    0x1600141F101F0012n, // 137: pshufb low qword
    0x1E1D1C1B1A191817n, // 138: pshufb high qword
    0x0C0B0A0908070605n, // 139: palignr 5 low qword
    0x0D0C0B0A09080706n, // 140: palignr 17 of the previous result (from its destination)
    0n,                  // 141: palignr 40 (zero)
    8n,                  // 142: pop m64 with a faulting destination: rsp advanced once
    0x1111111111111111n, // 143: pop m64 with a faulting destination: the top of the stack was stored
    0xBBBBBBBBCCCCDDDDn, // 144: xchg r8, rax (49 90): rax
    0x1111111122223333n, // 145: xchg r8, rax (49 90): r8
    0x00000000CCCCDDDDn, // 146: xchg r8d, eax (41 90): rax
    0x0000000022223333n, // 147: xchg r8d, eax (41 90): r8
    0x111111112222DDDDn, // 148: xchg r8w, ax (66 41 90): rax
    0xBBBBBBBBCCCC3333n, // 149: xchg r8w, ax (66 41 90): r8
    0x1111111122223333n, // 150: 66 90 is a nop
    0x40005n,            // 151: lock xadd m64 restarted after a write #PF: memory
    5n,                  // 152: ... rax is the old value
    0n,                  // 153: lock cmpxchg m64 (unequal) after a write #PF: memory unchanged
    0n,                  // 154: ... rax loaded with the old value
    0n,                  // 155: ... ZF clear
    7n,                  // 156: lock cmpxchg m64 (equal) after a write #PF: memory
    5n,                  // 157: ... rax
    1n,                  // 158: ... ZF set
    // 159-162: opcodes invalid in 64-bit mode raise #UD, one bit per encoding (UD64_VARIANT)
    0x1FFFFFFFFFFFn,     // 159: without prefix
    0x1FFFFFFFFFFFn,     // 160: with 66
    0x1FFFFFFFFFFFn,     // 161: with REX (40)
    0x1FFFFFFFFFFFn,     // 162: with REX.W (48)
    3n,                  // 163: compatibility mode runs them (passes that reached the end)
    0n,                  // 164: compatibility mode: eip of a #UD (none)
    0x0123456789ABCDEFn, // 165: movaps xmm2, xmm4 (0F 29, register form): low qword
    0xFEDCBA9876543210n, // 166: movaps xmm2, xmm4 (0F 29, register form): high qword
    0x0123456789ABCDEFn, // 167: movups xmm5, xmm2 (0F 11, register form): low qword
    0xFEDCBA9876543210n, // 168: movups xmm5, xmm2 (0F 11, register form): high qword
    0x0123456789ABCDEFn, // 169: movdqa xmm6, xmm5 (66 0F 7F, register form): low qword
    0xFEDCBA9876543210n, // 170: movdqa xmm6, xmm5 (66 0F 7F, register form): high qword
    0x0123456789ABCDEFn, // 171: movdqu xmm7, xmm6 (F3 0F 7F, register form): low qword
    0xFEDCBA9876543210n, // 172: movdqu xmm7, xmm6 (F3 0F 7F, register form): high qword
    0x0123456789ABCDEFn, // 173: movaps xmm9, xmm7 (41 0F 29): low qword
    0xFEDCBA9876543210n, // 174: movaps xmm9, xmm7 (41 0F 29): high qword
    0x0123456789ABCDEFn, // 175: movups xmm10, xmm9 (45 0F 11): low qword
    0xFEDCBA9876543210n, // 176: movups xmm10, xmm9 (45 0F 11): high qword
    0x0123456789ABCDEFn, // 177: movdqa xmm0, xmm10 (66 44 0F 7F): low qword
    0xFEDCBA9876543210n, // 178: movdqa xmm0, xmm10 (66 44 0F 7F): high qword
    0x0123456789ABCDEFn, // 179: movdqu xmm11, xmm0 (F3 41 0F 7F): low qword
    0xFEDCBA9876543210n, // 180: movdqu xmm11, xmm0 (F3 41 0F 7F): high qword
    // 181-193: left to issue #11's branch (far call/jmp through m16:32)
    0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n,
    // 194-197: jcc to the next instruction after a flag-setting instruction (issue #16), passes
    // with the right flags in the following block, entered from the jcc and from another path
    50000n,              // 194: cmp r8 + jl +0 (rel8), from the jcc
    50000n,              // 195: ... from a 32-bit cmp elsewhere
    50000n,              // 196: add r32 + jl +0 (rel32), from the jcc
    50000n,              // 197: ... from a popfq elsewhere
];

const emulator = new V86({
    bios: { url: bin_file },
    autostart: true,
    memory_size: 32 * 1024 * 1024,
    log_level: 0,
    disable_jit: +process.env.DISABLE_JIT,
    // V86_WASM_PATH overrides the core build, e.g. build/v86-mem64-debug.wasm
    wasm_path: process.env.V86_WASM_PATH,
});

// JIT64_CHAINING=0: don't chain modules (as on hosts without wasm tail calls)
if(process.env.JIT64_CHAINING === "0")
{
    emulator.bus.register("emulator-started", () => {
        emulator.v86.cpu.wm.exports["set_jit_config"](7, 0);
    });
}

// JIT32_CHAINING=0: don't chain 16/32-bit modules
if(process.env.JIT32_CHAINING === "0")
{
    emulator.bus.register("emulator-started", () => {
        emulator.v86.cpu.wm.exports["set_jit_config"](10, 0);
    });
}

// JIT_THRESHOLD=<n>: compile code after n executed instructions instead of the default
// (a low value exercises the jit on all of the test's code)
if(process.env.JIT_THRESHOLD)
{
    emulator.bus.register("emulator-started", () => {
        emulator.v86.cpu.wm.exports["set_jit_config"](5, +process.env.JIT_THRESHOLD);
    });
}

// JIT64_DEFER_FLAGS=0: eager lazy-flag stores instead of keeping them in wasm locals
if(process.env.JIT64_DEFER_FLAGS === "0")
{
    emulator.bus.register("emulator-started", () => {
        emulator.v86.cpu.wm.exports["set_jit_config"](8, 0);
    });
}

// JIT64_TLB_CACHE=0: no cached data tlb entry in the 64-bit jit
if(process.env.JIT64_TLB_CACHE === "0")
{
    emulator.bus.register("emulator-started", () => {
        emulator.v86.cpu.wm.exports["set_jit_config"](9, 0);
    });
}

const timeout = setTimeout(() => {
    throw new Error("Timeout waiting for longmode test to finish");
}, 60 * 1000);

emulator.add_listener("serial0-output-byte", function(byte)
{
    if(byte !== 0x4B) // 'K'
    {
        return;
    }
    clearTimeout(timeout);

    const data = emulator.read_memory(0x90000, expected.length * 8);
    const view = new BigUint64Array(data.buffer, data.byteOffset, expected.length);

    const failures = [];
    for(let i = 0; i < expected.length; i++)
    {
        if(view[i] !== expected[i])
        {
            failures.push({
                name: `test ${i}`,
                expected: "0x" + expected[i].toString(16),
                actual: "0x" + view[i].toString(16),
            });
        }
    }

    if(failures.length === 0)
    {
        console.log(`[+] All ${expected.length} longmode tests passed`);
        emulator.destroy();
        process.exit(0);
    }
    else
    {
        console.table(failures);
        console.error(`[-] ${failures.length}/${expected.length} longmode tests failed`);
        process.exit(1);
    }
});
