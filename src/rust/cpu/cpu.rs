#![allow(non_upper_case_globals)]

use crate::config;
use crate::cpu::fpu::fpu_set_tag_word;
use crate::cpu::global_pointers::*;
use crate::cpu::memory;
use crate::cpu::misc_instr::{
    adjust_stack_reg, get_stack_pointer, get_stack_pointer64, getaf, getcf, getof, getpf, getsf,
    getzf, pop16, pop32s, pop64, push16, push32,
};
use crate::cpu::modrm::{resolve_modrm16, resolve_modrm32, resolve_modrm64};
use crate::cpu::{apic, ioapic, pic};
use crate::dbg::dbg_trace;
use crate::gen;
use crate::jit;
use crate::jit::is_near_end_of_page;
use crate::opstats;
use crate::page::Page;
use crate::paging::OrPageFault;
use crate::prefix;
use crate::profiler;
use crate::profiler::stat;
use crate::softfloat;
use crate::state_flags::CachedStateFlags;

use crate::fxhash::HashSet;
use std::ptr;

mod wasm {
    extern "C" {
        pub fn call_indirect_jit32(f: i32, x: u16);
        pub fn call_indirect_jit64(f: i32, x: u16);
    }
}

pub mod js {
    #[link(wasm_import_module = "env")]
    extern "C" {
        pub fn cpu_exception_hook(interrupt: i32) -> bool;
        pub fn microtick() -> f64;
        pub fn run_hardware_timers(acpi_enabled: bool, t: f64) -> f64;
        pub fn cpu_event_halt();
        pub fn stop_idling();

        pub fn io_port_read8(port: i32) -> i32;
        pub fn io_port_read16(port: i32) -> i32;
        pub fn io_port_read32(port: i32) -> i32;

        pub fn io_port_write8(port: i32, value: i32);
        pub fn io_port_write16(port: i32, value: i32);
        pub fn io_port_write32(port: i32, value: i32);

        pub fn get_rand_int() -> i32;
    }
}

/// The offset for our generated functions in the wasm table. Every index less than this is
/// reserved for rustc's indirect functions
pub const WASM_TABLE_OFFSET: u32 = 1024;

#[derive(Copy, Clone)]
#[repr(C)]
#[repr(align(16))]
pub union reg128 {
    pub i8: [i8; 16],
    pub i16: [i16; 8],
    pub i32: [i32; 4],
    pub i64: [i64; 2],
    pub u8: [u8; 16],
    pub u16: [u16; 8],
    pub u32: [u32; 4],
    pub u64: [u64; 2],
    pub f32: [f32; 4],
    pub f64: [f64; 2],
}

pub const CHECK_MISSED_ENTRY_POINTS: bool = false;

pub const INTERPRETER_ITERATION_LIMIT: u32 = 100_001;

// How often, in milliseconds, to yield to the browser for rendering and running events
pub const TIME_PER_FRAME: f64 = 1.0;

pub const FLAG_SUB: i32 = -0x8000_0000;
pub const FLAG_CARRY: i32 = 1;
pub const FLAG_PARITY: i32 = 4;
pub const FLAG_ADJUST: i32 = 16;
pub const FLAG_ZERO: i32 = 64;
pub const FLAG_SIGN: i32 = 128;
pub const FLAG_TRAP: i32 = 256;
pub const FLAG_INTERRUPT: i32 = 512;
pub const FLAG_DIRECTION: i32 = 1024;
pub const FLAG_OVERFLOW: i32 = 2048;
pub const FLAG_IOPL: i32 = 1 << 12 | 1 << 13;
pub const FLAG_NT: i32 = 1 << 14;
pub const FLAG_RF: i32 = 1 << 16;
pub const FLAG_VM: i32 = 1 << 17;
pub const FLAG_AC: i32 = 1 << 18;
pub const FLAG_VIF: i32 = 1 << 19;
pub const FLAG_VIP: i32 = 1 << 20;
pub const FLAG_ID: i32 = 1 << 21;
pub const FLAGS_DEFAULT: i32 = 1 << 1;
pub const FLAGS_MASK: i32 = FLAG_CARRY
    | FLAG_PARITY
    | FLAG_ADJUST
    | FLAG_ZERO
    | FLAG_SIGN
    | FLAG_TRAP
    | FLAG_INTERRUPT
    | FLAG_DIRECTION
    | FLAG_OVERFLOW
    | FLAG_IOPL
    | FLAG_NT
    | FLAG_RF
    | FLAG_VM
    | FLAG_AC
    | FLAG_VIF
    | FLAG_VIP
    | FLAG_ID;
pub const FLAGS_ALL: i32 =
    FLAG_CARRY | FLAG_PARITY | FLAG_ADJUST | FLAG_ZERO | FLAG_SIGN | FLAG_OVERFLOW;
pub const OPSIZE_8: i32 = 7;
pub const OPSIZE_16: i32 = 15;
pub const OPSIZE_32: i32 = 31;
pub const OPSIZE_64: i32 = 63;

pub const EAX: i32 = 0;
pub const ECX: i32 = 1;
pub const EDX: i32 = 2;
pub const EBX: i32 = 3;
pub const ESP: i32 = 4;
pub const EBP: i32 = 5;
pub const ESI: i32 = 6;
pub const EDI: i32 = 7;

pub const AX: i32 = 0;
pub const CX: i32 = 1;
pub const DX: i32 = 2;
pub const BX: i32 = 3;
pub const SP: i32 = 4;
pub const BP: i32 = 5;
pub const SI: i32 = 6;
pub const DI: i32 = 7;

pub const AL: i32 = 0;
pub const CL: i32 = 1;
pub const DL: i32 = 2;
pub const BL: i32 = 3;
pub const AH: i32 = 4;
pub const CH: i32 = 5;
pub const DH: i32 = 6;
pub const BH: i32 = 7;

pub const ES: i32 = 0;
pub const CS: i32 = 1;
pub const SS: i32 = 2;
pub const DS: i32 = 3;
pub const FS: i32 = 4;
pub const GS: i32 = 5;
pub const TR: i32 = 6;

pub const LDTR: i32 = 7;
pub const PAGE_TABLE_PRESENT_MASK: i32 = 1 << 0;
pub const PAGE_TABLE_RW_MASK: i32 = 1 << 1;
pub const PAGE_TABLE_USER_MASK: i32 = 1 << 2;
pub const PAGE_TABLE_ACCESSED_MASK: i32 = 1 << 5;
pub const PAGE_TABLE_DIRTY_MASK: i32 = 1 << 6;
pub const PAGE_TABLE_PSE_MASK: i32 = 1 << 7;
pub const PAGE_TABLE_GLOBAL_MASK: i32 = 1 << 8;
// Reserved bits in 64-bit paging-structure entries (PAE/long mode): physical
// address bits PHYSICAL_ADDRESS_BITS..51. In long mode bits 52..62 are ignored
// (available to software), bit 63 is NX (handled separately against EFER.NXE)
pub const PHYSICAL_ADDRESS_BITS: u32 = 36;
pub const PAE_ENTRY_RSVD: u64 = 0x000F_FFFF_FFFF_FFFF & !((1 << PHYSICAL_ADDRESS_BITS) - 1);
// PAE paging outside of long mode also reserves bits 52..62
pub const PAE_LEGACY_ENTRY_RSVD: u64 = 0x7FFF_FFFF_FFFF_FFFF & !((1 << PHYSICAL_ADDRESS_BITS) - 1);
// The address field (bits 12..51) of 64-bit paging-structure entries
pub const PAE_ENTRY_ADDRESS: u64 = 0x000F_FFFF_FFFF_F000;
// Reserved bits 20:13 in 2 MiB PDEs (PAE/long mode); 4 MiB PDEs in legacy
// 32-bit mode use some of these as address bits (PSE-36)
pub const PAE_PDE_PS_RSVD: u64 = 0x1F_E000;
pub const MMAP_BLOCK_BITS: i32 = 17;
pub const MMAP_BLOCK_SIZE: i32 = 1 << MMAP_BLOCK_BITS;
pub const CR0_PE: i32 = 1;
pub const CR0_MP: i32 = 1 << 1;
pub const CR0_EM: i32 = 1 << 2;
pub const CR0_TS: i32 = 1 << 3;
pub const CR0_ET: i32 = 1 << 4;
pub const CR0_WP: i32 = 1 << 16;
pub const CR0_AM: i32 = 1 << 18;
pub const CR0_NW: i32 = 1 << 29;
pub const CR0_CD: i32 = 1 << 30;
pub const CR0_PG: i32 = 1 << 31;
pub const CR4_VME: i32 = 1;
pub const CR4_PVI: i32 = 1 << 1;
pub const CR4_TSD: i32 = 1 << 2;
pub const CR4_PSE: i32 = 1 << 4;
pub const CR4_DE: i32 = 1 << 3;
pub const CR4_PAE: i32 = 1 << 5;
pub const CR4_PGE: i32 = 1 << 7;
pub const CR4_OSFXSR: i32 = 1 << 9;
pub const CR4_OSXMMEXCPT: i32 = 1 << 10;
pub const CR4_SMEP: i32 = 1 << 20;

pub const TSR_BACKLINK: i32 = 0x00;
pub const TSR_CR3: i32 = 0x1C;
pub const TSR_EIP: i32 = 0x20;
pub const TSR_EFLAGS: i32 = 0x24;

pub const TSR_EAX: i32 = 0x28;
pub const TSR_ECX: i32 = 0x2c;
pub const TSR_EDX: i32 = 0x30;
pub const TSR_EBX: i32 = 0x34;
pub const TSR_ESP: i32 = 0x38;
pub const TSR_EBP: i32 = 0x3c;
pub const TSR_ESI: i32 = 0x40;
pub const TSR_EDI: i32 = 0x44;

pub const TSR_ES: i32 = 0x48;
pub const TSR_CS: i32 = 0x4c;
pub const TSR_SS: i32 = 0x50;
pub const TSR_DS: i32 = 0x54;
pub const TSR_FS: i32 = 0x58;
pub const TSR_GS: i32 = 0x5c;
pub const TSR_LDT: i32 = 0x60;

pub const IA32_TIME_STAMP_COUNTER: i32 = 0x10;
pub const IA32_PLATFORM_ID: i32 = 0x17;
pub const IA32_APIC_BASE: i32 = 0x1B;
pub const MSR_TEST_CTRL: i32 = 0x33;
pub const MSR_SMI_COUNT: i32 = 0x34;
pub const IA32_FEAT_CTL: i32 = 0x3A;
pub const IA32_SPEC_CTRL: i32 = 0x48;
pub const IA32_BIOS_UPDT_TRIG: i32 = 0x79;
pub const IA32_BIOS_SIGN_ID: i32 = 0x8B;
pub const IA32_PMC0: i32 = 0xC1;
pub const IA32_PMC1: i32 = 0xC2;
pub const MSR_PLATFORM_INFO: i32 = 0xCE;
pub const IA32_ARCH_CAPABILITIES: i32 = 0x10A;
pub const MSR_TSX_FORCE_ABORT: i32 = 0x10F;
pub const IA32_TSX_CTRL: i32 = 0x122;
pub const IA32_MCU_OPT_CTRL: i32 = 0x123;
pub const MISC_FEATURE_ENABLES: i32 = 0x140;
pub const IA32_SYSENTER_CS: i32 = 0x174;
pub const IA32_SYSENTER_ESP: i32 = 0x175;
pub const IA32_SYSENTER_EIP: i32 = 0x176;
pub const IA32_MCG_CAP: i32 = 0x179;
pub const IA32_PERFEVTSEL0: i32 = 0x186;
pub const IA32_PERFEVTSEL1: i32 = 0x187;
pub const IA32_MISC_ENABLE: i32 = 0x1A0;
pub const IA32_PAT: i32 = 0x277;
pub const IA32_RTIT_CTL: i32 = 0x570;
pub const MSR_PKG_C2_RESIDENCY: i32 = 0x60D;
pub const IA32_FS_BASE: i32 = 0xC0000100u32 as i32;
pub const IA32_GS_BASE: i32 = 0xC0000101u32 as i32;
pub const IA32_KERNEL_GS_BASE: i32 = 0xC0000102u32 as i32;
pub const MSR_AMD64_LS_CFG: i32 = 0xC0011020u32 as i32;
pub const MSR_AMD64_DE_CFG: i32 = 0xC0011029u32 as i32;

pub const IA32_EFER: i32 = 0xC0000080u32 as i32;
pub const IA32_STAR: i32 = 0xC0000081u32 as i32;
pub const IA32_LSTAR: i32 = 0xC0000082u32 as i32;
pub const IA32_CSTAR: i32 = 0xC0000083u32 as i32;
pub const IA32_SFMASK: i32 = 0xC0000084u32 as i32;

// IA32_ARCH_CAPABILITIES: the emulated cpu doesn't execute speculatively, so it
// reports itself as unaffected by the speculative-execution vulnerabilities. Without
// this Linux enables mitigations that are very expensive here, most of all page table
// isolation (two cr3 writes, i.e. tlb flushes, per syscall/interrupt from user mode),
// but also buffer clearing (verw) on kernel exit and the bhb clearing loop on entry.
pub const ARCH_CAPABILITIES: u64 = 1 << 0 // RDCL_NO (meltdown)
    | 1 << 4 // SSB_NO
    | 1 << 5 // MDS_NO
    | 1 << 6 // PSCHANGE_MC_NO (itlb multihit)
    | 1 << 8 // TAA_NO
    | 1 << 13 // SBDR_SSDP_NO
    | 1 << 14 // FBSDP_NO
    | 1 << 15 // PSDP_NO
    | 1 << 20 // BHI_NO
    | 1 << 24 // PBRSB_NO
    | 1 << 26 // GDS_NO
    | 1 << 27; // RFDS_NO

pub const EFER_SCE: u64 = 1 << 0;
pub const EFER_LME: u64 = 1 << 8;
pub const EFER_LMA: u64 = 1 << 10;
pub const EFER_NXE: u64 = 1 << 11;

pub const IA32_APIC_BASE_BSP: i32 = 1 << 8;
pub const IA32_APIC_BASE_EXTD: i32 = 1 << 10;
pub const IA32_APIC_BASE_EN: i32 = 1 << 11;

pub const IOAPIC_MEM_ADDRESS: u32 = 0xFEC00000;
pub const IOAPIC_MEM_SIZE: u32 = 32;
pub const APIC_MEM_ADDRESS: u32 = 0xFEE00000;
pub const APIC_MEM_SIZE: u32 = 0x1000;

pub const MXCSR_MASK: i32 = 0xffff;
pub const MXCSR_FZ: i32 = 1 << 15;
pub const MXCSR_DAZ: i32 = 1 << 6;
pub const MXCSR_RC_SHIFT: i32 = 13;

pub const VALID_TLB_ENTRY_MAX: i32 = 10000;
pub const TLB_VALID: i32 = 1 << 0;
pub const TLB_READONLY: i32 = 1 << 1;
pub const TLB_NO_USER: i32 = 1 << 2;
pub const TLB_IN_MAPPED_RANGE: i32 = 1 << 3;
pub const TLB_GLOBAL: i32 = 1 << 4;
pub const TLB_HAS_CODE: i32 = 1 << 5;
pub const TLB_NOT_EXECUTABLE: i32 = 1 << 6;
pub const IVT_SIZE: u32 = 0x400;
pub const CPU_EXCEPTION_DE: i32 = 0;
pub const CPU_EXCEPTION_DB: i32 = 1;
pub const CPU_EXCEPTION_NMI: i32 = 2;
pub const CPU_EXCEPTION_BP: i32 = 3;
pub const CPU_EXCEPTION_OF: i32 = 4;
pub const CPU_EXCEPTION_BR: i32 = 5;
pub const CPU_EXCEPTION_UD: i32 = 6;
pub const CPU_EXCEPTION_NM: i32 = 7;
pub const CPU_EXCEPTION_DF: i32 = 8;
pub const CPU_EXCEPTION_TS: i32 = 10;
pub const CPU_EXCEPTION_NP: i32 = 11;
pub const CPU_EXCEPTION_SS: i32 = 12;
pub const CPU_EXCEPTION_GP: i32 = 13;
pub const CPU_EXCEPTION_PF: i32 = 14;
pub const CPU_EXCEPTION_MF: i32 = 16;
pub const CPU_EXCEPTION_AC: i32 = 17;
pub const CPU_EXCEPTION_MC: i32 = 18;
pub const CPU_EXCEPTION_XM: i32 = 19;
pub const CPU_EXCEPTION_VE: i32 = 20;

pub const CHECK_TLB_INVARIANTS: bool = false;

pub const DEBUG: bool = cfg!(debug_assertions);

pub const LOOP_COUNTER: i32 = 100_003;

// should probably be kept in sync with APIC_TIMER_FREQ in apic.js
pub const TSC_RATE: f64 = 1_000_000.0;

pub static mut cpuid_level: u32 = 0x16;

pub static mut jit_block_boundary: bool = false;

const TSC_ENABLE_IMPRECISE_BROWSER_WORKAROUND: bool = true;

#[cfg(debug_assertions)]
const TSC_VERBOSE_LOGGING: bool = false;
#[cfg(debug_assertions)]
pub static mut tsc_last_extra: u64 = 0;

// the last value returned by rdtsc
pub static mut tsc_last_value: u64 = 0;
// the smallest difference between two rdtsc readings (depends on the browser's performance.now resolution)
pub static mut tsc_resolution: u64 = u64::MAX;
// how many times rdtsc was called and had to return the same value (due to browser's performance.now resolution)
pub static mut tsc_number_of_same_readings: u64 = 0;
// how often rdtsc was previously called without its value changing, used for interpolating quick
// consecutive calls between rdtsc (when it's called faster than the browser's performance.now
// changes)
pub static mut tsc_speed: u64 = 1;

// used for restoring the state
pub static mut tsc_offset: u64 = 0;

// repr(C): compiled 64-bit code reads it (jit64::gen_chain_to_next_module)
#[repr(C)]
pub struct Code {
    pub wasm_table_index: jit::WasmTableIndex,
    pub state_flags: CachedStateFlags,
    pub state_table: [u16; 0x1000],
}

/// Element type of tlb_data: i32 in the default build (host offsets are < 4 GiB), i64 in the
/// mem64 build (host offsets into the 64-bit guest memory may exceed 32 bits above 4 GiB).
/// All reads go through tlb_pick_entry (u64) and writes through tlb_put_entry.
#[cfg(not(feature = "mem64"))]
pub type TlbEntry = i32;
#[cfg(feature = "mem64")]
pub type TlbEntry = i64;

/// tlb entries carry unsigned address bits; zero-extend i32 entries (an i32 -> u64 cast would
/// sign-extend when bit 31 of the host address is set)
#[cfg(not(feature = "mem64"))]
#[inline(always)]
pub fn tlb_entry_to_u64(e: TlbEntry) -> u64 { e as u32 as u64 }
#[cfg(feature = "mem64")]
#[inline(always)]
pub fn tlb_entry_to_u64(e: TlbEntry) -> u64 { e as u64 }

/// Free list of Code allocations. Codes live in dedicated chunks that are never moved, so
/// the pointers in tlb_code / tlb_code_high (and in generated code) stay valid until freed.
/// Freed Codes are only pushed to this list, which keeps full tlb clears free of allocator
/// traffic (they free one entry per compiled page).
static mut code_free_list: Vec<ptr::NonNull<Code>> = Vec::new();

/// Allocate a Code (zeroed; set_tlb_code initialises all fields). Chunks are allocated
/// without dropping, so their addresses are stable for the lifetime of the emulator.
pub unsafe fn code_alloc() -> ptr::NonNull<Code> {
    #[allow(static_mut_refs)]
    if let Some(c) = code_free_list.pop() {
        return c;
    }
    const CHUNK: usize = 64;
    let bytes = std::mem::size_of::<Code>() * CHUNK;
    let align = std::mem::align_of::<Code>();
    let base = std::alloc::alloc_zeroed(std::alloc::Layout::from_size_align(bytes, align).unwrap())
        as *mut Code;
    dbg_assert!(!base.is_null());
    #[allow(static_mut_refs)]
    for i in (0..CHUNK).rev() {
        code_free_list.push(ptr::NonNull::new_unchecked(base.add(i)));
    }
    #[allow(static_mut_refs)]
    code_free_list.pop().unwrap()
}

/// Return a Code to the free list (the memory is not deallocated)
pub unsafe fn code_free(c: ptr::NonNull<Code>) {
    #[allow(static_mut_refs)]
    code_free_list.push(c)
}

pub static mut tlb_data: [TlbEntry; 0x100000] = [0; 0x100000];
pub static mut tlb_code: [Option<ptr::NonNull<Code>>; 0x100000] = [None; 0x100000];

// TLB for linear pages at or above 4 GiB (48-bit virtual addresses in long
// mode). Direct-mapped, keyed by the tag of the page (tlb_high_tag: the page
// number of the canonical, sign-extended address); an entry is valid if
// tlb_high_page[idx] == tlb_high_tag(page) and tlb_high_entry[idx] != 0. The entry format is
// the same as in tlb_data, but as u64 (the page number << 12 may exceed 32
// bits): memory::tlb_host_base(phys) ^ (page << 12) | info_bits, so the
// physical address is reconstructed by phys_of_tlb_entry.
pub const TLB_HIGH_SIZE: usize = 0x40000;
pub static mut tlb_high_page: [u64; TLB_HIGH_SIZE] = [0; TLB_HIGH_SIZE];
pub static mut tlb_high_entry: [u64; TLB_HIGH_SIZE] = [0; TLB_HIGH_SIZE];
// jitted code for the page in tlb_high_page[idx] (the counterpart of tlb_code
// for pages at or above 4 GiB); dropped whenever the slot is evicted
pub static mut tlb_code_high: [Option<ptr::NonNull<Code>>; TLB_HIGH_SIZE] = [None; TLB_HIGH_SIZE];

/// Remove the high TLB entry at idx, including its jitted code
#[inline]
pub unsafe fn tlb_high_evict(idx: usize) {
    tlb_high_page[idx] = 0;
    tlb_high_entry[idx] = 0;
    if let Some(c) = tlb_code_high[idx].take() {
        code_free(c);
    }
}

/// The jitted-code slot for a virtual page: pages below 4 GiB always have one,
/// higher pages only while they are in the TLB
#[inline]
pub unsafe fn tlb_code_slot(page: u64) -> Option<&'static mut Option<ptr::NonNull<Code>>> {
    if page < 0x10_0000 {
        Some(&mut *ptr::addr_of_mut!(tlb_code[page as usize]))
    }
    else {
        let idx = tlb_high_index(page);
        if tlb_high_page[idx] == tlb_high_tag(page) {
            Some(&mut *ptr::addr_of_mut!(tlb_code_high[idx]))
        }
        else {
            None
        }
    }
}

/// The jitted code for a virtual page, if any
#[inline]
pub unsafe fn tlb_code_get(page: u64) -> Option<ptr::NonNull<Code>> {
    match tlb_code_slot(page) {
        Some(slot) => *slot,
        None => None,
    }
}

/// Strip the canonical sign-extension bits (63:48), which don't take part in
/// translation, so that TLB page numbers and page-walk indices fit.
#[inline]
pub fn canonicalize_address(address: u64) -> u64 { address & 0x0000_FFFF_FFFF_FFFF }

#[inline]
pub fn is_canonical_address(address: u64) -> bool { (address << 16) as i64 >> 16 == address as i64 }

/// The tag of a (48-bit) page number in tlb_high_page: the page number of the canonical
/// address, i.e. sign-extended from bit 35. A memory access looks up its address' arithmetic
/// page number (address as i64 >> 12) directly, which is a tag only for canonical addresses:
/// non-canonical ones miss and reach do_page_walk, which raises #GP.
#[inline]
pub fn tlb_high_tag(page: u64) -> u64 { ((page << 28) as i64 >> 28) as u64 }

/// The slot of a tag in the high TLB (replicated by the jit, see jit64::gen_tlb_high_entry)
#[inline]
pub fn tlb_high_hash(tag: u64) -> usize {
    (tag.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 25) as usize & (TLB_HIGH_SIZE - 1)
}

#[inline]
pub fn tlb_high_index(page: u64) -> usize { tlb_high_hash(tlb_high_tag(page)) }

/// The TLB entry (with info bits) for the page containing address, or 0 if none. Both forms
/// of an address (canonical or with bits 63:48 stripped) find the same entry; memory accesses
/// use tlb_pick_entry_for_access instead.
#[inline]
pub unsafe fn tlb_pick_entry(address: u64) -> u64 {
    let address = canonicalize_address(address);
    let page = address >> 12;
    if page < 0x10_0000 {
        tlb_entry_to_u64(tlb_data[page as usize])
    }
    else {
        let idx = tlb_high_index(page);
        if tlb_high_page[idx] == tlb_high_tag(page) {
            tlb_high_entry[idx]
        }
        else {
            0
        }
    }
}

/// Like tlb_pick_entry, but 0 for non-canonical addresses, so that memory accesses to them
/// take the slow path (do_page_walk), which raises #GP
#[inline(always)]
pub unsafe fn tlb_pick_entry_for_access(address: u64) -> u64 {
    if address >> 32 == 0 {
        tlb_entry_to_u64(tlb_data[(address >> 12) as usize])
    }
    else {
        let tag = (address as i64 >> 12) as u64;
        let idx = tlb_high_hash(tag);
        if tlb_high_page[idx] == tag {
            tlb_high_entry[idx]
        }
        else {
            0
        }
    }
}

/// The physical address of a (virtual) address, given the valid TLB entry of its page
#[inline(always)]
pub fn phys_of_tlb_entry(entry: u64, address: u64) -> u64 {
    memory::phys_of_tlb_host_base((entry & !0xFFF) ^ canonicalize_address(address))
}

/// Insert a TLB entry (as produced by do_page_walk) for the page containing address.
#[inline]
pub unsafe fn tlb_put_entry(address: u64, tlb_entry: u64) {
    let address = canonicalize_address(address);
    let page = address >> 12;
    if page < 0x10_0000 {
        tlb_data[page as usize] = tlb_entry as TlbEntry;
    }
    else {
        let idx = tlb_high_index(page);
        if tlb_high_page[idx] != tlb_high_tag(page) {
            tlb_high_evict(idx);
        }
        tlb_high_page[idx] = tlb_high_tag(page);
        tlb_high_entry[idx] = tlb_entry;
    }
}

/// Invalidate the TLB entry for the page containing address (if any).
#[inline]
pub unsafe fn tlb_invalidate_page(address: u64) {
    let address = canonicalize_address(address);
    let page = address >> 12;
    if page < 0x10_0000 {
        clear_tlb_code(page);
        tlb_data[page as usize] = 0;
    }
    else {
        let idx = tlb_high_index(page);
        if tlb_high_page[idx] == tlb_high_tag(page) {
            tlb_high_evict(idx);
        }
    }
}

// TLB membership list: page numbers can be up to 36 bits (48-bit linear
// address space), so entries must be stored at full width or high pages would
// silently survive flush/invalidate
pub static mut valid_tlb_entries: [u64; 10000] = [0; 10000];
pub static mut valid_tlb_entries_count: i32 = 0;

pub static mut in_jit: bool = false;

pub enum JitExitReason {
    None,
    CpuException {
        code: i32,
        error_code: Option<i32>,
    },
    /// Leave compiled code without an exception; the interpreter continues at
    /// instruction_pointer (used for self-modifying code, and for page switches to pages that
    /// can't be executed, where the interpreter's instruction fetch raises the page fault)
    SelfModifyingCodeBail,
}

pub static mut jit_exit_reason: JitExitReason = JitExitReason::None;

pub enum LastJump {
    Interrupt {
        phys_addr: u64,
        int: u8,
        software: bool,
        error: Option<u32>,
    },
    Compiled {
        phys_addr: u64,
    },
    Interpreted {
        phys_addr: u64,
    },
    None,
}
impl LastJump {
    pub fn phys_address(&self) -> Option<u64> {
        match self {
            LastJump::Interrupt { phys_addr, .. } => Some(*phys_addr),
            LastJump::Compiled { phys_addr } => Some(*phys_addr),
            LastJump::Interpreted { phys_addr } => Some(*phys_addr),
            LastJump::None => None,
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            LastJump::Interrupt { .. } => "interrupt",
            LastJump::Compiled { .. } => "compiled",
            LastJump::Interpreted { .. } => "interpreted",
            LastJump::None => "none",
        }
    }
}
pub static mut debug_last_jump: LastJump = LastJump::None;

#[derive(Copy, Clone)]
pub struct SegmentSelector {
    raw: u16,
}

impl SegmentSelector {
    pub fn of_u16(raw: u16) -> SegmentSelector { SegmentSelector { raw } }
    pub fn rpl(&self) -> u8 { (self.raw & 3) as u8 }
    pub fn is_gdt(&self) -> bool { (self.raw & 4) == 0 }
    pub fn descriptor_offset(&self) -> u16 { (self.raw & !7) as u16 }

    pub fn is_null(&self) -> bool { self.is_gdt() && self.descriptor_offset() == 0 }
}

// Used to indicate early that the selector cannot be used to fetch a descriptor
#[derive(PartialEq)]
pub enum SelectorNullOrInvalid {
    IsNull,
    OutsideOfTableLimit,
}

/// 64-bit interrupt/trap gate descriptor (16 bytes in long mode).
pub struct InterruptDescriptor64 {
    pub low: u64,
    pub high: u64,
}

impl InterruptDescriptor64 {
    pub fn of_u128(low: u64, high: u64) -> InterruptDescriptor64 {
        InterruptDescriptor64 { low, high }
    }
    pub fn offset(&self) -> u64 {
        self.low & 0xFFFF | self.low >> 32 & 0xFFFF_0000 | self.high << 32
    }
    pub fn selector(&self) -> u16 { (self.low >> 16 & 0xFFFF) as u16 }
    pub fn ist(&self) -> u8 { (self.low >> 32 & 7) as u8 }
    pub fn access_byte(&self) -> u8 { (self.low >> 40 & 0xFF) as u8 }
    pub fn dpl(&self) -> u8 { (self.access_byte() >> 5 & 3) as u8 }
    pub fn gate_type(&self) -> u8 { self.access_byte() & 0xF }
    pub fn is_present(&self) -> bool { self.access_byte() & 0x80 == 0x80 }
    /// interrupt (0xE) or trap (0xF) gate; these are the only valid types in long mode
    pub fn is_valid_long_mode_gate(&self) -> bool { self.gate_type() & !1 == 0xE }
    pub fn is_trap_gate(&self) -> bool { self.gate_type() & 1 == 1 }
}

pub struct SegmentDescriptor {
    pub raw: u64,
}

impl SegmentDescriptor {
    pub fn of_u64(raw: u64) -> SegmentDescriptor { SegmentDescriptor { raw } }
    pub fn base(&self) -> i32 {
        ((self.raw >> 16) & 0xffff | (self.raw & 0xff_00000000) >> 16 | (self.raw >> 56 << 24))
            as i32
    }
    pub fn limit(&self) -> u32 { (self.raw & 0xffff | ((self.raw >> 48) & 0xf) << 16) as u32 }
    pub fn access_byte(&self) -> u8 { ((self.raw >> 40) & 0xff) as u8 }
    pub fn flags(&self) -> u8 { ((self.raw >> 48 >> 4) & 0xf) as u8 }

    pub fn is_system(&self) -> bool { self.access_byte() & 0x10 == 0 }
    pub fn system_type(&self) -> u8 { self.access_byte() & 0xF }

    pub fn accessed(&self) -> bool { self.access_byte() & 1 == 1 }
    pub fn is_rw(&self) -> bool { self.access_byte() & 2 == 2 }
    pub fn is_dc(&self) -> bool { self.access_byte() & 4 == 4 }
    pub fn is_executable(&self) -> bool { self.access_byte() & 8 == 8 }
    pub fn is_present(&self) -> bool { self.access_byte() & 0x80 == 0x80 }
    pub fn is_writable(&self) -> bool { self.is_rw() && !self.is_executable() }
    pub fn is_readable(&self) -> bool { self.is_rw() || !self.is_executable() }
    pub fn is_conforming_executable(&self) -> bool { self.is_dc() && self.is_executable() }
    pub fn dpl(&self) -> u8 { (self.access_byte() >> 5) & 3 }
    pub fn is_32(&self) -> bool { self.flags() & 4 == 4 }
    // the L bit (64-bit code segment); only meaningful when EFER.LMA=1
    pub fn is_64(&self) -> bool { self.flags() & 2 == 2 }
    pub fn effective_limit(&self) -> u32 {
        if self.flags() & 8 == 8 {
            self.limit() << 12 | 0xFFF
        }
        else {
            self.limit()
        }
    }
    pub fn set_busy(&self) -> SegmentDescriptor {
        SegmentDescriptor {
            raw: self.raw | 2 << 40,
        }
    }
    pub fn clear_busy(&self) -> SegmentDescriptor {
        SegmentDescriptor {
            raw: self.raw & !(2 << 40),
        }
    }
    pub fn set_accessed(&self) -> SegmentDescriptor {
        SegmentDescriptor {
            raw: self.raw | 1 << 40,
        }
    }
}

pub struct InterruptDescriptor {
    raw: u64,
}

impl InterruptDescriptor {
    pub fn of_u64(raw: u64) -> InterruptDescriptor { InterruptDescriptor { raw } }
    pub fn offset(&self) -> i32 { (self.raw & 0xffff | self.raw >> 32 & 0xffff0000) as i32 }
    pub fn selector(&self) -> u16 { (self.raw >> 16 & 0xffff) as u16 }
    pub fn access_byte(&self) -> u8 { (self.raw >> 40 & 0xff) as u8 }
    pub fn dpl(&self) -> u8 { (self.access_byte() >> 5 & 3) as u8 }
    pub fn gate_type(&self) -> u8 { self.access_byte() & 7 }
    pub fn is_32(&self) -> bool { self.access_byte() & 8 == 8 }
    pub fn is_present(&self) -> bool { self.access_byte() & 0x80 == 0x80 }
    pub fn reserved_zeros_are_valid(&self) -> bool { self.access_byte() & 16 == 0 }

    const TASK_GATE: u8 = 0b101;
    const INTERRUPT_GATE: u8 = 0b110;
    const TRAP_GATE: u8 = 0b111;
}

pub unsafe fn switch_cs_real_mode(selector: i32) {
    dbg_assert!(!*protected_mode || vm86_mode());

    *sreg.offset(CS as isize) = selector as u16;
    *segment_is_null.offset(CS as isize) = false;
    *segment_offsets.offset(CS as isize) = selector << 4;
    update_cs_size(false, false);
}

unsafe fn get_tss_ss_esp(dpl: u8) -> OrPageFault<(i32, i32)> {
    Ok(if *tss_size_32 {
        let tss_stack_offset = ((dpl << 3) + 4) as u32;
        if tss_stack_offset + 7 > *segment_limits.offset(TR as isize) {
            panic!("#TS handler");
        }
        let addr = translate_address_system_read(
            (*segment_offsets.offset(TR as isize) + tss_stack_offset as i32) as u32 as u64,
        )?;
        dbg_assert!(addr & 0xFFF <= 0x1000 - 6);
        (memory::read16(addr + 4), memory::read32s(addr))
    }
    else {
        let tss_stack_offset = ((dpl << 2) + 2) as u32;
        if tss_stack_offset + 3 > *segment_limits.offset(TR as isize) {
            panic!("#TS handler");
        }
        let addr = translate_address_system_read(
            (*segment_offsets.offset(TR as isize) + tss_stack_offset as i32) as u32 as u64,
        )?;
        dbg_assert!(addr & 0xFFF <= 0x1000 - 4);
        (memory::read16(addr + 2), memory::read16(addr))
    })
}

/// The RSP[dpl] stack pointer from the 64-bit TSS (used for privilege-level
/// switches when delivering interrupts in long mode).
unsafe fn get_tss_rsp64(dpl: u8) -> OrPageFault<u64> {
    // 64-bit TSS: rsp0/1/2 at offsets 4, 12, 20
    let tss_stack_offset = 4u32 + ((dpl as u32) << 3);
    if tss_stack_offset + 7 > *segment_limits.offset(TR as isize) {
        trigger_ts(0);
        return Err(());
    }
    let tss_base = *tss_base64;
    let addr = translate_address_system_read(tss_base + tss_stack_offset as u64)?;
    Ok(memory::read64s(addr) as u64)
}

/// The ISTn stack pointer from the 64-bit TSS (ist is 1-7).
unsafe fn get_tss_ist64(ist: u8) -> OrPageFault<u64> {
    dbg_assert!(ist >= 1 && ist <= 7);
    // 64-bit TSS: reserved@0x00, RSP0..2 at 0x04, IST1..7 at 0x1C
    let offset = 0x1cu32 + ((ist as u32 - 1) << 3);
    if offset + 7 > *segment_limits.offset(TR as isize) {
        trigger_ts(0);
        return Err(());
    }
    let tss_base = *tss_base64;
    let addr = translate_address_system_read(tss_base + offset as u64)?;
    Ok(memory::read64s(addr) as u64)
}

pub unsafe fn iret16() { iret(true); }
pub unsafe fn iret32() { iret(false); }

/// 64-bit interrupt return (iretq).
pub unsafe fn iret64() {
    dbg_assert!(*is_64);
    *nmi_blocked = false;

    let rsp = read_reg64(ESP);
    let new_rip = return_on_pagefault!(safe_read64s(rsp));
    let new_cs = return_on_pagefault!(safe_read64s(rsp + 8)) as u16;
    let new_flags = return_on_pagefault!(safe_read64s(rsp + 16)) as u32 as i32;

    let cs_selector = SegmentSelector::of_u16(new_cs);
    if cs_selector.is_null() {
        dbg_log!("#gp iret64: null cs selector");
        trigger_gp(0);
        return;
    }

    let cs_descriptor = match return_on_pagefault!(lookup_segment_selector(cs_selector)) {
        Ok((desc, _)) => desc,
        Err(SelectorNullOrInvalid::IsNull) => {
            trigger_gp(new_cs as i32 & !3);
            return;
        },
        Err(SelectorNullOrInvalid::OutsideOfTableLimit) => {
            dbg_log!("#gp iret64: cs selector outside of table limit");
            trigger_gp(new_cs as i32 & !3);
            return;
        },
    };

    if !cs_descriptor.is_executable() || !cs_descriptor.is_present() {
        dbg_log!("#gp iret64: cs not executable or not present");
        trigger_gp(new_cs as i32 & !3);
        return;
    }

    let new_cpl = cs_selector.rpl();
    if new_cpl < *cpl {
        dbg_log!("#gp iret64: rpl < cpl");
        trigger_gp(new_cs as i32 & !3);
        return;
    }
    if !cs_descriptor.is_dc() && cs_selector.rpl() != cs_descriptor.dpl() {
        dbg_log!("#gp iret64: non-conforming cs and rpl != dpl");
        trigger_gp(new_cs as i32 & !3);
        return;
    }
    if cs_descriptor.is_dc() && cs_descriptor.dpl() > cs_selector.rpl() {
        dbg_log!("#gp iret64: conforming cs and dpl > rpl");
        trigger_gp(new_cs as i32 & !3);
        return;
    }

    if new_cpl > *cpl {
        // outer privilege return: pop rsp and ss as well
        let new_rsp = return_on_pagefault!(safe_read64s(rsp + 24));
        let new_ss = return_on_pagefault!(safe_read64s(rsp + 32)) as u16;

        let ss_selector = SegmentSelector::of_u16(new_ss);
        if ss_selector.is_null() {
            // loading a null selector into ss is not allowed on iret
            dbg_log!("#gp iret64: null ss selector");
            trigger_gp(0);
            return;
        }
        let ss_descriptor = match return_on_pagefault!(lookup_segment_selector(ss_selector)) {
            Ok((desc, _)) => desc,
            Err(SelectorNullOrInvalid::IsNull)
            | Err(SelectorNullOrInvalid::OutsideOfTableLimit) => {
                dbg_log!("#gp iret64: invalid ss selector");
                trigger_gp(new_ss as i32 & !3);
                return;
            },
        };

        if ss_descriptor.is_system()
            || ss_selector.rpl() != new_cpl
            || !ss_descriptor.is_writable()
            || ss_descriptor.dpl() != new_cpl
        {
            dbg_log!("#gp iret64: invalid ss descriptor");
            trigger_gp(new_ss as i32 & !3);
            return;
        }
        if !ss_descriptor.is_present() {
            dbg_log!("#ss iret64: ss not present");
            trigger_ss(new_ss as i32 & !3);
            return;
        }

        // no exceptions below

        // rflags are updated with the permissions of the old cpl (at cpl 3,
        // update_eflags would keep if and iopl, like popf), as in iret32
        update_eflags_iret(new_flags);

        // switch_seg(SS) checks the descriptor against cpl; the checks in
        // switch_seg must see the *target* cpl (same order as iret32)
        *cpl = new_cpl;
        cpl_changed();

        write_reg64(ESP, new_rsp);

        if !switch_seg(SS, new_ss as i32) {
            dbg_assert!(false);
        }
    }
    else {
        // same privilege: the 64-bit frame is always 5 slots; the frame's RSP
        // is loaded even when returning to the same CPL (handlers may rewrite
        // it, e.g. the int3 selftest patching the pt_regs sp/rip), but SS is
        // left alone
        let new_rsp = return_on_pagefault!(safe_read64s(rsp + 24));
        write_reg64(ESP, new_rsp);

        update_eflags_iret(new_flags);
    }

    // no exceptions below

    *cpl = new_cpl;
    cpl_changed();

    *sreg.offset(CS as isize) = new_cs;
    *segment_is_null.offset(CS as isize) = false;
    *segment_limits.offset(CS as isize) = cs_descriptor.effective_limit();
    *segment_offsets.offset(CS as isize) = if *is_64 { 0 } else { cs_descriptor.base() };
    *segment_access_bytes.offset(CS as isize) = cs_descriptor.access_byte();

    update_cs_size(cs_descriptor.is_32(), cs_descriptor.is_64());

    *instruction_pointer = new_rip;

    update_state_flags();

    handle_irqs();
}

pub unsafe fn iret(is_16: bool) {
    *nmi_blocked = false;
    if vm86_mode() && getiopl() < 3 {
        // vm86 mode, iopl != 3
        dbg_log!("#gp iret vm86 mode, iopl != 3");
        trigger_gp(0);
        return;
    }

    let (new_eip, new_cs, mut new_flags) = if is_16 {
        (
            return_on_pagefault!(safe_read16(get_stack_pointer(0))),
            return_on_pagefault!(safe_read16(get_stack_pointer(2))),
            return_on_pagefault!(safe_read16(get_stack_pointer(4))),
        )
    }
    else {
        (
            return_on_pagefault!(safe_read32s(get_stack_pointer(0))),
            return_on_pagefault!(safe_read16(get_stack_pointer(4))),
            return_on_pagefault!(safe_read32s(get_stack_pointer(8))),
        )
    };

    if !*protected_mode || (vm86_mode() && getiopl() == 3) {
        if new_eip as u32 & 0xFFFF0000 != 0 {
            panic!("#GP handler");
        }

        switch_cs_real_mode(new_cs);
        *instruction_pointer = (get_seg_cs() as u32).wrapping_add(new_eip as u32) as u64;

        if is_16 {
            update_eflags_iret(new_flags | *flags & !0xFFFF);
            adjust_stack_reg(3 * 2);
        }
        else {
            if !*protected_mode {
                update_eflags_iret((new_flags & 0x257FD5) | (*flags & 0x1A0000));
            }
            else {
                update_eflags_iret(new_flags);
            }
            adjust_stack_reg(3 * 4);
        }

        update_state_flags();
        handle_irqs();
        return;
    }

    dbg_assert!(!vm86_mode());

    if *flags & FLAG_NT != 0 {
        // nested task: return to the task linked through the back-link field of the current tss
        let tss_offset = *segment_offsets.offset(TR as isize);
        let backlink = return_on_pagefault!(safe_read16((tss_offset + TSR_BACKLINK) as u32 as u64));
        do_task_switch(backlink, None, TaskSwitchSource::Iret);
        return;
    }

    if new_flags & FLAG_VM != 0 {
        if *cpl == 0 {
            // return to virtual 8086 mode

            // vm86 cannot be set in 16 bit flag
            dbg_assert!(!is_16);

            let temp_esp = return_on_pagefault!(safe_read32s(get_stack_pointer(12)));
            let temp_ss = return_on_pagefault!(safe_read16(get_stack_pointer(16)));

            let new_es = return_on_pagefault!(safe_read16(get_stack_pointer(20)));
            let new_ds = return_on_pagefault!(safe_read16(get_stack_pointer(24)));
            let new_fs = return_on_pagefault!(safe_read16(get_stack_pointer(28)));
            let new_gs = return_on_pagefault!(safe_read16(get_stack_pointer(32)));

            // no exceptions below

            update_eflags_iret(new_flags);
            *flags |= FLAG_VM;

            switch_cs_real_mode(new_cs);
            *instruction_pointer =
                (get_seg_cs() as u32).wrapping_add(new_eip as u32 & 0xFFFF) as u64;

            if !switch_seg(ES, new_es)
                || !switch_seg(DS, new_ds)
                || !switch_seg(FS, new_fs)
                || !switch_seg(GS, new_gs)
            {
                // XXX: Should be checked before side effects
                dbg_assert!(false);
            }

            adjust_stack_reg(9 * 4); // 9 dwords: eip, cs, flags, esp, ss, es, ds, fs, gs

            write_reg32(ESP, temp_esp);
            if !switch_seg(SS, temp_ss) {
                // XXX
                dbg_assert!(false);
            }

            *cpl = 3;
            cpl_changed();

            update_cs_size(false, false);
            update_state_flags();

            // iret end
            return;
        }
        else {
            dbg_log!("vm86 flag ignored because cpl != 0");
            new_flags &= !FLAG_VM;
        }
    }

    // protected mode return

    let cs_selector = SegmentSelector::of_u16(new_cs as u16);
    let cs_descriptor = match return_on_pagefault!(lookup_segment_selector(cs_selector)) {
        Ok((desc, _)) => desc,
        Err(SelectorNullOrInvalid::IsNull) => panic!("Unimplemented: CS selector is null"),
        Err(SelectorNullOrInvalid::OutsideOfTableLimit) => {
            panic!("Unimplemented: CS selector is invalid")
        },
    };

    if new_eip as u32 > cs_descriptor.effective_limit() {
        dbg_log!(
            "#gp iret: new_eip > cs_descriptor.effective_limit, new_eip={:x} cs_descriptor.effective_limit={:x}",
            new_eip as u32,
            cs_descriptor.effective_limit()
        );
        trigger_gp(new_cs & !3);
        return;
    }

    if !cs_descriptor.is_present() {
        panic!("not present");
    }
    if !cs_descriptor.is_executable() {
        panic!("not exec");
    }
    if cs_selector.rpl() < *cpl {
        panic!("rpl < cpl");
    }
    if cs_descriptor.is_dc() && cs_descriptor.dpl() > cs_selector.rpl() {
        panic!("conforming and dpl > rpl");
    }

    if !cs_descriptor.is_dc() && cs_selector.rpl() != cs_descriptor.dpl() {
        dbg_log!(
            "#gp iret: non-conforming cs and rpl != dpl, dpl={} rpl={}",
            cs_descriptor.dpl(),
            cs_selector.rpl()
        );
        trigger_gp(new_cs & !3);
        return;
    }

    if cs_selector.rpl() > *cpl {
        // outer privilege return
        let (temp_esp, temp_ss) = if is_16 {
            (
                return_on_pagefault!(safe_read16(get_stack_pointer(6))),
                return_on_pagefault!(safe_read16(get_stack_pointer(8))),
            )
        }
        else {
            (
                return_on_pagefault!(safe_read32s(get_stack_pointer(12))),
                return_on_pagefault!(safe_read16(get_stack_pointer(16))),
            )
        };

        let ss_selector = SegmentSelector::of_u16(temp_ss as u16);
        let ss_descriptor = match return_on_pagefault!(lookup_segment_selector(ss_selector)) {
            Ok((desc, _)) => desc,
            Err(SelectorNullOrInvalid::IsNull) => {
                dbg_log!("#GP for loading 0 in SS sel={:x}", temp_ss);
                dbg_trace();
                trigger_gp(0);
                return;
            },
            Err(SelectorNullOrInvalid::OutsideOfTableLimit) => {
                dbg_log!("#GP for loading invalid in SS sel={:x}", temp_ss);
                trigger_gp(temp_ss & !3);
                return;
            },
        };
        let new_cpl = cs_selector.rpl();

        if ss_descriptor.is_system()
            || ss_selector.rpl() != new_cpl
            || !ss_descriptor.is_writable()
            || ss_descriptor.dpl() != new_cpl
        {
            dbg_log!("#GP for loading invalid in SS sel={:x}", temp_ss);
            dbg_trace();
            trigger_gp(temp_ss & !3);
            return;
        }

        if !ss_descriptor.is_present() {
            dbg_log!("#SS for loading non-present in SS sel={:x}", temp_ss);
            dbg_trace();
            trigger_ss(temp_ss & !3);
            return;
        }

        // no exceptions below

        if is_16 {
            update_eflags_iret(new_flags | *flags & !0xFFFF);
        }
        else {
            update_eflags_iret(new_flags);
        }

        *cpl = cs_selector.rpl();
        cpl_changed();

        if !switch_seg(SS, temp_ss) {
            // XXX
            dbg_assert!(false);
        }

        set_stack_reg(temp_esp);

        if *cpl == 0 && !is_16 {
            *flags = *flags & !FLAG_VIF & !FLAG_VIP | (new_flags & (FLAG_VIF | FLAG_VIP));
        }

        for reg in [ES, DS, FS, GS] {
            let access = *segment_access_bytes.offset(reg as isize);
            let dpl = access >> 5 & 3;
            let executable = access & 8 == 8;
            let conforming = access & 4 == 4;
            if dpl < *cpl && !(executable && conforming) {
                //dbg_log!(
                //    "set segment to null sreg={} dpl={} executable={} conforming={}",
                //    reg,
                //    dpl,
                //    executable,
                //    conforming
                //);
                *segment_is_null.offset(reg as isize) = true;
                *sreg.offset(reg as isize) = 0;
            }
        }
    }
    else if cs_selector.rpl() == *cpl {
        // same privilege return
        // no exceptions below
        if is_16 {
            adjust_stack_reg(3 * 2);
            update_eflags_iret(new_flags | *flags & !0xFFFF);
        }
        else {
            adjust_stack_reg(3 * 4);
            update_eflags_iret(new_flags);
        }

        // update vip and vif, which are not changed by update_eflags
        if *cpl == 0 && !is_16 {
            *flags = *flags & !FLAG_VIF & !FLAG_VIP | (new_flags & (FLAG_VIF | FLAG_VIP));
        }
    }
    else {
        dbg_assert!(false);
    }

    *sreg.offset(CS as isize) = new_cs as u16;
    dbg_assert!((new_cs & 3) == *cpl as i32);

    update_cs_size(cs_descriptor.is_32(), cs_descriptor.is_64());

    *segment_limits.offset(CS as isize) = cs_descriptor.effective_limit();
    *segment_offsets.offset(CS as isize) = cs_descriptor.base();
    *segment_access_bytes.offset(CS as isize) = cs_descriptor.access_byte();

    *instruction_pointer = (get_seg_cs() as u32).wrapping_add(new_eip as u32) as u64;

    update_state_flags();

    // iret end

    handle_irqs();
}

/// Fault-class exceptions save RFLAGS with RF set in the pushed image, so
/// that an iret back to the faulting instruction doesn't immediately
/// re-trigger an instruction breakpoint (Intel SDM Vol. 3A, Table 6-1).
/// #DB is excluded: it is fault-class only for instruction breakpoints.
fn exception_is_fault(vector: i32) -> bool {
    matches!(
        vector,
        0 | 5 | 6 | 7 | 8 | 10 | 11 | 12 | 13 | 14 | 16 | 17 | 19 | 20 | 21
    )
}

pub unsafe fn call_interrupt_vector(
    interrupt_nr: i32,
    is_software_int: bool,
    error_code: Option<i32>,
) {
    // compiled 64-bit code that called into the interpreter must not continue
    crate::jit64::JIT64_EXIT = true;
    // a faulting/trapping instruction's watchpoint hits are discarded; its
    // re-execution (or the next instruction) records them again
    db_pending_data = 0;
    if *protected_mode {
        if *efer & EFER_LMA != 0 {
            // IA-32e mode (incl. compatibility mode): interrupt/exception
            // delivery uses 64-bit IDT gates and the 64-bit frame regardless
            // of the interrupted code segment's L bit
            let was_delivering = in_interrupt_delivery;
            in_interrupt_delivery = interrupt_nr;
            call_interrupt_vector64(interrupt_nr, is_software_int, error_code);
            in_interrupt_delivery = was_delivering;
            return;
        }

        if vm86_mode() && *cr.offset(4) & CR4_VME != 0 {
            panic!("Unimplemented: VME");
        }

        if vm86_mode() && is_software_int && getiopl() < 3 {
            dbg_log!("call_interrupt_vector #GP. vm86 && software int && iopl < 3");
            dbg_trace();
            trigger_gp(0);
            return;
        }

        if interrupt_nr << 3 | 7 > *idtr_size {
            dbg_log!("interrupt_nr={:x} idtr_size={:x}", interrupt_nr, *idtr_size);
            dbg_trace();
            panic!("Unimplemented: #GP handler");
        }

        let descriptor_address = return_on_pagefault!(translate_address_system_read(
            (*idtr_offset64).wrapping_add((interrupt_nr << 3) as u64)
        ));

        let descriptor = InterruptDescriptor::of_u64(memory::read64s(descriptor_address) as u64);

        let mut offset = descriptor.offset();
        let selector = descriptor.selector() as i32;
        let dpl = descriptor.dpl();
        let gate_type = descriptor.gate_type();

        if is_software_int && dpl < *cpl {
            dbg_log!("#gp software interrupt ({:x}) and dpl < cpl", interrupt_nr);
            dbg_trace();
            trigger_gp(interrupt_nr << 3 | 2);
            return;
        }

        if gate_type != InterruptDescriptor::TRAP_GATE
            && gate_type != InterruptDescriptor::INTERRUPT_GATE
            && gate_type != InterruptDescriptor::TASK_GATE
        {
            // invalid gate_type
            dbg_log!(
                "gate type invalid. gate_type=0b{:b} raw={:b}",
                gate_type,
                descriptor.raw
            );
            dbg_trace();
            panic!("Unimplemented: #GP handler");
        }

        if !descriptor.reserved_zeros_are_valid() {
            dbg_log!(
                "reserved 0s violated. gate_type=0b{:b} raw={:b}",
                gate_type,
                descriptor.raw
            );
            dbg_trace();
            panic!("Unimplemented: #GP handler");
        }

        if !descriptor.is_present() {
            // present bit not set
            dbg_log!("#np int descriptor not present, int={}", interrupt_nr);
            trigger_np(interrupt_nr << 3 | 2);
            return;
        }

        if gate_type == InterruptDescriptor::TASK_GATE {
            // task gate
            dbg_log!(
                "interrupt to task gate: int={:x} sel={:x} dpl={}",
                interrupt_nr,
                selector,
                dpl
            );
            dbg_trace();
            dbg_assert!(offset == 0, "TODO: Check this (likely #GP)");
            do_task_switch(selector, error_code, TaskSwitchSource::CallOrInt);
            return;
        }

        let cs_segment_descriptor = match return_on_pagefault!(lookup_segment_selector(
            SegmentSelector::of_u16(selector as u16)
        )) {
            Ok((desc, _)) => desc,
            Err(SelectorNullOrInvalid::IsNull) => {
                dbg_log!("is null");
                panic!("Unimplemented: #GP handler");
            },
            Err(SelectorNullOrInvalid::OutsideOfTableLimit) => {
                dbg_log!("is invalid");
                panic!("Unimplemented: #GP handler (error code)");
            },
        };

        dbg_assert!(offset as u32 <= cs_segment_descriptor.effective_limit());

        if !cs_segment_descriptor.is_executable() || cs_segment_descriptor.dpl() > *cpl {
            dbg_log!("not exec");
            panic!("Unimplemented: #GP handler");
        }
        if !cs_segment_descriptor.is_present() {
            // kvm-unit-test
            dbg_log!("not present");
            trigger_np(interrupt_nr << 3 | 2);
            return;
        }

        let mut old_flags = get_eflags();
        if exception_is_fault(interrupt_nr) || take_force_rf_in_exception_image() {
            old_flags |= FLAG_RF;
        }

        if !cs_segment_descriptor.is_dc() && cs_segment_descriptor.dpl() < *cpl {
            // inter privilege level interrupt
            // interrupt from vm86 mode

            if old_flags & FLAG_VM != 0 && cs_segment_descriptor.dpl() != 0 {
                panic!("Unimplemented: #GP handler for non-0 cs segment dpl when in vm86 mode");
            }

            let (new_ss, new_esp) =
                return_on_pagefault!(get_tss_ss_esp(cs_segment_descriptor.dpl()));

            let ss_segment_selector = SegmentSelector::of_u16(new_ss as u16);
            let ss_segment_descriptor =
                match return_on_pagefault!(lookup_segment_selector(ss_segment_selector)) {
                    Ok((desc, _)) => desc,
                    Err(
                        SelectorNullOrInvalid::IsNull | SelectorNullOrInvalid::OutsideOfTableLimit,
                    ) => {
                        panic!("Unimplemented: #TS handler");
                    },
                };

            if ss_segment_descriptor.is_dc() {
                dbg_assert!(new_esp as u32 > ss_segment_descriptor.effective_limit());
            }
            else {
                dbg_assert!(new_esp as u32 - 1 <= ss_segment_descriptor.effective_limit());
            }
            dbg_assert!(!ss_segment_descriptor.is_system() && ss_segment_descriptor.is_writable());

            if ss_segment_selector.rpl() != cs_segment_descriptor.dpl() {
                panic!("Unimplemented: #TS handler");
            }
            if ss_segment_descriptor.dpl() != cs_segment_descriptor.dpl()
                || !ss_segment_descriptor.is_rw()
            {
                panic!("Unimplemented: #TS handler");
            }
            if !ss_segment_descriptor.is_present() {
                panic!("Unimplemented: #TS handler");
            }

            let old_esp = read_reg32(ESP);
            let old_ss = *sreg.offset(SS as isize) as i32;

            let error_code_space = if error_code.is_some() { 1 } else { 0 };
            let vm86_space = if (old_flags & FLAG_VM) == FLAG_VM { 4 } else { 0 };
            let bytes_per_arg = if descriptor.is_32() { 4 } else { 2 };

            let stack_space = bytes_per_arg * (5 + error_code_space + vm86_space);
            let new_stack_pointer = ss_segment_descriptor.base()
                + if ss_segment_descriptor.is_32() {
                    new_esp - stack_space
                }
                else {
                    new_esp - stack_space & 0xFFFF
                };

            return_on_pagefault!(translate_address_system_write(
                new_stack_pointer as u32 as u64
            ));
            return_on_pagefault!(translate_address_system_write(
                (ss_segment_descriptor.base() + new_esp - 1) as u32 as u64
            ));

            // no exceptions below
            *cpl = cs_segment_descriptor.dpl();
            cpl_changed();

            update_cs_size(cs_segment_descriptor.is_32(), cs_segment_descriptor.is_64());

            *flags &= !FLAG_VM & !FLAG_RF;

            if !switch_seg(SS, new_ss) {
                // XXX
                dbg_assert!(false);
            }
            set_stack_reg(new_esp);

            // XXX: #SS if stack would cross stack limit

            if old_flags & FLAG_VM != 0 {
                if !descriptor.is_32() {
                    dbg_assert!(false);
                }
                else {
                    push32(*sreg.offset(GS as isize) as i32).unwrap();
                    push32(*sreg.offset(FS as isize) as i32).unwrap();
                    push32(*sreg.offset(DS as isize) as i32).unwrap();
                    push32(*sreg.offset(ES as isize) as i32).unwrap();
                }
            }

            if descriptor.is_32() {
                push32(old_ss).unwrap();
                push32(old_esp).unwrap();
            }
            else {
                push16(old_ss).unwrap();
                push16(old_esp & 0xFFFF).unwrap();
            }
        }
        else if cs_segment_descriptor.is_dc() || cs_segment_descriptor.dpl() == *cpl {
            // intra privilege level interrupt

            //dbg_log!("Intra privilege interrupt gate=" + h(selector, 4) + ":" + h(offset >>> 0, 8) +
            //        " gate_type=" + gate_type + " 16bit=" + descriptor.is_32() +
            //        " cpl=" + *cpl + " dpl=" + segment_descriptor.dpl() + " conforming=" + +segment_descriptor.is_dc(), );
            //debug.dump_regs_short();

            if *flags & FLAG_VM != 0 {
                dbg_assert!(false, "check error code");
                trigger_gp(selector & !3);
                return;
            }

            let bytes_per_arg = if descriptor.is_32() { 4 } else { 2 };
            let error_code_space = if error_code.is_some() { 1 } else { 0 };

            let stack_space = bytes_per_arg * (3 + error_code_space);

            // XXX: with current cpl or with cpl 0?
            return_on_pagefault!(writable_or_pagefault(
                get_stack_pointer(-stack_space),
                stack_space
            ));

        // no exceptions below
        }
        else {
            panic!("Unimplemented: #GP handler");
        }

        // XXX: #SS if stack would cross stack limit
        if descriptor.is_32() {
            push32(old_flags).unwrap();
            push32(*sreg.offset(CS as isize) as i32).unwrap();
            push32(get_real_eip()).unwrap();

            if let Some(ec) = error_code {
                push32(ec).unwrap();
            }
        }
        else {
            push16(old_flags & 0xFFFF).unwrap();
            push16(*sreg.offset(CS as isize) as i32).unwrap();
            push16(get_real_eip() & 0xFFFF).unwrap();

            if let Some(ec) = error_code {
                dbg_assert!(ec >= 0 && ec < 0x10000);
                push16(ec).unwrap();
            }

            offset &= 0xFFFF;
        }

        if old_flags & FLAG_VM != 0 {
            if !switch_seg(GS, 0) || !switch_seg(FS, 0) || !switch_seg(DS, 0) || !switch_seg(ES, 0)
            {
                // can't fail
                dbg_assert!(false);
            }
        }

        *sreg.offset(CS as isize) = (selector as u16) & !3 | *cpl as u16;
        dbg_assert!((*sreg.offset(CS as isize) & 3) == *cpl as u16);

        update_cs_size(cs_segment_descriptor.is_32(), cs_segment_descriptor.is_64());

        *segment_limits.offset(CS as isize) = cs_segment_descriptor.effective_limit();
        *segment_offsets.offset(CS as isize) = cs_segment_descriptor.base();
        *segment_access_bytes.offset(CS as isize) = cs_segment_descriptor.access_byte();

        *instruction_pointer = (get_seg_cs() as u32).wrapping_add(offset as u32) as u64;

        *flags &= !FLAG_NT & !FLAG_VM & !FLAG_RF & !FLAG_TRAP;

        if gate_type == InterruptDescriptor::INTERRUPT_GATE {
            // clear int flag for interrupt gates
            *flags &= !FLAG_INTERRUPT;
        }
        else {
            if *flags & FLAG_INTERRUPT != 0 && old_flags & FLAG_INTERRUPT == 0 {
                handle_irqs();
            }
        }

        update_state_flags();
    }
    else {
        // call 4 byte cs:ip interrupt vector from ivt at cpu.memory 0

        let index = (interrupt_nr << 2) as u64;
        let new_ip = memory::read16(index);
        let new_cs = memory::read16(index + 2);

        dbg_assert!(
            index | 3 <= IVT_SIZE as u64,
            "Unimplemented: #GP for interrupt number out of IVT bounds"
        );

        // XXX: #SS if stack would cross stack limit

        // push flags, cs:ip
        push16(get_eflags() & 0xFFFF).unwrap();
        push16(*sreg.offset(CS as isize) as i32).unwrap();
        push16(get_real_eip() & 0xFFFF).unwrap();

        *flags &= !FLAG_INTERRUPT & !FLAG_AC & !FLAG_TRAP;

        switch_cs_real_mode(new_cs);
        *instruction_pointer = (get_seg_cs() as u32).wrapping_add(new_ip as u32) as u64;
        update_state_flags();
    }
}

/// Interrupt/trap delivery in IA-32e mode (64-bit *and* compatibility
/// submode): 16-byte IDT gates, a 64-bit interrupt frame and IST stack
/// switches. The interrupted code may be running in compat mode (is_64=0).
pub unsafe fn call_interrupt_vector64(
    interrupt_nr: i32,
    is_software_int: bool,
    error_code: Option<i32>,
) {
    dbg_assert!(*protected_mode && *efer & EFER_LMA != 0);

    if (interrupt_nr << 4 | 15) as u32 > *idtr_size as u32 {
        dbg_log!("interrupt_nr={:x} idtr_size={:x}", interrupt_nr, *idtr_size);
        if is_software_int {
            trigger_gp(interrupt_nr << 4 | 2);
        }
        else {
            // an exception while delivering a hardware exception means a
            // triple fault in practice; shut down with a message instead of
            // recursing
            panic!(
                "Triple fault: exception {} during exception delivery",
                interrupt_nr
            );
        }
        return;
    }

    let descriptor_address = return_on_pagefault!(translate_address_system_read(
        (*idtr_offset64).wrapping_add((interrupt_nr << 4) as u64)
    ));

    let descriptor = InterruptDescriptor64::of_u128(
        memory::read64s(descriptor_address) as u64,
        memory::read64s(descriptor_address + 8) as u64,
    );

    let offset = descriptor.offset();
    if is_software_int && descriptor.dpl() < *cpl {
        dbg_log!("#gp software interrupt ({:x}) and dpl < cpl", interrupt_nr);
        trigger_gp(interrupt_nr << 4 | 2);
        return;
    }

    if !descriptor.is_valid_long_mode_gate() {
        dbg_log!(
            "gate type invalid. gate_type=0b{:b} access={:b}",
            descriptor.gate_type(),
            descriptor.access_byte()
        );
        if is_software_int {
            trigger_gp(interrupt_nr << 4 | 2);
        }
        else {
            panic!("Triple fault: invalid gate for exception {}", interrupt_nr);
        }
        return;
    }

    if !descriptor.is_present() {
        dbg_log!("#np int descriptor not present, int={}", interrupt_nr);
        trigger_np(interrupt_nr << 4 | 2);
        return;
    }

    let cs_selector = SegmentSelector::of_u16(descriptor.selector());
    let cs_descriptor = match return_on_pagefault!(lookup_segment_selector(cs_selector)) {
        Ok((desc, _)) => desc,
        Err(SelectorNullOrInvalid::IsNull) | Err(SelectorNullOrInvalid::OutsideOfTableLimit) => {
            dbg_log!(
                "#gp invalid cs selector in gate, sel={:x}",
                descriptor.selector()
            );
            trigger_gp(descriptor.selector() as i32 & !3);
            return;
        },
    };

    if !cs_descriptor.is_executable() {
        dbg_log!("#gp interrupt: cs not executable");
        trigger_gp(descriptor.selector() as i32 & !3);
        return;
    }
    if !cs_descriptor.is_present() {
        dbg_log!("#np interrupt: cs not present");
        trigger_np(descriptor.selector() as i32 & !3);
        return;
    }
    if !cs_descriptor.is_64() {
        dbg_log!("#gp interrupt: cs is not a 64-bit code segment");
        trigger_gp(descriptor.selector() as i32 & !3);
        return;
    }
    if !cs_descriptor.is_dc() && cs_descriptor.dpl() < *cpl {
        // inter-privilege-level interrupt; dpl is the new cpl
    }
    else if cs_descriptor.is_dc() || cs_descriptor.dpl() == *cpl {
        // intra-privilege-level
    }
    else {
        dbg_log!("#gp interrupt: cs dpl={} cpl={}", cs_descriptor.dpl(), *cpl);
        trigger_gp(descriptor.selector() as i32 & !3);
        return;
    }

    let mut old_flags = get_eflags();
    if exception_is_fault(interrupt_nr) || take_force_rf_in_exception_image() {
        old_flags |= FLAG_RF;
    }
    let old_rip = *instruction_pointer;
    let old_cs = *sreg.offset(CS as isize) as u64;
    let old_ss = *sreg.offset(SS as isize) as u64;
    let old_rsp = read_reg64(ESP);

    let new_cpl = cs_descriptor.dpl();
    let switch_stack = !cs_descriptor.is_dc() && new_cpl < *cpl;
    let ist = descriptor.ist();

    // In 64-bit mode the exception frame is always 5 slots: SS:RSP are pushed
    // regardless of privilege change (Intel SDM Vol. 3A, "Exception and
    // Interrupt Handling in 64-bit Mode")
    let mut frame_size: u64 = 40; // rip, cs, rflags, rsp, ss
    if error_code.is_some() {
        frame_size += 8;
    }

    let stack_base: u64 = if switch_stack || ist != 0 {
        if ist != 0 {
            return_on_pagefault!(get_tss_ist64(ist))
        }
        else {
            return_on_pagefault!(get_tss_rsp64(new_cpl))
        }
    }
    else {
        old_rsp
    };

    // Switch to the target privilege level *before* the workability checks and
    // pushes: the hardware pushes the interrupt frame with supervisor
    // privileges even when it the interrupted context was userland.
    *cpl = new_cpl;
    cpl_changed();

    // check that the frame fits (linear, ss base is 0 in 64-bit mode)
    return_on_pagefault!(writable_or_pagefault(
        stack_base.wrapping_sub(frame_size),
        frame_size as i32
    ));

    // no exceptions below

    *flags &= !FLAG_VM & !FLAG_RF;

    if switch_stack {
        // in long mode ss is set to null on an interrupt stack switch
        *sreg.offset(SS as isize) = 0;
        *segment_is_null.offset(SS as isize) = true;
    }

    // Push the frame the way the CPU does: ss, rsp, rflags, cs, rip and
    // finally the error code, so that (from low to high addresses) the frame
    // reads error, rip, cs, rflags, rsp, ss. SS:RSP are pushed unconditionally
    // in long mode (see frame_size above).
    let mut p = stack_base;
    p = p.wrapping_sub(8);
    return_on_pagefault!(safe_write64(p, old_ss));
    p = p.wrapping_sub(8);
    return_on_pagefault!(safe_write64(p, old_rsp));
    p = p.wrapping_sub(8);
    return_on_pagefault!(safe_write64(p, old_flags as u64));
    p = p.wrapping_sub(8);
    return_on_pagefault!(safe_write64(p, old_cs));
    p = p.wrapping_sub(8);
    return_on_pagefault!(safe_write64(p, old_rip));
    if let Some(ec) = error_code {
        p = p.wrapping_sub(8);
        return_on_pagefault!(safe_write64(p, ec as u64));
    }
    dbg_assert!(p == stack_base.wrapping_sub(frame_size));
    write_reg64(ESP, p);

    *sreg.offset(CS as isize) = descriptor.selector() & !3 | *cpl as u16;
    *segment_is_null.offset(CS as isize) = false;
    *segment_limits.offset(CS as isize) = cs_descriptor.effective_limit();
    *segment_offsets.offset(CS as isize) = 0;
    *segment_access_bytes.offset(CS as isize) = cs_descriptor.access_byte();

    update_cs_size(cs_descriptor.is_32(), cs_descriptor.is_64());

    *instruction_pointer = offset;

    *flags &= !FLAG_NT & !FLAG_RF & !FLAG_TRAP;

    if !descriptor.is_trap_gate() {
        // clear int flag for interrupt gates
        *flags &= !FLAG_INTERRUPT;
    }
    else {
        if *flags & FLAG_INTERRUPT != 0 && old_flags & FLAG_INTERRUPT == 0 {
            handle_irqs();
        }
    }

    update_state_flags();
}

pub unsafe fn far_jump(eip: i32, selector: i32, is_call: bool, is_osize_32: bool) {
    dbg_assert!(selector < 0x10000 && selector >= 0);

    if !*protected_mode || vm86_mode() {
        if is_call {
            if is_osize_32 {
                return_on_pagefault!(writable_or_pagefault(get_stack_pointer(-8), 8));

                push32(*sreg.offset(CS as isize) as i32).unwrap();
                push32(get_real_eip()).unwrap();
            }
            else {
                return_on_pagefault!(writable_or_pagefault(get_stack_pointer(-4), 4));

                push16(*sreg.offset(CS as isize) as i32).unwrap();
                push16(get_real_eip()).unwrap();
            }
        }
        switch_cs_real_mode(selector);
        *instruction_pointer = (get_seg_cs() as u32).wrapping_add(eip as u32) as u64;
        update_state_flags();
        return;
    }

    let cs_selector = SegmentSelector::of_u16(selector as u16);
    let info = match return_on_pagefault!(lookup_segment_selector(cs_selector)) {
        Ok((desc, _)) => desc,
        Err(SelectorNullOrInvalid::IsNull) => {
            dbg_log!("#gp null cs");
            trigger_gp(0);
            return;
        },
        Err(SelectorNullOrInvalid::OutsideOfTableLimit) => {
            dbg_log!("#gp invalid cs: {:x}", selector);
            trigger_gp(selector & !3);
            return;
        },
    };

    if info.is_system() {
        dbg_log!("system type cs: {:x}", selector);

        if info.system_type() == 0xC || info.system_type() == 4 {
            // call gate
            dbg_assert!(is_call, "TODO: Jump through call gate");

            let is_16 = info.system_type() == 4;

            if info.dpl() < *cpl || info.dpl() < cs_selector.rpl() {
                dbg_log!("#gp cs gate dpl < cpl or dpl < rpl: {:x}", selector);
                trigger_gp(selector & !3);
                return;
            }

            if !info.is_present() {
                dbg_log!("#NP for loading not-present in gate cs sel={:x}", selector);
                trigger_np(selector & !3);
                return;
            }

            let cs_selector = (info.raw >> 16) as i32;

            let cs_info = match return_on_pagefault!(lookup_segment_selector(
                SegmentSelector::of_u16(cs_selector as u16)
            )) {
                Ok((desc, _)) => desc,
                Err(SelectorNullOrInvalid::IsNull) => {
                    dbg_log!("#gp null cs");
                    trigger_gp(0);
                    return;
                },
                Err(SelectorNullOrInvalid::OutsideOfTableLimit) => {
                    dbg_log!("#gp invalid cs: {:x}", cs_selector);
                    trigger_gp(cs_selector & !3);
                    return;
                },
            };

            if cs_info.is_system() {
                dbg_log!("#gp non-code cs: {:x}", cs_selector);
                trigger_gp(cs_selector & !3);
                return;
            }

            if !cs_info.is_executable() {
                dbg_log!("#gp non-executable cs: {:x}", cs_selector);
                trigger_gp(cs_selector & !3);
                return;
            }

            if cs_info.dpl() > *cpl {
                dbg_log!("#gp dpl > cpl: {:x}", cs_selector);
                trigger_gp(cs_selector & !3);
                return;
            }

            if !cs_info.is_present() {
                dbg_log!("#NP for loading not-present in cs sel={:x}", cs_selector);
                trigger_np(cs_selector & !3);
                return;
            }

            if !cs_info.is_dc() && cs_info.dpl() < *cpl {
                dbg_log!(
                    "more privilege call gate is_16={} from={} to={}",
                    is_16,
                    *cpl,
                    cs_info.dpl()
                );
                let (new_ss, new_esp) = return_on_pagefault!(get_tss_ss_esp(cs_info.dpl()));

                let ss_selector = SegmentSelector::of_u16(new_ss as u16);
                let ss_info = match return_on_pagefault!(lookup_segment_selector(ss_selector)) {
                    Ok((desc, _)) => desc,
                    Err(SelectorNullOrInvalid::IsNull) => {
                        panic!("null ss: {}", new_ss);
                    },
                    Err(SelectorNullOrInvalid::OutsideOfTableLimit) => {
                        panic!("invalid ss: {}", new_ss);
                    },
                };

                if ss_info.is_dc() {
                    dbg_assert!(new_esp as u32 > ss_info.effective_limit());
                }
                else {
                    dbg_assert!(new_esp as u32 - 1 <= ss_info.effective_limit());
                }
                dbg_assert!(!ss_info.is_system() && ss_info.is_writable());

                if ss_selector.rpl() != cs_info.dpl()
                // xxx: 0 in v86 mode
                {
                    panic!("#TS handler");
                }
                if ss_info.dpl() != cs_info.dpl() || !ss_info.is_writable() {
                    panic!("#TS handler");
                }
                if !ss_info.is_present() {
                    panic!("#SS handler");
                }

                let parameter_count = (info.raw >> 32 & 0x1F) as i32;
                let mut stack_space = if is_16 { 4 } else { 8 };
                if is_call {
                    stack_space +=
                        if is_16 { 4 + 2 * parameter_count } else { 8 + 4 * parameter_count };
                }
                if ss_info.is_32() {
                    return_on_pagefault!(writable_or_pagefault_cpl(
                        cs_info.dpl(),
                        (ss_info.base() + new_esp - stack_space) as u32 as u64,
                        stack_space
                    ));
                }
                else {
                    return_on_pagefault!(writable_or_pagefault_cpl(
                        cs_info.dpl(),
                        (ss_info.base() + (new_esp - stack_space & 0xFFFF)) as u32 as u64,
                        stack_space
                    ));
                }

                let old_esp = read_reg32(ESP);
                let old_ss = *sreg.offset(SS as isize);
                let old_stack_pointer = get_stack_pointer(0);

                //dbg_log!("old_esp=" + h(old_esp));

                *cpl = cs_info.dpl();
                cpl_changed();

                update_cs_size(cs_info.is_32(), cs_info.is_64());

                dbg_assert!(new_ss & 3 == cs_info.dpl() as i32);
                // XXX: Should be checked before side effects
                if !switch_seg(SS, new_ss) {
                    dbg_assert!(false);
                };
                set_stack_reg(new_esp);

                //dbg_log!("parameter_count=" + parameter_count);
                //dbg_assert!(parameter_count == 0, "TODO");

                if is_16 {
                    push16(old_ss as i32).unwrap();
                    push16(old_esp).unwrap();
                }
                else {
                    push32(old_ss as i32).unwrap();
                    push32(old_esp).unwrap();
                }

                if is_call {
                    if is_16 {
                        for i in (0..parameter_count).rev() {
                            let parameter = safe_read16(old_stack_pointer + 2 * i as u64).unwrap();
                            push16(parameter).unwrap();
                        }

                        //writable_or_pagefault(get_stack_pointer(-4), 4);
                        push16(*sreg.offset(CS as isize) as i32).unwrap();
                        push16(get_real_eip()).unwrap();
                    }
                    else {
                        for i in (0..parameter_count).rev() {
                            let parameter = safe_read32s(old_stack_pointer + 4 * i as u64).unwrap();
                            push32(parameter).unwrap();
                        }

                        //writable_or_pagefault(get_stack_pointer(-8), 8);
                        push32(*sreg.offset(CS as isize) as i32).unwrap();
                        push32(get_real_eip()).unwrap();
                    }
                }
            }
            else {
                dbg_log!(
                    "same privilege call gate is_16={} from={} to={} conforming={}",
                    is_16,
                    *cpl,
                    cs_info.dpl(),
                    cs_info.is_dc()
                );

                if is_call {
                    if is_16 {
                        return_on_pagefault!(writable_or_pagefault(get_stack_pointer(-4), 4));

                        push16(*sreg.offset(CS as isize) as i32).unwrap();
                        push16(get_real_eip()).unwrap();
                    }
                    else {
                        return_on_pagefault!(writable_or_pagefault(get_stack_pointer(-8), 8));

                        push32(*sreg.offset(CS as isize) as i32).unwrap();
                        push32(get_real_eip()).unwrap();
                    }
                }

                dbg_assert!(*cpl == cs_info.dpl());
            }

            // Note: eip from call is ignored
            let mut new_eip = (info.raw & 0xFFFF) as i32;
            if !is_16 {
                new_eip |= ((info.raw >> 32) & 0xFFFF0000) as i32;
            }

            dbg_log!(
                "call gate eip={:x} cs={:x} conforming={}",
                new_eip as u32,
                cs_selector,
                cs_info.is_dc()
            );
            dbg_assert!((new_eip as u32) <= cs_info.effective_limit(), "todo: #gp");

            update_cs_size(cs_info.is_32(), cs_info.is_64());

            *segment_is_null.offset(CS as isize) = false;
            *segment_limits.offset(CS as isize) = cs_info.effective_limit();
            *segment_offsets.offset(CS as isize) = cs_info.base();
            *segment_access_bytes.offset(CS as isize) = cs_info.access_byte();
            *sreg.offset(CS as isize) = cs_selector as u16 & !3 | *cpl as u16;
            dbg_assert!(*sreg.offset(CS as isize) & 3 == *cpl as u16);

            *instruction_pointer = (get_seg_cs() as u32).wrapping_add(new_eip as u32) as u64;

            update_state_flags();
        }
        else if info.system_type() == 1 || info.system_type() == 9 {
            // available tss
            if info.dpl() < *cpl || info.dpl() < cs_selector.rpl() {
                dbg_log!("#gp tss dpl < cpl or dpl < rpl: {:x}", selector);
                trigger_gp(selector & !3);
                return;
            }

            if !info.is_present() {
                dbg_log!("#NP for loading not-present tss sel={:x}", selector);
                trigger_np(selector & !3);
                return;
            }

            do_task_switch(
                selector,
                None,
                if is_call { TaskSwitchSource::CallOrInt } else { TaskSwitchSource::Jump },
            );
        }
        else if info.system_type() == 5 {
            // task gate
            if info.dpl() < *cpl || info.dpl() < cs_selector.rpl() {
                dbg_log!("#gp task gate dpl < cpl or dpl < rpl: {:x}", selector);
                trigger_gp(selector & !3);
                return;
            }

            if !info.is_present() {
                dbg_log!("#NP for loading not-present task gate sel={:x}", selector);
                trigger_np(selector & !3);
                return;
            }

            let tss_selector = (info.raw >> 16) as i32 & 0xFFFF;
            do_task_switch(
                tss_selector,
                None,
                if is_call { TaskSwitchSource::CallOrInt } else { TaskSwitchSource::Jump },
            );
        }
        else {
            dbg_assert!(false, "TODO: #gp invalid system type");
        }
    }
    else {
        if !info.is_executable() {
            dbg_log!("#gp non-executable cs: {:x}", selector);
            trigger_gp(selector & !3);
            return;
        }

        if info.is_dc() {
            // conforming code segment
            if info.dpl() > *cpl {
                dbg_log!("#gp cs dpl > cpl: {:x}", selector);
                trigger_gp(selector & !3);
                return;
            }
        }
        else {
            // non-conforming code segment

            if cs_selector.rpl() > *cpl || info.dpl() != *cpl {
                dbg_log!("#gp cs rpl > cpl or dpl != cpl: {:x}", selector);
                trigger_gp(selector & !3);
                return;
            }
        }

        if !info.is_present() {
            dbg_log!("#NP for loading not-present in cs sel={:x}", selector);
            dbg_trace();
            trigger_np(selector & !3);
            return;
        }

        if is_call {
            if is_osize_32 {
                return_on_pagefault!(writable_or_pagefault(get_stack_pointer(-8), 8));

                push32(*sreg.offset(CS as isize) as i32).unwrap();
                push32(get_real_eip()).unwrap();
            }
            else {
                return_on_pagefault!(writable_or_pagefault(get_stack_pointer(-4), 4));

                push16(*sreg.offset(CS as isize) as i32).unwrap();
                push16(get_real_eip()).unwrap();
            }
        }

        dbg_assert!((eip as u32) <= info.effective_limit(), "todo: #gp");

        update_cs_size(info.is_32(), info.is_64());

        *segment_is_null.offset(CS as isize) = false;
        *segment_limits.offset(CS as isize) = info.effective_limit();
        *segment_access_bytes.offset(CS as isize) = info.access_byte();

        *segment_offsets.offset(CS as isize) = info.base();
        *sreg.offset(CS as isize) = selector as u16 & !3 | *cpl as u16;

        *instruction_pointer = (get_seg_cs() as u32).wrapping_add(eip as u32) as u64;

        update_state_flags();
    }
}

/// 64-bit far return (retfq): pop rip, cs (and rsp, ss on a privilege change).
/// rip is the value popped by the caller; the stack has been advanced past it.
pub unsafe fn far_return64(rip: u64, selector: i32, stack_adjust: u64) {
    let cs_selector = SegmentSelector::of_u16(selector as u16);
    let info = match return_on_pagefault!(lookup_segment_selector(cs_selector)) {
        Ok((desc, _)) => desc,
        Err(SelectorNullOrInvalid::IsNull) | Err(SelectorNullOrInvalid::OutsideOfTableLimit) => {
            dbg_log!("far return64: #gp invalid cs: {:x}", selector);
            trigger_gp(selector & !3);
            return;
        },
    };

    if info.is_system() || !info.is_executable() || !info.is_present() {
        dbg_log!("far return64: #gp cs is not a present code segment");
        trigger_gp(selector & !3);
        return;
    }
    if cs_selector.rpl() < *cpl {
        dbg_log!("far return64: #gp rpl < cpl");
        trigger_gp(selector & !3);
        return;
    }
    if !info.is_dc() && cs_selector.rpl() != info.dpl() {
        dbg_log!("far return64: #gp non-conforming cs and rpl != dpl");
        trigger_gp(selector & !3);
        return;
    }

    if cs_selector.rpl() > *cpl {
        // outer privilege return: pop rsp and ss
        let new_rsp = return_on_pagefault!(pop64());
        let new_ss = return_on_pagefault!(pop64()) as u16;
        let ss_selector = SegmentSelector::of_u16(new_ss);
        let ss_descriptor = match return_on_pagefault!(lookup_segment_selector(ss_selector)) {
            Ok((desc, _)) => desc,
            Err(SelectorNullOrInvalid::IsNull)
            | Err(SelectorNullOrInvalid::OutsideOfTableLimit) => {
                dbg_log!("far return64: #gp invalid ss: {:x}", new_ss);
                trigger_gp(new_ss as i32 & !3);
                return;
            },
        };
        if ss_descriptor.is_system()
            || ss_selector.rpl() != cs_selector.rpl()
            || !ss_descriptor.is_writable()
            || ss_descriptor.dpl() != cs_selector.rpl()
            || !ss_descriptor.is_present()
        {
            dbg_log!("far return64: #gp/#ss invalid ss descriptor");
            trigger_gp(new_ss as i32 & !3);
            return;
        }
        // no exceptions below
        *cpl = cs_selector.rpl();
        cpl_changed();
        write_reg64(ESP, new_rsp);
        if !switch_seg(SS, new_ss as i32) {
            dbg_assert!(false);
        }
    }
    else {
        write_reg64(ESP, read_reg64(ESP).wrapping_add(stack_adjust));
    }

    // no exceptions below

    *sreg.offset(CS as isize) = selector as u16 & !3 | *cpl as u16;
    *segment_is_null.offset(CS as isize) = false;
    *segment_limits.offset(CS as isize) = info.effective_limit();
    *segment_offsets.offset(CS as isize) = if *is_64 { 0 } else { info.base() };
    *segment_access_bytes.offset(CS as isize) = info.access_byte();

    update_cs_size(info.is_32(), info.is_64());

    *instruction_pointer = rip;

    update_state_flags();
}

/// Far call or jmp through a memory far pointer in 64-bit mode (FF /3 and FF /5): `offset` and
/// `selector` come from the m16:16, m16:32 or m16:64 operand of operand size `osize` (16 with
/// 0x66, 32 without a prefix, 64 with REX.W: in group 5 only near call/jmp and push default to 64
/// bits). A call pushes cs and the return address as two slots of `osize` bits on the 64-bit
/// stack, after all the checks, so that a fault leaves rsp and memory unchanged. The target can be
/// a 64-bit or a compatibility mode code segment. System descriptors raise #GP(selector): TSS and
/// task gates are invalid in IA-32e mode, and 64-bit call gates aren't implemented.
pub unsafe fn far_jump64(offset: u64, selector: i32, is_call: bool, osize: u32) {
    dbg_assert!(*is_64);
    dbg_assert!(osize == 16 || osize == 32 || osize == 64);
    dbg_assert!(selector >= 0 && selector < 0x10000);

    let cs_selector = SegmentSelector::of_u16(selector as u16);
    let info = match return_on_pagefault!(lookup_segment_selector(cs_selector)) {
        Ok((desc, _)) => desc,
        Err(SelectorNullOrInvalid::IsNull) => {
            dbg_log!("far jump64: #gp null cs");
            trigger_gp(0);
            return;
        },
        Err(SelectorNullOrInvalid::OutsideOfTableLimit) => {
            dbg_log!("far jump64: #gp invalid cs: {:x}", selector);
            trigger_gp(selector & !3);
            return;
        },
    };

    if info.is_system() {
        dbg_log!(
            "far jump64: #gp system descriptor type={:x} (64-bit call gates are unimplemented)",
            info.system_type()
        );
        trigger_gp(selector & !3);
        return;
    }
    if !info.is_executable() {
        dbg_log!("far jump64: #gp non-executable cs: {:x}", selector);
        trigger_gp(selector & !3);
        return;
    }
    if info.is_64() && info.is_32() {
        dbg_log!("far jump64: #gp cs with L and D set: {:x}", selector);
        trigger_gp(selector & !3);
        return;
    }
    if info.is_dc() {
        // conforming
        if info.dpl() > *cpl {
            dbg_log!("far jump64: #gp conforming cs dpl > cpl: {:x}", selector);
            trigger_gp(selector & !3);
            return;
        }
    }
    else if cs_selector.rpl() > *cpl || info.dpl() != *cpl {
        dbg_log!(
            "far jump64: #gp non-conforming cs rpl > cpl or dpl != cpl: {:x}",
            selector
        );
        trigger_gp(selector & !3);
        return;
    }
    if !info.is_present() {
        dbg_log!("far jump64: #np cs: {:x}", selector);
        trigger_np(selector & !3);
        return;
    }

    let target_is_64 = info.is_64();
    let mut new_ip = if osize == 16 { offset & 0xFFFF } else { offset };
    if target_is_64 {
        if !is_canonical_address(new_ip) {
            dbg_log!("far jump64: #gp non-canonical rip: {:x}", new_ip);
            trigger_gp(0);
            return;
        }
    }
    else {
        // compatibility mode
        new_ip &= 0xFFFF_FFFF;
        if new_ip > info.effective_limit() as u64 {
            dbg_log!("far jump64: #gp eip beyond the cs limit: {:x}", new_ip);
            trigger_gp(0);
            return;
        }
    }

    if is_call {
        let size = (osize / 8) as i32;
        let old_cs = *sreg.offset(CS as isize);
        let return_ip = *instruction_pointer;
        // both slots, so that a #PF on the second one doesn't leave the first one pushed
        return_on_pagefault!(writable_or_pagefault(
            get_stack_pointer64(-2 * size),
            2 * size
        ));
        let cs_slot = get_stack_pointer64(-size);
        let ip_slot = get_stack_pointer64(-2 * size);
        match osize {
            16 => {
                safe_write16(cs_slot, old_cs as i32).unwrap();
                safe_write16(ip_slot, return_ip as i32 & 0xFFFF).unwrap();
            },
            32 => {
                safe_write32(cs_slot, old_cs as i32).unwrap();
                safe_write32(ip_slot, return_ip as i32).unwrap();
            },
            _ => {
                safe_write64(cs_slot, old_cs as u64).unwrap();
                safe_write64(ip_slot, return_ip).unwrap();
            },
        }
        write_reg64(ESP, read_reg64(ESP).wrapping_sub(2 * size as u64));
    }

    // no exceptions below

    let base = if target_is_64 { 0 } else { info.base() };
    *sreg.offset(CS as isize) = selector as u16 & !3 | *cpl as u16;
    *segment_is_null.offset(CS as isize) = false;
    *segment_limits.offset(CS as isize) = info.effective_limit();
    *segment_offsets.offset(CS as isize) = base;
    *segment_access_bytes.offset(CS as isize) = info.access_byte();

    update_cs_size(info.is_32(), info.is_64());

    *instruction_pointer =
        if target_is_64 { new_ip } else { (base as u32).wrapping_add(new_ip as u32) as u64 };

    update_state_flags();
}

pub unsafe fn far_return(eip: i32, selector: i32, stack_adjust: i32, is_osize_32: bool) {
    dbg_assert!(selector < 0x10000 && selector >= 0);

    if !*protected_mode {
        dbg_assert!(!*is_32);
    }

    if !*protected_mode || vm86_mode() {
        switch_cs_real_mode(selector);
        *instruction_pointer = (get_seg_cs() as u32).wrapping_add(eip as u32) as u64;
        adjust_stack_reg(2 * (if is_osize_32 { 4 } else { 2 }) + stack_adjust);
        update_state_flags();
        return;
    }

    let cs_selector = SegmentSelector::of_u16(selector as u16);
    let info = match return_on_pagefault!(lookup_segment_selector(cs_selector)) {
        Ok((desc, _)) => desc,
        Err(SelectorNullOrInvalid::IsNull) => {
            dbg_log!("far return: #gp null cs");
            trigger_gp(0);
            return;
        },
        Err(SelectorNullOrInvalid::OutsideOfTableLimit) => {
            dbg_log!("far return: #gp invalid cs: {:x}", selector);
            trigger_gp(selector & !3);
            return;
        },
    };

    if info.is_system() {
        dbg_assert!(false, "is system in far return");
        trigger_gp(selector & !3);
        return;
    }

    if !info.is_executable() {
        dbg_log!("non-executable cs: {:x}", selector);
        trigger_gp(selector & !3);
        return;
    }

    if cs_selector.rpl() < *cpl {
        dbg_log!("cs rpl < cpl: {:x}", selector);
        trigger_gp(selector & !3);
        return;
    }

    if info.is_dc() && info.dpl() > cs_selector.rpl() {
        dbg_log!("cs conforming and dpl > rpl: {:x}", selector);
        trigger_gp(selector & !3);
        return;
    }

    if !info.is_dc() && info.dpl() != cs_selector.rpl() {
        dbg_log!("cs non-conforming and dpl != rpl: {:x}", selector);
        trigger_gp(selector & !3);
        return;
    }

    if !info.is_present() {
        dbg_log!("#NP for loading not-present in cs sel={:x}", selector);
        dbg_trace();
        trigger_np(selector & !3);
        return;
    }

    if cs_selector.rpl() > *cpl {
        dbg_log!(
            "far return privilege change cs: {:x} from={} to={} is_16={}",
            selector,
            *cpl,
            cs_selector.rpl(),
            is_osize_32
        );

        let temp_esp;
        let temp_ss;
        if is_osize_32 {
            //dbg_log!("esp read from " + h(translate_address_system_read(get_stack_pointer(stack_adjust + 8))))
            temp_esp = safe_read32s(get_stack_pointer(stack_adjust + 8)).unwrap();
            //dbg_log!("esp=" + h(temp_esp));
            temp_ss = safe_read16(get_stack_pointer(stack_adjust + 12)).unwrap();
        }
        else {
            //dbg_log!("esp read from " + h(translate_address_system_read(get_stack_pointer(stack_adjust + 4))));
            temp_esp = safe_read16(get_stack_pointer(stack_adjust + 4)).unwrap();
            //dbg_log!("esp=" + h(temp_esp));
            temp_ss = safe_read16(get_stack_pointer(stack_adjust + 6)).unwrap();
        }

        *cpl = cs_selector.rpl();
        cpl_changed();

        // XXX: This failure should be checked before side effects
        if !switch_seg(SS, temp_ss) {
            dbg_assert!(false);
        }
        set_stack_reg(temp_esp + stack_adjust);

        //if(is_osize_32)
        //{
        //    adjust_stack_reg(2 * 4);
        //}
        //else
        //{
        //    adjust_stack_reg(2 * 2);
        //}

        //throw debug.unimpl("privilege change");

        //adjust_stack_reg(stack_adjust);

        for reg in [ES, DS, FS, GS] {
            let access = *segment_access_bytes.offset(reg as isize);
            let dpl = access >> 5 & 3;
            let executable = access & 8 == 8;
            let conforming = access & 4 == 4;
            if dpl < *cpl && !(executable && conforming) {
                *segment_is_null.offset(reg as isize) = true;
                *sreg.offset(reg as isize) = 0;
            }
        }
    }
    else {
        if is_osize_32 {
            adjust_stack_reg(2 * 4 + stack_adjust);
        }
        else {
            adjust_stack_reg(2 * 2 + stack_adjust);
        }
    }

    //dbg_assert(*cpl == info.dpl);

    update_cs_size(info.is_32(), info.is_64());

    *segment_is_null.offset(CS as isize) = false;
    *segment_limits.offset(CS as isize) = info.effective_limit();
    *segment_access_bytes.offset(CS as isize) = info.access_byte();

    *segment_offsets.offset(CS as isize) = info.base();
    *sreg.offset(CS as isize) = selector as u16;
    dbg_assert!(selector & 3 == *cpl as i32);

    *instruction_pointer = (get_seg_cs() as u32).wrapping_add(eip as u32) as u64;

    update_state_flags();
}

#[derive(Copy, Clone, PartialEq)]
pub enum TaskSwitchSource {
    Jump,
    CallOrInt,
    Iret,
}

pub unsafe fn do_task_switch(selector: i32, error_code: Option<i32>, source: TaskSwitchSource) {
    dbg_log!("do_task_switch sel={:x}", selector);

    dbg_assert!(*tss_size_32, "TODO: 16-bit TSS in task switch");

    let selector = SegmentSelector::of_u16(selector as u16);
    let (descriptor, descriptor_address) =
        match lookup_segment_selector(selector).expect("TODO: handle pagefault") {
            Ok(desc) => desc,
            Err(_) => {
                panic!("#GP handler");
            },
        };

    dbg_assert!(selector.is_gdt());
    dbg_assert!((descriptor.system_type() & !2) == 1 || (descriptor.system_type() & !2) == 9);
    let tss_is_16 = descriptor.system_type() <= 3;
    let tss_is_busy = (descriptor.system_type() & 2) == 2;

    if source == TaskSwitchSource::Iret {
        if !tss_is_busy {
            // a task return must target a busy task
            panic!("#TS handler");
        }
    }
    else if tss_is_busy {
        // jump, call or int to a busy task
        panic!("#GP handler");
    }

    if !descriptor.is_present() {
        panic!("#NP handler");
    }

    if descriptor.effective_limit() < 103 {
        panic!("#NP handler");
    }

    let _tsr_size = *segment_limits.offset(TR as isize);
    let tsr_offset = *segment_offsets.offset(TR as isize);

    let mut old_eflags = get_eflags();

    if tss_is_busy {
        old_eflags &= !FLAG_NT;
    }

    writable_or_pagefault(tsr_offset as u32 as u64, 0x66).unwrap();

    //safe_write32((tsr_offset + TSR_CR3) as u32 as u64, *cr.offset(3));

    // TODO: Write 16 bit values if old tss is 16 bit
    safe_write32((tsr_offset + TSR_EIP) as u32 as u64, get_real_eip()).unwrap();
    safe_write32((tsr_offset + TSR_EFLAGS) as u32 as u64, old_eflags).unwrap();

    safe_write32((tsr_offset + TSR_EAX) as u32 as u64, read_reg32(EAX)).unwrap();
    safe_write32((tsr_offset + TSR_ECX) as u32 as u64, read_reg32(ECX)).unwrap();
    safe_write32((tsr_offset + TSR_EDX) as u32 as u64, read_reg32(EDX)).unwrap();
    safe_write32((tsr_offset + TSR_EBX) as u32 as u64, read_reg32(EBX)).unwrap();

    safe_write32((tsr_offset + TSR_ESP) as u32 as u64, read_reg32(ESP)).unwrap();
    safe_write32((tsr_offset + TSR_EBP) as u32 as u64, read_reg32(EBP)).unwrap();
    safe_write32((tsr_offset + TSR_ESI) as u32 as u64, read_reg32(ESI)).unwrap();
    safe_write32((tsr_offset + TSR_EDI) as u32 as u64, read_reg32(EDI)).unwrap();

    safe_write32(
        (tsr_offset + TSR_ES) as u32 as u64,
        *sreg.offset(ES as isize) as i32,
    )
    .unwrap();
    safe_write32(
        (tsr_offset + TSR_CS) as u32 as u64,
        *sreg.offset(CS as isize) as i32,
    )
    .unwrap();
    safe_write32(
        (tsr_offset + TSR_SS) as u32 as u64,
        *sreg.offset(SS as isize) as i32,
    )
    .unwrap();
    safe_write32(
        (tsr_offset + TSR_DS) as u32 as u64,
        *sreg.offset(DS as isize) as i32,
    )
    .unwrap();
    safe_write32(
        (tsr_offset + TSR_FS) as u32 as u64,
        *sreg.offset(FS as isize) as i32,
    )
    .unwrap();
    safe_write32(
        (tsr_offset + TSR_GS) as u32 as u64,
        *sreg.offset(GS as isize) as i32,
    )
    .unwrap();

    //safe_write32((tsr_offset + TSR_LDT) as u32 as u64, *sreg.offset(reg_ldtr));

    if source == TaskSwitchSource::Jump || source == TaskSwitchSource::Iret {
        // mark the old task as not busy
        let tr_selector = SegmentSelector::of_u16(*sreg.offset(TR as isize));
        let (tr_descriptor, tr_descriptor_address) =
            match lookup_segment_selector(tr_selector).expect("TODO: handle pagefault") {
                Ok(desc) => desc,
                Err(_) => {
                    panic!("#TS handler");
                },
            };
        safe_write64(tr_descriptor_address as u64, tr_descriptor.clear_busy().raw).unwrap();
    }

    if source != TaskSwitchSource::Iret {
        // jump, call and int mark the new task as busy (iret would not)
        safe_write64(descriptor_address as u64, descriptor.set_busy().raw).unwrap();
    }

    //let new_tsr_size = descriptor.effective_limit;
    let new_tsr_offset = descriptor.base();

    dbg_assert!(!tss_is_16, "unimplemented");

    if source == TaskSwitchSource::CallOrInt {
        safe_write16(
            (new_tsr_offset + TSR_BACKLINK) as u32 as u64,
            *sreg.offset(TR as isize) as i32,
        )
        .unwrap();
    }

    let new_cr3 = safe_read32s((new_tsr_offset + TSR_CR3) as u32 as u64).unwrap();

    *flags &= !FLAG_VM;

    let new_eip = safe_read32s((new_tsr_offset + TSR_EIP) as u32 as u64).unwrap();
    let new_cs = safe_read16((new_tsr_offset + TSR_CS) as u32 as u64).unwrap();
    let new_ldt = safe_read16((new_tsr_offset + TSR_LDT) as u32 as u64).unwrap();

    let mut new_eflags = safe_read32s((new_tsr_offset + TSR_EFLAGS) as u32 as u64).unwrap();
    if source == TaskSwitchSource::CallOrInt {
        new_eflags |= FLAG_NT;
    }

    load_ldt(new_ldt).unwrap();

    let new_cpl;
    if new_eflags & FLAG_VM != 0 {
        *segment_is_null.offset(CS as isize) = false;
        *segment_offsets.offset(CS as isize) = new_cs << 4;
        *sreg.offset(CS as isize) = new_cs as u16;
        update_cs_size(false, false);
        new_cpl = 3;
    }
    else {
        let new_cs_selector = SegmentSelector::of_u16(new_cs as u16);
        let new_cs_descriptor =
            match lookup_segment_selector(new_cs_selector).expect("TODO: handle pagefault") {
                Ok((desc, _)) => desc,
                Err(SelectorNullOrInvalid::IsNull) => {
                    dbg_log!("null cs");
                    panic!("#TS handler");
                },
                Err(SelectorNullOrInvalid::OutsideOfTableLimit) => {
                    dbg_log!("invalid cs: {:x}", new_cs);
                    panic!("#TS handler");
                },
            };

        if new_cs_descriptor.is_system() {
            panic!("#TS handler");
        }

        if !new_cs_descriptor.is_executable() {
            panic!("#TS handler");
        }

        if new_cs_descriptor.is_dc() && new_cs_descriptor.dpl() > new_cs_selector.rpl() {
            dbg_log!("cs conforming and dpl > rpl: {:x}", selector.raw);
            panic!("#TS handler");
        }

        if !new_cs_descriptor.is_dc() && new_cs_descriptor.dpl() != new_cs_selector.rpl() {
            dbg_log!("cs non-conforming and dpl != rpl: {:x}", selector.raw);
            panic!("#TS handler");
        }

        if !new_cs_descriptor.is_present() {
            dbg_log!("#NP for loading not-present in cs sel={:x}", selector.raw);
            panic!("#TS handler");
        }

        *segment_is_null.offset(CS as isize) = false;
        *segment_limits.offset(CS as isize) = new_cs_descriptor.effective_limit();
        *segment_offsets.offset(CS as isize) = new_cs_descriptor.base();
        *segment_access_bytes.offset(CS as isize) = new_cs_descriptor.access_byte();
        *sreg.offset(CS as isize) = new_cs as u16;

        dbg_assert!(
            new_eip as u32 <= new_cs_descriptor.effective_limit(),
            "todo: #gp"
        );
        update_cs_size(new_cs_descriptor.is_32(), new_cs_descriptor.is_64());

        new_cpl = new_cs_selector.rpl();
    }

    *cpl = 0; // run update_eflags at cpl 0
    update_eflags(new_eflags);

    if new_eflags & FLAG_VM != 0 {
        *flags |= FLAG_VM;
    }
    *cpl = new_cpl;
    cpl_changed();

    if source == TaskSwitchSource::CallOrInt {
        *flags |= FLAG_NT;
    }

    write_reg32(
        EAX,
        safe_read32s((new_tsr_offset + TSR_EAX) as u32 as u64).unwrap(),
    );
    write_reg32(
        ECX,
        safe_read32s((new_tsr_offset + TSR_ECX) as u32 as u64).unwrap(),
    );
    write_reg32(
        EDX,
        safe_read32s((new_tsr_offset + TSR_EDX) as u32 as u64).unwrap(),
    );
    write_reg32(
        EBX,
        safe_read32s((new_tsr_offset + TSR_EBX) as u32 as u64).unwrap(),
    );

    write_reg32(
        ESP,
        safe_read32s((new_tsr_offset + TSR_ESP) as u32 as u64).unwrap(),
    );
    write_reg32(
        EBP,
        safe_read32s((new_tsr_offset + TSR_EBP) as u32 as u64).unwrap(),
    );
    write_reg32(
        ESI,
        safe_read32s((new_tsr_offset + TSR_ESI) as u32 as u64).unwrap(),
    );
    write_reg32(
        EDI,
        safe_read32s((new_tsr_offset + TSR_EDI) as u32 as u64).unwrap(),
    );

    if !switch_seg(
        ES,
        safe_read16((new_tsr_offset + TSR_ES) as u32 as u64).unwrap(),
    ) || !switch_seg(
        SS,
        safe_read16((new_tsr_offset + TSR_SS) as u32 as u64).unwrap(),
    ) || !switch_seg(
        DS,
        safe_read16((new_tsr_offset + TSR_DS) as u32 as u64).unwrap(),
    ) || !switch_seg(
        FS,
        safe_read16((new_tsr_offset + TSR_FS) as u32 as u64).unwrap(),
    ) || !switch_seg(
        GS,
        safe_read16((new_tsr_offset + TSR_GS) as u32 as u64).unwrap(),
    ) {
        // XXX: Should be checked before side effects
        dbg_assert!(false);
    }

    *instruction_pointer = (get_seg_cs() as u32).wrapping_add(if new_eflags & FLAG_VM != 0 {
        new_eip as u32 & 0xFFFF
    }
    else {
        new_eip as u32
    }) as u64;

    *segment_offsets.offset(TR as isize) = descriptor.base();
    *segment_limits.offset(TR as isize) = descriptor.effective_limit();
    *sreg.offset(TR as isize) = selector.raw;

    set_cr3(new_cr3 as u32 as u64);

    *cr.offset(0) |= CR0_TS;

    if let Some(error_code) = error_code {
        if tss_is_16 {
            push16(error_code & 0xFFFF).unwrap();
        }
        else {
            push32(error_code).unwrap();
        }
    }

    update_state_flags();
}

pub unsafe fn after_block_boundary() { jit_block_boundary = true; }

#[no_mangle]
pub fn track_jit_exit(phys_addr: u64) {
    unsafe {
        debug_last_jump = LastJump::Compiled { phys_addr };
    }
}

#[no_mangle]
pub unsafe fn get_eflags() -> i32 {
    return *flags & !FLAGS_ALL
        | getcf() as i32
        | (getpf() as i32) << 2
        | (getaf() as i32) << 4
        | (getzf() as i32) << 6
        | (getsf() as i32) << 7
        | (getof() as i32) << 11;
}

pub unsafe fn readable_or_pagefault(addr: u64, size: i32) -> OrPageFault<()> {
    dbg_assert!(size < 0x1000);
    dbg_assert!(size > 0);

    let user = *cpl == 3;
    translate_address(addr, false, user, false, true)?;

    let end = addr.wrapping_add(size as u64).wrapping_sub(1) & !0xFFF;
    if addr & !0xFFF != end {
        translate_address(end, false, user, false, true)?;
    }

    return Ok(());
}

pub unsafe fn writable_or_pagefault(addr: u64, size: i32) -> OrPageFault<()> {
    writable_or_pagefault_cpl(*cpl, addr, size)
}

pub unsafe fn writable_or_pagefault_cpl(other_cpl: u8, addr: u64, size: i32) -> OrPageFault<()> {
    dbg_assert!(size < 0x1000);
    dbg_assert!(size > 0);

    let user = other_cpl == 3;
    translate_address(addr, true, user, false, true)?;

    let end = addr.wrapping_add(size as u64).wrapping_sub(1) & !0xFFF;
    if addr & !0xFFF != end {
        translate_address(end, true, user, false, true)?;
    }

    return Ok(());
}

pub fn translate_address_read_no_side_effects(address: u64) -> OrPageFault<u64> {
    unsafe { translate_address(address, false, *cpl == 3, false, false) }
}
#[inline]
pub fn translate_address_read(address: u64) -> OrPageFault<u64> {
    unsafe { translate_address(address, false, *cpl == 3, false, true) }
}
pub unsafe fn translate_address_read_jit(address: u64) -> OrPageFault<u64> {
    translate_address(address, false, *cpl == 3, true, true)
}

#[inline]
pub unsafe fn translate_address_write(address: u64) -> OrPageFault<u64> {
    translate_address(address, true, *cpl == 3, false, true)
}
pub unsafe fn translate_address_write_jit(address: u64, wasm_table_index: u16) -> OrPageFault<u64> {
    let mut entry = tlb_pick_entry_for_access(address);
    let user = *cpl == 3;
    if entry as i32 & (TLB_VALID | if user { TLB_NO_USER } else { 0 } | TLB_READONLY) != TLB_VALID {
        entry = do_page_walk(address, true, user, true, true, false)?;
    }
    let has_code = entry as i32 & TLB_HAS_CODE != 0;
    let phys_addr = phys_of_tlb_entry(entry, address);
    if !has_code {
        return Ok(phys_addr);
    }
    // mmio-backed pages never have code (do_page_walk), so phys_addr is
    // unusable for them: their host base doesn't survive the 32-bit tlb
    // entry encoding (see tlb_set_has_code). Computing a Page from it would
    // trip the page bounds assertion
    dbg_assert!(entry as i32 & TLB_IN_MAPPED_RANGE == 0);
    let page = Page::page_of(phys_addr);
    let is_smc = jit::jit_page_has_wasm_table_index(page, wasm_table_index);
    jit::jit_dirty_page(page);
    if !is_smc {
        return Ok(phys_addr);
    }
    dbg_log!(
        "SMC: write to addr phys={:x} virt={:x} of the running module {}, exiting",
        phys_addr,
        address as u32,
        wasm_table_index,
    );
    jit_exit_reason = JitExitReason::SelfModifyingCodeBail;
    Err(())
}

pub unsafe fn translate_address_system_read(address: u64) -> OrPageFault<u64> {
    translate_address(address, false, false, false, true)
}
pub unsafe fn translate_address_system_write(address: u64) -> OrPageFault<u64> {
    translate_address(address, true, false, false, true)
}

#[inline(always)]
pub unsafe fn translate_address(
    address: u64,
    for_writing: bool,
    user: bool,
    jit: bool,
    side_effects: bool,
) -> OrPageFault<u64> {
    let mut entry = tlb_pick_entry_for_access(address);
    if entry as i32
        & (TLB_VALID
            | if user { TLB_NO_USER } else { 0 }
            | if for_writing { TLB_READONLY } else { 0 })
        != TLB_VALID
    {
        entry = do_page_walk(address, for_writing, user, jit, side_effects, false)?;
    }
    Ok(phys_of_tlb_entry(entry, address))
}

pub unsafe fn translate_address_write_and_can_skip_dirty(address: u64) -> OrPageFault<(u64, bool)> {
    let mut entry = tlb_pick_entry_for_access(address);
    let user = *cpl == 3;
    if entry as i32 & (TLB_VALID | if user { TLB_NO_USER } else { 0 } | TLB_READONLY) != TLB_VALID {
        entry = do_page_walk(address, true, user, false, true, false)?;
    }
    Ok((
        phys_of_tlb_entry(entry, address),
        entry as i32 & TLB_HAS_CODE == 0,
    ))
}

// 32-bit paging:
// - 10 bits PD | 10 bits PT | 12 bits offset
// - 10 bits PD | 22 bits offset (4MB huge page)
//
// PAE paging:
// - 2 bits PDPT | 9 bits PD | 9 bits PT | 12 bits offset
// - 2 bits PDPT | 9 bits PD | 21 bits offset (2MB huge page)
//
// Long mode (4-level paging):
// - 9 bits PML4 | 9 bits PDPT | 9 bits PD | 9 bits PT | 12 bits offset
// - 9 bits PML4 | 9 bits PDPT | 9 bits PD | 21 bits offset (2MB huge page)
//
// Note that PAE entries are 64-bit, and can describe physical addresses over 32
// bits; the supported physical address width is given by PAE_ENTRY_RSVD.
//
// Not #[cold]: this is the only TLB fill path and runs constantly in walk-heavy phases.
pub unsafe fn do_page_walk(
    addr: u64,
    for_writing: bool,
    user: bool,
    jit: bool,
    side_effects: bool,
    for_fetch: bool,
) -> OrPageFault<u64> {
    let global;
    let mut allow_user = true;
    // accumulated write permission of upper paging levels (long mode only)
    let mut allow_write_upper = true;
    // accumulated NX permission of upper paging levels (long mode only)
    let mut allow_fetch = true;
    // CR2 receives the full linear address on #PF (real hardware stores the
    // effective address as generated, sign-extension bits included; Linux's
    // early_make_pgtable recovers the physical address from it, so squashing
    // to the low 48 bits breaks the early physmap demand-fault handler)
    let fault_addr = addr;
    if !is_canonical_address(addr) {
        // only reachable in 64-bit mode; memory references through SS should raise #SS(0)
        // instead, which isn't distinguished here
        if side_effects {
            trigger_gp_noncanonical(jit);
        }
        return Err(());
    }
    let addr = canonicalize_address(addr);
    let page = addr >> 12;
    let high;

    let cr0 = *cr;
    let cr4 = *cr.offset(4);
    // NX needs EFER.NXE and 64-bit page table entries (PAE or long mode, which implies PAE);
    // 32-bit entries are sign-extended, so their bit 63 isn't an NX bit
    let nx_enabled = *efer & EFER_NXE != 0 && cr4 & CR4_PAE != 0;
    // I/D (bit 4) of the page-fault error code: set for instruction fetches
    // when NX (see above) or CR4.SMEP is enabled
    let pfec_fetch = for_fetch && (nx_enabled || cr4 & CR4_SMEP != 0);
    // fetch faults determined by the final permissions (NX, SMEP): raised
    // before any accessed/dirty bits touch the entries
    let nx_fetch_fault = for_fetch && nx_enabled;
    let smep_fetch_fault = for_fetch && cr0 & CR0_PG != 0 && cr4 & CR4_SMEP != 0 && !user;

    if cr0 & CR0_PG == 0 {
        // paging disabled
        high = addr & 0xFFFF_F000;
        global = false
    }
    else {
        profiler::stat_increment(stat::TLB_MISS);

        let long_mode = *efer & EFER_LMA != 0;
        let pae = cr4 & CR4_PAE != 0;

        let (page_dir_addr, page_dir_entry) = if long_mode {
            // 4-level paging: PML4 → PDPT → PD (→ PT)
            dbg_assert!(pae, "Long mode requires PAE");

            let pml4_addr = (get_cr3() & PAE_ENTRY_ADDRESS) + ((addr >> 39 & 0x1FF) << 3);
            let pml4_entry = memory::read64s(pml4_addr) as u64;
            if pml4_entry & (PAGE_TABLE_PRESENT_MASK as u64) == 0 {
                if side_effects {
                    trigger_pagefault_extended(
                        fault_addr,
                        for_writing,
                        user,
                        pfec_fetch,
                        false,
                        false,
                        jit,
                    );
                }
                return Err(());
            }
            if pml4_entry & PAE_ENTRY_RSVD != 0
                || *efer & EFER_NXE == 0 && pml4_entry & (1u64 << 63) != 0
            {
                if side_effects {
                    trigger_pagefault_extended(
                        fault_addr,
                        for_writing,
                        user,
                        pfec_fetch,
                        true,
                        true,
                        jit,
                    );
                }
                return Err(());
            }
            allow_user &= pml4_entry & (PAGE_TABLE_USER_MASK as u64) != 0;
            allow_write_upper &= pml4_entry & (PAGE_TABLE_RW_MASK as u64) != 0;
            allow_fetch &= pml4_entry & 0x8000_0000_0000_0000u64 == 0;
            if side_effects && pml4_entry & (PAGE_TABLE_ACCESSED_MASK as u64) == 0 {
                memory::write8(
                    pml4_addr,
                    (pml4_entry | PAGE_TABLE_ACCESSED_MASK as u64) as i32,
                );
            }

            let pdpt_addr = (pml4_entry & PAE_ENTRY_ADDRESS) + (((addr >> 30) & 0x1FF) << 3);
            let pdpt_entry = memory::read64s(pdpt_addr) as u64;
            if pdpt_entry & (PAGE_TABLE_PRESENT_MASK as u64) == 0 {
                if side_effects {
                    trigger_pagefault_extended(
                        fault_addr,
                        for_writing,
                        user,
                        pfec_fetch,
                        false,
                        false,
                        jit,
                    );
                }
                return Err(());
            }
            if pdpt_entry & PAE_ENTRY_RSVD != 0
                || *efer & EFER_NXE == 0 && pdpt_entry & (1u64 << 63) != 0
            {
                if side_effects {
                    trigger_pagefault_extended(
                        fault_addr,
                        for_writing,
                        user,
                        pfec_fetch,
                        true,
                        true,
                        jit,
                    );
                }
                return Err(());
            }
            allow_user &= pdpt_entry & (PAGE_TABLE_USER_MASK as u64) != 0;
            allow_write_upper &= pdpt_entry & (PAGE_TABLE_RW_MASK as u64) != 0;
            allow_fetch &= pdpt_entry & 0x8000_0000_0000_0000u64 == 0;
            dbg_assert!(
                pdpt_entry & (PAGE_TABLE_PSE_MASK as u64) == 0,
                "TODO: 1GB pages in long mode"
            );
            if side_effects && pdpt_entry & (PAGE_TABLE_ACCESSED_MASK as u64) == 0 {
                memory::write8(
                    pdpt_addr,
                    (pdpt_entry | PAGE_TABLE_ACCESSED_MASK as u64) as i32,
                );
            }

            let page_dir_addr = (pdpt_entry & PAE_ENTRY_ADDRESS) + (((addr >> 21) & 0x1FF) << 3);
            let page_dir_entry = memory::read64s(page_dir_addr) as u64;

            (page_dir_addr, page_dir_entry)
        }
        else if pae {
            let pdpt_entry = *reg_pdpte.offset(((addr >> 30) & 3) as isize);
            if pdpt_entry as i32 & PAGE_TABLE_PRESENT_MASK == 0 {
                if side_effects {
                    trigger_pagefault_extended(
                        fault_addr,
                        for_writing,
                        user,
                        pfec_fetch,
                        false,
                        false,
                        jit,
                    );
                }
                return Err(());
            }

            let page_dir_addr =
                (pdpt_entry as u64 & PAE_ENTRY_ADDRESS) + (((addr >> 21) & 0x1FF) << 3);
            let page_dir_entry = memory::read64s(page_dir_addr) as u64;

            (page_dir_addr, page_dir_entry)
        }
        else {
            let page_dir_addr =
                (*cr.offset(3) as u32 & 0xFFFF_F000) as u64 + (((addr >> 22) & 0x3FF) << 2);
            let page_dir_entry = memory::read32s(page_dir_addr) as u64;
            (page_dir_addr, page_dir_entry)
        };

        if page_dir_entry as i32 & PAGE_TABLE_PRESENT_MASK == 0 {
            if side_effects {
                trigger_pagefault_extended(
                    fault_addr,
                    for_writing,
                    user,
                    pfec_fetch,
                    false,
                    false,
                    jit,
                );
            }
            return Err(());
        }
        // reserved-bit faults: physical address bits beyond the supported 32
        // bits (v86 maps memory below 4 GiB) and NX while EFER.NXE=0. Only
        // 64-bit-entry (PAE/long-mode) formats; 32-bit entries are
        // read32s-sign-extended, so gate on pae
        let entry_rsvd = if long_mode { PAE_ENTRY_RSVD } else { PAE_LEGACY_ENTRY_RSVD };
        if pae
            && (page_dir_entry & entry_rsvd != 0
                || *efer & EFER_NXE == 0 && page_dir_entry & (1u64 << 63) != 0)
        {
            if side_effects {
                trigger_pagefault_extended(
                    fault_addr,
                    for_writing,
                    user,
                    pfec_fetch,
                    true,
                    true,
                    jit,
                );
            }
            return Err(());
        }

        let kernel_write_override = !user && 0 == cr0 & CR0_WP;
        let mut allow_write = page_dir_entry as i32 & PAGE_TABLE_RW_MASK != 0;
        allow_write &= allow_write_upper;
        allow_user &= page_dir_entry as i32 & PAGE_TABLE_USER_MASK != 0;
        allow_fetch &= page_dir_entry & 0x8000_0000_0000_0000u64 == 0;

        // PAE (and long mode) always honour the size bit, 32-bit paging only with CR4.PSE
        if 0 != page_dir_entry as i32 & PAGE_TABLE_PSE_MASK && (pae || 0 != cr4 & CR4_PSE) {
            // size bit is set

            if pae && page_dir_entry & PAE_PDE_PS_RSVD != 0 {
                // reserved bits 20:13 in 2 MiB PDEs
                if side_effects {
                    trigger_pagefault_extended(
                        fault_addr,
                        for_writing,
                        user,
                        pfec_fetch,
                        true,
                        true,
                        jit,
                    );
                }
                return Err(());
            }
            if for_writing && !allow_write && !kernel_write_override || user && !allow_user {
                if side_effects {
                    trigger_pagefault_extended(
                        fault_addr,
                        for_writing,
                        user,
                        pfec_fetch,
                        true,
                        false,
                        jit,
                    );
                }
                return Err(());
            }
            if nx_fetch_fault && !allow_fetch || smep_fetch_fault && allow_user {
                // NX fetch fault / SMEP supervisor fetch of a user page: no
                // accessed/dirty bits are set on a faulting fetch
                if side_effects {
                    trigger_pagefault_nx(fault_addr);
                }
                return Err(());
            }

            // set the accessed and dirty bits

            let new_page_dir_entry = page_dir_entry
                | PAGE_TABLE_ACCESSED_MASK as u64
                | if for_writing { PAGE_TABLE_DIRTY_MASK as u64 } else { 0 };

            if side_effects && page_dir_entry != new_page_dir_entry {
                memory::write8(page_dir_addr, new_page_dir_entry as i32);
            }

            high = if pae {
                page_dir_entry & PAE_ENTRY_ADDRESS & !0x1F_FFFF | (addr & 0x1FF000)
            }
            else {
                (page_dir_entry as u32 & 0xFFC0_0000) as u64 | (addr & 0x3FF000)
            };
            global = page_dir_entry as i32 & PAGE_TABLE_GLOBAL_MASK == PAGE_TABLE_GLOBAL_MASK
        }
        else {
            let (page_table_addr, page_table_entry) = if pae {
                let page_table_addr =
                    (page_dir_entry & PAE_ENTRY_ADDRESS) + (((addr >> 12) & 0x1FF) << 3);
                let page_table_entry = memory::read64s(page_table_addr) as u64;

                (page_table_addr, page_table_entry)
            }
            else {
                let page_table_addr =
                    (page_dir_entry as u32 & 0xFFFF_F000) as u64 + (((addr >> 12) & 0x3FF) << 2);
                let page_table_entry = memory::read32s(page_table_addr) as u64;
                (page_table_addr, page_table_entry)
            };

            let present = page_table_entry as i32 & PAGE_TABLE_PRESENT_MASK != 0;
            if pae
                && present
                && (page_table_entry & entry_rsvd != 0
                    || *efer & EFER_NXE == 0 && page_table_entry & (1u64 << 63) != 0)
            {
                if side_effects {
                    trigger_pagefault_extended(
                        fault_addr,
                        for_writing,
                        user,
                        pfec_fetch,
                        true,
                        true,
                        jit,
                    );
                }
                return Err(());
            }
            allow_write &= page_table_entry as i32 & PAGE_TABLE_RW_MASK != 0;
            allow_user &= page_table_entry as i32 & PAGE_TABLE_USER_MASK != 0;
            allow_fetch &= page_table_entry & 0x8000_0000_0000_0000u64 == 0;

            if !present
                || for_writing && !allow_write && !kernel_write_override
                || user && !allow_user
            {
                if side_effects {
                    trigger_pagefault_extended(
                        fault_addr,
                        for_writing,
                        user,
                        pfec_fetch,
                        present,
                        false,
                        jit,
                    );
                }
                return Err(());
            }
            if nx_fetch_fault && !allow_fetch || smep_fetch_fault && allow_user {
                // NX fetch fault / SMEP supervisor fetch of a user page: no
                // accessed/dirty bits are set on a faulting fetch
                if side_effects {
                    trigger_pagefault_nx(fault_addr);
                }
                return Err(());
            }

            // Set the accessed and dirty bits
            // Note: dirty bit is only set on the page table entry
            let new_page_dir_entry = page_dir_entry | PAGE_TABLE_ACCESSED_MASK as u64;
            if side_effects && new_page_dir_entry != page_dir_entry {
                memory::write8(page_dir_addr, new_page_dir_entry as i32);
            }
            let new_page_table_entry = page_table_entry
                | PAGE_TABLE_ACCESSED_MASK as u64
                | if for_writing { PAGE_TABLE_DIRTY_MASK as u64 } else { 0 };
            if side_effects && page_table_entry != new_page_table_entry {
                memory::write8(page_table_addr, new_page_table_entry as i32);
            }

            high = if pae {
                page_table_entry & PAE_ENTRY_ADDRESS
            }
            else {
                (page_table_entry as u32 & 0xFFFF_F000) as u64
            };
            global = page_table_entry as i32 & PAGE_TABLE_GLOBAL_MASK == PAGE_TABLE_GLOBAL_MASK
        }
    }

    if side_effects && tlb_pick_entry(addr) == 0 {
        if valid_tlb_entries_count == VALID_TLB_ENTRY_MAX {
            profiler::stat_increment(stat::TLB_FULL);
            clear_tlb();
            // also clear global entries if tlb is almost full after clearing non-global pages
            if valid_tlb_entries_count > VALID_TLB_ENTRY_MAX * 3 / 4 {
                profiler::stat_increment(stat::TLB_GLOBAL_FULL);
                full_clear_tlb();
            }
        }
        dbg_assert!(valid_tlb_entries_count < VALID_TLB_ENTRY_MAX);
        valid_tlb_entries[valid_tlb_entries_count as usize] = page;
        valid_tlb_entries_count += 1;
        // TODO: Check that there are no duplicates in valid_tlb_entries
        // XXX: There will probably be duplicates due to invlpg deleting
        // entries from tlb_data but not from valid_tlb_entries
    }

    let is_in_mapped_range = memory::in_mapped_range(high);
    let has_code = if side_effects {
        !is_in_mapped_range && jit::jit_page_has_code(Page::page_of(high))
    }
    else {
        // If side_effects is false, don't call into jit::jit_page_has_code. This value is not used
        // anyway (we only get here by translate_address_read_no_side_effects, which only uses the
        // address part)
        true
    };
    // nx_enabled computed at the top of this function
    let info_bits = TLB_VALID
        | if for_writing { 0 } else { TLB_READONLY }
        | if allow_user { 0 } else { TLB_NO_USER }
        | if is_in_mapped_range { TLB_IN_MAPPED_RANGE } else { 0 }
        | if global && 0 != cr4 & CR4_PGE { TLB_GLOBAL } else { 0 }
        | if has_code { TLB_HAS_CODE } else { 0 }
        | if nx_enabled && !allow_fetch { TLB_NOT_EXECUTABLE } else { 0 };

    if side_effects {
        // bake in the host address (the addition with memory::mem8) to save an instruction
        // from the fast path of memory accesses
        let tlb_entry = memory::tlb_host_base(high) ^ (page << 12) | info_bits as u64;
        dbg_assert!(high & 0xFFF == 0);
        tlb_put_entry(addr, tlb_entry);

        jit::update_tlb_code(page, Page::page_of(high));
    }

    Ok(memory::tlb_host_base(high) ^ (page << 12) | info_bits as u64)
}

#[no_mangle]
pub unsafe fn full_clear_tlb() {
    profiler::stat_increment(stat::FULL_CLEAR_TLB);
    // clear tlb including global pages
    *last_virt_eip64 = -1;
    for i in 0..valid_tlb_entries_count {
        let page = valid_tlb_entries[i as usize] as u64;
        if page < 0x10_0000 {
            clear_tlb_code(page);
            tlb_data[page as usize] = 0;
        }
        else {
            let idx = tlb_high_index(page);
            if tlb_high_page[idx] == tlb_high_tag(page) {
                tlb_high_evict(idx);
            }
        }
    }
    valid_tlb_entries_count = 0;

    if CHECK_TLB_INVARIANTS {
        #[allow(static_mut_refs)]
        for &entry in tlb_data.iter() {
            dbg_assert!(entry == 0);
        }
    };
}

#[no_mangle]
pub unsafe fn clear_tlb() {
    profiler::stat_increment(stat::CLEAR_TLB);
    // clear tlb excluding global pages
    *last_virt_eip64 = -1;
    let mut global_page_offset = 0;
    for i in 0..valid_tlb_entries_count {
        let page = valid_tlb_entries[i as usize] as u64;
        // only the flag bits, which live in the low half of the entry
        let entry = if page < 0x10_0000 {
            tlb_data[page as usize] as u64 as u32 as i32
        }
        else {
            let idx = tlb_high_index(page);
            if tlb_high_page[idx] == tlb_high_tag(page) {
                tlb_high_entry[idx] as i32
            }
            else {
                0
            }
        };
        if 0 != entry & TLB_GLOBAL {
            // reinsert at the front
            valid_tlb_entries[global_page_offset as usize] = page;
            global_page_offset += 1;
        }
        else if page < 0x10_0000 {
            clear_tlb_code(page);
            tlb_data[page as usize] = 0;
        }
        else {
            let idx = tlb_high_index(page);
            if tlb_high_page[idx] == tlb_high_tag(page) {
                tlb_high_evict(idx);
            }
        }
    }
    valid_tlb_entries_count = global_page_offset;

    if CHECK_TLB_INVARIANTS {
        #[allow(static_mut_refs)]
        for &entry in tlb_data.iter() {
            dbg_assert!(entry == 0 || 0 != entry as u64 as u32 & TLB_GLOBAL as u32);
        }
    };
}

#[no_mangle]
pub unsafe fn trigger_de_jit(eip_offset_in_page: i32) {
    dbg_log!("#de in jit mode");
    dbg_assert!(eip_offset_in_page >= 0 && eip_offset_in_page < 0x1000);
    *instruction_pointer = *instruction_pointer & !0xFFF | eip_offset_in_page as u64;
    jit_exit_reason = JitExitReason::CpuException {
        code: CPU_EXCEPTION_DE,
        error_code: None,
    }
}

#[no_mangle]
pub unsafe fn trigger_ud_jit(eip_offset_in_page: i32) {
    dbg_log!("#ud in jit mode");
    dbg_assert!(eip_offset_in_page >= 0 && eip_offset_in_page < 0x1000);
    *instruction_pointer = *instruction_pointer & !0xFFF | eip_offset_in_page as u64;
    jit_exit_reason = JitExitReason::CpuException {
        code: CPU_EXCEPTION_UD,
        error_code: None,
    }
}

#[no_mangle]
pub unsafe fn trigger_nm_jit(eip_offset_in_page: i32) {
    dbg_log!("#nm in jit mode");
    dbg_assert!(eip_offset_in_page >= 0 && eip_offset_in_page < 0x1000);
    *instruction_pointer = *instruction_pointer & !0xFFF | eip_offset_in_page as u64;
    jit_exit_reason = JitExitReason::CpuException {
        code: CPU_EXCEPTION_NM,
        error_code: None,
    }
}

#[no_mangle]
pub unsafe fn trigger_gp_jit(code: i32, eip_offset_in_page: i32) {
    dbg_log!("#gp in jit mode");
    dbg_assert!(eip_offset_in_page >= 0 && eip_offset_in_page < 0x1000);
    *instruction_pointer = *instruction_pointer & !0xFFF | eip_offset_in_page as u64;
    jit_exit_reason = JitExitReason::CpuException {
        code: CPU_EXCEPTION_GP,
        error_code: Some(code),
    }
}

/// #GP(0) for a memory access to a non-canonical address (from do_page_walk)
#[cold]
unsafe fn trigger_gp_noncanonical(jit: bool) {
    if jit {
        dbg_log!("#gp (non-canonical address) in jit mode");
        jit_exit_reason = JitExitReason::CpuException {
            code: CPU_EXCEPTION_GP,
            error_code: Some(0),
        };
    }
    else {
        trigger_gp(0);
    }
}

#[no_mangle]
pub unsafe fn exit_jit() {
    #[allow(static_mut_refs)]
    let (code, error_code) = match std::mem::replace(&mut jit_exit_reason, JitExitReason::None) {
        JitExitReason::CpuException { code, error_code } => (code, error_code),
        JitExitReason::SelfModifyingCodeBail => return,
        JitExitReason::None => {
            dbg_assert!(false, "exit_jit without exit reason");
            return;
        },
    };
    if DEBUG {
        if js::cpu_exception_hook(code) {
            return;
        }
    }
    call_interrupt_vector(code, false, error_code);
}

// Set to the vector number while an exception/interrupt is being delivered
// (-1 otherwise). A page fault during delivery of a #PF is promoted to #DF
// (fault-class + fault-class), while a fault during #DF delivery shuts the
// machine down (triple fault). A page fault during delivery of a benign
// exception (e.g. #DE mid-delivery with a not-present stack page) is NOT
// escalated: the PF is delivered normally and the source instruction
// re-executes after the handler's iret, restarting the outer delivery.
pub static mut in_interrupt_delivery: i32 = -1;
pub static mut delivering_double_fault: bool = false;

/// An APIC INIT IPI requested a CPU-local reset; carried out at the next
/// instruction boundary (cycle_internal), like hardware delivering INIT
/// between instructions
pub static mut cpu_init_pending: bool = false;

#[no_mangle]
pub unsafe fn request_cpu_init() {
    cpu_init_pending = true;
    stop_jit_chaining();
}

/// True while do_many_cycles_native runs instructions. Device, APIC and port
/// handlers called from there run in the middle of an instruction (possibly
/// inside compiled code whose registers live in wasm locals), so interrupts
/// they raise must not be delivered until the next instruction boundary.
pub static mut in_cpu_execution: bool = false;

/// An interrupt may have become deliverable during the current instruction;
/// checked at the next instruction boundary (cycle_internal)
pub static mut irq_check_pending: bool = false;

/// Make compiled 64-bit code return to cycle_internal at its next chaining point
unsafe fn stop_jit_chaining() { jit_chain_instruction_limit = *instruction_counter; }

/// handle_irqs for callers that may run in the middle of an instruction:
/// delivering the interrupt there would push a stale rip and switch stacks
/// under compiled code that keeps running with its cached registers
pub unsafe fn handle_irqs_or_defer() {
    if in_cpu_execution {
        irq_check_pending = true;
        stop_jit_chaining();
        // compiled 64-bit code that called into the interpreter (e.g. popf setting IF, or an
        // out instruction raising an irq) leaves after the current instruction, so that
        // cycle_internal can deliver the interrupt (a no-op outside jit64)
        crate::jit64::JIT64_EXIT = true;
    }
    else {
        handle_irqs();
    }
}

/// Debug facilities (DR0-3 breakpoints via DR7, TF single-stepping, INT1).
/// When any of these are armed the CPU runs interpreter-only: JIT dispatch
/// and compilation are gated off so the per-instruction checks in
/// jit_run_interpreted see everything. Recomputed on every DR7 write.
// Debug facilities that force interpreter-only execution, maintained by
// dr7_update_armed (the only writer). Bit 0: instruction breakpoints armed, bit 1:
// data watchpoints armed.
pub static mut debug_bp_armed: u8 = 0;
/// Data-watchpoint hits (B0-B3 bits) accumulated by the current instruction;
/// delivered as a trap-class #DB after it completes. Cleared on any other
/// exception delivery (a faulting instruction is re-executed and re-hits).
pub static mut db_pending_data: i32 = 0;
/// Requests RF=1 in the next saved exception image (fault-class #DB from an
/// instruction breakpoint); consumed by call_interrupt_vector{,64}.
pub static mut force_rf_in_exception_image: bool = false;

unsafe fn take_force_rf_in_exception_image() -> bool {
    let force = force_rf_in_exception_image;
    force_rf_in_exception_image = false;
    force
}

pub const DR6_BS: i32 = 1 << 14;
/// DR6 bits that read as 1 (reserved)
pub const DR6_RESERVED_READ: i32 = 0xFFFF0FF0u32 as i32;

#[inline(always)]
pub unsafe fn debug_mode_active() -> bool {
    debug_bp_armed != 0 || *flags & (FLAG_TRAP | FLAG_RF) != 0
}

/// Deliver #DB. `matched` holds the B0-B3 bits of the breakpoints that fired.
/// Faults (instruction breakpoints) restart at the faulting instruction and
/// save an image with RF=1; traps (TF, data watchpoints, INT1) push the
/// address of the next instruction.
#[inline(never)]
#[cold]
unsafe fn deliver_db(matched: i32, single_step: bool, fault: bool) {
    dbg_log!(
        "#db matched={:x} ss={} fault={} ip={:x}",
        matched,
        single_step,
        fault,
        *instruction_pointer
    );
    // B0-B3 report only the current event; BS is sticky-or'ed (Intel SDM
    // 17.2.4: the processor updates B0-B3 and BS on every debug exception)
    let mut dr6 = *dreg.offset(6) & DR6_BS;
    if single_step {
        dr6 |= DR6_BS;
    }
    *dreg.offset(6) = dr6 | matched;
    db_pending_data = 0;
    if fault {
        force_rf_in_exception_image = true;
    }
    call_interrupt_vector(CPU_EXCEPTION_DB, false, None);
}

/// INT1 (icebp, opcode 0xF1): #DB trap with no condition bits; not a
/// software interrupt (no DPL check)
#[no_mangle]
pub unsafe fn int1() { deliver_db(0, false, false); }

/// Recompute the armed flags after a DR7 write and make sure no compiled
/// code keeps running while breakpoints are armed.
pub unsafe fn dr7_update_armed() {
    let dr7 = *dreg.offset(7);
    let mut exec = false;
    let mut data = false;
    for n in 0..4 {
        if dr7 >> (n * 2) & 3 != 0 {
            // Ln or Gn set
            if dr7 >> (16 + n * 4) & 3 == 0 {
                exec = true;
            }
            else {
                data = true;
            }
        }
    }
    debug_bp_armed = exec as u8 | (data as u8) << 1;
    if exec || data {
        // leave compiled 64-bit code (a no-op outside jit64); 0F 23 is a
        // block boundary, so the 32-bit JIT returns to the gated dispatcher
        // right after this instruction
        crate::jit64::JIT64_EXIT = true;
    }
}

/// Data-watchpoint check, called from the safe_read*/safe_write* family.
/// Only records the hit; the trap is delivered after the instruction.
#[inline(always)]
pub unsafe fn dbg_check_data_wp(addr: u64, size: u64, is_write: bool) {
    let dr7 = *dreg.offset(7);
    for n in 0..4 {
        if dr7 >> (n * 2) & 3 == 0 {
            continue; // breakpoint n not enabled
        }
        let rw = dr7 >> (16 + n * 4) & 3;
        if rw == 0 || rw == 2 {
            continue; // execution breakpoint, or i/o breakpoint (CR4.DE, not modelled)
        }
        if !is_write && rw == 1 {
            continue; // write-only watchpoint
        }
        let len = 1u64 << (dr7 >> (18 + n * 4) & 3);
        let base = *dreg.offset(n as isize) as u32 as u64;
        if addr < base.wrapping_add(len) && base < addr.wrapping_add(size) {
            // [addr, addr+size) overlaps [base, base+len)
            db_pending_data |= 1 << n;
        }
    }
}

/// Pagefault handling with the jit works as follows:
/// - If the slow path is taken, it calls safe_{read,write}*_jit
/// - safe_{read,write}*_jit call translate_address_{read,write}_jit
/// - translate_address_{read,write}_jit do the normal page walk and call this method with
///   jit=true when a page fault happens
/// - this method prepares a page fault by setting cr2, and writes the error code
///   into jit_exit_reason. This method *doesn't* trigger the interrupt, as registers are
///   still stored in the wasm module
/// - back in the wasm module, the generated code detects the page fault, restores the registers
///   and finally calls exit_jit, which does the interrupt
///
/// Non-jit resets the instruction pointer and does the PF interrupt directly
pub unsafe fn trigger_pagefault(addr: u64, present: bool, write: bool, user: bool, jit: bool) {
    trigger_pagefault_extended(addr, write, user, false, present, false, jit)
}

/// Page fault with full control over the error-code bits. `fetch` is the I/D
/// bit (instruction fetch), `reserved` is the RSVD bit (page fault caused by a
/// reserved bit in a present paging-structure entry). Note that P=1 is always
/// implied for these (a present entry caused the fault).
pub unsafe fn trigger_pagefault_extended(
    addr: u64,
    write: bool,
    user: bool,
    fetch: bool,
    present: bool,
    reserved: bool,
    jit: bool,
) {
    if config::LOG_PAGE_FAULTS {
        dbg_log!(
            "page fault{} w={} u={} p={} r={} i={} eip={:x} cr2={:x}",
            if jit { "jit" } else { "" },
            write as i32,
            user as i32,
            present as i32,
            reserved as i32,
            fetch as i32,
            *previous_ip,
            addr
        );
        dbg_trace();
    }
    profiler::stat_increment(stat::PAGE_FAULT);
    *cr.offset(2) = addr as u32 as i32;
    *cr2_64 = addr;
    // invalidate tlb entry
    tlb_invalidate_page(addr);
    let error_code = (user as i32) << 2
        | (write as i32) << 1
        | present as i32
        | (reserved as i32) << 3
        | (fetch as i32) << 4;
    if jit {
        jit_exit_reason = JitExitReason::CpuException {
            code: CPU_EXCEPTION_PF,
            error_code: Some(error_code),
        };
    }
    else {
        *instruction_pointer = *previous_ip;
        if DEBUG {
            if js::cpu_exception_hook(CPU_EXCEPTION_PF) {
                return;
            }
        }
        if delivering_double_fault {
            // a fault while delivering #DF: triple fault, shut down
            panic!(
                "Triple fault: page fault during #DF delivery, cr2={:x}",
                addr
            );
        }
        if in_interrupt_delivery == CPU_EXCEPTION_PF {
            // a page fault while delivering a page fault escalates to #DF
            // (fault-class + fault-class); page faults nest at most one level
            // in real hardware
            dbg_log!("#DF: page fault during #PF delivery, cr2={:x}", addr);
            delivering_double_fault = true;
            in_interrupt_delivery = -1;
            call_interrupt_vector(CPU_EXCEPTION_DF, false, Some(0));
            delivering_double_fault = false;
            return;
        }
        // note: a page fault during delivery of a benign exception (#DE, ...)
        // is not escalated — the PF is delivered normally and the outer
        // delivery restarts when the handler's iret re-runs the source
        call_interrupt_vector(CPU_EXCEPTION_PF, false, Some(error_code));
    }
}

pub fn tlb_set_has_code(physical_page: Page, has_code: bool) {
    for i in 0..unsafe { valid_tlb_entries_count } {
        let page = unsafe { valid_tlb_entries[i as usize] } as u64;
        let entry = unsafe { tlb_pick_entry(page << 12) };
        if 0 != entry {
            if entry as i32 & TLB_IN_MAPPED_RANGE != 0 {
                // The host base of mmio-backed pages (e.g. the BIOS alias at
                // 0xFFF00000) can exceed 4 GiB and doesn't survive the i32
                // entry encoding; they never have code (see do_page_walk), so
                // there is nothing to mark
                continue;
            }
            let tlb_physical_page = Page::page_of(phys_of_tlb_entry(entry, page << 12));
            if physical_page == tlb_physical_page {
                unsafe {
                    tlb_put_entry(
                        page << 12,
                        if has_code {
                            entry | TLB_HAS_CODE as u64
                        }
                        else {
                            entry & !(TLB_HAS_CODE as u64)
                        },
                    )
                }
                if !has_code {
                    clear_tlb_code(page);
                }
            }
        }
    }

    check_tlb_invariants();
}
pub fn tlb_set_has_code_multiple(physical_pages: &HashSet<Page>, has_code: bool) {
    let physical_pages: Vec<Page> = physical_pages.into_iter().copied().collect();
    for i in 0..unsafe { valid_tlb_entries_count } {
        let page = unsafe { valid_tlb_entries[i as usize] } as u64;
        let entry = unsafe { tlb_pick_entry(page << 12) };
        if 0 != entry {
            if entry as i32 & TLB_IN_MAPPED_RANGE != 0 {
                // see tlb_set_has_code
                continue;
            }
            let tlb_physical_page = Page::page_of(phys_of_tlb_entry(entry, page << 12));
            if physical_pages.contains(&tlb_physical_page) {
                unsafe {
                    tlb_put_entry(
                        page << 12,
                        if has_code {
                            entry | TLB_HAS_CODE as u64
                        }
                        else {
                            entry & !(TLB_HAS_CODE as u64)
                        },
                    )
                }
            }
        }
    }

    check_tlb_invariants();
}

pub fn check_tlb_invariants() {
    if !CHECK_TLB_INVARIANTS {
        return;
    }

    for i in 0..unsafe { valid_tlb_entries_count } {
        let page = unsafe { valid_tlb_entries[i as usize] } as u64;
        let entry = unsafe { tlb_pick_entry(page << 12) };

        if 0 == entry || 0 != entry as i32 & TLB_IN_MAPPED_RANGE {
            // there's no code in mapped memory
            continue;
        }

        let target = phys_of_tlb_entry(entry, page << 12);
        dbg_assert!(!memory::in_mapped_range(target));

        let entry_has_code = entry as i32 & TLB_HAS_CODE != 0;
        let has_code = jit::jit_page_has_code(Page::page_of(target));

        // If some code has been created in a page, the corresponding tlb entries must be marked
        dbg_assert!(!has_code || entry_has_code);
    }
}

pub const DISABLE_EIP_TRANSLATION_OPTIMISATION: bool = false;

#[inline]
pub unsafe fn read_imm8() -> OrPageFault<i32> {
    let eip = *instruction_pointer;
    if DISABLE_EIP_TRANSLATION_OPTIMISATION || 0 != eip & !0xFFF ^ *last_virt_eip64 as u64 {
        *eip_phys64 = translate_address_read(eip)? ^ eip;
        *last_virt_eip64 = (eip & !0xFFF) as i64
    }
    dbg_assert!(!memory::in_mapped_range(*eip_phys64 ^ eip));
    let data8 = memory::read8_no_mmap_check(*eip_phys64 ^ eip);
    *instruction_pointer = eip + 1;
    return Ok(data8);
}

#[inline]
pub unsafe fn read_imm8s() -> OrPageFault<i32> { return Ok(read_imm8()? << 24 >> 24); }

#[inline]
pub unsafe fn read_imm16() -> OrPageFault<i32> {
    // Two checks in one comparison:
    // 1. Did the high bits of eip change
    // or 2. Are the low 12 bits of eip 0xFFF (and this read crosses a page boundary)
    if DISABLE_EIP_TRANSLATION_OPTIMISATION
        || (*instruction_pointer ^ *last_virt_eip64 as u64) > 0xFFE
    {
        return Ok(read_imm8()? | read_imm8()? << 8);
    }
    else {
        let data16 = memory::read16(*eip_phys64 ^ *instruction_pointer);
        *instruction_pointer = *instruction_pointer + 2;
        return Ok(data16);
    };
}

#[inline]
pub unsafe fn read_imm32s() -> OrPageFault<i32> {
    // Analogue to the above comment
    if DISABLE_EIP_TRANSLATION_OPTIMISATION
        || (*instruction_pointer ^ *last_virt_eip64 as u64) > 0xFFC
    {
        return Ok(read_imm16()? | read_imm16()? << 16);
    }
    else {
        let data32 = memory::read32s(*eip_phys64 ^ *instruction_pointer);
        *instruction_pointer = *instruction_pointer + 4;
        return Ok(data32);
    };
}

#[inline]
pub unsafe fn read_imm64s() -> OrPageFault<u64> {
    let low = read_imm32s()? as u32 as u64;
    let high = read_imm32s()? as u32 as u64;
    Ok(low | high << 32)
}

#[inline]
pub unsafe fn is_osize_32() -> bool {
    dbg_assert!(!in_jit);
    if *is_64 {
        // in 64-bit mode the default operand size is 32 (0x66 -> 16,
        // REX.W -> 64, handled at dispatch)
        return *prefixes & prefix::PREFIX_MASK_OPSIZE == 0;
    }
    return *is_32 != (*prefixes & prefix::PREFIX_MASK_OPSIZE == prefix::PREFIX_MASK_OPSIZE);
}

#[inline]
pub unsafe fn is_asize_32() -> bool {
    dbg_assert!(!in_jit);
    if *is_64 {
        // in 64-bit mode the default address size is 64 (0x67 -> 32)
        return *prefixes & prefix::PREFIX_MASK_ADDRSIZE != 0;
    }
    return *is_32 != (*prefixes & prefix::PREFIX_MASK_ADDRSIZE == prefix::PREFIX_MASK_ADDRSIZE);
}

pub unsafe fn lookup_segment_selector(
    selector: SegmentSelector,
) -> OrPageFault<Result<(SegmentDescriptor, u64), SelectorNullOrInvalid>> {
    if selector.is_null() {
        return Ok(Err(SelectorNullOrInvalid::IsNull));
    }

    let (table_offset, table_limit) = if selector.is_gdt() {
        (*gdtr_offset64, *gdtr_size as u32)
    }
    else {
        (
            *segment_offsets.offset(LDTR as isize) as u32 as u64,
            *segment_limits.offset(LDTR as isize) as u32,
        )
    };

    if selector.descriptor_offset() as u32 > table_limit {
        dbg_log!(
            "segment outside of table limit: selector={:x} offset={:x} isgdt={} table_limit={:x}",
            selector.raw,
            selector.descriptor_offset(),
            selector.is_gdt(),
            table_limit
        );
        return Ok(Err(SelectorNullOrInvalid::OutsideOfTableLimit));
    }

    let descriptor_address =
        (table_offset).wrapping_add(selector.descriptor_offset() as u32 as u64);

    let descriptor = SegmentDescriptor::of_u64(memory::read64s(translate_address_system_read(
        descriptor_address,
    )?) as u64);

    Ok(Ok((descriptor, descriptor_address)))
}

#[inline(never)]
pub unsafe fn switch_seg(reg: i32, selector_raw: i32) -> bool {
    dbg_assert!(reg >= 0 && reg <= 5);
    dbg_assert!(reg != CS);
    dbg_assert!(selector_raw >= 0 && selector_raw < 0x10000);

    if vm86_mode() {
        // TODO: Should set segment_limits and segment_access_bytes if ever implemented in get_seg
        //       (only vm86, not in real mode)
    }

    if !*protected_mode || vm86_mode() {
        *sreg.offset(reg as isize) = selector_raw as u16;
        *segment_is_null.offset(reg as isize) = false;
        *segment_offsets.offset(reg as isize) = selector_raw << 4;

        if reg == SS {
            *stack_size_32 = false;
        }
        update_state_flags();
        return true;
    }

    let selector = SegmentSelector::of_u16(selector_raw as u16);
    let (mut descriptor, descriptor_address) =
        match return_on_pagefault!(lookup_segment_selector(selector), false) {
            Ok(desc) => desc,
            Err(SelectorNullOrInvalid::IsNull) => {
                if reg == SS && !(*is_64 && *cpl != 3) {
                    // a null SS is only permitted in 64-bit mode at cpl < 3
                    dbg_log!("#GP for loading 0 in SS sel={:x}", selector_raw);
                    dbg_trace();
                    trigger_gp(0);
                    return false;
                }
                // In 64-bit mode a null selector in SS is usable (the SS base
                // is ignored). The stack size attribute is not changed by a
                // null load.
                *sreg.offset(reg as isize) = selector_raw as u16;
                *segment_is_null.offset(reg as isize) = true;
                update_state_flags();
                return true;
            },
            Err(SelectorNullOrInvalid::OutsideOfTableLimit) => {
                dbg_log!(
                    "#GP for loading invalid in seg={} sel={:x}",
                    reg,
                    selector_raw,
                );
                dbg_trace();
                trigger_gp(selector_raw & !3);
                return false;
            },
        };

    if reg == SS {
        if descriptor.is_system()
            || selector.rpl() != *cpl
            || !descriptor.is_writable()
            || descriptor.dpl() != *cpl
        {
            dbg_log!("#GP for loading invalid in SS sel={:x}", selector_raw);
            trigger_gp(selector_raw & !3);
            return false;
        }

        if !descriptor.is_present() {
            dbg_log!("#SS for loading non-present in SS sel={:x}", selector_raw);
            trigger_ss(selector_raw & !3);
            return false;
        }

        *stack_size_32 = descriptor.is_32();
    }
    else {
        if descriptor.is_system()
            || !descriptor.is_readable()
            || (!descriptor.is_conforming_executable()
                && (selector.rpl() > descriptor.dpl() || *cpl > descriptor.dpl()))
        {
            dbg_log!(
                "#GP for loading invalid in seg {} sel={:x} sys={} readable={} dc={} exec={} rpl={} dpl={} cpl={} present={} paging={}",
                reg,
                selector_raw,
                descriptor.is_system(),
                descriptor.is_readable(),
                descriptor.is_dc(),
                descriptor.is_executable(),
                selector.rpl(),
                descriptor.dpl(),
                *cpl,
                descriptor.is_present(),
                *cr & CR0_PG != 0,
            );
            dbg_trace();
            trigger_gp(selector_raw & !3);
            return false;
        }

        if !descriptor.is_present() {
            dbg_log!(
                "#NP for loading not-present in seg {} sel={:x}",
                reg,
                selector_raw,
            );
            trigger_np(selector_raw & !3);
            return false;
        }
    }

    if !descriptor.accessed() {
        descriptor = descriptor.set_accessed();

        memory::write8(
            translate_address_system_write(descriptor_address.wrapping_add(5)).unwrap(),
            descriptor.access_byte() as i32,
        );
    }

    *segment_is_null.offset(reg as isize) = false;
    *segment_limits.offset(reg as isize) = descriptor.effective_limit();
    // in 64-bit mode the base of cs/ss/ds/es is ignored (forced to 0); only
    // fs and gs keep their base
    *segment_offsets.offset(reg as isize) =
        if *is_64 && reg != FS && reg != GS { 0 } else { descriptor.base() };
    *segment_access_bytes.offset(reg as isize) = descriptor.access_byte();
    *sreg.offset(reg as isize) = selector_raw as u16;

    update_state_flags();

    true
}

pub unsafe fn load_tr(selector: i32) {
    let selector = SegmentSelector::of_u16(selector as u16);
    dbg_assert!(selector.is_gdt(), "TODO: TR can only be loaded from GDT");

    let (descriptor, descriptor_address) =
        match return_on_pagefault!(lookup_segment_selector(selector)) {
            Ok((desc, addr)) => (desc, addr),
            Err(SelectorNullOrInvalid::IsNull) => {
                panic!("TODO: null TR");
            },
            Err(SelectorNullOrInvalid::OutsideOfTableLimit) => {
                panic!("TODO: TR selector outside of table limit");
            },
        };

    //dbg_log!(
    //    "load tr: {:x} offset={:x} limit={:x} is32={}",
    //    selector.raw,
    //    descriptor.base(),
    //    descriptor.effective_limit(),
    //    descriptor.system_type() == 9,
    //);

    if !descriptor.is_system() {
        panic!("#GP | ltr: not a system entry (happens when running kvm-unit-test without ACPI)");
    }

    if descriptor.system_type() != 9 && descriptor.system_type() != 1 {
        // 0xB: busy 386 TSS (GP)
        // 0x9: 386 TSS
        // 0x3: busy 286 TSS (GP)
        // 0x1: 286 TSS (??)
        panic!(
            "#GP | ltr: invalid type (type = 0x{:x})",
            descriptor.system_type()
        );
    }

    if !descriptor.is_present() {
        panic!("#NT | present bit not set (ltr)");
    }

    *tss_size_32 = descriptor.system_type() == 9;
    *segment_limits.offset(TR as isize) = descriptor.effective_limit();
    *segment_offsets.offset(TR as isize) = descriptor.base();
    if *is_64 {
        // A 64-bit TSS descriptor is 16 bytes: the high dword of the 64-bit
        // base address sits in the second qword of the descriptor.
        let descriptor_high_phys = return_on_pagefault!(translate_address_system_read(
            descriptor_address.wrapping_add(8),
        ));
        let descriptor_high = memory::read64s(descriptor_high_phys) as u64;
        *tss_base64 = (descriptor.base() as u32 as u64) | ((descriptor_high & 0xFFFF_FFFF) << 32);
    }
    *sreg.offset(TR as isize) = selector.raw;

    // Mark task as busy
    memory::write8(
        translate_address_system_write(descriptor_address.wrapping_add(5)).unwrap(),
        descriptor.set_busy().access_byte() as i32,
    );
}

pub unsafe fn load_ldt(selector: i32) -> OrPageFault<()> {
    let selector = SegmentSelector::of_u16(selector as u16);

    if selector.is_null() {
        dbg_log!("lldt: null loaded");
        *segment_limits.offset(LDTR as isize) = 0;
        *segment_offsets.offset(LDTR as isize) = 0;
        *sreg.offset(LDTR as isize) = selector.raw;
        return Ok(());
    }

    dbg_assert!(selector.is_gdt(), "TODO: LDT can only be loaded from GDT");

    let (descriptor, _) = match lookup_segment_selector(selector)? {
        Ok((desc, addr)) => (desc, addr),
        Err(SelectorNullOrInvalid::IsNull) => {
            panic!("TODO: null TR");
        },
        Err(SelectorNullOrInvalid::OutsideOfTableLimit) => {
            panic!("TODO: TR selector outside of table limit");
        },
    };

    if !descriptor.is_present() {
        panic!("#NT | present bit not set (lldt)");
    }

    if !descriptor.is_system() {
        panic!("#GP | lldt: not a system entry");
    }

    if descriptor.system_type() != 2 {
        panic!(
            "#GP | lldt: invalid type (type = 0x{:x})",
            descriptor.system_type()
        );
    }

    dbg_log!(
        "lldt: {:x} offset={:x} limit={:x}",
        selector.raw,
        descriptor.base(),
        descriptor.effective_limit()
    );
    *segment_limits.offset(LDTR as isize) = descriptor.effective_limit();
    *segment_offsets.offset(LDTR as isize) = descriptor.base();
    *sreg.offset(LDTR as isize) = selector.raw;

    Ok(())
}

#[no_mangle]
#[cfg(feature = "profiler")]
pub unsafe fn log_segment_null(segment: i32) {
    dbg_assert!(segment >= 0 && segment < 8);
    if *segment_is_null.offset(segment as isize) {
        dbg_assert!(segment != CS && segment != SS);
        dbg_log!("#gp: Access null segment in jit");
    }
}

pub unsafe fn get_seg(segment: i32) -> OrPageFault<i32> {
    dbg_assert!(segment >= 0 && segment < 8);
    if *segment_is_null.offset(segment as isize) {
        dbg_assert!(segment != CS && segment != SS);
        dbg_log!("#gp: Access null segment {}", segment);
        dbg_trace();
        dbg_assert!(!in_jit);
        trigger_gp(0);
        return Err(());
    }
    return Ok(*segment_offsets.offset(segment as isize));
}

pub unsafe fn set_cr0(cr0: i32) {
    let old_cr0 = *cr;

    if old_cr0 & CR0_AM == 0 && cr0 & CR0_AM != 0 {
        dbg_log!("Warning: Unimplemented: cr0 alignment mask");
    }
    if (cr0 & (CR0_PE | CR0_PG)) == CR0_PG {
        panic!("cannot load PG without PE");
    }

    *cr = cr0;
    *cr |= CR0_ET;

    if old_cr0 & (CR0_PG | CR0_WP) != cr0 & (CR0_PG | CR0_WP) {
        full_clear_tlb();
    }

    if *cr.offset(4) & CR4_PAE != 0
        && *efer & EFER_LME == 0
        && old_cr0 & (CR0_CD | CR0_NW | CR0_PG) != cr0 & (CR0_CD | CR0_NW | CR0_PG)
    {
        load_pdpte(get_cr3())
    }

    *protected_mode = (*cr & CR0_PE) == CR0_PE;
    *segment_access_bytes.offset(CS as isize) = 0x80 | 0x10 | 0x08 | 0x02; // P dpl0 S E RW

    let had_lma = *efer & EFER_LMA != 0;
    update_efer_lma();
    if had_lma && *efer & EFER_LMA == 0 && *is_64 {
        // leaving long mode (paging disabled while executing 64-bit code)
        dbg_log!("Leaving long mode");
        *is_64 = false;
        *is_32 = true;
        update_state_flags();
    }
}

// EFER.LMA follows EFER.LME && CR0.PG
pub unsafe fn update_efer_lma() {
    if *efer & EFER_LME != 0 && *cr & CR0_PG != 0 {
        *efer |= EFER_LMA;
    }
    else {
        *efer &= !EFER_LMA;
    }
}

/// The full 64-bit cr3 (cr[3] holds the low half)
#[inline(always)]
pub unsafe fn get_cr3() -> u64 { *cr.offset(3) as u32 as u64 | (*cr3_high as u64) << 32 }

pub unsafe fn set_cr3(mut cr3: u64) {
    if cfg!(debug_assertions) && cr3 > 0x1000000 {
        let pml4e0 = memory::read64s(cr3 & PAE_ENTRY_ADDRESS) as u64;
        let pml4e2 = memory::read64s((cr3 & PAE_ENTRY_ADDRESS) + 16) as u64;
        dbg_log!("cr3 <- {:x} pml4[0]={:x} pml4[2]={:x}", cr3, pml4e0, pml4e2);
    }
    if *cr.offset(4) & CR4_PAE != 0 {
        cr3 &= !0b1111;
        if *efer & EFER_LME == 0 {
            // 32-bit PAE: CR3 points to the PDPT and the PDPTEs are cached in
            // the CPU. In long mode (LME set) CR3 points to the PML4 instead
            // and there is no PDPTE cache.
            cr3 &= 0xFFFF_FFFF;
            load_pdpte(cr3);
        }
    }
    else {
        cr3 &= !0b111111100111 & 0xFFFF_FFFF;
        dbg_assert!(cr3 & 0xFFF == 0, "TODO");
    }
    *cr.offset(3) = cr3 as i32;
    *cr3_high = (cr3 >> 32) as u32;
    clear_tlb();
}

pub unsafe fn load_pdpte(cr3: u64) {
    dbg_assert!(cr3 & 0b1111 == 0);
    for i in 0..4 {
        let mut pdpt_entry = memory::read64s(cr3 + 8 * i as u64) as u64;
        pdpt_entry &= !0b1110_0000_0000;
        // bits 3-4 (PWT/PCD) are cache attributes we don't model
        pdpt_entry &= !0b1_1000u64;
        if pdpt_entry as i32 & PAGE_TABLE_PRESENT_MASK != 0 {
            // bits 2:1 (rw/us) are ignored in PDPTEs; setting truly reserved
            // bits should #gp, but tolerating them is harmless for bring-up
            let reserved = pdpt_entry & 0b1_1110_0000_0110;
            if reserved != 0 {
                dbg_log!("reserved bits in pdpte: {:x}", reserved);
            }
        }
        dbg_assert!(
            pdpt_entry & 0x7FF0_0000_0000_0000 == 0,
            "Unsupported: PDPT entry beyond 52 bits"
        );
        *reg_pdpte.offset(i) = pdpt_entry;
    }
}

pub unsafe fn cpl_changed() { *last_virt_eip64 = -1 }

pub unsafe fn update_cs_size(new_size: bool, new_is_64: bool) {
    let new_is_64 = new_is_64 && *efer & EFER_LMA != 0;
    dbg_assert!(
        !new_is_64 || !new_size,
        "64-bit code segment with D bit set"
    );
    if *is_32 != new_size || *is_64 != new_is_64 {
        *is_32 = new_size;
        *is_64 = new_is_64;
    }
}

#[inline(never)]
pub unsafe fn test_privileges_for_io(port: i32, size: i32) -> bool {
    if *protected_mode && (*cpl > getiopl() as u8 || (*flags & FLAG_VM != 0)) {
        if !*tss_size_32 {
            dbg_log!("#GP for port io, 16-bit TSS  port={:x} size={}", port, size);
            trigger_gp(0);
            return false;
        }

        let tsr_size = *segment_limits.offset(TR as isize);
        let tsr_offset = *segment_offsets.offset(TR as isize);

        if tsr_size >= 0x67 {
            dbg_assert!(tsr_offset + 0x64 + 2 & 0xFFF < 0xFFF);

            let iomap_base = memory::read16(return_on_pagefault!(
                translate_address_system_read((tsr_offset + 0x64 + 2) as u32 as u64),
                false
            ));
            let high_port = port + size - 1;

            if tsr_size >= (iomap_base + (high_port >> 3)) as u32 {
                let mask = ((1 << size) - 1) << (port & 7);
                let addr = return_on_pagefault!(
                    translate_address_system_read(
                        (tsr_offset + iomap_base + (port >> 3)) as u32 as u64,
                    ),
                    false
                );
                let port_info =
                    if mask & 0xFF00 != 0 { memory::read16(addr) } else { memory::read8(addr) };

                dbg_assert!(addr & 0xFFF < 0xFFF);

                if port_info & mask == 0 {
                    return true;
                }
            }
        }

        dbg_log!("#GP for port io  port={:x} size={}", port, size);
        trigger_gp(0);
        return false;
    }

    return true;
}

pub unsafe fn popa16() {
    return_on_pagefault!(readable_or_pagefault(get_stack_pointer(0), 16));

    write_reg16(DI, pop16().unwrap());
    write_reg16(SI, pop16().unwrap());
    write_reg16(BP, pop16().unwrap());
    adjust_stack_reg(2);
    write_reg16(BX, pop16().unwrap());
    write_reg16(DX, pop16().unwrap());
    write_reg16(CX, pop16().unwrap());
    write_reg16(AX, pop16().unwrap());
}

pub unsafe fn popa32() {
    return_on_pagefault!(readable_or_pagefault(get_stack_pointer(0), 32));

    write_reg32(EDI, pop32s().unwrap());
    write_reg32(ESI, pop32s().unwrap());
    write_reg32(EBP, pop32s().unwrap());
    adjust_stack_reg(4);
    write_reg32(EBX, pop32s().unwrap());
    write_reg32(EDX, pop32s().unwrap());
    write_reg32(ECX, pop32s().unwrap());
    write_reg32(EAX, pop32s().unwrap());
}

pub fn get_state_flags() -> CachedStateFlags { unsafe { *state_flags } }

#[no_mangle]
pub fn get_seg_cs() -> i32 {
    // in 64-bit mode the cs base is forced to 0
    unsafe {
        if *is_64 {
            0
        }
        else {
            *segment_offsets.offset(CS as isize)
        }
    }
}

pub unsafe fn get_seg_ss() -> i32 { return *segment_offsets.offset(SS as isize); }

pub unsafe fn segment_prefix(default_segment: i32) -> i32 {
    let prefix = *prefixes & prefix::PREFIX_MASK_SEGMENT;
    if 0 != prefix {
        dbg_assert!(prefix != prefix::SEG_PREFIX_ZERO);
        prefix as i32 - 1
    }
    else {
        default_segment
    }
}

#[inline]
pub unsafe fn get_seg_prefix(default_segment: i32) -> OrPageFault<i32> {
    dbg_assert!(!in_jit);
    let prefix = *prefixes & prefix::PREFIX_MASK_SEGMENT;
    if 0 != prefix {
        if prefix == prefix::SEG_PREFIX_ZERO {
            return Ok(0);
        }
        else {
            return get_seg(prefix as i32 - 1);
        }
    }
    else {
        return get_seg(default_segment);
    };
}

/// The segment base as a full 64-bit linear address part. In long mode the
/// FS/GS bases come from their 64-bit MSRs; all other bases are forced to 0 in
/// 64-bit mode (and every base fits in 32 bits in compatibility mode).
#[inline]
pub unsafe fn get_seg64(segment: i32) -> u64 {
    if *is_64 {
        match segment {
            FS => *fs_base,
            GS => *gs_base,
            _ => 0,
        }
    }
    else {
        *segment_offsets.offset(segment as isize) as u32 as u64
    }
}

#[inline]
pub unsafe fn get_seg_prefix64(default_segment: i32) -> OrPageFault<u64> {
    dbg_assert!(!in_jit);
    let prefix = *prefixes & prefix::PREFIX_MASK_SEGMENT;
    if 0 != prefix {
        if prefix == prefix::SEG_PREFIX_ZERO {
            return Ok(0);
        }
        else {
            return Ok(get_seg64(prefix as i32 - 1));
        }
    }
    else {
        return Ok(get_seg64(default_segment));
    };
}

#[inline]
pub unsafe fn get_seg_prefix_ds(offset: u64) -> OrPageFault<u64> {
    Ok(get_seg_prefix64(DS)?.wrapping_add(offset))
}

#[inline]
pub unsafe fn get_seg_prefix_ss(offset: u64) -> OrPageFault<u64> {
    Ok(get_seg_prefix64(SS)?.wrapping_add(offset))
}

// imm_len: number of immediate bytes following modrm/sib/disp, needed for
// RIP-relative addressing in 64-bit mode (ignored in 16/32-bit modes)
#[inline]
pub unsafe fn modrm_resolve(modrm_byte: i32, imm_len: i32) -> OrPageFault<u64> {
    if *is_64 {
        if *prefixes & prefix::PREFIX_MASK_ADDRSIZE != 0 {
            // 0x67: 32-bit addressing in 64-bit mode
            // (note: REX.B/X register extension not applied here)
            Ok(resolve_modrm32(modrm_byte)? as u32 as u64)
        }
        else {
            resolve_modrm64(modrm_byte, imm_len)
        }
    }
    else if is_asize_32() {
        Ok(resolve_modrm32(modrm_byte)? as u32 as u64)
    }
    else {
        Ok(resolve_modrm16(modrm_byte)? as u32 as u64)
    }
}

#[inline]
pub unsafe fn run_instruction(opcode: i32) { gen::interpreter::run(opcode as u32) }
#[inline]
pub unsafe fn run_instruction0f_16(opcode: i32) { gen::interpreter0f::run(opcode as u32) }
#[inline]
pub unsafe fn run_instruction0f_32(opcode: i32) {
    if *is_64 && gen::interpreter0f::is_default_64_operand_size(opcode as u32) {
        // default-64-bit 0F instructions (push/pop fs/gs) in long mode
        gen::interpreter0f::run(opcode as u32 | 0x200)
    }
    else {
        gen::interpreter0f::run(opcode as u32 | 0x100)
    }
}

pub unsafe fn cycle_internal() {
    profiler::stat_increment(stat::CYCLE_INTERNAL);
    if cpu_init_pending {
        // APIC INIT: CPU-local reset (devices and RAM are not touched)
        cpu_init_pending = false;
        reset_cpu();
        return;
    }
    if irq_check_pending {
        irq_check_pending = false;
        handle_irqs();
        if *in_hlt {
            return;
        }
    }
    let mut jit_entry = None;
    let initial_eip = *instruction_pointer;
    let initial_state_flags = *state_flags;

    // Breakpoints and single-stepping are only checked by the interpreter
    if !debug_mode_active() {
        match tlb_code_get(canonicalize_address(initial_eip) >> 12) {
            None => {},
            Some(c) => {
                let c = c.as_ref();

                if initial_state_flags == c.state_flags {
                    let state = c.state_table[initial_eip as usize & 0xFFF];
                    if state != u16::MAX {
                        jit_entry = Some((c.wasm_table_index.to_u16(), state));
                    }
                    else {
                        profiler::stat_increment(if is_near_end_of_page(initial_eip) {
                            stat::RUN_INTERPRETED_NEAR_END_OF_PAGE
                        }
                        else {
                            stat::RUN_INTERPRETED_PAGE_HAS_CODE
                        })
                    }
                }
                else {
                    profiler::stat_increment(stat::RUN_INTERPRETED_DIFFERENT_STATE);
                    let s = *state_flags;
                    if c.state_flags.cpl3() != s.cpl3() {
                        profiler::stat_increment(stat::RUN_INTERPRETED_DIFFERENT_STATE_CPL3);
                    }
                    if c.state_flags.has_flat_segmentation() != s.has_flat_segmentation() {
                        profiler::stat_increment(stat::RUN_INTERPRETED_DIFFERENT_STATE_FLAT);
                    }
                    if c.state_flags.is_32() != s.is_32() {
                        profiler::stat_increment(stat::RUN_INTERPRETED_DIFFERENT_STATE_IS32);
                    }
                    if c.state_flags.ssize_32() != s.ssize_32() {
                        profiler::stat_increment(stat::RUN_INTERPRETED_DIFFERENT_STATE_SS32);
                    }
                }
            },
        }
    }

    if let Some((wasm_table_index, initial_state)) = jit_entry {
        if jit::CHECK_JIT_STATE_INVARIANTS {
            match get_phys_eip() {
                Err(()) => dbg_assert!(false),
                Ok(phys_eip) => {
                    let entry = jit::jit_find_cache_entry(phys_eip, initial_state_flags);
                    dbg_assert!(entry.wasm_table_index.to_u16() == wasm_table_index);
                    dbg_assert!(entry.initial_state == initial_state);
                },
            }
        }
        profiler::stat_increment(stat::RUN_FROM_CACHE);
        let initial_instruction_counter = *instruction_counter;
        #[cfg(debug_assertions)]
        {
            in_jit = true;
        }
        if initial_state_flags.is_64() {
            // modules for 64-bit code get the registers as arguments
            wasm::call_indirect_jit64(
                wasm_table_index as i32 + WASM_TABLE_OFFSET as i32,
                initial_state,
            );
        }
        else {
            // modules for 16/32-bit code get the registers as arguments
            wasm::call_indirect_jit32(
                wasm_table_index as i32 + WASM_TABLE_OFFSET as i32,
                initial_state,
            );
        }
        #[cfg(debug_assertions)]
        {
            in_jit = false;
        }
        profiler::stat_increment_by(
            stat::RUN_FROM_CACHE_STEPS,
            (*instruction_counter - initial_instruction_counter) as u64,
        );
        dbg_assert!(
            *instruction_counter != initial_instruction_counter,
            "Instruction counter didn't change"
        );

        if cfg!(feature = "profiler") {
            dbg_assert!(match debug_last_jump {
                LastJump::Compiled { .. } => true,
                _ => false,
            });
            #[allow(static_mut_refs)]
            let last_jump_addr = debug_last_jump.phys_address().unwrap();
            let last_jump_opcode = if last_jump_addr != 0 {
                memory::read32s(last_jump_addr)
            }
            else {
                // Happens during exit due to loop iteration limit
                0
            };

            opstats::record_opstat_jit_exit(last_jump_opcode as u32);
        }

        if is_near_end_of_page(*instruction_pointer) {
            profiler::stat_increment(stat::RUN_FROM_CACHE_EXIT_NEAR_END_OF_PAGE);
        }
        else if (initial_eip ^ *instruction_pointer) & !0xFFF == 0 {
            profiler::stat_increment(stat::RUN_FROM_CACHE_EXIT_SAME_PAGE);
        }
        else {
            profiler::stat_increment(stat::RUN_FROM_CACHE_EXIT_DIFFERENT_PAGE);
        }
    }
    else {
        *previous_ip = initial_eip;
        let phys_addr = return_on_pagefault!(get_phys_eip());

        if !debug_mode_active() {
            match tlb_code_get(canonicalize_address(initial_eip) >> 12) {
                None => {},
                Some(c) => {
                    let c = c.as_ref();

                    if initial_state_flags == c.state_flags
                        && c.state_table[initial_eip as usize & 0xFFF] != u16::MAX
                    {
                        profiler::stat_increment(
                            stat::RUN_INTERPRETED_PAGE_HAS_ENTRY_AFTER_PAGE_WALK,
                        );
                        return;
                    }
                },
            }
        }

        #[cfg(feature = "profiler")]
        {
            if CHECK_MISSED_ENTRY_POINTS {
                jit::check_missed_entry_points(phys_addr, initial_state_flags);
            }
        }

        let initial_instruction_counter = *instruction_counter;
        jit_run_interpreted(phys_addr);

        jit::jit_increase_hotness_and_maybe_compile(
            initial_eip,
            phys_addr,
            get_seg_cs() as u32,
            initial_state_flags,
            *instruction_counter - initial_instruction_counter,
        );

        profiler::stat_increment_by(
            stat::RUN_INTERPRETED_STEPS,
            (*instruction_counter - initial_instruction_counter) as u64,
        );
        dbg_assert!(
            *instruction_counter != initial_instruction_counter,
            "Instruction counter didn't change"
        );
    };
}

#[inline]
pub unsafe fn get_phys_eip() -> OrPageFault<u64> {
    let eip = *instruction_pointer;
    if 0 != eip & !0xFFF ^ *last_virt_eip64 as u64 {
        *eip_phys64 = translate_address_read_code(eip)? ^ eip;
        *last_virt_eip64 = (eip & !0xFFF) as i64
    }
    let phys_addr = *eip_phys64 ^ eip;
    dbg_assert!(!memory::in_mapped_range(phys_addr));
    return Ok(phys_addr);
}

/// Like translate_address_read, but for instruction fetches: additionally
/// faults (with the I/D bit set in the error code) when the page is not
/// executable (NX bit, long mode) or when a supervisor fetches from a user
/// page with CR4.SMEP set, and sets I/D on fetch page faults in general when
/// EFER.NXE or CR4.SMEP is enabled.
pub fn translate_address_read_code(address: u64) -> OrPageFault<u64> {
    unsafe {
        let user = *cpl == 3;
        let mut full_entry = tlb_pick_entry_for_access(address);
        if full_entry as i32 & (TLB_VALID | if user { TLB_NO_USER } else { 0 }) != TLB_VALID {
            full_entry = do_page_walk(address, false, user, false, true, true)?;
        }
        let phys = phys_of_tlb_entry(full_entry, address);
        let entry = full_entry as i32;
        let check_nx = *efer & EFER_NXE != 0 && *cr.offset(4) & CR4_PAE != 0;
        let check_smep = *cr.offset(0) & CR0_PG != 0 && *cr.offset(4) & CR4_SMEP != 0 && *cpl != 3;
        if check_nx && entry & (TLB_VALID | TLB_NOT_EXECUTABLE) == TLB_VALID | TLB_NOT_EXECUTABLE {
            trigger_pagefault_nx(address);
            return Err(());
        }
        if check_smep && entry & (TLB_VALID | TLB_NO_USER) == TLB_VALID {
            // SMEP: supervisor-mode instruction fetch from a user page
            trigger_pagefault_extended(address, false, false, true, true, false, false);
            return Err(());
        }
        Ok(phys)
    }
}

/// Whether an instruction fetch from a page with the given tlb entry violates NX or SMEP (the
/// faults raised by translate_address_read_code after a successful translation)
unsafe fn fetch_violates_nx_or_smep(entry: i32) -> bool {
    let check_nx = *efer & EFER_NXE != 0 && *cr.offset(4) & CR4_PAE != 0;
    let check_smep = *cr.offset(0) & CR0_PG != 0 && *cr.offset(4) & CR4_SMEP != 0 && *cpl != 3;
    check_nx && entry & (TLB_VALID | TLB_NOT_EXECUTABLE) == TLB_VALID | TLB_NOT_EXECUTABLE
        || check_smep && entry & (TLB_VALID | TLB_NO_USER) == TLB_VALID
}

/// Like translate_address_read_code, but without side effects: no page fault is raised and no
/// accessed bits or tlb entries are written. Err if an instruction fetch would fault.
pub fn translate_address_read_code_no_side_effects(address: u64) -> OrPageFault<u64> {
    unsafe {
        let user = *cpl == 3;
        let mut entry = tlb_pick_entry_for_access(address);
        if entry as i32 & (TLB_VALID | if user { TLB_NO_USER } else { 0 }) != TLB_VALID {
            entry = do_page_walk(address, false, user, false, false, false)?;
        }
        if fetch_violates_nx_or_smep(entry as i32) {
            return Err(());
        }
        Ok(phys_of_tlb_entry(entry, address))
    }
}

unsafe fn jit_run_interpreted(mut phys_addr: u64) {
    profiler::stat_increment(stat::RUN_INTERPRETED);
    dbg_assert!(!memory::in_mapped_range(phys_addr));

    jit_block_boundary = false;
    let mut i = 0;

    loop {
        if CHECK_MISSED_ENTRY_POINTS {
            let entry = jit::jit_find_cache_entry(phys_addr, *state_flags);
            if entry != jit::CachedCode::NONE {
                profiler::stat_increment(
                    stat::RUN_INTERPRETED_MISSED_COMPILED_ENTRY_RUN_INTERPRETED,
                );
            }
        }

        i += 1;
        let start_eip = *instruction_pointer;

        let debug_mode = debug_mode_active();
        if debug_mode {
            if *flags & FLAG_RF != 0 {
                // RF suppresses instruction-address breakpoints for exactly
                // one instruction; it is cleared once that instruction ran
                *flags &= !FLAG_RF;
            }
            else if debug_bp_armed & 1 != 0 {
                let linear_ip = if *is_64 {
                    start_eip
                }
                else {
                    (get_seg_cs() as u32 as u64).wrapping_add(start_eip) & 0xFFFF_FFFF
                };
                let dr7 = *dreg.offset(7);
                let mut n = 0;
                let mut matched = 0;
                while n < 4 {
                    if dr7 >> (n * 2) & 3 != 0
                        && dr7 >> (16 + n * 4) & 3 == 0
                        && *dreg.offset(n) as u32 as u64 == linear_ip
                    {
                        matched |= 1 << n;
                    }
                    n += 1;
                }
                if matched != 0 {
                    // instruction-address breakpoint: fault-class #DB, the
                    // saved rip points at the faulting instruction
                    *previous_ip = start_eip;
                    *instruction_pointer = start_eip;
                    deliver_db(matched, false, true);
                    break;
                }
            }
        }

        let opcode = memory::read8_no_mmap_check(phys_addr);
        // TF is sampled *before* the instruction runs: an instruction that
        // sets TF (popf/iret) doesn't trap after itself, and one that clears
        // TF still traps (Intel SDM 17.3.1.4)
        let tf_before_instruction = *flags & FLAG_TRAP != 0;
        if tf_before_instruction
            && opcode == 0x0F
            && phys_addr & 0xFFF != 0xFFF
            && matches!(memory::read8_no_mmap_check(phys_addr + 1), 0x05 | 0x07)
        {
            // syscall/sysret are single-stepped *preemptively*: the #DB is
            // delivered instead of executing the instruction, in the
            // pre-syscall context (the saved rip points past it). This is the
            // Intel behaviour the kvm-unit-tests syscall test checks (the
            // #DB frame must show the user CS, not the kernel's).
            *instruction_pointer += 2;
            deliver_db(0, true, false);
            break;
        }
        *instruction_pointer += 1;
        dbg_assert!(*prefixes == 0);
        if *is_64 && opcode & 0xF0 == 0x40 {
            // REX prefix
            *prefixes = prefix::PREFIX_REX_PRESENT | ((opcode as u16 & 0xF) << 8);
            run_prefix_instruction();
            *prefixes = 0;
        }
        else if *is_64 {
            // in 64-bit mode the default operand size tier is 32 bit
            if gen::interpreter::is_default_64_operand_size(opcode as u32) {
                run_instruction(opcode | 0x200);
            }
            else {
                run_instruction(opcode | 0x100);
            }
        }
        else {
            run_instruction(opcode | (*is_32 as i32) << 8);
        }
        dbg_assert!(*prefixes == 0);

        if debug_mode {
            // trap-class debug events are delivered after the instruction:
            // data-watchpoint hits recorded by the safe_read/write family,
            // and single-stepping when TF was set at instruction start
            // (syscall/sysret instead key on TF at the *end* of the
            // instruction, per Intel behaviour tested by kvm-unit-tests)
            let tf_trap = tf_before_instruction;
            let pending = db_pending_data;
            if pending != 0 || tf_trap {
                deliver_db(pending, tf_trap, false);
                break;
            }
        }

        if jit_block_boundary
            || (start_eip ^ *instruction_pointer) & !0xFFF != 0
                // Limit the number of iterations, as jumps within the same page are not counted as
                // block boundaries for the interpreter, but only on the next backwards jump
            || (i >= INTERPRETER_ITERATION_LIMIT && start_eip >= *instruction_pointer)
        {
            break;
        }

        *previous_ip = *instruction_pointer;
        phys_addr = return_on_pagefault!(get_phys_eip());
    }

    if cfg!(debug_assertions) {
        debug_last_jump = LastJump::Interpreted { phys_addr };
    }

    *instruction_counter += i;
}

/// Execute one instruction with the same dispatch rules as the interpreter
/// loop. Used for the interrupt shadow after STI: pending IRQs are serviced
/// only after the instruction following STI completes.
pub unsafe fn run_shadow_instruction() {
    let opcode = return_on_pagefault!(read_imm8());
    if *is_64 && opcode & 0xF0 == 0x40 {
        // REX prefix
        *prefixes = prefix::PREFIX_REX_PRESENT | ((opcode as u16 & 0xF) << 8);
        run_prefix_instruction();
        *prefixes = 0;
    }
    else if *is_64 {
        if gen::interpreter::is_default_64_operand_size(opcode as u32) {
            run_instruction(opcode | 0x200);
        }
        else {
            run_instruction(opcode | 0x100);
        }
    }
    else {
        run_instruction(opcode | (*is_32 as i32) << 8);
    }
}

#[no_mangle]
pub fn update_state_flags() {
    unsafe {
        *state_flags = CachedStateFlags::of_u32(
            (*is_32 as u32) << 0
                | (*stack_size_32 as u32) << 1
                | ((*cpl == 3) as u32) << 2
                | (has_flat_segmentation() as u32) << 3
                | (*is_64 as u32) << 4,
        )
    }
}

#[no_mangle]
pub unsafe fn has_flat_segmentation() -> bool {
    // cs/ss can't be null
    return *segment_offsets.offset(SS as isize) == 0
        && !*segment_is_null.offset(DS as isize)
        && *segment_offsets.offset(DS as isize) == 0
        && *segment_offsets.offset(CS as isize) == 0;
}

pub unsafe fn run_prefix_instruction() {
    let opcode = return_on_pagefault!(read_imm8());
    if *is_64 && opcode & 0xF0 == 0x40 {
        // REX prefix (0x40-0x4F in 64-bit mode); only the last REX before the
        // opcode takes effect, a legacy prefix after a REX annuls it
        *prefixes = *prefixes & !prefix::PREFIX_MASK_REX
            | prefix::PREFIX_REX_PRESENT
            | ((opcode as u16 & 0xF) << 8);
        run_prefix_instruction();
        *prefixes = 0;
        return;
    }
    if *prefixes & prefix::PREFIX_REX_W != 0
        || *is_64 && is_osize_32() && gen::interpreter::is_default_64_operand_size(opcode as u32)
    {
        // REX.W (overrides the 0x66 prefix) or a default-64-bit instruction
        // (push/pop/call/ret/... in long mode)
        run_instruction(opcode | 0x200);
    }
    else {
        run_instruction(opcode | (is_osize_32() as i32) << 8);
    }
}

pub unsafe fn segment_prefix_op(seg: i32) {
    dbg_assert!(seg <= 5 && seg >= 0);
    // a legacy prefix after a REX prefix annuls the REX prefix
    *prefixes =
        *prefixes & !(prefix::PREFIX_MASK_SEGMENT | prefix::PREFIX_MASK_REX) | (seg as u16 + 1);
    run_prefix_instruction();
    *prefixes = 0
}

#[no_mangle]
pub unsafe fn main_loop() -> f64 {
    profiler::stat_increment(stat::MAIN_LOOP);

    let start = js::microtick();

    if *in_hlt {
        if cpu_init_pending {
            // INIT interrupts hlt like any other reset event
            *in_hlt = false;
        }
        else if *flags & FLAG_INTERRUPT != 0 {
            let t = js::run_hardware_timers(*acpi_enabled, start);
            handle_irqs();
            if *in_hlt {
                profiler::stat_increment(stat::MAIN_LOOP_IDLE);
                return t;
            }
        }
        else {
            // dead
            return 100.0;
        }
    }

    loop {
        do_many_cycles_native();

        let now = js::microtick();
        let t = js::run_hardware_timers(*acpi_enabled, now);
        handle_irqs();
        if *in_hlt {
            return t;
        }

        if now - start > TIME_PER_FRAME {
            break;
        }
    }

    return 0.0;
}

/// Compiled 64-bit code may continue directly with the next module (instead of returning to
/// cycle_internal) while the instruction counter is below this value
pub static mut jit_chain_instruction_limit: u32 = 0;

pub unsafe fn do_many_cycles_native() {
    profiler::stat_increment(stat::DO_MANY_CYCLES);
    let initial_instruction_counter = *instruction_counter;
    jit_chain_instruction_limit = initial_instruction_counter.wrapping_add(LOOP_COUNTER as u32);
    in_cpu_execution = true;
    while (*instruction_counter).wrapping_sub(initial_instruction_counter) < LOOP_COUNTER as u32
        && !*in_hlt
    {
        cycle_internal();
    }
    in_cpu_execution = false;
    if irq_check_pending {
        irq_check_pending = false;
        handle_irqs();
    }
}

#[cold]
pub unsafe fn trigger_de() {
    dbg_log!("#de");
    *instruction_pointer = *previous_ip;
    if DEBUG {
        if js::cpu_exception_hook(CPU_EXCEPTION_DE) {
            return;
        }
    }
    call_interrupt_vector(CPU_EXCEPTION_DE, false, None);
}

#[inline(never)]
pub unsafe fn trigger_ud() {
    dbg_log!("#ud at {:x}", *previous_ip);
    dbg_trace();
    *instruction_pointer = *previous_ip;
    if DEBUG {
        if js::cpu_exception_hook(CPU_EXCEPTION_UD) {
            return;
        }
    }
    call_interrupt_vector(CPU_EXCEPTION_UD, false, None);
}

/// For the opcodes that are invalid in 64-bit mode ("i64" in Intel SDM Vol. 2, Table A-2):
/// raise #UD in 64-bit mode and return true, so that the handler returns without running the
/// instruction. Their 64-bit operand size handlers (REX.W, instructions_64.rs) raise #UD
/// unconditionally; the handlers that 64-bit code shares with 16/32-bit code (no prefix, 0x66,
/// REX without W, and the opcodes without an operand size) call this first. Compatibility mode
/// isn't 64-bit mode: there they run as in protected mode
#[inline]
pub unsafe fn trigger_ud_if_64() -> bool {
    if *is_64 {
        trigger_ud();
        return true;
    }
    false
}

#[inline(never)]
pub unsafe fn trigger_nm() {
    dbg_log!("#nm eip={:x}", *previous_ip);
    dbg_trace();
    *instruction_pointer = *previous_ip;
    if DEBUG {
        if js::cpu_exception_hook(CPU_EXCEPTION_NM) {
            return;
        }
    }
    call_interrupt_vector(CPU_EXCEPTION_NM, false, None);
}

#[inline(never)]
pub unsafe fn trigger_gp(code: i32) {
    dbg_log!("#gp code={:x} at ip={:x}", code, *previous_ip);
    *instruction_pointer = *previous_ip;
    if DEBUG {
        if js::cpu_exception_hook(CPU_EXCEPTION_GP) {
            return;
        }
    }
    call_interrupt_vector(CPU_EXCEPTION_GP, false, Some(code));
}

#[cold]
pub unsafe fn virt_boundary_read16(low: u64, high: u64) -> i32 {
    dbg_assert!(low & 0xFFF == 0xFFF);
    dbg_assert!(high & 0xFFF == 0);
    return memory::read8(low) | memory::read8(high) << 8;
}

#[cold]
pub unsafe fn virt_boundary_read32s(low: u64, high: u64) -> i32 {
    dbg_assert!(low & 0xFFF >= 0xFFD);
    dbg_assert!(high - 3 & 0xFFF == low & 0xFFF);
    let mid;
    if 0 != low & 1 {
        if 0 != low & 2 {
            // 0xFFF
            mid = memory::read16(high - 2)
        }
        else {
            // 0xFFD
            mid = memory::read16(low + 1)
        }
    }
    else {
        // 0xFFE
        mid = virt_boundary_read16(low + 1, high - 1)
    }
    return memory::read8(low) | mid << 8 | memory::read8(high) << 24;
}

#[cold]
pub unsafe fn virt_boundary_write16(low: u64, high: u64, value: i32) {
    dbg_assert!(low & 0xFFF == 0xFFF);
    dbg_assert!(high & 0xFFF == 0);
    memory::write8(low, value);
    memory::write8(high, value >> 8);
}

#[cold]
pub unsafe fn virt_boundary_write32(low: u64, high: u64, value: i32) {
    dbg_assert!(low & 0xFFF >= 0xFFD);
    dbg_assert!(high - 3 & 0xFFF == low & 0xFFF);
    memory::write8(low, value);
    if 0 != low & 1 {
        if 0 != low & 2 {
            // 0xFFF
            memory::write8(high - 2, value >> 8);
            memory::write8(high - 1, value >> 16);
        }
        else {
            // 0xFFD
            memory::write8(low + 1, value >> 8);
            memory::write8(low + 2, value >> 16);
        }
    }
    else {
        // 0xFFE
        memory::write8(low + 1, value >> 8);
        memory::write8(high - 1, value >> 16);
    }
    memory::write8(high, value >> 24);
}

pub unsafe fn safe_read8(addr: u64) -> OrPageFault<i32> {
    if debug_bp_armed & 2 != 0 {
        dbg_check_data_wp(addr, 1, false);
    }
    Ok(memory::read8(translate_address_read(addr)?))
}

pub unsafe fn safe_read16(addr: u64) -> OrPageFault<i32> {
    if debug_bp_armed & 2 != 0 {
        dbg_check_data_wp(addr, 2, false);
    }
    if addr & 0xFFF == 0xFFF {
        Ok(safe_read8(addr)? | safe_read8(addr + 1)? << 8)
    }
    else {
        Ok(memory::read16(translate_address_read(addr)?))
    }
}

pub unsafe fn safe_read32s(addr: u64) -> OrPageFault<i32> {
    if debug_bp_armed & 2 != 0 {
        dbg_check_data_wp(addr, 4, false);
    }
    if addr & 0xFFF >= 0xFFD {
        Ok(safe_read16(addr)? | safe_read16(addr + 2)? << 16)
    }
    else {
        Ok(memory::read32s(translate_address_read(addr)?))
    }
}

pub unsafe fn safe_read_f32(addr: u64) -> OrPageFault<f32> {
    Ok(f32::from_bits(i32::cast_unsigned(safe_read32s(addr)?)))
}

pub unsafe fn safe_read64s(addr: u64) -> OrPageFault<u64> {
    if debug_bp_armed & 2 != 0 {
        dbg_check_data_wp(addr, 8, false);
    }
    if addr & 0xFFF > 0x1000 - 8 {
        Ok(safe_read32s(addr)? as u32 as u64 | (safe_read32s(addr + 4)? as u32 as u64) << 32)
    }
    else {
        Ok(memory::read64s(translate_address_read(addr)?) as u64)
    }
}

pub unsafe fn safe_read128s(addr: u64) -> OrPageFault<reg128> {
    if debug_bp_armed & 2 != 0 {
        dbg_check_data_wp(addr, 16, false);
    }
    if addr & 0xFFF > 0x1000 - 16 {
        Ok(reg128 {
            u64: [safe_read64s(addr)?, safe_read64s(addr + 8)?],
        })
    }
    else {
        Ok(memory::read128(translate_address_read(addr)?))
    }
}

#[no_mangle]
#[cfg(feature = "profiler")]
pub fn report_safe_read_jit_slow(address: u32, entry: i32) {
    if entry & TLB_VALID == 0 {
        profiler::stat_increment(stat::SAFE_READ_SLOW_NOT_VALID);
    }
    else if entry & TLB_IN_MAPPED_RANGE != 0 {
        profiler::stat_increment(stat::SAFE_READ_SLOW_IN_MAPPED_RANGE);
    }
    else if entry & TLB_NO_USER != 0 {
        profiler::stat_increment(stat::SAFE_READ_SLOW_NOT_USER);
    }
    else if address & 0xFFF > 0x1000 - 16 {
        profiler::stat_increment(stat::SAFE_READ_SLOW_PAGE_CROSSED);
    }
    else {
        dbg_log!("Unexpected entry bit: {:x} (read at {:x})", entry, address);
        dbg_assert!(false);
    }
}

#[no_mangle]
#[cfg(feature = "profiler")]
pub fn report_safe_write_jit_slow(address: u32, entry: i32) {
    if entry & TLB_VALID == 0 {
        profiler::stat_increment(stat::SAFE_WRITE_SLOW_NOT_VALID);
    }
    else if entry & TLB_IN_MAPPED_RANGE != 0 {
        profiler::stat_increment(stat::SAFE_WRITE_SLOW_IN_MAPPED_RANGE);
    }
    else if entry & TLB_HAS_CODE != 0 {
        profiler::stat_increment(stat::SAFE_WRITE_SLOW_HAS_CODE);
    }
    else if entry & TLB_READONLY != 0 {
        profiler::stat_increment(stat::SAFE_WRITE_SLOW_READ_ONLY);
    }
    else if entry & TLB_NO_USER != 0 {
        profiler::stat_increment(stat::SAFE_WRITE_SLOW_NOT_USER);
    }
    else if address & 0xFFF > 0x1000 - 16 {
        profiler::stat_increment(stat::SAFE_WRITE_SLOW_PAGE_CROSSED);
    }
    else {
        dbg_assert!(false);
    }
}

#[no_mangle]
#[cfg(feature = "profiler")]
pub fn report_safe_read_write_jit_slow(address: u32, entry: i32) {
    if entry & TLB_VALID == 0 {
        profiler::stat_increment(stat::SAFE_READ_WRITE_SLOW_NOT_VALID);
    }
    else if entry & TLB_IN_MAPPED_RANGE != 0 {
        profiler::stat_increment(stat::SAFE_READ_WRITE_SLOW_IN_MAPPED_RANGE);
    }
    else if entry & TLB_HAS_CODE != 0 {
        profiler::stat_increment(stat::SAFE_READ_WRITE_SLOW_HAS_CODE);
    }
    else if entry & TLB_READONLY != 0 {
        profiler::stat_increment(stat::SAFE_READ_WRITE_SLOW_READ_ONLY);
    }
    else if entry & TLB_NO_USER != 0 {
        profiler::stat_increment(stat::SAFE_READ_WRITE_SLOW_NOT_USER);
    }
    else if address & 0xFFF > 0x1000 - 16 {
        profiler::stat_increment(stat::SAFE_READ_WRITE_SLOW_PAGE_CROSSED);
    }
    else {
        dbg_assert!(false);
    }
}

#[cfg(not(feature = "mem64"))]
#[repr(align(0x1000))]
struct ScratchBuffer([u8; 0x1000 * 2]);
#[cfg(not(feature = "mem64"))]
static mut jit_paging_scratch_buffer: ScratchBuffer = ScratchBuffer([0; 2 * 0x1000]);

/// The slow paths of jitted memory accesses return a "pointer" (host_base ^ addr) & !0xFFF,
/// which the fast path xors with the address again to get the location to read from or write
/// to. Under mem64 that value needs 64 bits and the jit function types have no i64 return
/// variant, so it is stashed here and the helpers return 0 (1 still means fault). (Unused in
/// the default build.)
pub static mut jit_slow_path_entry: u64 = 0;

/// Where a jit slow path stages or sinks a value that the fast path then loads from/stores to.
/// Default build: the page-aligned static above. mem64: physical 0xA0000 in the VGA hole —
/// guest RAM that is never accessed as RAM (its reads/writes divert to the VGA device).
#[cfg(not(feature = "mem64"))]
#[inline(always)]
unsafe fn jit_scratch_base() -> u64 { &raw const jit_paging_scratch_buffer.0 as u64 }
#[cfg(feature = "mem64")]
#[inline(always)]
fn jit_scratch_base() -> u64 { 0xA0000 }

#[cfg(not(feature = "mem64"))]
#[inline(always)]
unsafe fn jit_scratch_store8(offset: u64, value: u8) {
    *(jit_scratch_base() as *mut u8).offset(offset as isize) = value
}
#[cfg(feature = "mem64")]
#[inline(always)]
unsafe fn jit_scratch_store8(offset: u64, value: u8) {
    memory::write8_no_mmap_or_dirty_check(jit_scratch_base() + offset, value as i32)
}
#[cfg(not(feature = "mem64"))]
#[inline(always)]
unsafe fn jit_scratch_store16(offset: u64, value: u16) {
    ptr::write_unaligned((jit_scratch_base() + offset) as *mut u16, value)
}
#[cfg(feature = "mem64")]
#[inline(always)]
unsafe fn jit_scratch_store16(offset: u64, value: u16) {
    memory::write16_no_mmap_or_dirty_check(jit_scratch_base() + offset, value as i32)
}
#[cfg(not(feature = "mem64"))]
#[inline(always)]
unsafe fn jit_scratch_store32(offset: u64, value: i32) {
    ptr::write_unaligned((jit_scratch_base() + offset) as *mut i32, value)
}
#[cfg(feature = "mem64")]
#[inline(always)]
unsafe fn jit_scratch_store32(offset: u64, value: i32) {
    memory::write32_no_mmap_or_dirty_check(jit_scratch_base() + offset, value)
}
#[cfg(not(feature = "mem64"))]
#[inline(always)]
unsafe fn jit_scratch_store64(offset: u64, value: i64) {
    ptr::write_unaligned((jit_scratch_base() + offset) as *mut i64, value)
}
#[cfg(feature = "mem64")]
#[inline(always)]
unsafe fn jit_scratch_store64(offset: u64, value: i64) {
    memory::write64_no_mmap_or_dirty_check(jit_scratch_base() + offset, value as u64)
}

/// Hand a slow path's result to the fast path: returned directly in the default build;
/// stashed under mem64 (see jit_slow_path_entry).
#[cfg(not(feature = "mem64"))]
#[inline(always)]
fn jit_slow_result(result: u64) -> i32 { result as i32 }
#[cfg(feature = "mem64")]
#[inline(always)]
fn jit_slow_result(result: u64) -> i32 {
    unsafe { jit_slow_path_entry = result };
    0
}

pub unsafe fn safe_read_slow_jit(
    addr: i32,
    bitsize: i32,
    is_write: bool,
    eip_offset_in_page_and_wasm_table_index: i32,
) -> i32 {
    safe_read_slow_jit_impl(
        addr as u32 as u64,
        bitsize,
        is_write,
        eip_offset_in_page_and_wasm_table_index,
    )
}

/// Slow path of jitted memory reads (and the read part of read-modify-write accesses). Returns
/// (pointer ^ address) & !0xFFF, where pointer is where the fast path reads the value from, or
/// 1 for an exception (which is delivered by exit_jit). Only the low 32 bits of the address
/// matter for the returned value.
pub unsafe fn safe_read_slow_jit_impl(
    addr: u64,
    bitsize: i32,
    is_write: bool,
    eip_offset_in_page_and_wasm_table_index: i32,
) -> i32 {
    let wasm_table_index = (eip_offset_in_page_and_wasm_table_index >> 16) as u16;
    let eip_offset_in_page = eip_offset_in_page_and_wasm_table_index & 0xFFFF;
    dbg_assert!(eip_offset_in_page >= 0 && eip_offset_in_page < 0x1000);
    dbg_assert!(u32::from(wasm_table_index) < jit::WASM_TABLE_SIZE);

    let crosses_page = (addr & 0xFFF) as i32 + bitsize / 8 > 0x1000;
    let addr_low = match if is_write {
        translate_address_write_jit(addr, wasm_table_index)
    }
    else {
        translate_address_read_jit(addr)
    } {
        Err(()) => {
            *instruction_pointer = *instruction_pointer & !0xFFF | eip_offset_in_page as u64;
            return 1;
        },
        Ok(addr) => addr,
    };
    if crosses_page {
        let boundary_addr = (addr | 0xFFF).wrapping_add(1);
        let addr_high = match if is_write {
            translate_address_write_jit(boundary_addr, wasm_table_index)
        }
        else {
            translate_address_read_jit(boundary_addr)
        } {
            Err(()) => {
                *instruction_pointer = *instruction_pointer & !0xFFF | eip_offset_in_page as u64;
                return 1;
            },
            Ok(addr) => addr,
        };
        // TODO: Could check if virtual pages point to consecutive physical and go to fast path
        // do read, write into scratch buffer

        // TODO: Could check if virtual pages point to consecutive physical and go to fast path
        // do read, write into scratch buffer

        dbg_assert!(jit_scratch_base() & 0xFFF == 0);

        for s in addr_low..((addr_low | 0xFFF) + 1) {
            jit_scratch_store8(s & 0xFFF, memory::read8(s) as u8)
        }
        for s in addr_high..(addr_high + ((addr as i32 + bitsize / 8) & 0xFFF) as u64) {
            jit_scratch_store8(0x1000 | s & 0xFFF, memory::read8(s) as u8)
        }

        jit_slow_result((jit_scratch_base() ^ addr) & !0xFFF)
    }
    else if memory::in_mapped_range(addr_low) {
        let off = addr_low & 0xFFF;

        match bitsize {
            128 => {
                let v = memory::read128(addr_low);
                jit_scratch_store64(off, unsafe { v.u64[0] } as i64);
                jit_scratch_store64(off + 8, unsafe { v.u64[1] } as i64);
            },
            64 => jit_scratch_store64(off, memory::read64s(addr_low)),
            32 => jit_scratch_store32(off, memory::read32s(addr_low)),
            16 => jit_scratch_store16(off, memory::read16(addr_low) as u16),
            8 => jit_scratch_store8(off, memory::read8(addr_low) as u8),
            _ => {
                dbg_assert!(false);
            },
        }

        jit_slow_result((jit_scratch_base() ^ addr) & !0xFFF)
    }
    else {
        jit_slow_result((memory::tlb_host_base(addr_low) ^ addr) & !0xFFF)
    }
}

#[no_mangle]
pub unsafe fn safe_read8_slow_jit(addr: i32, eip: i32) -> i32 {
    safe_read_slow_jit(addr, 8, false, eip)
}
#[no_mangle]
pub unsafe fn safe_read16_slow_jit(addr: i32, eip: i32) -> i32 {
    safe_read_slow_jit(addr, 16, false, eip)
}
#[no_mangle]
pub unsafe fn safe_read32s_slow_jit(addr: i32, eip: i32) -> i32 {
    safe_read_slow_jit(addr, 32, false, eip)
}
#[no_mangle]
pub unsafe fn safe_read64s_slow_jit(addr: i32, eip: i32) -> i32 {
    safe_read_slow_jit(addr, 64, false, eip)
}
#[no_mangle]
pub unsafe fn safe_read128s_slow_jit(addr: i32, eip: i32) -> i32 {
    safe_read_slow_jit(addr, 128, false, eip)
}

#[no_mangle]
pub unsafe fn get_phys_eip_slow_jit(addr: i32) -> i32 {
    match translate_address_read_jit(addr as u32 as u64) {
        Err(()) => 1,
        Ok(_) if fetch_violates_nx_or_smep(tlb_pick_entry(addr as u32 as u64) as i32) => {
            // a page switch to a page that can't be executed: leave the compiled code, the
            // interpreter's instruction fetch raises the page fault
            jit_exit_reason = JitExitReason::SelfModifyingCodeBail;
            1
        },
        Ok(addr_low) => {
            dbg_assert!(!memory::in_mapped_range(addr_low)); // same assumption as in read_imm8
            jit_slow_result((memory::tlb_host_base(addr_low) ^ addr as u32 as u64) & !0xFFF)
        },
    }
}

#[no_mangle]
pub unsafe fn safe_read_write8_slow_jit(addr: i32, eip_and_wasm_table_index: i32) -> i32 {
    safe_read_slow_jit(addr, 8, true, eip_and_wasm_table_index)
}
#[no_mangle]
pub unsafe fn safe_read_write16_slow_jit(addr: i32, eip_and_wasm_table_index: i32) -> i32 {
    safe_read_slow_jit(addr, 16, true, eip_and_wasm_table_index)
}
#[no_mangle]
pub unsafe fn safe_read_write32s_slow_jit(addr: i32, eip_and_wasm_table_index: i32) -> i32 {
    safe_read_slow_jit(addr, 32, true, eip_and_wasm_table_index)
}
#[no_mangle]
pub unsafe fn safe_read_write64_slow_jit(addr: i32, eip_and_wasm_table_index: i32) -> i32 {
    safe_read_slow_jit(addr, 64, true, eip_and_wasm_table_index)
}

#[no_mangle]
pub unsafe fn readable_or_pagefault_jit(addr: i32, size: i32, eip_offset_in_page: i32) -> i32 {
    dbg_assert!(size > 0 && size < 0x1000);
    dbg_assert!(eip_offset_in_page >= 0 && eip_offset_in_page < 0x1000);
    let crosses_page = (addr & 0xFFF) + size > 0x1000;
    if translate_address_read_jit(addr as u32 as u64).is_err()
        || crosses_page && translate_address_read_jit(((addr | 0xFFF) + 1) as u32 as u64).is_err()
    {
        *instruction_pointer = *instruction_pointer & !0xFFF | eip_offset_in_page as u64;
        return 1;
    }
    0
}

pub unsafe fn safe_write_slow_jit(
    addr: i32,
    bitsize: i32,
    value_low: u64,
    value_high: u64,
    eip_offset_in_page_and_wasm_table_index: i32,
) -> i32 {
    safe_write_slow_jit_impl(
        addr as u32 as u64,
        bitsize,
        value_low,
        value_high,
        eip_offset_in_page_and_wasm_table_index,
    )
}

/// Slow path of jitted memory writes, see safe_read_slow_jit_impl. Writes that cross a page or
/// go to memory-mapped io are done here, the returned pointer then points to a scratch buffer.
pub unsafe fn safe_write_slow_jit_impl(
    addr: u64,
    bitsize: i32,
    value_low: u64,
    value_high: u64,
    eip_offset_in_page_and_wasm_table_index: i32,
) -> i32 {
    let wasm_table_index = (eip_offset_in_page_and_wasm_table_index >> 16) as u16;
    let eip_offset_in_page = eip_offset_in_page_and_wasm_table_index & 0xFFFF;
    dbg_assert!(eip_offset_in_page >= 0 && eip_offset_in_page < 0x1000);
    dbg_assert!(u32::from(wasm_table_index) < jit::WASM_TABLE_SIZE);

    let crosses_page = (addr & 0xFFF) as i32 + bitsize / 8 > 0x1000;
    let addr_low = match translate_address_write_jit(addr, wasm_table_index) {
        Err(()) => {
            *instruction_pointer = *instruction_pointer & !0xFFF | eip_offset_in_page as u64;
            return 1;
        },
        Ok(x) => x,
    };
    if crosses_page {
        let addr_high =
            match translate_address_write_jit((addr | 0xFFF).wrapping_add(1), wasm_table_index) {
                Err(()) => {
                    *instruction_pointer =
                        *instruction_pointer & !0xFFF | eip_offset_in_page as u64;
                    return 1;
                },
                Ok(x) => x,
            };
        // TODO: Could check if virtual pages point to consecutive physical and go to fast path

        // do write, return dummy pointer for fast path to write into

        match bitsize {
            128 => safe_write128(
                addr,
                reg128 {
                    u64: [value_low, value_high],
                },
            )
            .unwrap(),
            64 => safe_write64(addr, value_low).unwrap(),
            32 => virt_boundary_write32(addr_low, addr_high | (addr + 3 & 3), value_low as i32),
            16 => virt_boundary_write16(addr_low, addr_high, value_low as i32),
            8 => {
                dbg_assert!(false);
            },
            _ => {
                dbg_assert!(false);
            },
        }

        dbg_assert!(jit_scratch_base() & 0xFFF == 0);
        jit_slow_result((jit_scratch_base() ^ addr) & !0xFFF)
    }
    else if memory::in_mapped_range(addr_low) {
        match bitsize {
            128 => memory::mmap_write128(addr_low, value_low, value_high),
            64 => memory::mmap_write64(addr_low, value_low),
            32 => memory::mmap_write32(addr_low, value_low as i32),
            16 => memory::mmap_write16(addr_low, (value_low & 0xFFFF) as i32),
            8 => memory::mmap_write8(addr_low, (value_low & 0xFF) as i32),
            _ => {
                dbg_assert!(false);
            },
        }

        jit_slow_result((jit_scratch_base() ^ addr) & !0xFFF)
    }
    else {
        jit_slow_result((memory::tlb_host_base(addr_low) ^ addr) & !0xFFF)
    }
}

// 64-bit address variants of the slow paths, for the 64-bit jit
#[no_mangle]
pub unsafe fn safe_read8_slow_jit64(addr: u64, eip: i32) -> i32 {
    safe_read_slow_jit_impl(addr, 8, false, eip)
}
#[no_mangle]
pub unsafe fn safe_read16_slow_jit64(addr: u64, eip: i32) -> i32 {
    safe_read_slow_jit_impl(addr, 16, false, eip)
}
#[no_mangle]
pub unsafe fn safe_read32s_slow_jit64(addr: u64, eip: i32) -> i32 {
    safe_read_slow_jit_impl(addr, 32, false, eip)
}
#[no_mangle]
pub unsafe fn safe_read64s_slow_jit64(addr: u64, eip: i32) -> i32 {
    safe_read_slow_jit_impl(addr, 64, false, eip)
}
#[no_mangle]
pub unsafe fn safe_read_write8_slow_jit64(addr: u64, eip_and_wasm_table_index: i32) -> i32 {
    safe_read_slow_jit_impl(addr, 8, true, eip_and_wasm_table_index)
}
#[no_mangle]
pub unsafe fn safe_read_write16_slow_jit64(addr: u64, eip_and_wasm_table_index: i32) -> i32 {
    safe_read_slow_jit_impl(addr, 16, true, eip_and_wasm_table_index)
}
#[no_mangle]
pub unsafe fn safe_read_write32s_slow_jit64(addr: u64, eip_and_wasm_table_index: i32) -> i32 {
    safe_read_slow_jit_impl(addr, 32, true, eip_and_wasm_table_index)
}
#[no_mangle]
pub unsafe fn safe_read_write64_slow_jit64(addr: u64, eip_and_wasm_table_index: i32) -> i32 {
    safe_read_slow_jit_impl(addr, 64, true, eip_and_wasm_table_index)
}
#[no_mangle]
pub unsafe fn safe_write8_slow_jit64(addr: u64, value: i32, eip_and_wasm_table_index: i32) -> i32 {
    safe_write_slow_jit_impl(addr, 8, value as u32 as u64, 0, eip_and_wasm_table_index)
}
#[no_mangle]
pub unsafe fn safe_write16_slow_jit64(addr: u64, value: i32, eip_and_wasm_table_index: i32) -> i32 {
    safe_write_slow_jit_impl(addr, 16, value as u32 as u64, 0, eip_and_wasm_table_index)
}
#[no_mangle]
pub unsafe fn safe_write32_slow_jit64(addr: u64, value: i32, eip_and_wasm_table_index: i32) -> i32 {
    safe_write_slow_jit_impl(addr, 32, value as u32 as u64, 0, eip_and_wasm_table_index)
}
#[no_mangle]
pub unsafe fn safe_write64_slow_jit64(addr: u64, value: u64, eip_and_wasm_table_index: i32) -> i32 {
    safe_write_slow_jit_impl(addr, 64, value, 0, eip_and_wasm_table_index)
}

#[no_mangle]
pub unsafe fn safe_write8_slow_jit(addr: i32, value: u32, eip_and_wasm_table_index: i32) -> i32 {
    safe_write_slow_jit(addr, 8, value as u64, 0, eip_and_wasm_table_index)
}
#[no_mangle]
pub unsafe fn safe_write16_slow_jit(addr: i32, value: u32, eip_and_wasm_table_index: i32) -> i32 {
    safe_write_slow_jit(addr, 16, value as u64, 0, eip_and_wasm_table_index)
}
#[no_mangle]
pub unsafe fn safe_write32_slow_jit(addr: i32, value: u32, eip_and_wasm_table_index: i32) -> i32 {
    safe_write_slow_jit(addr, 32, value as u64, 0, eip_and_wasm_table_index)
}
#[no_mangle]
pub unsafe fn safe_write64_slow_jit(addr: i32, value: u64, eip_and_wasm_table_index: i32) -> i32 {
    safe_write_slow_jit(addr, 64, value, 0, eip_and_wasm_table_index)
}
#[no_mangle]
pub unsafe fn safe_write128_slow_jit(
    addr: i32,
    low: u64,
    high: u64,
    eip_and_wasm_table_index: i32,
) -> i32 {
    safe_write_slow_jit(addr, 128, low, high, eip_and_wasm_table_index)
}

#[no_mangle]
pub unsafe fn writable_or_pagefault_jit(
    addr: i32,
    size: i32,
    eip_offset_in_page_and_wasm_table_index: i32,
) -> i32 {
    let wasm_table_index = (eip_offset_in_page_and_wasm_table_index >> 16) as u16;
    let eip_offset_in_page = eip_offset_in_page_and_wasm_table_index & 0xFFFF;
    dbg_assert!(size > 0 && size < 0x1000);
    dbg_assert!(eip_offset_in_page >= 0 && eip_offset_in_page < 0x1000);
    dbg_assert!(u32::from(wasm_table_index) < jit::WASM_TABLE_SIZE);
    let crosses_page = (addr & 0xFFF) + size > 0x1000;
    if translate_address_write_jit(addr as u32 as u64, wasm_table_index).is_err()
        || crosses_page
            && translate_address_write_jit(((addr | 0xFFF) + 1) as u32 as u64, wasm_table_index)
                .is_err()
    {
        *instruction_pointer = *instruction_pointer & !0xFFF | eip_offset_in_page as u64;
        return 1;
    }
    0
}

pub unsafe fn safe_write8(addr: u64, value: i32) -> OrPageFault<()> {
    if debug_bp_armed & 2 != 0 {
        dbg_check_data_wp(addr, 1, true);
    }
    let (phys_addr, can_skip_dirty_page) = translate_address_write_and_can_skip_dirty(addr)?;
    if memory::in_mapped_range(phys_addr) {
        memory::mmap_write8(phys_addr, value);
    }
    else {
        if !can_skip_dirty_page {
            jit::jit_dirty_page(Page::page_of(phys_addr));
        }
        else {
            dbg_assert!(!jit::jit_page_has_code(Page::page_of(phys_addr)));
        }
        memory::write8_no_mmap_or_dirty_check(phys_addr, value);
    };
    Ok(())
}

pub unsafe fn safe_write16(addr: u64, value: i32) -> OrPageFault<()> {
    if debug_bp_armed & 2 != 0 {
        dbg_check_data_wp(addr, 2, true);
    }
    let (phys_addr, can_skip_dirty_page) = translate_address_write_and_can_skip_dirty(addr)?;
    dbg_assert!(value >= 0 && value < 0x10000);
    if addr & 0xFFF == 0xFFF {
        virt_boundary_write16(phys_addr, translate_address_write(addr + 1)?, value);
    }
    else if memory::in_mapped_range(phys_addr) {
        memory::mmap_write16(phys_addr, value);
    }
    else {
        if !can_skip_dirty_page {
            jit::jit_dirty_page(Page::page_of(phys_addr));
        }
        else {
            dbg_assert!(!jit::jit_page_has_code(Page::page_of(phys_addr)));
        }
        memory::write16_no_mmap_or_dirty_check(phys_addr, value);
    };
    Ok(())
}

pub unsafe fn safe_write32(addr: u64, value: i32) -> OrPageFault<()> {
    if debug_bp_armed & 2 != 0 {
        dbg_check_data_wp(addr, 4, true);
    }
    let (phys_addr, can_skip_dirty_page) = translate_address_write_and_can_skip_dirty(addr)?;
    if addr & 0xFFF > 0x1000 - 4 {
        virt_boundary_write32(
            phys_addr,
            translate_address_write(addr + 3 & !3)? | (addr + 3 & 3),
            value,
        );
    }
    else if memory::in_mapped_range(phys_addr) {
        memory::mmap_write32(phys_addr, value);
    }
    else {
        if !can_skip_dirty_page {
            jit::jit_dirty_page(Page::page_of(phys_addr));
        }
        else {
            dbg_assert!(!jit::jit_page_has_code(Page::page_of(phys_addr)));
        }
        memory::write32_no_mmap_or_dirty_check(phys_addr, value);
    };
    Ok(())
}

pub unsafe fn safe_write64(addr: u64, value: u64) -> OrPageFault<()> {
    if debug_bp_armed & 2 != 0 {
        dbg_check_data_wp(addr, 8, true);
    }
    if addr & 0xFFF > 0x1000 - 8 {
        writable_or_pagefault(addr, 8)?;
        safe_write32(addr, value as i32).unwrap();
        safe_write32(addr + 4, (value >> 32) as i32).unwrap();
    }
    else {
        let (phys_addr, can_skip_dirty_page) = translate_address_write_and_can_skip_dirty(addr)?;
        if memory::in_mapped_range(phys_addr) {
            memory::mmap_write64(phys_addr, value);
        }
        else {
            if !can_skip_dirty_page {
                jit::jit_dirty_page(Page::page_of(phys_addr));
            }
            else {
                dbg_assert!(!jit::jit_page_has_code(Page::page_of(phys_addr)));
            }
            memory::write64_no_mmap_or_dirty_check(phys_addr, value);
        }
    };
    Ok(())
}

pub unsafe fn safe_write128(addr: u64, value: reg128) -> OrPageFault<()> {
    if debug_bp_armed & 2 != 0 {
        dbg_check_data_wp(addr, 16, true);
    }
    if addr & 0xFFF > 0x1000 - 16 {
        writable_or_pagefault(addr, 16)?;
        safe_write64(addr, value.u64[0]).unwrap();
        safe_write64(addr + 8, value.u64[1]).unwrap();
    }
    else {
        let (phys_addr, can_skip_dirty_page) = translate_address_write_and_can_skip_dirty(addr)?;
        if memory::in_mapped_range(phys_addr) {
            memory::mmap_write128(phys_addr, value.u64[0], value.u64[1]);
        }
        else {
            if !can_skip_dirty_page {
                jit::jit_dirty_page(Page::page_of(phys_addr));
            }
            else {
                dbg_assert!(!jit::jit_page_has_code(Page::page_of(phys_addr)));
            }
            memory::write128_no_mmap_or_dirty_check(phys_addr, value);
        }
    };
    Ok(())
}

#[inline(always)]
pub unsafe fn safe_read_write8(addr: u64, instruction: &dyn Fn(i32) -> i32) {
    if debug_bp_armed & 2 != 0 {
        dbg_check_data_wp(addr, 1, true);
    }
    let (phys_addr, can_skip_dirty_page) =
        return_on_pagefault!(translate_address_write_and_can_skip_dirty(addr));
    let x = memory::read8(phys_addr);
    let value = instruction(x);
    dbg_assert!(value >= 0 && value < 0x100);
    if memory::in_mapped_range(phys_addr) {
        memory::mmap_write8(phys_addr, value);
    }
    else {
        if !can_skip_dirty_page {
            jit::jit_dirty_page(Page::page_of(phys_addr));
        }
        else {
            dbg_assert!(!jit::jit_page_has_code(Page::page_of(phys_addr)));
        }
        memory::write8_no_mmap_or_dirty_check(phys_addr, value);
    }
}

#[inline(always)]
pub unsafe fn safe_read_write16(addr: u64, instruction: &dyn Fn(i32) -> i32) {
    if debug_bp_armed & 2 != 0 {
        dbg_check_data_wp(addr, 2, true);
    }
    let (phys_addr, can_skip_dirty_page) =
        return_on_pagefault!(translate_address_write_and_can_skip_dirty(addr));
    if phys_addr & 0xFFF == 0xFFF {
        let phys_addr_high = return_on_pagefault!(translate_address_write(addr + 1));
        let x = virt_boundary_read16(phys_addr, phys_addr_high);
        virt_boundary_write16(phys_addr, phys_addr_high, instruction(x));
    }
    else {
        let x = memory::read16(phys_addr);
        let value = instruction(x);
        dbg_assert!(value >= 0 && value < 0x10000);
        if memory::in_mapped_range(phys_addr) {
            memory::mmap_write16(phys_addr, value);
        }
        else {
            if !can_skip_dirty_page {
                jit::jit_dirty_page(Page::page_of(phys_addr));
            }
            else {
                dbg_assert!(!jit::jit_page_has_code(Page::page_of(phys_addr)));
            }
            memory::write16_no_mmap_or_dirty_check(phys_addr, value);
        };
    }
}

#[inline(always)]
pub unsafe fn safe_read_write32(addr: u64, instruction: &dyn Fn(i32) -> i32) {
    if debug_bp_armed & 2 != 0 {
        dbg_check_data_wp(addr, 4, true);
    }
    let (phys_addr, can_skip_dirty_page) =
        return_on_pagefault!(translate_address_write_and_can_skip_dirty(addr));
    if phys_addr & 0xFFF >= 0xFFD {
        let phys_addr_high = return_on_pagefault!(translate_address_write(addr + 3 & !3));
        let phys_addr_high = phys_addr_high | addr + 3 & 3;
        let x = virt_boundary_read32s(phys_addr, phys_addr_high);
        virt_boundary_write32(phys_addr, phys_addr_high, instruction(x));
    }
    else {
        let x = memory::read32s(phys_addr);
        let value = instruction(x);
        if memory::in_mapped_range(phys_addr) {
            memory::mmap_write32(phys_addr, value);
        }
        else {
            if !can_skip_dirty_page {
                jit::jit_dirty_page(Page::page_of(phys_addr));
            }
            else {
                dbg_assert!(!jit::jit_page_has_code(Page::page_of(phys_addr)));
            }
            memory::write32_no_mmap_or_dirty_check(phys_addr, value);
        };
    }
}

#[inline(always)]
pub unsafe fn safe_read_write64(addr: u64, instruction: &dyn Fn(u64) -> u64) {
    if debug_bp_armed & 2 != 0 {
        dbg_check_data_wp(addr, 8, true);
    }
    let (phys_addr, can_skip_dirty_page) =
        return_on_pagefault!(translate_address_write_and_can_skip_dirty(addr));
    if phys_addr & 0xFFF > 0x1000 - 8 {
        // crosses a page boundary (cold). Both pages must be writable before the
        // instruction runs: it can change registers and flags, and a fault on the
        // second page must leave them untouched so the restart sees the same state
        return_on_pagefault!(writable_or_pagefault(addr, 8));
        let x = return_on_pagefault!(safe_read64s(addr));
        let value = instruction(x);
        return_on_pagefault!(safe_write64(addr, value));
    }
    else {
        let x = memory::read64s(phys_addr) as u64;
        let value = instruction(x);
        if memory::in_mapped_range(phys_addr) {
            memory::mmap_write64(phys_addr, value);
        }
        else {
            if !can_skip_dirty_page {
                jit::jit_dirty_page(Page::page_of(phys_addr));
            }
            else {
                dbg_assert!(!jit::jit_page_has_code(Page::page_of(phys_addr)));
            }
            memory::write64_no_mmap_or_dirty_check(phys_addr, value);
        };
    }
}

// REX extension bits of the current instruction's prefixes (64-bit mode only):
// return 8 when the corresponding bit is set, 0 otherwise
#[inline]
pub unsafe fn rex_r() -> i32 { (*prefixes & prefix::PREFIX_REX_R) as i32 >> 7 }
#[inline]
pub unsafe fn rex_x() -> i32 { (*prefixes & prefix::PREFIX_REX_X) as i32 >> 6 }
#[inline]
pub unsafe fn rex_b() -> i32 { (*prefixes & prefix::PREFIX_REX_B) as i32 >> 5 }
#[inline]
pub unsafe fn rex_w() -> bool { *prefixes & prefix::PREFIX_REX_W != 0 }

#[inline]
fn get_reg8_index(index: i32) -> i32 { return index << 3 & 24 | index >> 2 & 1; }

#[inline]
pub unsafe fn read_reg8(index: i32) -> i32 {
    dbg_assert!(index >= 0 && index < 16);
    if *prefixes & prefix::PREFIX_REX_PRESENT != 0 {
        // with a REX prefix, indices 4-15 encode the low bytes of rsp-r15
        // (ah/ch/dh/bh are not encodable)
        return *reg8.offset((index << 3) as isize) as i32;
    }
    dbg_assert!(index < 8);
    return *reg8.offset(get_reg8_index(index) as isize) as i32;
}

#[inline]
pub unsafe fn write_reg8(index: i32, value: i32) {
    dbg_assert!(index >= 0 && index < 16);
    if *prefixes & prefix::PREFIX_REX_PRESENT != 0 {
        *reg8.offset((index << 3) as isize) = value as u8;
        return;
    }
    dbg_assert!(index < 8);
    *reg8.offset(get_reg8_index(index) as isize) = value as u8;
}

#[inline]
fn get_reg16_index(index: i32) -> i32 { return index << 2; }

#[inline]
pub unsafe fn read_reg16(index: i32) -> i32 {
    dbg_assert!(index >= 0 && index < 16);
    return *reg16.offset(get_reg16_index(index) as isize) as i32;
}

#[inline]
pub unsafe fn write_reg16(index: i32, value: i32) {
    dbg_assert!(index >= 0 && index < 16);
    *reg16.offset(get_reg16_index(index) as isize) = value as u16;
}

// 8-bit access to the legacy high byte registers (AH/CH/DH/BH) by
// instructions that hardcode them (sahf, div8, ...). Unlike
// read_reg8/write_reg8 these are NOT remapped by a REX prefix.
#[inline]
pub unsafe fn read_reg8_legacy_high(index: i32) -> i32 {
    dbg_assert!(index >= 4 && index < 8);
    *reg8.offset(((index - 4) * 8 + 1) as isize) as i32
}
#[inline]
pub unsafe fn write_reg8_legacy_high(index: i32, value: i32) {
    dbg_assert!(index >= 4 && index < 8);
    *reg8.offset(((index - 4) * 8 + 1) as isize) = value as u8;
}

#[inline]
pub unsafe fn read_reg32(index: i32) -> i32 {
    dbg_assert!(index >= 0 && index < 16);
    *reg32.offset((index << 1) as isize)
}

#[inline]
pub unsafe fn write_reg32(index: i32, value: i32) {
    dbg_assert!(index >= 0 && index < 16);
    // 32-bit writes zero-extend into the full 64-bit register
    *reg64.offset(index as isize) = value as u32 as u64;
}

#[inline]
pub unsafe fn read_reg64(index: i32) -> u64 {
    dbg_assert!(index >= 0 && index < 16);
    *reg64.offset(index as isize)
}

#[inline]
pub unsafe fn write_reg64(index: i32, value: u64) {
    dbg_assert!(index >= 0 && index < 16);
    *reg64.offset(index as isize) = value;
}

// MMX register accessors: REX.R/REX.B do not extend MMX register fields, so
// the index is masked to 3 bits (the generated modrm decode has REX folded in)
#[inline]
pub unsafe fn read_mmx32s(r: i32) -> i32 { (*fpu_st.offset((r & 7) as isize)).mantissa as i32 }

#[inline]
pub unsafe fn read_mmx64s(r: i32) -> u64 { (*fpu_st.offset((r & 7) as isize)).mantissa }

#[inline]
pub unsafe fn write_mmx_reg64(r: i32, data: u64) {
    let r = r & 7;
    *fpu_st.offset(r as isize) = softfloat::F80 {
        mantissa: data,
        sign_exponent: 0xFFFF,
    };
}

// xmm register accessor: xmm0-7 live at reg_xmm, xmm8-15 (64-bit mode, REX
// extension) at reg_xmm_high
#[inline]
pub unsafe fn reg_xmm_ptr(r: i32) -> *mut reg128 {
    dbg_assert!(r >= 0 && r < 16);
    if r < 8 {
        reg_xmm.offset(r as isize)
    }
    else {
        reg_xmm_high.offset(r as isize - 8)
    }
}

#[inline]
pub unsafe fn read_xmm_f32(r: i32) -> f32 { return (*reg_xmm_ptr(r)).f32[0]; }

#[inline]
pub unsafe fn read_xmm32(r: i32) -> i32 { return (*reg_xmm_ptr(r)).u32[0] as i32; }

#[inline]
pub unsafe fn read_xmm64s(r: i32) -> u64 { (*reg_xmm_ptr(r)).u64[0] }

#[inline]
pub unsafe fn read_xmm128s(r: i32) -> reg128 { return *reg_xmm_ptr(r); }

#[inline]
pub unsafe fn write_xmm_f32(r: i32, data: f32) { (*reg_xmm_ptr(r)).f32[0] = data; }

#[inline]
pub unsafe fn write_xmm32(r: i32, data: i32) { (*reg_xmm_ptr(r)).i32[0] = data; }

#[inline]
pub unsafe fn write_xmm64(r: i32, data: u64) { (*reg_xmm_ptr(r)).u64[0] = data }
#[inline]
pub unsafe fn write_xmm_f64(r: i32, data: f64) { (*reg_xmm_ptr(r)).f64[0] = data }

#[inline]
pub unsafe fn write_xmm128(r: i32, i0: i32, i1: i32, i2: i32, i3: i32) {
    let x = reg128 {
        u32: [i0 as u32, i1 as u32, i2 as u32, i3 as u32],
    };
    *reg_xmm_ptr(r) = x;
}

#[inline]
pub unsafe fn write_xmm128_2(r: i32, i0: u64, i1: u64) {
    *reg_xmm_ptr(r) = reg128 { u64: [i0, i1] };
}

#[inline]
pub unsafe fn write_xmm_reg128(r: i32, data: reg128) { *reg_xmm_ptr(r) = data; }

/// Set the fpu tag word to valid and the top-of-stack to 0 on mmx instructions
#[no_mangle]
pub fn transition_fpu_to_mmx() {
    unsafe {
        fpu_set_tag_word(0);
        *fpu_stack_ptr = 0;
    }
}

pub unsafe fn task_switch_test() -> bool {
    if 0 != *cr & (CR0_EM | CR0_TS) {
        trigger_nm();
        return false;
    }
    else {
        return true;
    };
}

pub unsafe fn set_mxcsr(new_mxcsr: i32) {
    dbg_assert!(new_mxcsr & !MXCSR_MASK == 0); // checked by caller

    if *mxcsr & MXCSR_DAZ == 0 && new_mxcsr & MXCSR_DAZ != 0 {
        dbg_log!("Warning: Unimplemented MXCSR bit: Denormals Are Zero");
    }
    if *mxcsr & MXCSR_FZ == 0 && new_mxcsr & MXCSR_FZ != 0 {
        dbg_log!("Warning: Unimplemented MXCSR bit: Flush To Zero");
    }

    let rounding_mode = new_mxcsr >> MXCSR_RC_SHIFT & 3;
    if *mxcsr >> MXCSR_RC_SHIFT & 3 == 0 && rounding_mode != 0 {
        dbg_log!(
            "Warning: Unimplemented MXCSR rounding mode: {}",
            rounding_mode
        );
    }

    let exception_mask = new_mxcsr >> 7 & 0b111111;
    if *mxcsr >> 7 & 0b111111 != exception_mask && exception_mask != 0b111111 {
        dbg_log!(
            "Warning: Unimplemented MXCSR exception mask: 0b{:b}",
            exception_mask
        );
    }

    *mxcsr = new_mxcsr;
}

#[no_mangle]
pub unsafe fn task_switch_test_jit(eip_offset_in_page: i32) {
    dbg_assert!(0 != *cr & (CR0_EM | CR0_TS));
    dbg_assert!(eip_offset_in_page >= 0 && eip_offset_in_page < 0x1000);
    trigger_nm_jit(eip_offset_in_page);
}

pub unsafe fn task_switch_test_mmx() -> bool {
    if *cr.offset(4) & CR4_OSFXSR == 0 {
        dbg_log!("Warning: Unimplemented task switch test with cr4.osfxsr=0");
    }
    if 0 != *cr & CR0_EM {
        trigger_ud();
        return false;
    }
    else if 0 != *cr & CR0_TS {
        trigger_nm();
        return false;
    }
    else {
        return true;
    };
}

#[no_mangle]
pub unsafe fn task_switch_test_mmx_jit(eip_offset_in_page: i32) {
    dbg_assert!(eip_offset_in_page >= 0 && eip_offset_in_page < 0x1000);
    if *cr.offset(4) & CR4_OSFXSR == 0 {
        dbg_log!("Warning: Unimplemented task switch test with cr4.osfxsr=0");
    }
    if 0 != *cr & CR0_EM {
        trigger_ud_jit(eip_offset_in_page);
    }
    else if 0 != *cr & CR0_TS {
        trigger_nm_jit(eip_offset_in_page);
    }
    else {
        dbg_assert!(false);
    }
}

pub unsafe fn read_moffs() -> OrPageFault<u64> {
    // In 64-bit mode the moffs immediate is 8 bytes (4 with 0x67); otherwise 2
    // or 4 bytes, depending on the address size attribute
    if *is_64 {
        if is_asize_32() {
            Ok(read_imm32s()? as u32 as u64)
        }
        else {
            read_imm64s()
        }
    }
    else if is_asize_32() {
        Ok(read_imm32s()? as u32 as u64)
    }
    else {
        Ok(read_imm16()? as u32 as u64)
    }
}

#[no_mangle]
pub unsafe fn get_real_eip() -> i32 {
    // Returns the 'real' instruction pointer, without segment offset
    return (*instruction_pointer).wrapping_sub(get_seg_cs() as u64) as u32 as i32;
}

pub unsafe fn get_real_eip64() -> u64 {
    // Returns the 'real' instruction pointer, without segment offset
    return (*instruction_pointer).wrapping_sub(get_seg_cs() as u64);
}

pub unsafe fn get_stack_reg() -> i32 {
    if *stack_size_32 {
        return read_reg32(ESP);
    }
    else {
        return read_reg16(SP);
    };
}

pub unsafe fn set_stack_reg(value: i32) {
    if *stack_size_32 {
        write_reg32(ESP, value)
    }
    else {
        write_reg16(SP, value)
    };
}

pub unsafe fn get_reg_asize(reg: i32) -> u64 {
    dbg_assert!(reg == ECX || reg == ESI || reg == EDI);
    if *is_64 && *prefixes & prefix::PREFIX_MASK_ADDRSIZE == 0 {
        // 64-bit address size in long mode
        return read_reg64(reg);
    }
    let r = read_reg32(reg);
    (if is_asize_32() { r } else { r & 0xFFFF }) as u32 as u64
}

pub unsafe fn set_reg_asize(is_asize_32: bool, reg: i32, value: u64) {
    dbg_assert!(reg == ECX || reg == ESI || reg == EDI);
    if *is_64 && !is_asize_32 {
        // default 64-bit address size in long mode
        write_reg64(reg, value)
    }
    else if is_asize_32 {
        write_reg32(reg, value as u32 as i32)
    }
    else {
        write_reg16(reg, value as u32 as i32)
    };
}

pub unsafe fn decr_ecx_asize(is_asize_32: bool) -> u64 {
    if *is_64 && *prefixes & prefix::PREFIX_MASK_ADDRSIZE == 0 {
        // 64-bit address size in long mode: loop/jrcxz use rcx
        let c = read_reg64(ECX).wrapping_sub(1);
        write_reg64(ECX, c);
        return c;
    }
    if is_asize_32 {
        write_reg32(ECX, read_reg32(ECX).wrapping_sub(1));
        read_reg32(ECX) as u32 as u64
    }
    else {
        write_reg16(CX, read_reg16(CX) - 1);
        read_reg16(CX) as u32 as u64
    }
}

#[no_mangle]
pub unsafe fn set_tsc(low: u32, high: u32) {
    let new_value = low as u64 | (high as u64) << 32;
    let current_value = read_tsc();
    tsc_offset = current_value - new_value;
}

#[no_mangle]
pub unsafe fn read_tsc() -> u64 {
    let value = (js::microtick() * TSC_RATE) as u64 - tsc_offset;

    if !TSC_ENABLE_IMPRECISE_BROWSER_WORKAROUND {
        return value;
    }

    if value == tsc_last_value {
        // If the browser returns the same value as last time, extrapolate based on the number of
        // rdtsc calls between the last two changes
        tsc_number_of_same_readings += 1;
        let extra = (tsc_number_of_same_readings * tsc_resolution) / tsc_speed;
        let extra = u64::min(extra, tsc_resolution - 1);
        #[cfg(debug_assertions)]
        {
            tsc_last_extra = extra;
        }
        return value + extra;
    }

    #[cfg(debug_assertions)]
    if tsc_last_extra != 0 {
        if TSC_VERBOSE_LOGGING || tsc_last_extra >= tsc_resolution {
            dbg_log!(
                "rdtsc: jump from {}+{} to {} (diff {}, {}%)",
                tsc_last_value as u64,
                tsc_last_extra as u64,
                value,
                value - (tsc_last_value + tsc_last_extra),
                (100 * tsc_last_extra) / tsc_resolution,
            );
            dbg_assert!(tsc_last_extra < tsc_resolution, "XXX: Overshot tsc");
        }
        tsc_last_extra = 0;
    }

    let d = value - tsc_last_value;
    if d < tsc_resolution {
        dbg_log!("rdtsc resolution: {}", d);
    }
    tsc_resolution = tsc_resolution.min(d);
    tsc_last_value = value;
    if tsc_number_of_same_readings != 0 {
        tsc_speed = tsc_number_of_same_readings;
        tsc_number_of_same_readings = 0;
    }

    value
}

pub unsafe fn vm86_mode() -> bool { return *flags & FLAG_VM == FLAG_VM; }

#[no_mangle]
pub unsafe fn getiopl() -> i32 { return *flags >> 12 & 3; }

#[no_mangle]
#[cfg(feature = "profiler")]
pub unsafe fn get_opstats_buffer(
    compiled: bool,
    jit_exit: bool,
    unguarded_register: bool,
    wasm_size: bool,
    opcode: u8,
    is_0f: bool,
    is_mem: bool,
    fixed_g: u8,
) -> f64 {
    {
        let index = (is_0f as usize) << 12
            | (opcode as usize) << 4
            | (is_mem as usize) << 3
            | fixed_g as usize;
        (if compiled {
            opstats::opstats_compiled_buffer[index]
        }
        else if jit_exit {
            opstats::opstats_jit_exit_buffer[index]
        }
        else if unguarded_register {
            opstats::opstats_unguarded_register_buffer[index]
        }
        else if wasm_size {
            opstats::opstats_wasm_size[index]
        }
        else {
            opstats::opstats_buffer[index]
        }) as f64
    }
}

#[no_mangle]
#[cfg(not(feature = "profiler"))]
pub unsafe fn get_opstats_buffer() -> f64 { 0.0 }

pub fn clear_tlb_code(page: u64) {
    unsafe {
        if let Some(slot) = tlb_code_slot(page) {
            if let Some(c) = slot.take() {
                code_free(c);
            }
        }
    }
}

pub unsafe fn invlpg(addr: u64) {
    tlb_invalidate_page(addr);
    *last_virt_eip64 = -1;
}

#[no_mangle]
pub unsafe fn update_eflags(new_flags: i32) {
    let mut dont_update = FLAG_RF | FLAG_VM | FLAG_VIP | FLAG_VIF;
    let mut clear = !FLAG_VIP & !FLAG_VIF & FLAGS_MASK;
    if 0 != *flags & FLAG_VM {
        // other case needs to be handled in popf or iret
        dbg_assert!(getiopl() == 3);
        dont_update |= FLAG_IOPL;
        // don't clear vip or vif
        clear |= FLAG_VIP | FLAG_VIF
    }
    else {
        if !*protected_mode {
            dbg_assert!(*cpl == 0);
        }
        if 0 != *cpl {
            // cpl > 0
            // cannot update iopl
            dont_update |= FLAG_IOPL;
            if *cpl as i32 > getiopl() {
                // cpl > iopl
                // cannot update interrupt flag
                dont_update |= FLAG_INTERRUPT
            }
        }
    }
    *flags = (new_flags ^ (*flags ^ new_flags) & dont_update) & clear | FLAGS_DEFAULT;
    *flags_changed = 0;

    if *flags & FLAG_TRAP != 0 {
        // Single-stepping armed: the interpreter loop in jit_run_interpreted
        // delivers a #DB trap after every instruction. Leave compiled 64-bit
        // code (no-op outside jit64); the 32-bit JIT's popf checks TF inline
        // and iret is a block boundary, and JIT dispatch is gated on
        // debug_mode_active, so no compiled code runs while TF is set.
        crate::jit64::JIT64_EXIT = true;
    }
}

/// iret (unlike popf and retf) restores RF from the saved flags image.
pub unsafe fn update_eflags_iret(new_flags: i32) {
    update_eflags(new_flags);
    *flags = *flags & !FLAG_RF | new_flags & FLAG_RF;
}

#[no_mangle]
pub unsafe fn get_valid_tlb_entries_count() -> i32 {
    if !cfg!(feature = "profiler") {
        return 0;
    }
    let mut result = 0;
    for i in 0..valid_tlb_entries_count {
        let page = valid_tlb_entries[i as usize];
        if page >= 0x10_0000 {
            continue;
        }
        let entry = tlb_data[page as usize];
        if 0 != entry as u64 {
            result += 1
        }
    }
    return result;
}

#[no_mangle]
pub unsafe fn get_valid_global_tlb_entries_count() -> i32 {
    if !cfg!(feature = "profiler") {
        return 0;
    }
    let mut result = 0;
    for i in 0..valid_tlb_entries_count {
        let page = valid_tlb_entries[i as usize];
        if page >= 0x10_0000 {
            continue;
        }
        let entry = tlb_data[page as usize];
        if 0 != entry as u64 as u32 & TLB_GLOBAL as u32 {
            result += 1
        }
    }
    return result;
}

#[inline(never)]
pub unsafe fn trigger_np(code: i32) {
    dbg_log!("#np");
    *instruction_pointer = *previous_ip;
    if DEBUG {
        if js::cpu_exception_hook(CPU_EXCEPTION_NP) {
            return;
        }
    }
    call_interrupt_vector(CPU_EXCEPTION_NP, false, Some(code));
}

#[inline(never)]
pub unsafe fn trigger_ss(code: i32) {
    dbg_log!("#ss");
    *instruction_pointer = *previous_ip;
    if DEBUG {
        if js::cpu_exception_hook(CPU_EXCEPTION_SS) {
            return;
        }
    }
    call_interrupt_vector(CPU_EXCEPTION_SS, false, Some(code));
}

/// Page fault caused by an instruction fetch from an NX page.
unsafe fn trigger_pagefault_nx(addr: u64) {
    if config::LOG_PAGE_FAULTS {
        dbg_log!("page fault nx eip={:x} cr2={:x}", *previous_ip, addr);
        dbg_trace();
    }
    profiler::stat_increment(stat::PAGE_FAULT);
    if cfg!(debug_assertions) {
        let a = canonicalize_address(addr);
        let pml4e = memory::read64s(get_cr3() & PAE_ENTRY_ADDRESS) as u64;
        let pdpte = memory::read64s((pml4e & PAE_ENTRY_ADDRESS) + ((a >> 30 & 0x1FF) << 3)) as u64;
        let pde = memory::read64s((pdpte & PAE_ENTRY_ADDRESS) + ((a >> 21 & 0x1FF) << 3)) as u64;
        dbg_log!(
            "pfwalk: cr3={:x} pml4e={:x} pdpte={:x} pde={:x}",
            get_cr3(),
            pml4e,
            pdpte,
            pde
        );
    }
    *cr.offset(2) = addr as u32 as i32;
    *cr2_64 = addr;
    tlb_invalidate_page(addr);
    // user<<2 | write<<1 | present | instruction fetch
    let error_code = ((*cpl == 3) as i32) << 2 | 1 | 1 << 4;
    *instruction_pointer = *previous_ip;
    if DEBUG {
        if js::cpu_exception_hook(CPU_EXCEPTION_PF) {
            return;
        }
    }
    call_interrupt_vector(CPU_EXCEPTION_PF, false, Some(error_code));
}

#[inline(never)]
pub unsafe fn trigger_ts(code: i32) {
    dbg_log!("#ts");
    *instruction_pointer = *previous_ip;
    if DEBUG {
        if js::cpu_exception_hook(CPU_EXCEPTION_TS) {
            return;
        }
    }
    call_interrupt_vector(CPU_EXCEPTION_TS, false, Some(code));
}

/// #DF (double fault). Always delivered with a zero error code; the IDT gate
/// for #DF must not itself fault (a second double fault while delivering #DF
/// shuts the processor down, which we model as a panic).
pub unsafe fn trigger_df(code: i32) {
    dbg_log!("#df");
    *instruction_pointer = *previous_ip;
    if DEBUG {
        if js::cpu_exception_hook(CPU_EXCEPTION_DF) {
            return;
        }
    }
    call_interrupt_vector(CPU_EXCEPTION_DF, false, Some(code));
}

#[no_mangle]
pub unsafe fn store_current_tsc() { *current_tsc = read_tsc(); }

#[no_mangle]
pub unsafe fn handle_irqs() {
    if *nmi_pending && !*nmi_blocked {
        *nmi_pending = false;
        *nmi_blocked = true;
        *previous_ip = *instruction_pointer;
        if *in_hlt {
            js::stop_idling();
            *in_hlt = false;
        }
        call_interrupt_vector(CPU_EXCEPTION_NMI, false, None);
    }
    if *flags & FLAG_INTERRUPT != 0 {
        if let Some(irq) = pic::pic_acknowledge_irq() {
            pic_call_irq(irq)
        }
        else if *apic_enabled {
            if let Some(irq) = apic::acknowledge_irq() {
                pic_call_irq(irq)
            }
        }
    }
}

/// Latch an NMI (vector 2 through the IDT, not maskable by IF); it is
/// delivered on the next instruction boundary, unless another NMI is in flight
#[no_mangle]
pub unsafe fn raise_nmi() {
    *nmi_pending = true;
    handle_irqs_or_defer();
}

unsafe fn pic_call_irq(interrupt_nr: u8) {
    *previous_ip = *instruction_pointer; // XXX: What if called after instruction (port IO)
    if *in_hlt {
        js::stop_idling();
        *in_hlt = false;
    }
    call_interrupt_vector(interrupt_nr as i32, false, None);
}

#[no_mangle]
unsafe fn device_raise_irq(i: u8) {
    pic::set_irq(i);
    // the ioapic is present regardless of ACPI (its redirection entries
    // reset to masked, so a guest that never programs it sees nothing)
    ioapic::set_irq(i);
    handle_irqs_or_defer()
}

#[no_mangle]
unsafe fn device_lower_irq(i: u8) {
    pic::clear_irq(i);
    ioapic::clear_irq(i);
    handle_irqs_or_defer()
}

pub fn io_port_read8(port: i32) -> i32 {
    unsafe {
        match port {
            0x20 => pic::port20_read() as i32,
            0x21 => pic::port21_read() as i32,
            0xA0 => pic::portA0_read() as i32,
            0xA1 => pic::portA1_read() as i32,
            0x4D0 => pic::port4D0_read() as i32,
            0x4D1 => pic::port4D1_read() as i32,
            _ => js::io_port_read8(port),
        }
    }
}
pub fn io_port_read16(port: i32) -> i32 { unsafe { js::io_port_read16(port) } }
pub fn io_port_read32(port: i32) -> i32 { unsafe { js::io_port_read32(port) } }

pub fn io_port_write8(port: i32, value: i32) {
    unsafe {
        match port {
            0x20 | 0x21 | 0xA0 | 0xA1 | 0x4D0 | 0x4D1 => {
                match port {
                    0x20 => pic::port20_write(value as u8),
                    0x21 => pic::port21_write(value as u8),
                    0xA0 => pic::portA0_write(value as u8),
                    0xA1 => pic::portA1_write(value as u8),
                    0x4D0 => pic::port4D0_write(value as u8),
                    0x4D1 => pic::port4D1_write(value as u8),
                    _ => dbg_assert!(false),
                };
                handle_irqs_or_defer()
            },
            _ => js::io_port_write8(port, value),
        }
    }
}
pub fn io_port_write16(port: i32, value: i32) { unsafe { js::io_port_write16(port, value) } }
pub fn io_port_write32(port: i32, value: i32) { unsafe { js::io_port_write32(port, value) } }

#[no_mangle]
#[cfg(debug_assertions)]
pub unsafe fn check_page_switch(block_addr: u64, next_block_addr: u64) {
    let x = translate_address_read_jit(*instruction_pointer);
    if x != Ok(next_block_addr) {
        dbg_log!(
            "page switch from={:x} to={:x} prev_eip={:x} eip={:x} phys_eip={:x}",
            block_addr,
            next_block_addr,
            *previous_ip,
            *instruction_pointer,
            x.unwrap_or(0),
        );
    }
    dbg_assert!(next_block_addr & 0xFFF == *instruction_pointer & 0xFFF);
    dbg_assert!(x.is_ok());
    dbg_assert!(x == Ok(next_block_addr));
}

#[no_mangle]
pub unsafe fn reset_cpu() {
    for i in 0..8 {
        *segment_is_null.offset(i) = false;
        *segment_limits.offset(i) = 0;
        *segment_offsets.offset(i) = 0;
        *segment_access_bytes.offset(i) = 0x80 | (0 << 5) | 0x10 | 0x02; // P dpl0 S RW

        *reg64.offset(i) = 0;

        *sreg.offset(i) = 0;
        *dreg.offset(i) = 0;

        write_xmm128_2(i as i32, 0, 0);

        *fpu_st.offset(i) = softfloat::F80::ZERO;
    }
    for i in 8..16 {
        *reg64.offset(i) = 0;
        write_xmm128_2(i as i32, 0, 0);
    }

    *segment_access_bytes.offset(CS as isize) = 0x80 | (0 << 5) | 0x10 | 0x08 | 0x02; // P dpl0 S E RW

    for i in 0..4 {
        *reg_pdpte.offset(i) = 0
    }

    *fpu_stack_empty = 0xFF;
    *fpu_stack_ptr = 0;
    *fpu_control_word = 0x37F;
    *fpu_status_word = 0;
    *fpu_ip = 0;
    *fpu_ip_selector = 0;
    *fpu_opcode = 0;
    *fpu_dp = 0;
    *fpu_dp_selector = 0;

    *mxcsr = 0x1F80;

    full_clear_tlb();

    *protected_mode = false;

    // http://www.sandpile.org/x86/initial.htm
    *idtr_size = 0;
    *idtr_offset = 0;
    *idtr_offset64 = 0;

    *gdtr_size = 0;
    *gdtr_offset = 0;
    *gdtr_offset64 = 0;

    *cr = 1 << 30 | 1 << 29 | 1 << 4;
    *cr.offset(2) = 0;
    *cr.offset(3) = 0;
    *cr3_high = 0;
    *cr.offset(4) = 0;
    *dreg.offset(6) = 0xFFFF0FF0u32 as i32;
    *dreg.offset(7) = 0x400;
    // EFER is cleared on reset (SCE/LME/NXE/etc. all off); without this a
    // reset from long mode would leave LMA set and the next boot's page
    // walks would run in the wrong mode
    *efer = 0;
    *cpl = 0;

    *is_32 = false;
    *is_64 = false;
    *stack_size_32 = false;
    *prefixes = 0;

    *last_virt_eip64 = -1;

    *instruction_counter = 0;
    *previous_ip = 0;
    *in_hlt = false;

    *sysenter_cs = 0;
    *sysenter_esp = 0;
    *sysenter_eip = 0;
    *sysenter_esp64 = 0;
    *sysenter_eip64 = 0;
    *misc_enable = 0;

    *flags = FLAGS_DEFAULT;
    *flags_changed = 0;
    *last_result = 0;
    *last_op1 = 0;
    *last_op_size = 0;

    *pat = 0x0007_0406_0007_0406;

    // the local APIC is enabled at reset (IA32_APIC_BASE = 0xFEE00900 for the BSP)
    *apic_enabled = true;

    *nmi_pending = false;
    *nmi_blocked = false;
    in_interrupt_delivery = -1;
    delivering_double_fault = false;

    set_tsc(0, 0);

    *instruction_pointer = 0xFFFF0;
    switch_cs_real_mode(0xF000);

    switch_seg(SS, 0x30);
    write_reg32(ESP, 0x100);

    update_state_flags();

    jit::jit_clear_cache_js();
}

#[no_mangle]
pub unsafe fn set_cpuid_level(level: u32) { cpuid_level = level }
