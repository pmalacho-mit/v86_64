use crate::fxhash::{HashMap, HashSet};
use std::collections::{BTreeMap, VecDeque};
use std::iter::FromIterator;
use std::mem::{self, MaybeUninit};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

use crate::analysis;
use crate::analysis::AnalysisType;
use crate::codegen;
use crate::control_flow;
use crate::control_flow::WasmStructure;
use crate::cpu::cpu;
use crate::cpu::memory;
use crate::cpu_context::CpuContext;
use crate::jit_instructions;
use crate::opstats;
use crate::page::Page;
use crate::profiler;
use crate::profiler::stat;
use crate::state_flags::CachedStateFlags;
use crate::wasmgen::wasm_builder::{Label, WasmBuilder, WasmLocal, WasmLocalI64};

#[derive(Copy, Clone, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct WasmTableIndex(u16);
impl WasmTableIndex {
    pub fn to_u16(self) -> u16 { self.0 }
}

mod unsafe_jit {
    use super::{CachedStateFlags, WasmTableIndex};

    #[link(wasm_import_module = "env")]
    extern "C" {
        pub fn codegen_finalize(
            wasm_table_index: WasmTableIndex,
            phys_addr: f64,
            state_flags: CachedStateFlags,
            ptr: u32,
            len: u32,
        ) -> bool;
        pub fn jit_clear_func(wasm_table_index: WasmTableIndex);
    }
}

/// Returns true if the module was compiled synchronously, in which case the caller must call
/// finish_compilation. Otherwise codegen_finalize_finished is called asynchronously
fn codegen_finalize(
    wasm_table_index: WasmTableIndex,
    phys_addr: u64,
    state_flags: CachedStateFlags,
    ptr: u32,
    len: u32,
) -> bool {
    unsafe {
        unsafe_jit::codegen_finalize(wasm_table_index, phys_addr as f64, state_flags, ptr, len)
    }
}

pub fn jit_clear_func(wasm_table_index: WasmTableIndex) {
    unsafe { unsafe_jit::jit_clear_func(wasm_table_index) }
}

static mut JIT_DISABLED: bool = false;
// compile 64-bit code (see jit64.rs); when disabled, 64-bit code runs in the interpreter
static mut JIT64_DISABLED: bool = false;
// let modules for 64-bit code continue directly with the next module through a tail call
// (jit64::gen_chain_to_next_module). Enabled by the host if it supports wasm tail calls.
static mut JIT64_CHAINING: bool = false;
// the same for modules for 16/32-bit code (codegen::gen_chain_to_next_module32)
static mut JIT32_CHAINING: bool = false;

// Maximum number of pages per wasm module. Necessary for the following reasons:
// - There is an upper limit on the size of a single function in wasm (currently ~7MB in all browsers)
//   See https://github.com/WebAssembly/design/issues/1138
// - v8 poorly handles large br_table elements and OOMs on modules much smaller than the above limit
//   See https://bugs.chromium.org/p/v8/issues/detail?id=9697 and https://bugs.chromium.org/p/v8/issues/detail?id=9141
//   Will hopefully be fixed in the near future by generating direct control flow
static mut MAX_PAGES: u32 = 3;

static mut JIT_USE_LOOP_SAFETY: bool = true;

pub static mut MAX_EXTRA_BASIC_BLOCKS: u32 = 250;

pub const JIT_THRESHOLD: u32 = 200 * 1000;
// can be lowered for tests
static mut JIT_HOTNESS_THRESHOLD: u32 = JIT_THRESHOLD;

// less branches will generate if-else, more will generate brtable
pub const BRTABLE_CUTOFF: usize = 10;

// needs to be synced to const.js
pub const WASM_TABLE_SIZE: u32 = 900;

pub const CHECK_JIT_STATE_INVARIANTS: bool = false;

const MAX_INSTRUCTION_LENGTH: u32 = 16;

static mut JIT_STATE: MaybeUninit<JitState> = MaybeUninit::uninit();

/// Access the global JIT state.
///
/// Zero synchronisation: the wasm32 target is single-threaded and the JIT is never reentered —
/// none of the functions called while a `JitStateRef` is alive calls back into `jit_state()`
/// (the reachable JS imports, `codegen_finalize` and `jit_clear_func`, only link/clear wasm
/// modules; `codegen_finalize_finished` runs on a later event-loop turn). The old
/// Mutex+v86-debug-assert wrapped this same invariant; it still holds and is checked in debug
/// builds via `JIT_STATE_BORROWED`.
#[inline(always)]
fn jit_state() -> JitStateRef { JitStateRef::new() }

#[cfg(debug_assertions)]
static mut JIT_STATE_BORROWED: bool = false;

struct JitStateRef(&'static mut JitState);

impl JitStateRef {
    #[inline(always)]
    #[allow(static_mut_refs)]
    fn new() -> Self {
        #[cfg(debug_assertions)]
        unsafe {
            dbg_assert!(!JIT_STATE_BORROWED, "reentrant access to JIT state");
            JIT_STATE_BORROWED = true;
        }
        JitStateRef(unsafe { JIT_STATE.assume_init_mut() })
    }
}
#[cfg(debug_assertions)]
impl Drop for JitStateRef {
    fn drop(&mut self) {
        #[allow(static_mut_refs)]
        unsafe {
            JIT_STATE_BORROWED = false
        }
    }
}
impl Deref for JitStateRef {
    type Target = JitState;
    fn deref(&self) -> &Self::Target { self.0 }
}
impl DerefMut for JitStateRef {
    fn deref_mut(&mut self) -> &mut Self::Target { self.0 }
}

#[no_mangle]
pub fn rust_init() {
    dbg_assert!(std::mem::size_of::<[Option<NonNull<cpu::Code>>; 0x100000]>() == 0x100000 * 4);

    #[allow(static_mut_refs)]
    unsafe {
        JIT_STATE.write(JitState::create_and_initialise());
    }

    use std::panic;

    panic::set_hook(Box::new(|panic_info| {
        console_log!("{}", panic_info.to_string());
    }));
}

struct PageInfo {
    wasm_table_index: WasmTableIndex,
    hidden_wasm_table_indices: Vec<WasmTableIndex>,
    entry_points: Vec<(u16, u16)>,
    state_flags: CachedStateFlags,
}

enum CompilingPageState {
    Compiling { pages: HashMap<Page, PageInfo> },
    CompilingWritten,
}

/// Flat per-physical-page JIT bookkeeping: the interpreter's hotness counter and a handle to
/// the page's entry-point set. One per covered physical page (see JitState::page_states).
#[derive(Clone, Copy)]
struct PageJitSlot {
    hotness: u32,
    /// index+1 into entry_sets, 0 = the page has no entry-point set. The empty slot (0) implies
    /// hotness == 0 (hotness and the set are created together, and clearing a page zeroes both)
    /// and no PageInfo / no compilation in progress for the page (compiled and compiling pages
    /// always have entry-point sets) — dirtying such a page cannot invalidate anything.
    entry_set: u32,
}
impl PageJitSlot {
    const EMPTY: PageJitSlot = PageJitSlot {
        hotness: 0,
        entry_set: 0,
    };
}

struct EntryPointSet {
    page: Page,
    offsets: HashSet<u16>,
}

struct JitState {
    wasm_builder: WasmBuilder,

    /// Hotness counters and entry-point-set handles per physical page, indexed directly by page
    /// number and sized once the memory size is known (init_page_state_coverage): the low 4 GiB
    /// (which includes all executable mmio aliases, e.g. the BIOS at 0xFFFFF) plus, in mem64
    /// builds, the high RAM. Pages outside the covered range (never RAM: only reachable when a
    /// guest maps nonexistent memory above the RAM top, or any page before coverage is
    /// established) go through fallback_hotness/fallback_entry_points instead — the fallback
    /// path is exactly the old HashMap behaviour.
    page_states: Vec<PageJitSlot>,
    /// Arena backing the per-page sets referenced by page_states (and the history of visited
    /// instruction offsets — entry points — on each executed page). A per-page set is created
    /// lazily when the interpreter first runs a block on the page, so pages that are never
    /// executed (and never compiled) cost only 8 bytes in page_states.
    entry_sets: Vec<Option<EntryPointSet>>,
    entry_sets_free: Vec<u32>,
    fallback_hotness: HashMap<Page, u32>,
    fallback_entry_points: HashMap<Page, HashSet<u16>>,
    pages: HashMap<Page, PageInfo>,
    wasm_table_index_free_list: Vec<WasmTableIndex>,
    compiling: Option<(WasmTableIndex, CompilingPageState)>,
    #[cfg(debug_assertions)]
    wasm_table_index_to_page: HashMap<WasmTableIndex, HashSet<Page>>,
}

/// The page_states index of a page, if the page is covered by the flat table
#[inline(always)]
fn page_slot(states: &Vec<PageJitSlot>, page: Page) -> Option<usize> {
    let i = page.to_u32() as usize;
    if i < states.len() {
        Some(i)
    }
    else {
        None
    }
}

/// Mutable access to a page's hotness counter (flat table or fallback map)
#[inline(always)]
fn hotness_mut(ctx: &mut JitState, page: Page) -> &mut u32 {
    match page_slot(&ctx.page_states, page) {
        Some(i) => &mut ctx.page_states[i].hotness,
        None => ctx.fallback_hotness.entry(page).or_insert(0),
    }
}

/// Zero a page's hotness counter (as the old code did by resetting the entry in place)
#[inline]
fn reset_hotness(ctx: &mut JitState, page: Page) {
    match page_slot(&ctx.page_states, page) {
        Some(i) => ctx.page_states[i].hotness = 0,
        None => {
            if let Some(h) = ctx.fallback_hotness.get_mut(&page) {
                *h = 0;
            }
        },
    }
}

fn get_entry_set<'a>(ctx: &'a JitState, page: Page) -> Option<&'a HashSet<u16>> {
    match page_slot(&ctx.page_states, page) {
        Some(i) => {
            let e = ctx.page_states[i].entry_set;
            if e == 0 {
                None
            }
            else {
                ctx.entry_sets[(e - 1) as usize]
                    .as_ref()
                    .map(|s| &s.offsets)
            }
        },
        None => ctx.fallback_entry_points.get(&page),
    }
}

fn has_entry_points(ctx: &JitState, page: Page) -> bool {
    match page_slot(&ctx.page_states, page) {
        Some(i) => ctx.page_states[i].entry_set != 0,
        None => ctx.fallback_entry_points.contains_key(&page),
    }
}

/// Get or create the page's entry-point set; the second return value is true when it was
/// created by this call. Creation has no other side effects; the caller at the block-exit hot
/// path additionally marks the page's tlb entries (as the old or_insert_with closure did).
fn get_or_create_entry_set(ctx: &mut JitState, page: Page) -> (&mut HashSet<u16>, bool) {
    match page_slot(&ctx.page_states, page) {
        Some(i) => {
            let e = ctx.page_states[i].entry_set;
            if e != 0 {
                let idx = (e - 1) as usize;
                return (&mut ctx.entry_sets[idx].as_mut().unwrap().offsets, false);
            }
            let idx = match ctx.entry_sets_free.pop() {
                Some(idx) => idx as usize,
                None => {
                    let idx = ctx.entry_sets.len();
                    ctx.entry_sets.push(None);
                    idx
                },
            };
            ctx.entry_sets[idx] = Some(EntryPointSet {
                page,
                offsets: HashSet::default(),
            });
            ctx.page_states[i].entry_set = idx as u32 + 1;
            (&mut ctx.entry_sets[idx].as_mut().unwrap().offsets, true)
        },
        None => match ctx.fallback_entry_points.entry(page) {
            std::collections::hash_map::Entry::Occupied(o) => (o.into_mut(), false),
            std::collections::hash_map::Entry::Vacant(v) => (v.insert(HashSet::default()), true),
        },
    }
}

/// Drop a page's entry-point set (if any) and zero its hotness counter; returns whether the
/// page had an entry-point set. Old equivalent: removing the page from the entry_points map.
fn clear_page_hotness_and_entry_points(ctx: &mut JitState, page: Page) -> bool {
    match page_slot(&ctx.page_states, page) {
        Some(i) => {
            let slot = &mut ctx.page_states[i];
            slot.hotness = 0;
            let e = mem::replace(&mut slot.entry_set, 0);
            if e == 0 {
                false
            }
            else {
                let idx = (e - 1) as usize;
                dbg_assert!(ctx.entry_sets[idx].as_ref().unwrap().page == page);
                ctx.entry_sets[idx] = None;
                ctx.entry_sets_free.push(e - 1);
                true
            }
        },
        None => {
            ctx.fallback_hotness.remove(&page);
            ctx.fallback_entry_points.remove(&page).is_some()
        },
    }
}

/// Size page_states to cover the given number of physical pages. Called once from
/// memory::allocate_memory; before that (and for pages beyond the covered range) every page
/// takes the fallback path.
pub fn init_page_state_coverage(pages: u32) {
    let mut ctx = jit_state();
    dbg_assert!(ctx.page_states.is_empty());
    ctx.page_states = vec![PageJitSlot::EMPTY; pages as usize];
}

fn check_jit_state_invariants(ctx: &mut JitState) {
    if !CHECK_JIT_STATE_INVARIANTS {
        return;
    }

    match &ctx.compiling {
        Some((_, CompilingPageState::Compiling { pages })) => {
            dbg_assert!(pages.keys().all(|page| has_entry_points(ctx, *page)));
        },
        _ => {},
    }

    let free: HashSet<WasmTableIndex> =
        HashSet::from_iter(ctx.wasm_table_index_free_list.iter().copied());
    let used = HashSet::from_iter(ctx.pages.values().map(|info| info.wasm_table_index));
    let compiling = HashSet::from_iter(ctx.compiling.as_ref().map(|&(index, _)| index));
    dbg_assert!(free.intersection(&used).next().is_none());
    dbg_assert!(used.intersection(&compiling).next().is_none());
    dbg_assert!(free.len() + used.len() + compiling.len() == (WASM_TABLE_SIZE - 1) as usize);

    let hidden: HashSet<WasmTableIndex> = ctx
        .pages
        .values()
        .flat_map(|info| info.hidden_wasm_table_indices.iter().copied())
        .collect();
    dbg_assert!(free.intersection(&hidden).next().is_none());
    dbg_assert!(hidden.is_subset(&used));

    #[cfg(debug_assertions)]
    for (wasm_table_index, pages) in &ctx.wasm_table_index_to_page {
        for page in pages {
            match ctx.pages.get(page) {
                Some(info) => dbg_assert!(
                    info.wasm_table_index == *wasm_table_index
                        || info.hidden_wasm_table_indices.contains(wasm_table_index)
                ),
                None => dbg_assert!(false),
            }
        }
    }

    match &ctx.compiling {
        Some((_, CompilingPageState::Compiling { pages })) => {
            dbg_assert!(pages.keys().all(|page| has_entry_points(ctx, *page)));
        },
        _ => {},
    }

    for i in 0..unsafe { cpu::valid_tlb_entries_count } {
        let page = unsafe { cpu::valid_tlb_entries[i as usize] };
        // pages >= 4 GiB live in the secondary TLB and never hold jittered code
        if page >= 0x10_0000 {
            continue;
        }
        let entry = unsafe { cpu::tlb_pick_entry(page << 12) };
        if 0 != entry {
            let tlb_physical_page = Page::page_of(cpu::phys_of_tlb_entry(entry, page << 12));
            let w = match unsafe { cpu::tlb_code[page as usize] } {
                None => None,
                Some(c) => unsafe { Some(c.as_ref().wasm_table_index) },
            };
            let tlb_has_code = entry as i32 & cpu::TLB_HAS_CODE == cpu::TLB_HAS_CODE;

            let infos = ctx.pages.get(&tlb_physical_page);
            let has_entry_points = has_entry_points(ctx, tlb_physical_page);
            dbg_assert!(tlb_has_code || !w.is_some());
            dbg_assert!(tlb_has_code || !infos.is_some());
            dbg_assert!(tlb_has_code || !has_entry_points);
            //dbg_assert!((w.is_some() || page.is_some() || has_entry_points) == tlb_has_code); // XXX: check this
        }
    }
}

impl JitState {
    pub fn create_and_initialise() -> JitState {
        // don't assign 0 (XXX: Check)
        let wasm_table_indices = (1..=(WASM_TABLE_SIZE - 1) as u16).map(|x| WasmTableIndex(x));

        JitState {
            wasm_builder: WasmBuilder::new(),

            page_states: Vec::new(),
            entry_sets: Vec::new(),
            entry_sets_free: Vec::new(),
            fallback_hotness: HashMap::default(),
            fallback_entry_points: HashMap::default(),
            pages: HashMap::default(),

            wasm_table_index_free_list: Vec::from_iter(wasm_table_indices),
            compiling: None,

            #[cfg(debug_assertions)]
            wasm_table_index_to_page: HashMap::default(),
        }
    }
}

#[derive(PartialEq, Eq)]
pub enum BasicBlockType {
    Normal {
        next_block_addr: Option<u64>,
        jump_offset: i32,
        jump_offset_is_32: bool,
    },
    ConditionalJump {
        next_block_addr: Option<u64>,
        next_block_branch_taken_addr: Option<u64>,
        condition: u8,
        jump_offset: i32,
        jump_offset_is_32: bool,
    },
    // Set eip to an absolute value (ret, jmp r/m, call r/m)
    AbsoluteEip,
    Exit,
}

pub struct BasicBlock {
    /// physical address
    pub addr: u64,
    /// virtual address (64-bit in long mode)
    pub virt_addr: u64,
    pub last_instruction_addr: u64,
    pub end_addr: u64,
    pub is_entry_block: bool,
    pub ty: BasicBlockType,
    pub has_sti: bool,
    pub number_of_instructions: u32,
}

#[derive(Copy, Clone, PartialEq)]
pub struct CachedCode {
    pub wasm_table_index: WasmTableIndex,
    pub initial_state: u16,
}

impl CachedCode {
    pub const NONE: CachedCode = CachedCode {
        wasm_table_index: WasmTableIndex(0),
        initial_state: 0,
    };
}

#[derive(PartialEq)]
pub enum InstructionOperandDest {
    WasmLocal(WasmLocal),
    Other,
}
#[derive(PartialEq)]
pub enum InstructionOperand {
    WasmLocal(WasmLocal),
    Immediate(i32),
    Other,
}
impl InstructionOperand {
    pub fn is_zero(&self) -> bool {
        match self {
            InstructionOperand::Immediate(0) => true,
            _ => false,
        }
    }
}
impl Into<InstructionOperand> for InstructionOperandDest {
    fn into(self: InstructionOperandDest) -> InstructionOperand {
        match self {
            InstructionOperandDest::WasmLocal(l) => InstructionOperand::WasmLocal(l),
            InstructionOperandDest::Other => InstructionOperand::Other,
        }
    }
}
pub enum Instruction {
    Cmp {
        dest: InstructionOperandDest,
        source: InstructionOperand,
        opsize: i32,
    },
    Sub {
        dest: InstructionOperandDest,
        source: InstructionOperand,
        opsize: i32,
        is_dec: bool,
    },
    Add {
        dest: InstructionOperandDest,
        source: InstructionOperand,
        opsize: i32,
        is_inc: bool,
    },
    AdcSbb {
        dest: InstructionOperandDest,
        #[allow(dead_code)]
        source: InstructionOperand,
        opsize: i32,
    },
    NonZeroShift {
        dest: InstructionOperandDest,
        opsize: i32,
    },
    Bitwise {
        dest: InstructionOperandDest,
        opsize: i32,
    },
    Other,
}

pub struct JitContext<'a> {
    pub cpu: &'a mut CpuContext,
    pub builder: &'a mut WasmBuilder,
    pub register_locals: &'a mut Vec<WasmLocal>,
    pub start_of_current_instruction: u64,
    pub exit_with_fault_label: Label,
    pub exit_label: Label,
    pub current_instruction: Instruction,
    pub previous_instruction: Instruction,
    pub instruction_counter: WasmLocal,
    pub wasm_table_index: WasmTableIndex,
    /// 64-bit jit: the operation that last set the lazy flags in the current block, if known
    pub flags64: crate::jit64::Flags64,
    /// 64-bit jit: the function-wide locals that hold deferred lazy-flag values
    pub jit64_flag_locals: Option<crate::jit64::FlagLocals64>,
    /// 64-bit jit: the deferred lazy-flag state, if the values differ from memory (the
    /// compile-time part; the values live in jit64_flag_locals)
    pub jit64_deferred: Option<crate::jit64::DeferredFlags64>,
    /// 64-bit jit: the function-wide locals of the data tlb cache (jit64::TlbCacheLocals)
    pub jit64_tlb_cache: Option<crate::jit64::TlbCacheLocals>,
    /// 64-bit jit: the 16 general purpose registers
    pub register_locals64: Vec<WasmLocalI64>,
    /// 64-bit jit: registers whose local may differ from memory (bitmask, within a block)
    pub dirty_registers64: u16,
    /// 64-bit jit: the (physical) end address of the current basic block
    pub block_end: u64,
}
impl<'a> JitContext<'a> {
    pub fn reg(&self, i: u32) -> WasmLocal {
        match self.register_locals.get(i as usize) {
            Some(x) => x.unsafe_clone(),
            None => {
                dbg_assert!(false);
                unsafe { std::hint::unreachable_unchecked() }
            },
        }
    }
}

pub const JIT_INSTR_BLOCK_BOUNDARY_FLAG: u32 = 1 << 0;

pub fn is_near_end_of_page(address: u64) -> bool {
    address & 0xFFF >= 0x1000 - MAX_INSTRUCTION_LENGTH as u64
}

pub fn jit_find_cache_entry(phys_address: u64, state_flags: CachedStateFlags) -> CachedCode {
    // TODO: dedup with jit_find_cache_entry_in_page?
    // NOTE: This is currently only used for invariant/missed-entry-point checking
    let ctx = jit_state();

    match ctx.pages.get(&Page::page_of(phys_address)) {
        Some(PageInfo {
            wasm_table_index,
            state_flags: s,
            entry_points,
            hidden_wasm_table_indices: _,
        }) => {
            if *s == state_flags {
                let page_offset = phys_address as u16 & 0xFFF;
                if let Some(&(_, initial_state)) =
                    entry_points.iter().find(|(p, _)| p == &page_offset)
                {
                    return CachedCode {
                        wasm_table_index: *wasm_table_index,
                        initial_state,
                    };
                }
            }
        },
        None => {},
    }

    return CachedCode::NONE;
}

#[no_mangle]
pub fn jit_find_cache_entry_in_page(
    virt_address: u32,
    wasm_table_index: WasmTableIndex,
    state_flags: u32,
) -> i32 {
    // TODO: generate code for this
    profiler::stat_increment(stat::INDIRECT_JUMP);

    let state_flags = CachedStateFlags::of_u32(state_flags);

    unsafe {
        match cpu::tlb_code_get(virt_address as u64 >> 12) {
            None => {},
            Some(c) => {
                let c = c.as_ref();
                if state_flags == c.state_flags && wasm_table_index == c.wasm_table_index {
                    let state = c.state_table[virt_address as usize & 0xFFF];
                    if state != u16::MAX {
                        return state.into();
                    }
                }
            },
        }
    }

    profiler::stat_increment(stat::INDIRECT_JUMP_NO_ENTRY);

    return -1;
}

/// The target of a relative jump from virt_addr (the end of the jump instruction)
fn jump_target(cpu: &CpuContext, virt_addr: u64, offset: i32, is_32: bool) -> u64 {
    if cpu.state_flags.is_64() {
        dbg_assert!(is_32);
        virt_addr.wrapping_add(offset as i64 as u64)
    }
    else if is_32 {
        (virt_addr as u32).wrapping_add(offset as u32) as u64
    }
    else {
        let cs_offset = cpu.cs_offset;
        cs_offset.wrapping_add(
            (virt_addr as u32)
                .wrapping_sub(cs_offset)
                .wrapping_add(offset as u32)
                & 0xFFFF,
        ) as u64
    }
}

#[no_mangle]
pub fn jit_find_cache_entry_in_page64(
    virt_address: u64,
    wasm_table_index: WasmTableIndex,
    state_flags: u32,
) -> i32 {
    profiler::stat_increment(stat::INDIRECT_JUMP);

    let state_flags = CachedStateFlags::of_u32(state_flags);

    unsafe {
        match cpu::tlb_code_get(cpu::canonicalize_address(virt_address) >> 12) {
            None => {},
            Some(c) => {
                let c = c.as_ref();
                if state_flags == c.state_flags && wasm_table_index == c.wasm_table_index {
                    let state = c.state_table[virt_address as usize & 0xFFF];
                    if state != u16::MAX {
                        return state.into();
                    }
                }
            },
        }
    }

    profiler::stat_increment(stat::INDIRECT_JUMP_NO_ENTRY);

    return -1;
}

fn jit_find_basic_blocks(
    ctx: &mut JitState,
    entry_points: HashSet<u64>,
    cpu: CpuContext,
) -> Vec<BasicBlock> {
    fn follow_jump(
        virt_target: u64,
        ctx: &mut JitState,
        pages: &mut HashSet<Page>,
        page_blacklist: &mut HashSet<Page>,
        max_pages: u32,
        marked_as_entry: &mut HashSet<u64>,
        to_visit_stack: &mut Vec<u64>,
    ) -> Option<u64> {
        if is_near_end_of_page(virt_target) {
            return None;
        }
        let phys_target = match cpu::translate_address_read_no_side_effects(virt_target) {
            Err(()) => {
                dbg_log!("Not analysing {:x} (page not mapped)", virt_target);
                return None;
            },
            Ok(t) => t,
        };

        let phys_page = Page::page_of(phys_target);

        if !pages.contains(&phys_page) && pages.len() as u32 == max_pages
            || page_blacklist.contains(&phys_page)
        {
            return None;
        }

        if !pages.contains(&phys_page) {
            // page seen for the first time, handle entry points
            if let Some(entry_points) = get_entry_set(ctx, phys_page) {
                let existing_entry_points = match ctx.pages.get(&phys_page) {
                    Some(PageInfo { entry_points, .. }) => {
                        HashSet::from_iter(entry_points.iter().map(|x| x.0))
                    },
                    None => HashSet::default(),
                };

                if entry_points
                    .iter()
                    .all(|entry_point| existing_entry_points.contains(entry_point))
                {
                    page_blacklist.insert(phys_page);
                    return None;
                }

                // XXX: Remove this paragraph
                //let old_length = entry_points.len();
                //entry_points.extend(existing_entry_points);
                //dbg_assert!(
                //    entry_points.union(&existing_entry_points).count() == entry_points.len()
                //);

                reset_hotness(ctx, phys_page);
                let entry_points = get_entry_set(ctx, phys_page).unwrap();

                for &addr_low in entry_points.iter() {
                    let addr = virt_target & !0xFFF | addr_low as u64;
                    to_visit_stack.push(addr);
                    marked_as_entry.insert(addr);
                }
            }
            else {
                // no entry points: ignore this page?
                page_blacklist.insert(phys_page);
                return None;
            }

            pages.insert(phys_page);
            dbg_assert!(pages.len() as u32 <= max_pages);
        }

        to_visit_stack.push(virt_target);
        Some(phys_target)
    }

    let mut to_visit_stack: Vec<u64> = Vec::new();
    let mut marked_as_entry: HashSet<u64> = HashSet::default();
    let mut basic_blocks: BTreeMap<u64, BasicBlock> = BTreeMap::new();
    let mut pages: HashSet<Page> = HashSet::default();
    let mut page_blacklist = HashSet::default();

    // 16-bit doesn't work correctly, most likely due to instruction pointer wrap-around
    let max_pages =
        if cpu.state_flags.is_32() || cpu.state_flags.is_64() { unsafe { MAX_PAGES } } else { 1 };

    for virt_addr in entry_points {
        let ok = follow_jump(
            virt_addr,
            ctx,
            &mut pages,
            &mut page_blacklist,
            max_pages,
            &mut marked_as_entry,
            &mut to_visit_stack,
        );
        dbg_assert!(ok.is_some());
        dbg_assert!(marked_as_entry.contains(&virt_addr));
    }

    while let Some(to_visit) = to_visit_stack.pop() {
        let phys_addr = match cpu::translate_address_read_no_side_effects(to_visit) {
            Err(()) => {
                dbg_log!("Not analysing {:x} (page not mapped)", to_visit);
                continue;
            },
            Ok(phys_addr) => phys_addr,
        };

        if basic_blocks.contains_key(&phys_addr) {
            continue;
        }

        if is_near_end_of_page(phys_addr) {
            // Empty basic block, don't insert
            profiler::stat_increment(stat::COMPILE_CUT_OFF_AT_END_OF_PAGE);
            continue;
        }

        let mut current_address = phys_addr;
        let mut current_block = BasicBlock {
            addr: current_address,
            virt_addr: to_visit,
            last_instruction_addr: 0,
            end_addr: 0,
            ty: BasicBlockType::Exit,
            is_entry_block: false,
            has_sti: false,
            number_of_instructions: 0,
        };
        loop {
            let addr_before_instruction = current_address;
            let mut cpu = &mut CpuContext {
                eip: current_address,
                ..cpu
            };
            let analysis = analysis::analyze_step(&mut cpu);
            let has_next_instruction = !analysis.no_next_instruction;
            current_address = cpu.eip;

            dbg_assert!(Page::page_of(current_address) == Page::page_of(addr_before_instruction));
            let current_virt_addr = to_visit & !0xFFF | current_address as u64 & 0xFFF;

            if analysis.ty == AnalysisType::STI && is_near_end_of_page(current_address) {
                // cut off before the STI so that it is handled by interpreted mode
                profiler::stat_increment(stat::COMPILE_CUT_OFF_AT_END_OF_PAGE);
                break;
            }

            current_block.number_of_instructions += 1;
            current_block.last_instruction_addr = addr_before_instruction;
            current_block.end_addr = current_address;

            match analysis.ty {
                AnalysisType::Normal | AnalysisType::STI => {
                    dbg_assert!(has_next_instruction);
                    dbg_assert!(!analysis.absolute_jump);

                    if current_block.has_sti {
                        // Convert next instruction after STI (i.e., the current instruction) into block boundary

                        marked_as_entry.insert(current_virt_addr);
                        to_visit_stack.push(current_virt_addr);

                        break;
                    }

                    if analysis.ty == AnalysisType::STI {
                        dbg_assert!(
                            !is_near_end_of_page(current_address),
                            "should be handled above"
                        );

                        current_block.has_sti = true;
                    }
                    else {
                        // Only split non-STI blocks (one instruction needs to run after STI before
                        // handle_irqs may be called)

                        if basic_blocks.contains_key(&current_address) {
                            dbg_assert!(!is_near_end_of_page(current_address));
                            current_block.ty = BasicBlockType::Normal {
                                next_block_addr: Some(current_address),
                                jump_offset: 0,
                                jump_offset_is_32: true,
                            };
                            break;
                        }
                    }
                },
                AnalysisType::Jump {
                    offset,
                    is_32,
                    condition: Some(condition),
                } => {
                    dbg_assert!(!analysis.absolute_jump);
                    // conditional jump: continue at next and continue at jump target

                    let jump_target = jump_target(&cpu, current_virt_addr, offset, is_32);

                    dbg_assert!(has_next_instruction);
                    to_visit_stack.push(current_virt_addr);

                    let next_block_addr = if is_near_end_of_page(current_address) {
                        None
                    }
                    else {
                        Some(current_address)
                    };

                    current_block.ty = BasicBlockType::ConditionalJump {
                        next_block_addr,
                        next_block_branch_taken_addr: follow_jump(
                            jump_target,
                            ctx,
                            &mut pages,
                            &mut page_blacklist,
                            max_pages,
                            &mut marked_as_entry,
                            &mut to_visit_stack,
                        ),
                        condition,
                        jump_offset: offset,
                        jump_offset_is_32: is_32,
                    };

                    break;
                },
                AnalysisType::Jump {
                    offset,
                    is_32,
                    condition: None,
                } => {
                    dbg_assert!(!analysis.absolute_jump);
                    // non-conditional jump: continue at jump target

                    let jump_target = jump_target(&cpu, current_virt_addr, offset, is_32);

                    if has_next_instruction {
                        // Execution will eventually come back to the next instruction (CALL)
                        marked_as_entry.insert(current_virt_addr);
                        to_visit_stack.push(current_virt_addr);
                    }

                    current_block.ty = BasicBlockType::Normal {
                        next_block_addr: follow_jump(
                            jump_target,
                            ctx,
                            &mut pages,
                            &mut page_blacklist,
                            max_pages,
                            &mut marked_as_entry,
                            &mut to_visit_stack,
                        ),
                        jump_offset: offset,
                        jump_offset_is_32: is_32,
                    };

                    break;
                },
                AnalysisType::BlockBoundary => {
                    // a block boundary but not a jump, get out

                    if has_next_instruction {
                        // block boundary, but execution will eventually come back
                        // to the next instruction. Create a new basic block
                        // starting at the next instruction and register it as an
                        // entry point
                        marked_as_entry.insert(current_virt_addr);
                        to_visit_stack.push(current_virt_addr);
                    }

                    if analysis.absolute_jump {
                        current_block.ty = BasicBlockType::AbsoluteEip;
                    }

                    break;
                },
            }

            if is_near_end_of_page(current_address) {
                profiler::stat_increment(stat::COMPILE_CUT_OFF_AT_END_OF_PAGE);
                break;
            }
        }

        if current_block.number_of_instructions == 0 {
            // Empty basic block, don't insert (only happens when STI is found near end of page)
            continue;
        }

        let previous_block = basic_blocks
            .range(..current_block.addr)
            .next_back()
            .filter(|(_, previous_block)| !previous_block.has_sti)
            .map(|(_, previous_block)| previous_block);

        if let Some(previous_block) = previous_block {
            if current_block.addr < previous_block.end_addr {
                // If this block overlaps with the previous block, re-analyze the previous block
                to_visit_stack.push(previous_block.virt_addr);

                let addr = previous_block.addr;
                let old_block = basic_blocks.remove(&addr);
                dbg_assert!(old_block.is_some());

                // Note that this does not ensure the invariant that two consecutive blocks don't
                // overlay. For that, we also need to check the following block.
            }
        }

        dbg_assert!(current_block.addr < current_block.end_addr);
        dbg_assert!(current_block.addr <= current_block.last_instruction_addr);
        dbg_assert!(current_block.last_instruction_addr < current_block.end_addr);

        basic_blocks.insert(current_block.addr, current_block);
    }

    dbg_assert!(pages.len() as u32 <= max_pages);

    for block in basic_blocks.values_mut() {
        if marked_as_entry.contains(&block.virt_addr) {
            block.is_entry_block = true;
        }
    }

    // delete edges pointing to blocks that were dropped (currently only due to STI near the end of a page)
    let known_addresses: HashSet<u64> = basic_blocks.keys().copied().collect();
    for block in basic_blocks.values_mut() {
        match &mut block.ty {
            BasicBlockType::Normal {
                next_block_addr, ..
            } => {
                if next_block_addr.map_or(false, |a| !known_addresses.contains(&a)) {
                    *next_block_addr = None;
                }
            },
            BasicBlockType::ConditionalJump {
                next_block_addr,
                next_block_branch_taken_addr,
                ..
            } => {
                if next_block_addr.map_or(false, |a| !known_addresses.contains(&a)) {
                    *next_block_addr = None;
                }
                if next_block_branch_taken_addr.map_or(false, |a| !known_addresses.contains(&a)) {
                    *next_block_branch_taken_addr = None;
                }
            },
            BasicBlockType::Exit | BasicBlockType::AbsoluteEip => {},
        }
    }

    let basic_blocks: Vec<BasicBlock> = basic_blocks.into_iter().map(|(_, block)| block).collect();

    for i in 0..basic_blocks.len() - 1 {
        let next_block_addr = basic_blocks[i + 1].addr;
        let next_block_end_addr = basic_blocks[i + 1].end_addr;
        let next_block_is_entry = basic_blocks[i + 1].is_entry_block;
        let block = &basic_blocks[i];
        dbg_assert!(block.addr < next_block_addr);
        if next_block_addr < block.end_addr {
            dbg_log!(
                "Overlapping first=[from={:x} to={:x} is_entry={}] second=[from={:x} to={:x} is_entry={}]",
                block.addr,
                block.end_addr,
                block.is_entry_block as u8,
                next_block_addr,
                next_block_end_addr,
                next_block_is_entry as u8
            );
        }
    }

    basic_blocks
}

#[no_mangle]
#[cfg(debug_assertions)]
pub fn jit_force_generate_unsafe(virt_addr: i32) {
    dbg_assert!(
        !is_near_end_of_page(virt_addr as u32 as u64),
        "cannot force compile near end of page"
    );
    let phys_addr = cpu::translate_address_read(virt_addr as u32 as u64).unwrap();
    jit_increase_hotness_and_maybe_compile(
        virt_addr as u32 as u64,
        phys_addr,
        cpu::get_seg_cs() as u32,
        cpu::get_state_flags(),
        JIT_THRESHOLD,
    );
    // in progress, or already finished (synchronous compilation)
    let ctx = jit_state();
    dbg_assert!(ctx.compiling.is_some() || ctx.pages.contains_key(&Page::page_of(phys_addr)));
}

#[inline(never)]
fn jit_analyze_and_generate(
    ctx: &mut JitState,
    virt_entry_point: u64,
    phys_entry_point: u64,
    cs_offset: u32,
    state_flags: CachedStateFlags,
) {
    let page = Page::page_of(phys_entry_point);

    dbg_assert!(ctx.compiling.is_none());

    let entry_points = match get_entry_set(ctx, page) {
        None => return,
        Some(entry_points) => entry_points,
    };

    let existing_entry_points = match ctx.pages.get(&page) {
        Some(PageInfo { entry_points, .. }) => HashSet::from_iter(entry_points.iter().map(|x| x.0)),
        None => HashSet::default(),
    };

    if entry_points
        .iter()
        .all(|entry_point| existing_entry_points.contains(entry_point))
    {
        profiler::stat_increment(stat::COMPILE_SKIPPED_NO_NEW_ENTRY_POINTS);
        return;
    }

    // XXX: check and remove
    //let old_length = entry_points.len();
    //entry_points.extend(existing_entry_points);
    //dbg_log!(
    //    "{} + {} = {}",
    //    entry_points.len(),
    //    existing_entry_points.len(),
    //    entry_points.union(&existing_entry_points).count()
    //);
    //dbg_assert!(entry_points.union(&existing_entry_points).count() == entry_points.len());

    profiler::stat_increment(stat::COMPILE);

    let cpu = CpuContext {
        eip: 0,
        prefixes: 0,
        cs_offset,
        state_flags,
    };

    dbg_assert!(
        cpu::translate_address_read_no_side_effects(virt_entry_point).unwrap() == phys_entry_point
    );
    let virt_page = virt_entry_point & !0xFFF;
    let entry_points: HashSet<u64> = entry_points.iter().map(|e| virt_page | *e as u64).collect();
    let basic_blocks = jit_find_basic_blocks(ctx, entry_points, cpu.clone());

    let mut pages = HashSet::default();

    for b in basic_blocks.iter() {
        // Remove this assertion once page-crossing jit is enabled
        dbg_assert!(Page::page_of(b.addr) == Page::page_of(b.end_addr));
        pages.insert(Page::page_of(b.addr));
    }

    let print = false;

    for b in basic_blocks.iter() {
        if !print {
            break;
        }
        let last_instruction_opcode = memory::read32s(b.last_instruction_addr);
        let op = opstats::decode(last_instruction_opcode as u32);
        dbg_log!(
            "BB: 0x{:x} {}{:02x} {} {}",
            b.addr,
            if op.is_0f { "0f" } else { "" },
            op.opcode,
            if b.is_entry_block { "entry" } else { "noentry" },
            match &b.ty {
                BasicBlockType::ConditionalJump {
                    next_block_addr: Some(next_block_addr),
                    next_block_branch_taken_addr: Some(next_block_branch_taken_addr),
                    ..
                } => format!(
                    "0x{:x} 0x{:x}",
                    next_block_addr, next_block_branch_taken_addr
                ),
                BasicBlockType::ConditionalJump {
                    next_block_addr: None,
                    next_block_branch_taken_addr: Some(next_block_branch_taken_addr),
                    ..
                } => format!("0x{:x}", next_block_branch_taken_addr),
                BasicBlockType::ConditionalJump {
                    next_block_addr: Some(next_block_addr),
                    next_block_branch_taken_addr: None,
                    ..
                } => format!("0x{:x}", next_block_addr),
                BasicBlockType::ConditionalJump {
                    next_block_addr: None,
                    next_block_branch_taken_addr: None,
                    ..
                } => format!(""),
                BasicBlockType::Normal {
                    next_block_addr: Some(next_block_addr),
                    ..
                } => format!("0x{:x}", next_block_addr),
                BasicBlockType::Normal {
                    next_block_addr: None,
                    ..
                } => format!(""),
                BasicBlockType::Exit => format!(""),
                BasicBlockType::AbsoluteEip => format!(""),
            }
        );
    }

    let graph = control_flow::make_graph(&basic_blocks);
    let mut structure = control_flow::loopify(&graph);

    if print {
        dbg_log!("before blockify:");
        for group in &structure {
            dbg_log!("=> Group");
            group.print(0);
        }
    }

    control_flow::blockify(&mut structure, &graph);

    if cfg!(debug_assertions) {
        control_flow::assert_invariants(&structure);
    }

    if print {
        dbg_log!("after blockify:");
        for group in &structure {
            dbg_log!("=> Group");
            group.print(0);
        }
    }

    if ctx.wasm_table_index_free_list.is_empty() {
        dbg_log!("wasm_table_index_free_list empty, clearing cache");

        // When no free slots are available, delete all cached modules. We could increase the
        // size of the table, but this way the initial size acts as an upper bound for the
        // number of wasm modules that we generate, which we want anyway to avoid getting our
        // tab killed by browsers due to memory constraints.
        jit_clear_cache(ctx);

        profiler::stat_increment(stat::INVALIDATE_ALL_MODULES_NO_FREE_WASM_INDICES);

        dbg_log!(
            "after jit_clear_cache: {} free",
            ctx.wasm_table_index_free_list.len(),
        );

        // This assertion can fail if all entries are pending (not possible unless
        // WASM_TABLE_SIZE is set very low)
        dbg_assert!(!ctx.wasm_table_index_free_list.is_empty());
    }

    // allocate an index in the wasm table
    let wasm_table_index = ctx
        .wasm_table_index_free_list
        .pop()
        .expect("allocate wasm table index");
    dbg_assert!(wasm_table_index != WasmTableIndex(0));

    dbg_assert!(!pages.is_empty());
    dbg_assert!(pages.len() <= unsafe { MAX_PAGES } as usize);

    let basic_block_by_addr: HashMap<u64, BasicBlock> =
        basic_blocks.into_iter().map(|b| (b.addr, b)).collect();

    let entries = jit_generate_module(
        structure,
        &basic_block_by_addr,
        cpu,
        &mut ctx.wasm_builder,
        wasm_table_index,
        state_flags,
    );
    dbg_assert!(!entries.is_empty());

    let mut page_info = HashMap::default();
    for &p in &pages {
        page_info.entry(p).or_insert_with(|| PageInfo {
            wasm_table_index,
            state_flags,
            entry_points: Vec::new(),
            hidden_wasm_table_indices: Vec::new(),
        });
        get_or_create_entry_set(ctx, p);
    }
    for &(addr, state) in &entries {
        let code = page_info.get_mut(&Page::page_of(addr)).unwrap();
        code.entry_points.push((addr as u16 & 0xFFF, state));
    }

    profiler::stat_increment_by(
        stat::COMPILE_WASM_TOTAL_BYTES,
        ctx.wasm_builder.get_output_len() as u64,
    );
    profiler::stat_increment_by(stat::COMPILE_PAGE, pages.len() as u64);

    cpu::tlb_set_has_code_multiple(&pages, true);

    dbg_assert!(ctx.compiling.is_none());
    ctx.compiling = Some((
        wasm_table_index,
        CompilingPageState::Compiling { pages: page_info },
    ));

    let phys_addr = page.to_address();

    // either compiles synchronously, or calls codegen_finalize_finished asynchronously when
    // finished
    if codegen_finalize(
        wasm_table_index,
        phys_addr,
        state_flags,
        ctx.wasm_builder.get_output_ptr() as u32,
        ctx.wasm_builder.get_output_len(),
    ) {
        finish_compilation(ctx, wasm_table_index, phys_addr, state_flags);
    }

    check_jit_state_invariants(ctx);
}

#[no_mangle]
pub fn codegen_finalize_finished(
    wasm_table_index: WasmTableIndex,
    phys_addr: f64,
    state_flags: CachedStateFlags,
) {
    finish_compilation(
        &mut jit_state(),
        wasm_table_index,
        phys_addr as u64,
        state_flags,
    )
}

fn finish_compilation(
    ctx: &mut JitState,
    wasm_table_index: WasmTableIndex,
    phys_addr: u64,
    state_flags: CachedStateFlags,
) {
    dbg_assert!(wasm_table_index != WasmTableIndex(0));

    dbg_log!(
        "Finished compiling for page at {:x}",
        Page::page_of(phys_addr).to_address()
    );

    let pages = match mem::replace(&mut ctx.compiling, None) {
        None => {
            dbg_assert!(false);
            return;
        },
        Some((in_progress_wasm_table_index, CompilingPageState::CompilingWritten)) => {
            dbg_assert!(wasm_table_index == in_progress_wasm_table_index);

            profiler::stat_increment(stat::INVALIDATE_MODULE_WRITTEN_WHILE_COMPILED);
            free_wasm_table_index(ctx, wasm_table_index);
            check_jit_state_invariants(ctx);
            return;
        },
        Some((in_progress_wasm_table_index, CompilingPageState::Compiling { pages })) => {
            dbg_assert!(wasm_table_index == in_progress_wasm_table_index);
            dbg_assert!(!pages.is_empty());
            pages
        },
    };

    for i in 0..unsafe { cpu::valid_tlb_entries_count } {
        let page = unsafe { cpu::valid_tlb_entries[i as usize] } as u64;
        let entry = unsafe { cpu::tlb_pick_entry(page << 12) };
        if 0 != entry {
            let tlb_physical_page = Page::page_of(cpu::phys_of_tlb_entry(entry, page << 12));
            if let Some(info) = pages.get(&tlb_physical_page) {
                set_tlb_code(page, wasm_table_index, &info.entry_points, state_flags);
            }
        }
    }

    #[cfg(debug_assertions)]
    if CHECK_JIT_STATE_INVARIANTS {
        ctx.wasm_table_index_to_page
            .insert(wasm_table_index, pages.keys().copied().collect());
    }

    let mut check_for_unused_wasm_table_index = HashSet::default();

    for (page, mut info) in pages {
        if let Some(old_entry) = ctx.pages.remove(&page) {
            info.hidden_wasm_table_indices
                .extend(old_entry.hidden_wasm_table_indices);
            info.hidden_wasm_table_indices
                .push(old_entry.wasm_table_index);
            check_for_unused_wasm_table_index.insert(old_entry.wasm_table_index);
        }
        ctx.pages.insert(page, info);
    }

    let unused: Vec<&WasmTableIndex> = check_for_unused_wasm_table_index
        .iter()
        .filter(|&&i| ctx.pages.values().all(|page| page.wasm_table_index != i))
        .collect();

    for &index in unused {
        for p in ctx.pages.values_mut() {
            p.hidden_wasm_table_indices.retain(|&w| w != index);
        }

        dbg_log!("unused after overwrite {}", index.to_u16());
        profiler::stat_increment(stat::INVALIDATE_MODULE_UNUSED_AFTER_OVERWRITE);
        free_wasm_table_index(ctx, index);
    }

    check_jit_state_invariants(ctx);
}

/// virt_page: the (up to 36-bit) page number of the virtual page
pub fn update_tlb_code(virt_page: u64, phys_page: Page) {
    let ctx = jit_state();

    match ctx.pages.get(&phys_page) {
        Some(PageInfo {
            wasm_table_index,
            entry_points,
            state_flags,
            hidden_wasm_table_indices: _,
        }) => set_tlb_code(virt_page, *wasm_table_index, entry_points, *state_flags),
        None => cpu::clear_tlb_code(virt_page),
    };
}

/// virt_page: the (up to 36-bit) page number of the virtual page, which must be in the TLB
pub fn set_tlb_code(
    virt_page: u64,
    wasm_table_index: WasmTableIndex,
    entries: &Vec<(u16, u16)>,
    state_flags: CachedStateFlags,
) {
    // no entry points for non-executable pages: cycle_internal enters the jitted code through
    // them without fetching (and checking nx) first
    let entry = unsafe { cpu::tlb_pick_entry(virt_page << 12) } as i32;
    if entry & (cpu::TLB_VALID | cpu::TLB_NOT_EXECUTABLE) != cpu::TLB_VALID {
        cpu::clear_tlb_code(virt_page);
        return;
    }

    let slot = match unsafe { cpu::tlb_code_slot(virt_page) } {
        Some(slot) => slot,
        None => {
            dbg_assert!(false, "set_tlb_code: page not in tlb");
            return;
        },
    };
    let c = match *slot {
        None => unsafe {
            let mut c = cpu::code_alloc();
            *slot = Some(c);
            let c = c.as_mut();
            c.state_table.fill(u16::MAX);
            c.state_flags = state_flags;
            c.wasm_table_index = wasm_table_index;
            c
        },
        Some(mut c) => unsafe {
            let c = c.as_mut();
            c.state_table.fill(u16::MAX);
            c.state_flags = state_flags;
            c.wasm_table_index = wasm_table_index;
            c
        },
    };

    for &(addr, state) in entries {
        dbg_assert!(state != u16::MAX);
        c.state_table[addr as usize] = state;
    }
}

fn jit_generate_module(
    structure: Vec<WasmStructure>,
    basic_blocks: &HashMap<u64, BasicBlock>,
    mut cpu: CpuContext,
    builder: &mut WasmBuilder,
    wasm_table_index: WasmTableIndex,
    state_flags: CachedStateFlags,
) -> Vec<(u64, u16)> {
    builder.reset();

    // Modules take the guest registers as parameters of the module's function (the 64-bit jit
    // in 16 i64 parameters (register_locals64), the 32-bit jit in 8 i32 parameters
    // (register_locals)); the initial state is parameter 0. The signature doesn't vary with
    // chaining, so that tail calls between modules compiled with different chaining settings
    // are valid; chaining only decides whether the table is imported and the exit code
    // tail-calls the next module.
    let chaining =
        if state_flags.is_64() { unsafe { JIT64_CHAINING } } else { unsafe { JIT32_CHAINING } };
    if state_flags.is_64() {
        builder.set_jit64_function(chaining);
    }
    else {
        builder.set_jit32_function(chaining);
    }
    let register_locals64: Vec<WasmLocalI64> = if state_flags.is_64() {
        (0..16).map(|i| builder.arg_local_reg64(i)).collect()
    }
    else {
        Vec::new()
    };
    let mut register_locals = if state_flags.is_64() {
        Vec::new()
    }
    else {
        (0..8).map(|i| builder.arg_local_reg32(i)).collect()
    };

    builder.const_i32(0);
    let instruction_counter = builder.set_new_local();

    // the 64-bit jit keeps lazy-flag values in these locals within a basic block
    // (jit64::gen_defer_flags)
    let jit64_flag_locals = if state_flags.is_64() {
        Some(crate::jit64::FlagLocals64 {
            op1: builder.new_local_i64(),
            result: builder.new_local_i64(),
            cf: builder.new_local(),
        })
    }
    else {
        None
    };
    // ... and the data tlb cache in these (jit64::gen_tlb_cache_hit_condition)
    let jit64_tlb_cache = if state_flags.is_64() {
        let vbase = builder.new_local_i64();
        #[cfg(feature = "mem64")]
        let entry = builder.new_local_i64();
        #[cfg(not(feature = "mem64"))]
        let entry = builder.new_local();
        Some(crate::jit64::TlbCacheLocals { vbase, entry })
    }
    else {
        None
    };

    // with chaining, the exit code looks up the next entry point and continues here if it's
    // in this module (see jit64::gen_chain_to_next_module / codegen::gen_chain_to_next_module32)
    let reenter_label = if chaining { Some(builder.loop_void()) } else { None };
    let exit_label = builder.block_void();
    let exit_with_fault_label = builder.block_void();
    let main_loop_label = builder.loop_void();
    if unsafe { JIT_USE_LOOP_SAFETY } {
        builder.get_local(&instruction_counter);
        builder.const_i32(cpu::LOOP_COUNTER);
        builder.geu_i32();
        if cfg!(feature = "profiler") {
            builder.if_void();
            codegen::gen_debug_track_jit_exit(builder, 0);
            builder.br(exit_label);
            builder.block_end();
        }
        else {
            builder.br_if(exit_label);
        }
    }
    let brtable_default = builder.block_void();

    let ctx = &mut JitContext {
        cpu: &mut cpu,
        builder,
        register_locals: &mut register_locals,
        start_of_current_instruction: 0,
        exit_with_fault_label,
        exit_label,
        current_instruction: Instruction::Other,
        previous_instruction: Instruction::Other,
        instruction_counter,
        wasm_table_index,
        flags64: crate::jit64::Flags64::Unknown,
        jit64_flag_locals,
        jit64_deferred: None,
        jit64_tlb_cache,
        register_locals64,
        dirty_registers64: 0xFFFF,
        block_end: 0,
    };

    let entry_blocks = {
        let mut nodes = &structure;
        let result;
        loop {
            match &nodes[0] {
                WasmStructure::Dispatcher(e) => {
                    result = e.clone();
                    break;
                },
                WasmStructure::Loop { .. } => {
                    dbg_assert!(false);
                },
                WasmStructure::BasicBlock(_) => {
                    dbg_assert!(false);
                },
                // Note: We could use these blocks as entry points, which will yield
                // more entries for free, but it requires adding those to the dispatcher
                // It's to be investigated if this yields a performance improvement
                // See also the comment at the bottom of this function when creating entry
                // points
                WasmStructure::Block(children) => {
                    nodes = children;
                },
            }
        }
        result
    };

    let mut index_for_addr = HashMap::default();
    for (i, &addr) in entry_blocks.iter().enumerate() {
        dbg_assert!(i < 0x10000);
        index_for_addr.insert(addr, i as u16);
    }
    for b in basic_blocks.values() {
        if !index_for_addr.contains_key(&b.addr) {
            let i = index_for_addr.len();
            dbg_assert!(i < 0x10000);
            index_for_addr.insert(b.addr, i as u16);
        }
    }

    let mut label_for_addr: HashMap<u64, (Label, Option<u16>)> = HashMap::default();

    enum Work {
        WasmStructure(WasmStructure),
        BlockEnd {
            label: Label,
            targets: Vec<u64>,
            olds: HashMap<u64, (Label, Option<u16>)>,
        },
        LoopEnd {
            label: Label,
            entries: Vec<u64>,
            olds: HashMap<u64, (Label, Option<u16>)>,
        },
    }
    let mut work: VecDeque<Work> = structure
        .into_iter()
        .map(|x| Work::WasmStructure(x))
        .collect();

    while let Some(block) = work.pop_front() {
        let next_addr: Option<Vec<u64>> = work.iter().find_map(|x| match x {
            Work::WasmStructure(l) => Some(l.head().collect()),
            _ => None,
        });
        let target_block = &ctx.builder.arg_local_initial_state.unsafe_clone();

        match block {
            Work::WasmStructure(WasmStructure::BasicBlock(addr)) => {
                let block = basic_blocks.get(&addr).unwrap();
                jit_generate_basic_block(ctx, block);

                // 64-bit jit: commit the deferred lazy flags at the end of the block; the
                // conditional-jump glue instead commits in gen_condition_fn, after the
                // condition has been computed from the locals
                if !matches!(block.ty, BasicBlockType::ConditionalJump { .. }) {
                    crate::jit64::gen_commit_deferred_flags(ctx);
                }

                if block.has_sti {
                    match block.ty {
                        BasicBlockType::ConditionalJump {
                            condition,
                            jump_offset,
                            jump_offset_is_32,
                            ..
                        } => {
                            codegen::gen_set_eip_low_bits(
                                ctx.builder,
                                block.end_addr as i32 & 0xFFF,
                            );
                            codegen::gen_condition_fn(ctx, condition);
                            ctx.builder.if_void();
                            if jump_offset_is_32 {
                                codegen::gen_relative_jump_ctx(ctx, jump_offset);
                            }
                            else {
                                codegen::gen_jmp_rel16(ctx.builder, jump_offset as u16);
                            }
                            ctx.builder.block_end();
                        },
                        BasicBlockType::Normal {
                            jump_offset,
                            jump_offset_is_32,
                            ..
                        } => {
                            if jump_offset_is_32 {
                                codegen::gen_set_eip_low_bits_and_jump_rel(
                                    ctx,
                                    block.end_addr as i32 & 0xFFF,
                                    jump_offset,
                                );
                            }
                            else {
                                codegen::gen_set_eip_low_bits(
                                    ctx.builder,
                                    block.end_addr as i32 & 0xFFF,
                                );
                                codegen::gen_jmp_rel16(ctx.builder, jump_offset as u16);
                            }
                        },
                        BasicBlockType::Exit => {},
                        BasicBlockType::AbsoluteEip => {},
                    };
                    codegen::gen_debug_track_jit_exit(ctx.builder, block.last_instruction_addr);
                    codegen::gen_move_registers_from_locals_to_memory(ctx);
                    codegen::gen_fn0_const(ctx.builder, "handle_irqs");
                    codegen::gen_update_instruction_counter(ctx);
                    ctx.builder.return_();
                    continue;
                }

                match &block.ty {
                    BasicBlockType::Exit => {
                        // Exit this function
                        codegen::gen_debug_track_jit_exit(ctx.builder, block.last_instruction_addr);
                        codegen::gen_profiler_stat_increment(ctx.builder, stat::DIRECT_EXIT);
                        ctx.builder.br(ctx.exit_label);
                    },
                    BasicBlockType::AbsoluteEip if chaining => {
                        // the exit code finds the entry point, in this module or another
                        codegen::gen_debug_track_jit_exit(ctx.builder, block.last_instruction_addr);
                        ctx.builder.br(ctx.exit_label);
                    },
                    BasicBlockType::AbsoluteEip => {
                        // Check if we can stay in this module, if not exit
                        if state_flags.is_64() {
                            codegen::gen_get_eip64(ctx.builder);
                            ctx.builder.const_i32(wasm_table_index.to_u16() as i32);
                            ctx.builder.const_i32(state_flags.to_u32() as i32);
                            ctx.builder
                                .call_fn3_i64_i32_i32_ret("jit_find_cache_entry_in_page64");
                        }
                        else {
                            codegen::gen_get_eip(ctx.builder);
                            ctx.builder.const_i32(wasm_table_index.to_u16() as i32);
                            ctx.builder.const_i32(state_flags.to_u32() as i32);
                            ctx.builder.call_fn3_ret("jit_find_cache_entry_in_page");
                        }
                        ctx.builder.tee_local(target_block);
                        ctx.builder.const_i32(0);
                        ctx.builder.ge_i32();
                        // TODO: Could make this unconditional by including exit_label in the main br_table
                        ctx.builder.br_if(main_loop_label);

                        codegen::gen_debug_track_jit_exit(ctx.builder, block.last_instruction_addr);
                        ctx.builder.br(ctx.exit_label);
                    },
                    &BasicBlockType::Normal {
                        next_block_addr: None,
                        jump_offset,
                        jump_offset_is_32,
                    } => {
                        if jump_offset_is_32 {
                            codegen::gen_set_eip_low_bits_and_jump_rel(
                                ctx,
                                block.end_addr as i32 & 0xFFF,
                                jump_offset,
                            );
                        }
                        else {
                            codegen::gen_set_eip_low_bits(
                                ctx.builder,
                                block.end_addr as i32 & 0xFFF,
                            );
                            codegen::gen_jmp_rel16(ctx.builder, jump_offset as u16);
                        }

                        codegen::gen_debug_track_jit_exit(ctx.builder, block.last_instruction_addr);
                        codegen::gen_profiler_stat_increment(ctx.builder, stat::DIRECT_EXIT);
                        ctx.builder.br(ctx.exit_label);
                    },
                    &BasicBlockType::Normal {
                        next_block_addr: Some(next_block_addr),
                        jump_offset,
                        jump_offset_is_32,
                    } => {
                        // Unconditional jump to next basic block
                        // - All instructions that don't change eip
                        // - Unconditional jumps

                        if Page::page_of(next_block_addr) != Page::page_of(block.addr) {
                            if jump_offset_is_32 {
                                codegen::gen_set_eip_low_bits_and_jump_rel(
                                    ctx,
                                    block.end_addr as i32 & 0xFFF,
                                    jump_offset,
                                );
                            }
                            else {
                                codegen::gen_set_eip_low_bits(
                                    ctx.builder,
                                    block.end_addr as i32 & 0xFFF,
                                );
                                codegen::gen_jmp_rel16(ctx.builder, jump_offset as u16);
                            }

                            codegen::gen_profiler_stat_increment(
                                ctx.builder,
                                stat::NORMAL_PAGE_CHANGE,
                            );

                            codegen::gen_page_switch_check(
                                ctx,
                                next_block_addr,
                                block.last_instruction_addr,
                            );

                            #[cfg(debug_assertions)]
                            codegen::gen_check_page_switch(
                                ctx.builder,
                                block.addr,
                                next_block_addr,
                            );
                        }

                        if next_addr
                            .as_ref()
                            .map_or(false, |n| n.contains(&next_block_addr))
                        {
                            // Blocks are consecutive
                            if next_addr.unwrap().len() > 1 {
                                let target_index = *index_for_addr.get(&next_block_addr).unwrap();
                                if cfg!(feature = "profiler") {
                                    ctx.builder.const_i32(target_index.into());
                                    ctx.builder.call_fn1("debug_set_dispatcher_target");
                                }
                                ctx.builder.const_i32(target_index.into());
                                ctx.builder.set_local(target_block);
                                codegen::gen_profiler_stat_increment(
                                    ctx.builder,
                                    stat::NORMAL_FALLTHRU_WITH_TARGET_BLOCK,
                                );
                            }
                            else {
                                codegen::gen_profiler_stat_increment(
                                    ctx.builder,
                                    stat::NORMAL_FALLTHRU,
                                );
                            }
                        }
                        else {
                            let &(br, target_index) = label_for_addr.get(&next_block_addr).unwrap();
                            if let Some(target_index) = target_index {
                                if cfg!(feature = "profiler") {
                                    ctx.builder.const_i32(target_index.into());
                                    ctx.builder.call_fn1("debug_set_dispatcher_target");
                                }
                                ctx.builder.const_i32(target_index.into());
                                ctx.builder.set_local(target_block);
                                codegen::gen_profiler_stat_increment(
                                    ctx.builder,
                                    stat::NORMAL_BRANCH_WITH_TARGET_BLOCK,
                                );
                            }
                            else {
                                codegen::gen_profiler_stat_increment(
                                    ctx.builder,
                                    stat::NORMAL_BRANCH,
                                );
                            }
                            ctx.builder.br(br);
                        }
                    },
                    &BasicBlockType::ConditionalJump {
                        next_block_addr,
                        next_block_branch_taken_addr,
                        condition,
                        jump_offset,
                        jump_offset_is_32,
                    } => {
                        // Conditional jump to next basic block
                        // - jnz, jc, loop, jcxz, etc.

                        // Generate:
                        // (1) condition()
                        // (2) br_if()
                        // (3) br()
                        // Except:
                        // If we need to update eip in case (2), it's replaced by if { update_eip(); br() }
                        // If case (3) can fall through to the next basic block, the branch is eliminated
                        // Dispatcher target writes can be generated in either case
                        // Condition may be inverted if it helps generate a fallthrough instead of the second branch

                        codegen::gen_profiler_stat_increment(ctx.builder, stat::CONDITIONAL_JUMP);

                        #[derive(PartialEq)]
                        enum Case {
                            BranchTaken,
                            BranchNotTaken,
                        }

                        let mut handle_case = |case: Case, is_first| {
                            // first case generates condition and *has* to branch away,
                            // second case branches unconditionally or falls through

                            if is_first {
                                if case == Case::BranchNotTaken {
                                    codegen::gen_condition_fn_negated(ctx, condition);
                                }
                                else {
                                    codegen::gen_condition_fn(ctx, condition);
                                }
                            }

                            let next_block_addr = if case == Case::BranchTaken {
                                next_block_branch_taken_addr
                            }
                            else {
                                next_block_addr
                            };

                            if let Some(next_block_addr) = next_block_addr {
                                if Page::page_of(next_block_addr) != Page::page_of(block.addr) {
                                    dbg_assert!(case == Case::BranchTaken); // currently not possible in other case
                                    if is_first {
                                        ctx.builder.if_i32();
                                    }
                                    if jump_offset_is_32 {
                                        codegen::gen_set_eip_low_bits_and_jump_rel(
                                            ctx,
                                            block.end_addr as i32 & 0xFFF,
                                            jump_offset,
                                        );
                                    }
                                    else {
                                        codegen::gen_set_eip_low_bits(
                                            ctx.builder,
                                            block.end_addr as i32 & 0xFFF,
                                        );
                                        codegen::gen_jmp_rel16(ctx.builder, jump_offset as u16);
                                    }

                                    codegen::gen_profiler_stat_increment(
                                        ctx.builder,
                                        stat::CONDITIONAL_JUMP_PAGE_CHANGE,
                                    );
                                    codegen::gen_page_switch_check(
                                        ctx,
                                        next_block_addr,
                                        block.last_instruction_addr,
                                    );

                                    #[cfg(debug_assertions)]
                                    codegen::gen_check_page_switch(
                                        ctx.builder,
                                        block.addr,
                                        next_block_addr,
                                    );

                                    if is_first {
                                        ctx.builder.const_i32(1);
                                        ctx.builder.else_();
                                        ctx.builder.const_i32(0);
                                        ctx.builder.block_end();
                                    }
                                }

                                if next_addr
                                    .as_ref()
                                    .map_or(false, |n| n.contains(&next_block_addr))
                                {
                                    // blocks are consecutive

                                    // fallthrough, has to be second
                                    dbg_assert!(!is_first);

                                    if next_addr.as_ref().unwrap().len() > 1 {
                                        let target_index =
                                            *index_for_addr.get(&next_block_addr).unwrap();
                                        if cfg!(feature = "profiler") {
                                            ctx.builder.const_i32(target_index.into());
                                            ctx.builder.call_fn1("debug_set_dispatcher_target");
                                        }
                                        ctx.builder.const_i32(target_index.into());
                                        ctx.builder.set_local(target_block);
                                        codegen::gen_profiler_stat_increment(
                                            ctx.builder,
                                            stat::CONDITIONAL_JUMP_FALLTHRU_WITH_TARGET_BLOCK,
                                        );
                                    }
                                    else {
                                        codegen::gen_profiler_stat_increment(
                                            ctx.builder,
                                            stat::CONDITIONAL_JUMP_FALLTHRU,
                                        );
                                    }
                                }
                                else {
                                    let &(br, target_index) =
                                        label_for_addr.get(&next_block_addr).unwrap();
                                    if let Some(target_index) = target_index {
                                        if cfg!(feature = "profiler") {
                                            // Note: Currently called unconditionally, even if the
                                            // br_if below doesn't branch
                                            ctx.builder.const_i32(target_index.into());
                                            ctx.builder.call_fn1("debug_set_dispatcher_target");
                                        }
                                        ctx.builder.const_i32(target_index.into());
                                        ctx.builder.set_local(target_block);
                                    }

                                    if is_first {
                                        if cfg!(feature = "profiler") {
                                            ctx.builder.if_void();
                                            codegen::gen_profiler_stat_increment(
                                                ctx.builder,
                                                if target_index.is_some() {
                                                    stat::CONDITIONAL_JUMP_BRANCH_WITH_TARGET_BLOCK
                                                }
                                                else {
                                                    stat::CONDITIONAL_JUMP_BRANCH
                                                },
                                            );
                                            ctx.builder.br(br);
                                            ctx.builder.block_end();
                                        }
                                        else {
                                            ctx.builder.br_if(br);
                                        }
                                    }
                                    else {
                                        codegen::gen_profiler_stat_increment(
                                            ctx.builder,
                                            if target_index.is_some() {
                                                stat::CONDITIONAL_JUMP_BRANCH_WITH_TARGET_BLOCK
                                            }
                                            else {
                                                stat::CONDITIONAL_JUMP_BRANCH
                                            },
                                        );
                                        ctx.builder.br(br);
                                    }
                                }
                            }
                            else {
                                // target is outside of this module, update eip and exit
                                if is_first {
                                    ctx.builder.if_void();
                                }

                                if case == Case::BranchTaken {
                                    if jump_offset_is_32 {
                                        codegen::gen_set_eip_low_bits_and_jump_rel(
                                            ctx,
                                            block.end_addr as i32 & 0xFFF,
                                            jump_offset,
                                        );
                                    }
                                    else {
                                        codegen::gen_set_eip_low_bits(
                                            ctx.builder,
                                            block.end_addr as i32 & 0xFFF,
                                        );
                                        codegen::gen_jmp_rel16(ctx.builder, jump_offset as u16);
                                    }
                                }
                                else {
                                    codegen::gen_set_eip_low_bits(
                                        ctx.builder,
                                        block.end_addr as i32 & 0xFFF,
                                    );
                                }

                                codegen::gen_debug_track_jit_exit(
                                    ctx.builder,
                                    block.last_instruction_addr,
                                );
                                codegen::gen_profiler_stat_increment(
                                    ctx.builder,
                                    stat::CONDITIONAL_JUMP_EXIT,
                                );
                                ctx.builder.br(ctx.exit_label);

                                if is_first {
                                    ctx.builder.block_end();
                                }
                            }
                        };

                        let branch_taken_is_fallthrough = next_block_branch_taken_addr
                            .map_or(false, |addr| {
                                next_addr.as_ref().map_or(false, |n| n.contains(&addr))
                            });
                        let branch_not_taken_is_fallthrough = next_block_addr
                            .map_or(false, |addr| {
                                next_addr.as_ref().map_or(false, |n| n.contains(&addr))
                            });

                        if branch_not_taken_is_fallthrough && branch_taken_is_fallthrough {
                            let next_block_addr = next_block_addr.unwrap();
                            let next_block_branch_taken_addr =
                                next_block_branch_taken_addr.unwrap();

                            dbg_log!(
                                "Conditional control flow: fallthrough in both cases, page_switch={} next_is_multi={}",
                                Page::page_of(next_block_branch_taken_addr)
                                    != Page::page_of(block.addr),
                                next_addr.as_ref().unwrap().len() > 1,
                            );

                            dbg_assert!(
                                Page::page_of(next_block_addr) == Page::page_of(block.addr)
                            ); // currently not possible

                            if Page::page_of(next_block_branch_taken_addr)
                                != Page::page_of(block.addr)
                            {
                                codegen::gen_condition_fn(ctx, condition);
                                ctx.builder.if_void();

                                if jump_offset_is_32 {
                                    codegen::gen_set_eip_low_bits_and_jump_rel(
                                        ctx,
                                        block.end_addr as i32 & 0xFFF,
                                        jump_offset,
                                    );
                                }
                                else {
                                    codegen::gen_set_eip_low_bits(
                                        ctx.builder,
                                        block.end_addr as i32 & 0xFFF,
                                    );
                                    codegen::gen_jmp_rel16(ctx.builder, jump_offset as u16);
                                }

                                codegen::gen_profiler_stat_increment(
                                    ctx.builder,
                                    stat::CONDITIONAL_JUMP_PAGE_CHANGE,
                                );
                                codegen::gen_page_switch_check(
                                    ctx,
                                    next_block_branch_taken_addr,
                                    block.last_instruction_addr,
                                );

                                #[cfg(debug_assertions)]
                                codegen::gen_check_page_switch(
                                    ctx.builder,
                                    block.addr,
                                    next_block_branch_taken_addr,
                                );

                                dbg_assert!(next_addr.unwrap().len() > 1);

                                let target_index_taken =
                                    *index_for_addr.get(&next_block_branch_taken_addr).unwrap();
                                let target_index_not_taken =
                                    *index_for_addr.get(&next_block_addr).unwrap();

                                ctx.builder.const_i32(target_index_taken.into());
                                ctx.builder.set_local(target_block);

                                ctx.builder.else_();
                                ctx.builder.const_i32(target_index_not_taken.into());
                                ctx.builder.set_local(target_block);

                                ctx.builder.block_end();
                            }
                            else if next_addr.unwrap().len() > 1 {
                                let target_index_taken =
                                    *index_for_addr.get(&next_block_branch_taken_addr).unwrap();
                                let target_index_not_taken =
                                    *index_for_addr.get(&next_block_addr).unwrap();

                                codegen::gen_condition_fn(ctx, condition);
                                ctx.builder.if_i32();
                                ctx.builder.const_i32(target_index_taken.into());
                                ctx.builder.else_();
                                ctx.builder.const_i32(target_index_not_taken.into());
                                ctx.builder.block_end();
                                ctx.builder.set_local(target_block);
                            }
                            else {
                                // Both targets are the one block that follows (a jcc to the
                                // next instruction): no condition is generated, so the 64-bit
                                // jit's deferred lazy flags, otherwise committed by
                                // gen_condition_fn, are committed here (issue #16). Like every
                                // block, the following one starts with the flags in memory: it
                                // may also be entered from elsewhere
                                crate::jit64::gen_commit_deferred_flags(ctx);
                            }
                        }
                        else if branch_taken_is_fallthrough {
                            handle_case(Case::BranchNotTaken, true);
                            handle_case(Case::BranchTaken, false);
                        }
                        else {
                            handle_case(Case::BranchTaken, true);
                            handle_case(Case::BranchNotTaken, false);
                        }
                    },
                }
            },
            Work::WasmStructure(WasmStructure::Dispatcher(entries)) => {
                profiler::stat_increment(stat::COMPILE_DISPATCHER);

                if cfg!(feature = "profiler") {
                    ctx.builder.get_local(target_block);
                    ctx.builder.const_i32(index_for_addr.len() as i32);
                    ctx.builder.call_fn2("check_dispatcher_target");
                }

                if entries.len() > BRTABLE_CUTOFF {
                    // generate a brtable
                    codegen::gen_profiler_stat_increment(ctx.builder, stat::DISPATCHER_LARGE);
                    let mut cases = Vec::new();
                    for &addr in &entries {
                        let &(label, target_index) = label_for_addr.get(&addr).unwrap();
                        let &index = index_for_addr.get(&addr).unwrap();
                        dbg_assert!(target_index.is_none() || target_index == Some(index));
                        while index as usize >= cases.len() {
                            cases.push(brtable_default);
                        }
                        cases[index as usize] = label;
                    }
                    ctx.builder.get_local(target_block);
                    ctx.builder.brtable(brtable_default, &mut cases.iter());
                }
                else {
                    // generate a if target == block.addr then br block.label ...
                    codegen::gen_profiler_stat_increment(ctx.builder, stat::DISPATCHER_SMALL);
                    let nexts: HashSet<u64> = next_addr
                        .as_ref()
                        .map_or(HashSet::default(), |nexts| nexts.iter().copied().collect());
                    for &addr in &entries {
                        if nexts.contains(&addr) {
                            continue;
                        }
                        let index = *index_for_addr.get(&addr).unwrap();
                        let &(label, _) = label_for_addr.get(&addr).unwrap();
                        ctx.builder.get_local(target_block);
                        ctx.builder.const_i32(index.into());
                        ctx.builder.eq_i32();
                        ctx.builder.br_if(label);
                    }
                }
            },
            Work::WasmStructure(WasmStructure::Loop(children)) => {
                profiler::stat_increment(stat::COMPILE_WASM_LOOP);

                let entries: Vec<u64> = children[0].head().collect();
                let label = ctx.builder.loop_void();
                codegen::gen_profiler_stat_increment(ctx.builder, stat::LOOP);

                if entries.len() == 1 {
                    let addr = entries[0];
                    codegen::gen_set_eip_low_bits(ctx.builder, addr as i32 & 0xFFF);
                    profiler::stat_increment(stat::COMPILE_WITH_LOOP_SAFETY);
                    codegen::gen_profiler_stat_increment(ctx.builder, stat::LOOP_SAFETY);
                    if unsafe { JIT_USE_LOOP_SAFETY } {
                        ctx.builder.get_local(&ctx.instruction_counter);
                        ctx.builder.const_i32(cpu::LOOP_COUNTER);
                        ctx.builder.geu_i32();
                        if cfg!(feature = "profiler") {
                            ctx.builder.if_void();
                            codegen::gen_debug_track_jit_exit(ctx.builder, addr);
                            ctx.builder.br(exit_label);
                            ctx.builder.block_end();
                        }
                        else {
                            ctx.builder.br_if(exit_label);
                        }
                    }
                }

                let mut olds = HashMap::default();
                for &target in entries.iter() {
                    let index = if entries.len() == 1 {
                        None
                    }
                    else {
                        Some(*index_for_addr.get(&target).unwrap())
                    };
                    let old = label_for_addr.insert(target, (label, index));
                    if let Some(old) = old {
                        olds.insert(target, old);
                    }
                }

                work.push_front(Work::LoopEnd {
                    label,
                    entries,
                    olds,
                });
                for c in children.into_iter().rev() {
                    work.push_front(Work::WasmStructure(c));
                }
            },
            Work::LoopEnd {
                label,
                entries,
                olds,
            } => {
                for target in entries {
                    let old = label_for_addr.remove(&target);
                    dbg_assert!(old.map(|(l, _)| l) == Some(label));
                }
                for (target, old) in olds {
                    let old = label_for_addr.insert(target, old);
                    dbg_assert!(old.is_none());
                }

                ctx.builder.block_end();
            },
            Work::WasmStructure(WasmStructure::Block(children)) => {
                profiler::stat_increment(stat::COMPILE_WASM_BLOCK);

                let targets = next_addr.clone().unwrap();
                let label = ctx.builder.block_void();
                let mut olds = HashMap::default();
                for &target in targets.iter() {
                    let index = if targets.len() == 1 {
                        None
                    }
                    else {
                        Some(*index_for_addr.get(&target).unwrap())
                    };
                    let old = label_for_addr.insert(target, (label, index));
                    if let Some(old) = old {
                        olds.insert(target, old);
                    }
                }

                work.push_front(Work::BlockEnd {
                    label,
                    targets,
                    olds,
                });
                for c in children.into_iter().rev() {
                    work.push_front(Work::WasmStructure(c));
                }
            },
            Work::BlockEnd {
                label,
                targets,
                olds,
            } => {
                for target in targets {
                    let old = label_for_addr.remove(&target);
                    dbg_assert!(old.map(|(l, _)| l) == Some(label));
                }
                for (target, old) in olds {
                    let old = label_for_addr.insert(target, old);
                    dbg_assert!(old.is_none());
                }

                ctx.builder.block_end();
            },
        }
    }

    dbg_assert!(label_for_addr.is_empty());

    {
        ctx.builder.block_end(); // default case for the brtable
        ctx.builder.unreachable();
    }
    {
        ctx.builder.block_end(); // main loop
    }
    {
        // exit with exception or due to smc
        ctx.builder.block_end();
        crate::jit64::gen_commit_deferred_flags(ctx);
        codegen::gen_move_registers_from_locals_to_memory(ctx);
        codegen::gen_fn0_const(ctx.builder, "exit_jit");
        codegen::gen_update_instruction_counter(ctx);
        ctx.builder.return_();
    }
    {
        // exit
        ctx.builder.block_end();
        crate::jit64::gen_commit_deferred_flags(ctx);
        if let Some(reenter_label) = reenter_label {
            codegen::gen_update_instruction_counter(ctx);
            if state_flags.is_64() {
                crate::jit64::gen_chain_to_next_module(ctx, reenter_label);
            }
            else {
                codegen::gen_chain_to_next_module32(ctx, reenter_label);
            }
            codegen::gen_move_registers_from_locals_to_memory(ctx);
            ctx.builder.block_end(); // reenter_label
        }
        else {
            codegen::gen_move_registers_from_locals_to_memory(ctx);
            codegen::gen_update_instruction_counter(ctx);
        }
    }

    // the 32-bit registers are parameters of the module's function, not allocated locals
    ctx.register_locals.clear();
    if let Some(fl) = ctx.jit64_flag_locals.take() {
        ctx.builder.free_local_i64(fl.op1);
        ctx.builder.free_local_i64(fl.result);
        ctx.builder.free_local(fl.cf);
    }
    if let Some(c) = ctx.jit64_tlb_cache.take() {
        ctx.builder.free_local_i64(c.vbase);
        #[cfg(feature = "mem64")]
        ctx.builder.free_local_i64(c.entry);
        #[cfg(not(feature = "mem64"))]
        ctx.builder.free_local(c.entry);
    }
    // parameters, not allocated locals
    ctx.register_locals64.clear();
    ctx.builder
        .free_local(ctx.instruction_counter.unsafe_clone());

    ctx.builder.finish();

    let entries = Vec::from_iter(entry_blocks.iter().map(|addr| {
        let block = basic_blocks.get(&addr).unwrap();
        let index = *index_for_addr.get(&addr).unwrap();

        profiler::stat_increment(stat::COMPILE_ENTRY_POINT);

        dbg_assert!(block.addr < block.end_addr);
        // Note: We also insert blocks that weren't originally marked as entries here
        //       This doesn't have any downside, besides making the hash table slightly larger

        (block.addr, index)
    }));

    for b in basic_blocks.values() {
        if b.is_entry_block {
            dbg_assert!(entries.iter().find(|(addr, _)| *addr == b.addr).is_some());
        }
    }

    return entries;
}

fn jit_generate_basic_block(ctx: &mut JitContext, block: &BasicBlock) {
    let needs_eip_updated = match block.ty {
        BasicBlockType::Exit => true,
        _ => false,
    };

    profiler::stat_increment(stat::COMPILE_BASIC_BLOCK);

    let start_addr = block.addr;
    let last_instruction_addr = block.last_instruction_addr;
    let stop_addr = block.end_addr;

    // First iteration of do-while assumes the caller confirms this condition
    dbg_assert!(!is_near_end_of_page(start_addr));

    if cfg!(feature = "profiler") {
        ctx.builder.const_i64(start_addr as i64);
        ctx.builder.call_fn1_i64("enter_basic_block");
    }

    ctx.builder.get_local(&ctx.instruction_counter);
    ctx.builder.const_i32(block.number_of_instructions as i32);
    ctx.builder.add_i32();
    ctx.builder.set_local(&ctx.instruction_counter);

    ctx.cpu.eip = start_addr;
    ctx.current_instruction = Instruction::Other;
    ctx.previous_instruction = Instruction::Other;
    ctx.flags64 = crate::jit64::Flags64::Unknown;
    // deferred lazy flags are committed at the end of every block (below and in the
    // conditional-jump glue), so a block always starts with a clean state
    dbg_assert!(ctx.jit64_deferred.is_none());
    ctx.dirty_registers64 = 0xFFFF;
    ctx.block_end = stop_addr;

    loop {
        let mut instruction = 0;
        if cfg!(feature = "profiler") {
            instruction = memory::read32s(ctx.cpu.eip) as u32;
            opstats::gen_opstats(ctx.builder, instruction);
            opstats::record_opstat_compiled(instruction);
        }

        if ctx.cpu.eip == last_instruction_addr {
            // Before the last instruction:
            // - Set eip to *after* the instruction
            // - Set previous_eip to *before* the instruction
            if needs_eip_updated {
                codegen::gen_set_previous_eip_offset_from_eip_with_low_bits_ctx(
                    ctx,
                    last_instruction_addr as i32 & 0xFFF,
                );
                codegen::gen_set_eip_low_bits(ctx.builder, stop_addr as i32 & 0xFFF);
            }
        }

        let wasm_length_before = ctx.builder.instruction_body_length();

        ctx.start_of_current_instruction = ctx.cpu.eip;
        let start_eip = ctx.cpu.eip;
        let mut instruction_flags = 0;
        if ctx.cpu.is_64() {
            crate::jit64::jit_instruction(ctx, &mut instruction_flags);
        }
        else {
            jit_instructions::jit_instruction(ctx, &mut instruction_flags);
        }
        let end_eip = ctx.cpu.eip;

        let instruction_length = end_eip - start_eip;
        let was_block_boundary = instruction_flags & JIT_INSTR_BLOCK_BOUNDARY_FLAG != 0;

        let wasm_length = ctx.builder.instruction_body_length() - wasm_length_before;
        opstats::record_opstat_size_wasm(instruction, wasm_length as u64);

        dbg_assert!((end_eip == stop_addr) == (start_eip == last_instruction_addr));
        dbg_assert!(instruction_length < MAX_INSTRUCTION_LENGTH as u64);

        let end_addr = ctx.cpu.eip;

        if end_addr == stop_addr {
            // no page was crossed
            dbg_assert!(Page::page_of(end_addr) == Page::page_of(start_addr));
            break;
        }

        if was_block_boundary || is_near_end_of_page(end_addr) || end_addr > stop_addr {
            dbg_log!(
                "Overlapping basic blocks start={:x} expected_end={:x} end={:x} was_block_boundary={} near_end_of_page={}",
                start_addr,
                stop_addr,
                end_addr,
                was_block_boundary,
                is_near_end_of_page(end_addr)
            );
            dbg_assert!(false);
            break;
        }

        ctx.previous_instruction = mem::replace(&mut ctx.current_instruction, Instruction::Other);
    }
}

pub fn jit_increase_hotness_and_maybe_compile(
    virt_address: u64,
    phys_address: u64,
    cs_offset: u32,
    state_flags: CachedStateFlags,
    heat: u32,
) {
    if unsafe { JIT_DISABLED } {
        return;
    }

    if state_flags.is_64() && unsafe { JIT64_DISABLED } {
        return;
    }

    // breakpoints/single-stepping are only checked by the interpreter;
    // don't compile code while they're armed
    if unsafe { cpu::debug_mode_active() } {
        return;
    }

    let mut ctx = jit_state();
    let is_compiling = ctx.compiling.is_some();
    let page = Page::page_of(phys_address);
    let (entry_points, created) = get_or_create_entry_set(&mut ctx, page);
    if created {
        cpu::tlb_set_has_code(page, true);
        profiler::stat_increment(stat::RUN_INTERPRETED_NEW_PAGE);
    }

    if !is_near_end_of_page(phys_address) {
        entry_points.insert(phys_address as u16 & 0xFFF);
    }

    let hotness = hotness_mut(&mut ctx, page);
    *hotness = hotness.saturating_add(heat);
    if *hotness >= unsafe { JIT_HOTNESS_THRESHOLD } {
        if is_compiling {
            return;
        }
        // only try generating if we're in the correct address space
        if cpu::translate_address_read_no_side_effects(virt_address) == Ok(phys_address) {
            reset_hotness(&mut ctx, page);
            jit_analyze_and_generate(&mut ctx, virt_address, phys_address, cs_offset, state_flags)
        }
        else {
            profiler::stat_increment(stat::COMPILE_WRONG_ADDRESS_SPACE);
        }
    }
}

fn free_wasm_table_index(ctx: &mut JitState, wasm_table_index: WasmTableIndex) {
    if CHECK_JIT_STATE_INVARIANTS {
        dbg_assert!(!ctx.wasm_table_index_free_list.contains(&wasm_table_index));

        match &ctx.compiling {
            Some((wasm_table_index_compiling, _)) => {
                dbg_assert!(
                    *wasm_table_index_compiling != wasm_table_index,
                    "Attempt to free wasm table index that is currently being compiled"
                );
            },
            _ => {},
        }

        dbg_assert!(!ctx
            .pages
            .values()
            .any(|info| info.wasm_table_index == wasm_table_index));

        dbg_assert!(!ctx
            .pages
            .values()
            .any(|info| info.hidden_wasm_table_indices.contains(&wasm_table_index)));

        for i in 0..unsafe { cpu::valid_tlb_entries_count } {
            let page = unsafe { cpu::valid_tlb_entries[i as usize] };
            unsafe {
                // tlb_code_slot covers the low (flat) table and the high
                // (hashed) table — kernel pages live above 4 GiB
                match cpu::tlb_code_slot(page) {
                    None => {},
                    Some(slot) => match slot {
                        None => {},
                        Some(c) => {
                            dbg_assert!(c.as_ref().wasm_table_index != wasm_table_index);
                        },
                    },
                }
            }
        }
    }

    #[cfg(debug_assertions)]
    ctx.wasm_table_index_to_page.remove(&wasm_table_index);

    ctx.wasm_table_index_free_list.push(wasm_table_index);

    // It is not strictly necessary to clear the function, but it will fail more predictably if we
    // accidentally use the function and may garbage collect unused modules earlier
    jit_clear_func(wasm_table_index);
}

/// Register a write in this page: Delete all present code
fn jit_dirty_page_ctx(ctx: &mut JitState, page: Page) {
    // Fast path: a page without an entry-point set holds no JIT state at all — no hotness
    // counter (created together with the set), no PageInfo (compiled pages keep their set) and
    // it cannot be under compilation (compiling pages hold sets) — so dirtying it can't
    // invalidate anything. This covers virtually all writes: data pages, page-table walks'
    // accessed/dirty-bit updates, DMA, ... Pages outside the flat table's coverage take the
    // general path (their state lives in the fallback maps).
    if let Some(i) = page_slot(&ctx.page_states, page) {
        if ctx.page_states[i].entry_set == 0 {
            profiler::stat_increment(stat::DIRTY_PAGE_DID_NOT_HAVE_CODE);
            return;
        }
    }

    let mut did_have_code = false;

    if let Some(PageInfo {
        wasm_table_index,
        hidden_wasm_table_indices,
        state_flags: _,
        entry_points: _,
    }) = ctx.pages.remove(&page)
    {
        profiler::stat_increment(stat::INVALIDATE_PAGE_HAD_CODE);
        did_have_code = true;

        free(ctx, wasm_table_index);
        for wasm_table_index in hidden_wasm_table_indices {
            free(ctx, wasm_table_index);
        }

        fn free(ctx: &mut JitState, wasm_table_index: WasmTableIndex) {
            ctx.pages.retain(|_, info| {
                if info.wasm_table_index != wasm_table_index {
                    return true;
                }
                match info.hidden_wasm_table_indices.pop() {
                    Some(new_primary) => {
                        info.wasm_table_index = new_primary;
                        info.entry_points.clear();
                        true
                    },
                    None => false,
                }
            });

            for info in ctx.pages.values_mut() {
                info.hidden_wasm_table_indices
                    .retain(|&w| w != wasm_table_index)
            }

            for i in 0..unsafe { cpu::valid_tlb_entries_count } {
                let page = unsafe { cpu::valid_tlb_entries[i as usize] };
                let entry = unsafe { cpu::tlb_pick_entry(page << 12) };
                if 0 != entry {
                    let tlb_physical_page =
                        Page::page_of(cpu::phys_of_tlb_entry(entry, page << 12));
                    let slot = match unsafe { cpu::tlb_code_slot(page) } {
                        Some(slot) => slot,
                        None => continue,
                    };
                    match *slot {
                        None => {},
                        Some(c) => unsafe {
                            let w = c.as_ref().wasm_table_index;
                            if wasm_table_index == w {
                                cpu::code_free(c);
                                *slot = None;
                                if !has_entry_points(ctx, tlb_physical_page)
                                    && !ctx.pages.contains_key(&tlb_physical_page)
                                {
                                    cpu::tlb_put_entry(
                                        page << 12,
                                        entry & !(cpu::TLB_HAS_CODE as u64),
                                    );
                                }
                            }
                        },
                    }
                }
            }

            free_wasm_table_index(ctx, wasm_table_index);
        }
    }

    if clear_page_hotness_and_entry_points(ctx, page) {
        profiler::stat_increment(stat::INVALIDATE_PAGE_HAD_ENTRY_POINTS);
        did_have_code = true;

        match &ctx.compiling {
            Some((index, CompilingPageState::Compiling { pages })) => {
                if pages.contains_key(&page) {
                    ctx.compiling = Some((*index, CompilingPageState::CompilingWritten));
                }
            },
            _ => {},
        }
    }

    match &ctx.compiling {
        Some((_, CompilingPageState::Compiling { pages })) => {
            dbg_assert!(!pages.contains_key(&page));
        },
        _ => {},
    }

    check_jit_state_invariants(ctx);

    dbg_assert!(!jit_page_has_code_ctx(ctx, page));

    if did_have_code {
        cpu::tlb_set_has_code(page, false);
        // compiled 64-bit code may be running and must not continue after this instruction
        unsafe { crate::jit64::JIT64_EXIT = true };
    }

    if !did_have_code {
        profiler::stat_increment(stat::DIRTY_PAGE_DID_NOT_HAVE_CODE);
    }
}

pub fn jit_dirty_cache(start_addr: u64, end_addr: u64) {
    dbg_assert!(start_addr < end_addr);

    let start_page = Page::page_of(start_addr);
    let end_page = Page::page_of(end_addr - 1);

    for page in start_page.to_u32()..end_page.to_u32() + 1 {
        jit_dirty_page_ctx(&mut jit_state(), Page::of_u32(page));
    }
}
/// Physical addresses as f64: may exceed 32 bits (and end_addr may be exactly 4 GiB)
#[export_name = "jit_dirty_cache"]
pub fn jit_dirty_cache_js(start_addr: f64, end_addr: f64) {
    jit_dirty_cache(start_addr as u64, end_addr as u64)
}

#[no_mangle]
pub fn jit_dirty_page(page: Page) { jit_dirty_page_ctx(&mut jit_state(), page) }

/// dirty pages in the range of start_addr and end_addr, which must span at most two pages
pub fn jit_dirty_cache_small(start_addr: u64, end_addr: u64) {
    dbg_assert!(start_addr < end_addr);

    let start_page = Page::page_of(start_addr);
    let end_page = Page::page_of(end_addr - 1);

    let mut ctx = jit_state();
    jit_dirty_page_ctx(&mut ctx, start_page);

    // Note: This can't happen when paging is enabled, as writes across
    //       boundaries are split up on two pages
    if start_page != end_page {
        dbg_assert!(start_page.to_u32() + 1 == end_page.to_u32());
        jit_dirty_page_ctx(&mut ctx, end_page);
    }
}

#[no_mangle]
pub fn jit_clear_cache_js() { jit_clear_cache(&mut jit_state()) }

fn jit_clear_cache(ctx: &mut JitState) {
    let mut pages_with_code = HashSet::default();

    for entry_set in ctx.entry_sets.iter().flatten() {
        pages_with_code.insert(entry_set.page);
    }
    for &p in ctx.fallback_entry_points.keys() {
        pages_with_code.insert(p);
    }
    for &p in ctx.pages.keys() {
        pages_with_code.insert(p);
    }

    for page in pages_with_code {
        jit_dirty_page_ctx(ctx, page);
    }
}

pub fn jit_page_has_code(page: Page) -> bool { jit_page_has_code_ctx(&mut jit_state(), page) }

fn jit_page_has_code_ctx(ctx: &JitState, page: Page) -> bool {
    ctx.pages.contains_key(&page) || has_entry_points(ctx, page)
}

pub fn jit_page_has_wasm_table_index(page: Page, wasm_table_index: u16) -> bool {
    let ctx = jit_state();
    match ctx.pages.get(&page) {
        Some(info) => {
            info.wasm_table_index.to_u16() == wasm_table_index
                || info
                    .hidden_wasm_table_indices
                    .iter()
                    .any(|w| w.to_u16() == wasm_table_index)
        },
        None => false,
    }
}

#[no_mangle]
pub fn jit_get_wasm_table_index_free_list_count() -> u32 {
    if cfg!(feature = "profiler") {
        jit_state().wasm_table_index_free_list.len() as u32
    }
    else {
        0
    }
}
#[no_mangle]
pub fn jit_get_cache_size() -> u32 {
    if cfg!(feature = "profiler") {
        jit_state()
            .pages
            .values()
            .map(|p| p.entry_points.len() as u32)
            .sum()
    }
    else {
        0
    }
}

#[cfg(feature = "profiler")]
pub fn check_missed_entry_points(phys_address: u64, state_flags: CachedStateFlags) {
    let ctx = jit_state();

    if let Some(infos) = ctx.pages.get(&Page::page_of(phys_address)) {
        if infos.state_flags != state_flags {
            return;
        }

        #[allow(static_mut_refs)]
        let last_jump_type = unsafe { cpu::debug_last_jump.name() };
        #[allow(static_mut_refs)]
        let last_jump_addr = unsafe { cpu::debug_last_jump.phys_address() }.unwrap_or(0);
        let last_jump_opcode =
            if last_jump_addr != 0 { memory::read32s(last_jump_addr) } else { 0 };

        let opcode = memory::read32s(phys_address);
        dbg_log!(
            "Compiled exists, but no entry point, \
                 phys_addr={:x} opcode={:02x} {:02x} {:02x} {:02x}. \
                 Last jump at {:x} ({}) opcode={:02x} {:02x} {:02x} {:02x}",
            phys_address,
            opcode & 0xFF,
            opcode >> 8 & 0xFF,
            opcode >> 16 & 0xFF,
            opcode >> 16 & 0xFF,
            last_jump_addr,
            last_jump_type,
            last_jump_opcode & 0xFF,
            last_jump_opcode >> 8 & 0xFF,
            last_jump_opcode >> 16 & 0xFF,
            last_jump_opcode >> 16 & 0xFF,
        );
    }
}

#[no_mangle]
#[cfg(feature = "profiler")]
pub fn debug_set_dispatcher_target(_target_index: i32) {
    //dbg_log!("About to call dispatcher target_index={}", target_index);
}

#[no_mangle]
#[cfg(feature = "profiler")]
pub fn check_dispatcher_target(target_index: i32, max: i32) {
    //dbg_log!("Dispatcher called target={}", target_index);
    dbg_assert!(target_index >= 0);
    dbg_assert!(target_index < max);
}

#[no_mangle]
#[cfg(feature = "profiler")]
pub fn enter_basic_block(phys_eip: u64) {
    let eip = unsafe {
        cpu::translate_address_read(*crate::cpu::global_pointers::instruction_pointer).unwrap()
    };
    if Page::page_of(eip) != Page::page_of(phys_eip) {
        dbg_log!(
            "enter basic block failed block=0x{:x} actual eip=0x{:x}",
            phys_eip,
            eip
        );
        panic!();
    }
}

#[no_mangle]
pub unsafe fn set_jit_config(index: u32, value: u32) {
    match index {
        0 => JIT_DISABLED = value != 0,
        1 => MAX_PAGES = value,
        2 => JIT_USE_LOOP_SAFETY = value != 0,
        3 => MAX_EXTRA_BASIC_BLOCKS = value,
        4 => JIT64_DISABLED = value != 0,
        5 => JIT_HOTNESS_THRESHOLD = value,
        6 => crate::jit64::PROFILE_GENERIC = value != 0,
        7 => JIT64_CHAINING = value != 0,
        8 => crate::jit64::JIT64_DEFER_FLAGS = value != 0,
        9 => crate::jit64::JIT64_TLB_CACHE = value != 0,
        10 => JIT32_CHAINING = value != 0,
        _ => dbg_assert!(false),
    }
}

#[no_mangle]
pub unsafe fn get_jit_config(index: u32) -> u32 {
    match index {
        0 => JIT_DISABLED as u32,
        1 => MAX_PAGES as u32,
        2 => JIT_USE_LOOP_SAFETY as u32,
        3 => MAX_EXTRA_BASIC_BLOCKS as u32,
        4 => JIT64_DISABLED as u32,
        5 => JIT_HOTNESS_THRESHOLD,
        7 => JIT64_CHAINING as u32,
        8 => crate::jit64::JIT64_DEFER_FLAGS as u32,
        9 => crate::jit64::JIT64_TLB_CACHE as u32,
        10 => JIT32_CHAINING as u32,
        _ => 0,
    }
}
