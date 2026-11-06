#!/usr/bin/env node
// Compiles every opcode of gen/x86_table.js with the 64-bit jit, in many encodings (register and
// memory operands, operand size and REX prefixes, legacy prefixes), and checks that every
// generated module validates and instantiates. A module the host refuses ends the emulator with an
// uncaught CompileError, as in issue #8: the register form of the store-form xmm moves (0F 29,
// 0F 11, 66 0F 7F, F3 0F 7F with mod=11) pushed the operands of a wasm store in the wrong order.
//
// The code isn't run: jit_force_generate (debug builds only) compiles the page while the cpu is in
// 64-bit state, at cpl 0 and at cpl 3. JIT64_VALIDATE_FULL=1 tries more REX/prefix combinations.

import url from "node:url";
import path from "node:path";

const __dirname = url.fileURLToPath(new URL(".", import.meta.url));
const root_path = path.join(__dirname, "..", "..");

process.on("unhandledRejection", exn => { throw exn; });

if(+process.env.TEST_RELEASE_BUILD)
{
    console.log("jit_force_generate is only in debug builds, test skipped");
    process.exit(0);
}

const { V86 } = await import(root_path + "/src/main.js");
const { default: x86_table } = await import(root_path + "/gen/x86_table.js");

const FULL = +process.env.JIT64_VALIDATE_FULL;

const REX_PREFIXES = FULL ?
    [null, 0x40, 0x41, 0x42, 0x44, 0x45, 0x48, 0x49, 0x4C, 0x4D, 0x4F] :
    [null, 0x41, 0x44, 0x48, 0x4F];
const LEGACY_PREFIXES = FULL ?
    [[], [0x66], [0xF3], [0xF2], [0xF0], [0x64], [0x65], [0x67]] :
    [[], [0x66], [0xF3], [0xF2], [0xF0]];
// 64-bit state at cpl 0, and as in issue #8 (state flags 0x16: cpl 3, 32-bit stack)
const STATES = [{ cpl: 0, stack_size_32: 0 }, { cpl: 3, stack_size_32: 1 }];

function opcode_bytes(e)
{
    const bytes = [];
    if(e.opcode > 0xFFFF) bytes.push(e.opcode >> 16 & 0xFF);
    if(e.opcode > 0xFF) bytes.push(e.opcode >> 8 & 0xFF);
    if(e.map) bytes.push(e.map);
    bytes.push(e.opcode & 0xFF);
    return bytes;
}

function immediate_length(e, operand_size)
{
    let length = 0;
    if(e.imm8 || e.imm8s) length += 1;
    else if(e.immaddr) length += 8;
    else if(e.imm16) length += 2;
    else if(e.imm3264 && operand_size === 64) length += 8;
    else if(e.imm1632) length += operand_size === 16 ? 2 : 4;
    else if(e.imm32) length += 4;
    if(e.extra_imm8) length += 1;
    if(e.extra_imm16) length += 2;
    return length;
}

function modrm_forms(e)
{
    const reg = e.fixed_g === undefined ? 2 : e.fixed_g;
    const forms = [
        [0xC0 | reg << 3 | 1],                                // register
        [0xC0 | reg << 3 | 4],                                // register (rsp/xmm4/...)
        [0x00 | reg << 3 | 1],                                // [rcx]
        [0x40 | reg << 3 | 5, 0x10],                          // [rbp+0x10]
        [0x00 | reg << 3 | 5, 0x00, 0x01, 0x00, 0x00],        // [rip+0x100]
        [0x44 | reg << 3, 0x24, 0x08],                        // [rsp+8]
        [0x84 | reg << 3, 0x88, 0x00, 0x02, 0x00, 0x00],      // [rax+rcx*4+0x200]
        [0x04 | reg << 3, 0x25, 0x00, 0x03, 0x00, 0x00],      // [0x300]
    ];
    if(e.fixed_g === undefined)
    {
        forms.push([0xC0 | 7 << 3 | 0], [0xC0 | 0 << 3 | 7]);
    }
    return forms;
}

function* encodings()
{
    for(const e of x86_table)
    {
        if(e.prefix) continue;
        const bytes = opcode_bytes(e);
        // 66/F2/F3 that select the instruction come before REX
        const mandatory = e.opcode > 0xFFFF || e.opcode > 0xFF && (e.opcode >> 8) !== 0x0F ? [bytes.shift()] : [];
        const legacy_sets = e.sse || mandatory.length ? [[]] : LEGACY_PREFIXES;
        for(const legacy of legacy_sets)
        {
            for(const rex of REX_PREFIXES)
            {
                const operand_size = rex !== null && rex & 8 && e.os ? 64 : legacy.includes(0x66) ? 16 : 32;
                const imm = new Array(immediate_length(e, operand_size)).fill(0x01);
                const prefixes = [...legacy, ...mandatory, ...(rex === null ? [] : [rex])];
                for(const modrm of e.e ? modrm_forms(e) : [[]])
                {
                    yield { e, bytes: [...prefixes, ...bytes, ...modrm, ...imm] };
                }
            }
        }
    }
}

// Issue #16: a jcc to the next instruction after a flag-setting instruction. With both targets the
// following block, the jcc generated no condition, and the lazy flags the 64-bit jit keeps in wasm
// locals weren't committed at the end of the block (dbg_assert jit64_deferred.is_none() at the
// start of the next one). Each instruction below, followed by each jcc +0 (rel8 and rel32), is
// compiled too. They leave different deferred lazy-flag states (operation, operand size, carry
// kept by inc/dec, flags cleared by logic operations, memory operand), or none (adc, sbb and shl
// store their flags eagerly, nop sets none)
const FLAG_SETTERS = [
    [0x3A, 0xE3],                               // cmp ah, bl (issue #16)
    [0x01, 0xD6],                               // add esi, edx (issue #16)
    [0x66, 0x29, 0xD6],                         // sub si, dx
    [0x48, 0x39, 0xD6],                         // cmp rsi, rdx
    [0x48, 0x3D, 0x01, 0x00, 0x00, 0x00],       // cmp rax, 1
    [0x48, 0x83, 0xC6, 0x01],                   // add rsi, 1
    [0x48, 0x39, 0x06],                         // cmp [rsi], rax
    [0x48, 0x11, 0xD6],                         // adc rsi, rdx
    [0x19, 0xD6],                               // sbb esi, edx
    [0x48, 0xF7, 0xDE],                         // neg rsi
    [0x30, 0xD6],                               // xor dh, dl
    [0x09, 0xD6],                               // or esi, edx
    [0x48, 0x21, 0xD6],                         // and rsi, rdx
    [0x48, 0x85, 0xD6],                         // test rsi, rdx
    [0xFE, 0xC0],                               // inc al
    [0xFF, 0xC6],                               // inc esi
    [0x48, 0xFF, 0xCE],                         // dec rsi
    [0x48, 0x39, 0xD6, 0xFF, 0xC6],             // cmp rsi, rdx; inc esi
    [0x48, 0xD1, 0xE6],                         // shl rsi, 1
    [0x90],                                     // nop
];

function* flag_jcc_sequences()
{
    for(const setter of FLAG_SETTERS)
    {
        for(let cc = 0; cc < 16; cc++)
        {
            yield { label: "flags + jcc +0 (rel8)", bytes: [...setter, 0x70 | cc, 0x00] };
            yield { label: "flags + jcc +0 (rel32)", bytes: [...setter, 0x0F, 0x80 | cc, 0, 0, 0, 0] };
        }
    }
}

function create_emulator()
{
    return new Promise(resolve => {
        const emulator = new V86({
            autostart: false,
            memory_size: 16 * 1024 * 1024,
            log_level: 0,
            // V86_WASM_PATH overrides the core build, e.g. build/v86-mem64-debug.wasm
            wasm_path: process.env.V86_WASM_PATH,
        });
        emulator.add_listener("emulator-loaded", () => resolve(emulator));
    });
}

let emulator;
let cpu;
let module_result = null;

async function reset_emulator()
{
    if(emulator) emulator.destroy();
    emulator = await create_emulator();
    cpu = emulator.v86.cpu;
    const codegen_finalize = cpu.codegen_finalize;
    cpu.codegen_finalize = function(wasm_table_index, start, state_flags, ptr, len)
    {
        const code = new Uint8Array(this.wasm_memory.buffer, ptr >>> 0, len >>> 0).slice();
        if(WebAssembly.validate(code))
        {
            module_result = { error: null };
            return codegen_finalize.call(this, wasm_table_index, start, state_flags, ptr, len);
        }
        try
        {
            new WebAssembly.Module(code);
            module_result = { error: "WebAssembly.validate failed" };
        }
        catch(e)
        {
            module_result = { error: e.message.replace(/ @\+\d+$/, "").replace(/function #\d+/, "function") };
        }
        // leave the module uninstalled, but finish the compilation so the jit stays consistent
        Promise.resolve().then(() => this.codegen_finalize_finished(wasm_table_index, start, state_flags));
        return false;
    };
}

const PAGE = 0x100000;
const failures = new Map();
let cases = 0;
let compiled = 0;

function record_failure(key, bytes, state)
{
    if(!failures.has(key)) failures.set(key, []);
    failures.get(key).push(Buffer.from(bytes).toString("hex") + (state.cpl ? " (cpl 3)" : ""));
}

// label: the name failures are reported under, instead of e's opcode
async function compile(e, bytes, state, label)
{
    cpu.is_64[0] = 1;
    cpu.is_32[0] = 0;
    cpu.cpl[0] = state.cpl;
    cpu.stack_size_32[0] = state.stack_size_32;
    cpu.update_state_flags();

    cpu.jit_dirty_cache(PAGE, PAGE + 0x1000);
    cpu.mem8.fill(0x90, PAGE, PAGE + 0x40); // nops, then hlt to end the block
    cpu.mem8.fill(0xF4, PAGE + 0x40, PAGE + 0x1000);
    cpu.mem8.set(bytes, PAGE);

    const name = label || ("opcode " + e.opcode.toString(16).toUpperCase() +
        (e.map ? " (map " + e.map.toString(16).toUpperCase() + ")" : "") +
        (e.fixed_g === undefined ? "" : "/" + e.fixed_g));
    cases++;
    module_result = null;
    try
    {
        cpu.jit_force_generate(PAGE);
        await Promise.resolve();
    }
    catch(err)
    {
        // a rust panic (dbg_assert) or a module that failed to instantiate: the emulator's state
        // can't be trusted after it
        record_failure(name + ": " + err.message, bytes, state);
        await reset_emulator();
        return;
    }
    if(!module_result)
    {
        record_failure(name + ": no module generated", bytes, state);
        return;
    }
    compiled++;
    if(module_result.error)
    {
        record_failure(name + ": " + module_result.error, bytes, state);
    }
}

await reset_emulator();
const start = Date.now();
for(const state of STATES)
{
    for(const { e, bytes } of encodings())
    {
        await compile(e, bytes, state);
    }
}
for(const state of STATES)
{
    for(const { label, bytes } of flag_jcc_sequences())
    {
        await compile(null, bytes, state, label);
    }
}
emulator.destroy();

const seconds = ((Date.now() - start) / 1000).toFixed(1);
if(failures.size === 0)
{
    console.log(`[+] All ${cases} jit64 modules validated (${seconds} s)`);
    process.exit(0);
}
for(const [key, list] of failures)
{
    console.error(`${key}\n    ${list.length} encodings, e.g. ${list.slice(0, 6).join(", ")}`);
}
console.error(`[-] ${failures.size} distinct jit64 module failures in ${cases} encodings (${compiled} compiled)`);
process.exit(1);
