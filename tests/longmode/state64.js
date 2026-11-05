#!/usr/bin/env node
// save_state/restore_state must round-trip the 64-bit CPU state (idtr/gdtr/tss bases, fs/gs bases,
// syscall MSRs, ...), also through a serialized image, and an image without those fields must
// still load (32-bit state images).

import url from "node:url";
import path from "node:path";
import assert from "node:assert/strict";

const __dirname = url.fileURLToPath(new URL(".", import.meta.url));
const root_path = path.join(__dirname, "..", "..");

process.on("unhandledRejection", exn => { throw exn; });

const TEST_RELEASE_BUILD = +process.env.TEST_RELEASE_BUILD;
const { V86 } = await import(TEST_RELEASE_BUILD ? root_path + "/build/libv86.mjs" : root_path + "/src/main.js");

const config = {
    bios: { url: root_path + "/bios/seabios.bin" },
    vga_bios: { url: root_path + "/bios/vgabios.bin" },
    autostart: true,
    memory_size: 32 * 1024 * 1024,
    log_level: 0,
    disable_jit: +process.env.DISABLE_JIT,
    wasm_path: process.env.V86_WASM_PATH,
};

async function create(extra)
{
    const emulator = new V86({ ...config, ...extra });
    await new Promise(resolve => emulator.add_listener("emulator-loaded", resolve));
    return emulator;
}

const values = {
    idtr_offset64: [0x00000000, 0xfffffe00],
    gdtr_offset64: [0x00001000, 0xfffffe00],
    tss_base64: [0x00002000, 0xfffffe00],
    fs_base: [0x12345678, 0x00007f00],
    gs_base: [0x9abcdef0, 0xffff8880],
    kernel_gs_base: [0x0badf00d, 0xffff8881],
    star: [0, 0x00230010],
    lstar: [0x81000000, 0xffffffff],
    cstar: [0x81000100, 0xffffffff],
    sfmask: [0x47700, 0],
    sysenter_esp64: [0x11111111, 0xffff8000],
    sysenter_eip64: [0x22222222, 0xffff8000],
    cr2_64: [0xdeadbeef, 0x00007fff],
    cr3_high: [0x5],
    last_op1_64: [0x1, 0x80000000],
    last_result_64: [0x2, 0x80000000],
    misc_enable: [0x850089, 0],
    nmi_flags: [1, 1],
};

function snapshot(cpu)
{
    return Object.fromEntries(Object.keys(values).map(k => [k, Array.from(cpu[k])]));
}

const a = await create();
await new Promise(resolve => setTimeout(resolve, 1000));
await a.stop();
const cpu_a = a.v86.cpu;
for(const [k, v] of Object.entries(values)) cpu_a[k].set(v);
const expected = snapshot(cpu_a);

// restore into the running emulator, after clobbering
const state = await a.save_state();
for(const k of Object.keys(values)) cpu_a[k].fill(0);
await a.restore_state(state);
assert.deepEqual(snapshot(cpu_a), expected, "restore_state into the same emulator");

// restore into a fresh emulator (initial_state)
const b = await create({ initial_state: { buffer: state } });
assert.deepEqual(snapshot(b.v86.cpu), expected, "fresh emulator with initial_state");

// older image without the 64-bit fields: loads, 64-bit slots derive from the 32-bit mirrors
const orig_get_state = cpu_a.get_state;
cpu_a.get_state = function()
{
    const cpu_state = orig_get_state.call(this);
    for(let i = 99; i <= 116; i++) delete cpu_state[i];
    cpu_state[5] = 0x1234;
    cpu_state[7] = 0x5678;
    return cpu_state;
};
const old_state = await a.save_state();
cpu_a.get_state = orig_get_state;
await a.restore_state(old_state);
assert.equal(cpu_a.idtr_offset64[0], 0x1234);
assert.equal(cpu_a.idtr_offset64[1], 0);
assert.equal(cpu_a.gdtr_offset64[0], 0x5678);
assert.equal(cpu_a.lstar[0], 0);

a.destroy();
b.destroy();
console.log("[+] 64-bit CPU state round-trips");
process.exit(0);
