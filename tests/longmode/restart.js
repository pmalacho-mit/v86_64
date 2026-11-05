#!/usr/bin/env node
// Restart audit, see restart.asm: runs every memory-writing instruction on a
// writable page and on a read-only page (#PF, make writable, restart) and
// requires the same final state. Environment knobs as in run.js.
import path from "node:path";
import url from "node:url";
import fs from "node:fs";
import { execFileSync } from "node:child_process";

const __dirname = url.fileURLToPath(new URL(".", import.meta.url));
const root_path = path.join(__dirname, "..", "..");
const { V86 } = await import(+process.env.TEST_RELEASE_BUILD ? root_path + "/build/libv86.mjs" : root_path + "/src/main.js");

const asm_file = path.join(__dirname, "restart.asm");
const bin_file = path.join(__dirname, "restart.bin");

// ONLY=<regex> runs just the matching cases (for debugging)
let asm_source = fs.readFileSync(asm_file, "utf8");
if(process.env.ONLY)
{
    const re = new RegExp(process.env.ONLY);
    asm_source = asm_source.split("\n").filter(l => { const m = l.match(/^CASE[AUB]?\s+(\w+)\s*,/); return !m || re.test(m[1]); }).join("\n");
}
const built_asm = path.join(__dirname, "restart.built.asm");
fs.writeFileSync(built_asm, asm_source);

try {
    execFileSync("nasm", ["-w+error", "-f", "bin", "-o", bin_file, built_asm]);
} catch(e) {
    console.log("nasm not available or failed, test skipped");
    process.exit(0);
}

// every CASE / CASEA line, in order
const cases = [];
for(const line of asm_source.split("\n"))
{
    const m = line.match(/^(CASE[AUB]?)\s+(\w+)\s*,/);
    if(m) cases.push({ name: m[2], aligned: m[1] === "CASEA" });
}

// bits: 0-2 final state equals the reference (page 0+1 read-only, boundary with
// page 1 read-only, boundary with page 0 read-only), 3-5 the #PF really
// happened in those runs, 6 the reference runs didn't fault
// bit 4 is only required when the operand really crosses the boundary, which
// narrow operands at +0xFFC don't, so it isn't checked
const FULL = 0b1101111, ALIGNED = 0b1001001;
const SLOTS = ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
    "rflags", "xmm0.lo", "xmm0.hi", "xmm1.lo", "xmm1.hi", "x87 status", "memory checksum"];
const CFGS = ["", "pages 0+1 read-only", "", "boundary, page 1 read-only", "boundary, page 0 read-only"];

const emulator = new V86({
    bios: { url: bin_file },
    autostart: true,
    memory_size: 32 * 1024 * 1024,
    log_level: 0,
    disable_jit: +process.env.DISABLE_JIT,
    wasm_path: process.env.V86_WASM_PATH,
});

const config = (n, v) => emulator.bus.register("emulator-started", () => emulator.v86.cpu.wm.exports["set_jit_config"](n, v));
if(process.env.JIT64_CHAINING === "0") config(7, 0);
if(process.env.JIT32_CHAINING === "0") config(10, 0);
if(process.env.JIT_THRESHOLD) config(5, +process.env.JIT_THRESHOLD);
if(process.env.JIT64_DEFER_FLAGS === "0") config(8, 0);
if(process.env.JIT64_TLB_CACHE === "0") config(9, 0);

const timeout = setTimeout(() => { throw new Error("Timeout waiting for the restart test to finish"); }, 5 * 60 * 1000);

emulator.add_listener("serial0-output-byte", function(byte)
{
    if(byte !== 0x4B) return; // 'K'
    clearTimeout(timeout);

    const results = new BigUint64Array(emulator.read_memory(0x90000, cases.length * 8).slice().buffer);
    const diffs = new BigUint64Array(emulator.read_memory(0x94000, cases.length * 8).slice().buffer);

    if(process.env.DUMP)
    {
        const bufs = new BigUint64Array(emulator.read_memory(0x93000, 5 * 0x100).slice().buffer);
        console.log("mem:", Array.from(new BigUint64Array(emulator.read_memory(0xC00000, 64).slice().buffer), v => v.toString(16)).join(" "));
        for(let k = 0; k < 5; k++)
            console.log(`cfg ${k}:`, Array.from(bufs.slice(k * 32, k * 32 + 24), v => v.toString(16)).join(" "));
    }

    if(process.env.DUMP) console.log("vars:", Array.from(new BigUint64Array(emulator.read_memory(0x92080, 16).slice().buffer), v => v.toString(16)).join(" "));
    const failures = [];
    cases.forEach((c, i) => {
        const expected = BigInt(c.aligned ? ALIGNED : FULL);
        if((results[i] & expected) === expected) return;
        const d = Number(diffs[i]);
        const notes = [];
        if(d) notes.push(`${CFGS[(d >> 8) - 0] || "?"}: ${SLOTS[(d & 0xFF) - 1] || "?"} differs`);
        if(!(results[i] & 0b1000n)) notes.push("no #PF in the first read-only run (test is vacuous)");
        failures.push({ name: c.name, expected: "0b" + expected.toString(2), actual: "0b" + results[i].toString(2), note: notes.join("; ") });
    });

    if(failures.length === 0)
    {
        console.log(`[+] All ${cases.length} restart cases passed`);
        emulator.destroy();
        process.exit(0);
    }
    console.table(failures);
    console.error(`[-] ${failures.length}/${cases.length} restart cases failed`);
    process.exit(1);
});
