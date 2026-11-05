#!/usr/bin/env node
// Boots Alpine x86_64 to a root shell, saves the state, then restores it both into a fresh
// emulator (initial_state) and into the running one (restore_state), running user-space
// programs (system calls: lstar, kernel_gs_base, TSS) after each restore.
//
//   ALPINE_ISO=path/to/alpine-virt-3.19.x-x86_64.iso ./tests/longmode/alpine-state.js
//
// By default the kernel and initramfs are taken from the ISO's boot directory by the BIOS
// (SeaBIOS + ISOLINUX). With ALPINE_KERNEL and ALPINE_INITRD set, they're booted directly
// (with the ISO attached as cdrom, for the root filesystem) and the console is on ttyS0.
// Skipped if ALPINE_ISO isn't set. DISABLE_JIT=1 runs everything in the interpreter.

import url from "node:url";
import path from "node:path";
import fs from "node:fs";

const __dirname = url.fileURLToPath(new URL(".", import.meta.url));
const root_path = path.join(__dirname, "..", "..");

process.on("unhandledRejection", exn => { throw exn; });

const ISO = process.env.ALPINE_ISO;
if(!ISO || !fs.existsSync(ISO))
{
    console.log("ALPINE_ISO not set or not found, test skipped");
    process.exit(0);
}

const TEST_RELEASE_BUILD = +process.env.TEST_RELEASE_BUILD;
const { V86 } = await import(TEST_RELEASE_BUILD ? root_path + "/build/libv86.mjs" : root_path + "/src/main.js");

const direct = process.env.ALPINE_KERNEL && process.env.ALPINE_INITRD;
const TIMEOUT = (+process.env.TIMEOUT || 900) * 1000;


const config = {
    bios: { url: root_path + "/bios/seabios.bin" },
    vga_bios: { url: root_path + "/bios/vgabios.bin" },
    cdrom: { url: ISO },
    ...(direct ? {
        bzimage: { url: process.env.ALPINE_KERNEL },
        initrd: { url: process.env.ALPINE_INITRD },
        cmdline: "modules=loop,squashfs,sd-mod,usb-storage console=ttyS0",
    } : {}),
    memory_size: 512 * 1024 * 1024,
    autostart: true,
    log_level: 0,
    disable_jit: +process.env.DISABLE_JIT,
};

const start = Date.now();
const elapsed = () => ((Date.now() - start) / 1000).toFixed(0) + "s";

let output = "";
const timeout = setTimeout(() => {
    console.error(`[-] Timeout after ${elapsed()}, serial output:\n` + output.slice(-2000));
    process.exit(1);
}, TIMEOUT);

function wait_for(emulator, regex)
{
    return new Promise(resolve => {
        const listener = byte => {
            output += String.fromCharCode(byte);
            const m = output.match(regex);
            if(m)
            {
                emulator.remove_listener("serial0-output-byte", listener);
                resolve(m);
            }
        };
        emulator.add_listener("serial0-output-byte", listener);
    });
}

async function run_command(emulator, n)
{
    output = "";
    const done = wait_for(emulator, /ARCH=(\S+) SUM=(\d+) LS=(\d+)\r?\n/);
    // uname and ls are user-space programs, so they make system calls
    emulator.serial0_send(`echo ARCH=$(uname -m) SUM=$((40+${n})) LS=$(ls / | wc -l)\n`);
    const m = await done;
    if(m[1] !== "x86_64" || +m[2] !== 40 + n || +m[3] < 5)
    {
        console.error(`[-] Unexpected output: ${m[0]}`);
        process.exit(1);
    }
}

const emulator = new V86(config);
await wait_for(emulator, /login: $/);
console.log(`[+] Login prompt after ${elapsed()}`);
emulator.serial0_send("root\n");
await wait_for(emulator, /# $/);
await run_command(emulator, 2);

const state = await emulator.save_state();
console.log(`[+] Saved state (${(state.byteLength / 1048576).toFixed(0)} MB)`);

// restore into a fresh emulator
await emulator.stop();
const restored = new V86({ ...config, initial_state: { buffer: state } });
await run_command(restored, 3);
console.log(`[+] Fresh emulator with initial_state works (${elapsed()})`);

// restore into the running emulator
await restored.restore_state(state);
await run_command(restored, 4);
console.log(`[+] restore_state into running emulator works (${elapsed()})`);

clearTimeout(timeout);
restored.destroy();
emulator.destroy();
process.exit(0);
