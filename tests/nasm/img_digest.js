#!/usr/bin/env node
// Path-independent digests of the nasmtests images in tests/nasm/build/.
//
// The .img files embed the absolute path of their .asm source (a FILE symbol),
// so their sha256 differs between checkouts (and the ELF headers differ
// between ld versions). This digest covers only what the tests load: the ELF
// entry point and every allocated section (type, flags, address, size and
// contents). Two builds with equal digests run the same code on the same data,
// so fixtures recorded for one are valid for the other.
//
//   tests/nasm/img_digest.js > digests.txt      print "<digest>  <name>.img" lines
//   tests/nasm/img_digest.js --check digests.txt
//                                               compare; exit 1 on any mismatch,
//                                               missing or extra image

import fs from "node:fs";
import path from "node:path";
import url from "node:url";
import crypto from "node:crypto";

const __dirname = url.fileURLToPath(new URL(".", import.meta.url));
const BUILD_DIR = path.join(__dirname, "build");

function digest(file)
{
    const buf = fs.readFileSync(file);
    const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    if(view.getUint32(0, false) !== 0x7F454C46 || buf[4] !== 1 || buf[5] !== 1)
    {
        throw new Error(file + ": not a little-endian ELF32 file");
    }
    const entry = view.getUint32(24, true);
    const shoff = view.getUint32(32, true);
    const shentsize = view.getUint16(46, true);
    const shnum = view.getUint16(48, true);

    const SHF_ALLOC = 2, SHT_NOBITS = 8;
    const hash = crypto.createHash("sha256");
    const header = Buffer.alloc(4);
    header.writeUInt32LE(entry);
    hash.update(header);
    for(let i = 0; i < shnum; i++)
    {
        const at = shoff + i * shentsize;
        const type = view.getUint32(at + 4, true);
        const flags = view.getUint32(at + 8, true);
        if(!(flags & SHF_ALLOC)) continue; // symbols, debug info: not loaded
        const offset = view.getUint32(at + 16, true);
        const size = view.getUint32(at + 20, true);
        // type, flags, addr (sh_name is an offset into a string table: skipped)
        hash.update(buf.subarray(at + 4, at + 16));
        hash.update(buf.subarray(at + 20, at + 24));
        if(type !== SHT_NOBITS) hash.update(buf.subarray(offset, offset + size));
    }
    return hash.digest("hex");
}

const images = fs.readdirSync(BUILD_DIR).filter(f => f.endsWith(".img")).sort();
const lines = new Map(images.map(f => [f, digest(path.join(BUILD_DIR, f))]));

const check_index = process.argv.indexOf("--check");
if(check_index === -1)
{
    for(const [name, d] of lines) console.log(d + "  " + name);
}
else
{
    const expected = new Map(
        fs.readFileSync(process.argv[check_index + 1], "utf8").split("\n").filter(Boolean)
            .map(l => { const [d, name] = l.split(/\s+/); return [name, d]; }));
    let bad = 0;
    for(const [name, d] of expected)
    {
        if(!lines.has(name)) { console.log("missing: " + name); bad++; }
        else if(lines.get(name) !== d) { console.log("differs: " + name); bad++; }
    }
    for(const name of lines.keys())
    {
        if(!expected.has(name)) { console.log("extra: " + name); bad++; }
    }
    console.log(bad ? `[-] ${bad} of ${expected.size} images don't match` : `[+] all ${expected.size} images match`);
    process.exit(bad ? 1 : 0);
}
