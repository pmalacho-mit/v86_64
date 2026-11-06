#!/usr/bin/env node

// Generates the instruction tables of the 64-bit jit (src/rust/gen/jit64.rs, jit64_0f*.rs) and
// the exported wrappers around the interpreter's instruction handlers that the compiled code
// calls (src/rust/gen/jit64_wrappers.rs). See src/rust/jit64.rs.
//
// The tables are indexed like the interpreter's: opcode | tier << 8, where the tier is the
// operand size (0: 16 bit, 1: 32 bit, 2: 64 bit).

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import url from "node:url";

import x86_table from "./x86_table.js";
import * as rust_ast from "./rust_ast.js";
import { hex, get_switch_value, get_switch_exist, finalize_table_rust, OPCODE_MAPS, group_by_opcode_map, map_name_part } from "./util.js";

const __dirname = url.fileURLToPath(new URL(".", import.meta.url));
const OUT_DIR = path.join(__dirname, "..", "src/rust/gen/");

fs.mkdirSync(OUT_DIR, { recursive: true });

const table_arg = get_switch_value("--table");
const gen_all = get_switch_exist("--all");
const to_generate = {
    jit64: gen_all || table_arg === "jit64",
    jit64_0f: gen_all || table_arg === "jit64_0f",
    jit64_0f38: gen_all || table_arg === "jit64_0f38",
    jit64_0f3a: gen_all || table_arg === "jit64_0f3a",
    jit64_wrappers: gen_all || table_arg === "jit64_wrappers",
};

assert(
    Object.keys(to_generate).some(k => to_generate[k]),
    "Pass --table [jit64|jit64_0f|jit64_0f38|jit64_0f3a|jit64_wrappers] or --all to pick which tables to generate"
);

// name -> { module, params: [rust type], task_switch_test, sse }
const wrappers = new Map();

// Returns [rust expression, jit64::A variant] or undefined
function gen_read_imm(op, size_variant)
{
    const size = (op.os || op.opcode % 2 === 1) ? size_variant : 8;

    if(op.imm8) return ["ctx.cpu.read_imm8() as i32", "I32"];
    if(op.imm8s) return ["ctx.cpu.read_imm8s() as i32", "I32"];
    if(op.immaddr) return ["ctx.cpu.read_moffs64()", "I64"];
    if(op.imm16 || op.imm1632 && size === 16) return ["ctx.cpu.read_imm16() as i32", "I32"];
    if(op.imm3264 && size === 64) return ["ctx.cpu.read_imm64()", "I64"];
    if(op.imm32 || op.imm1632) return ["ctx.cpu.read_imm32() as i32", "I32"];
    return undefined;
}

function make_instruction_name(encoding, size)
{
    const suffix = encoding.os ? String(size) : "";
    const opcode_hex = hex(encoding.opcode & 0xFF, 2);
    const first_prefix = (encoding.opcode & 0xFF00) === 0 ? "" : hex(encoding.opcode >> 8 & 0xFF, 2);
    const second_prefix = (encoding.opcode & 0xFF0000) === 0 ? "" : hex(encoding.opcode >> 16 & 0xFF, 2);
    const fixed_g_suffix = encoding.fixed_g === undefined ? "" : `_${encoding.fixed_g}`;

    assert(first_prefix === "" || first_prefix === "0F" || first_prefix === "F2" || first_prefix === "F3");
    assert(second_prefix === "" || second_prefix === "66" || second_prefix === "F2" || second_prefix === "F3");
    assert(!encoding.map || first_prefix === "0F");

    if(encoding.ud)
    {
        // undefined opcode of a three-byte map
        return "instr_ud";
    }

    return `instr${suffix}_${second_prefix}${first_prefix}${map_name_part(encoding)}${opcode_hex}${fixed_g_suffix}`;
}

function instruction_module(encoding)
{
    return (encoding.opcode & 0xFF00) === 0x0F00 || (encoding.opcode & 0xFF0000) === 0x0F0000 ?
        "instructions_0f" : "instructions";
}

function register_wrapper(encoding, handler, params)
{
    const name = "jit64_" + handler;
    const existing = wrappers.get(name);
    const wrapper = {
        handler: instruction_module(encoding) + "::" + handler,
        params,
        task_switch_test: !!encoding.task_switch_test,
        sse: !!encoding.sse,
    };
    if(existing)
    {
        assert.deepEqual(existing, wrapper, "conflicting wrapper " + name);
    }
    wrappers.set(name, wrapper);
    return name;
}

const RUST_TYPE = { I32: "i32", I64: "u64" };

const JUMP_OPCODES = new Set([
    ...Array.from({ length: 16 }, (_, i) => 0x70 | i),
    ...Array.from({ length: 16 }, (_, i) => 0x0F80 | i),
    0xE9, 0xEB,
]);

function gen_instruction_body(encodings, size)
{
    const encoding = encodings[0];

    let has_66 = [];
    let has_f2 = [];
    let has_f3 = [];
    let no_prefix = [];

    for(let e of encodings)
    {
        if((e.opcode >>> 16) === 0x66) has_66.push(e);
        else if((e.opcode >>> 8 & 0xFF) === 0xF2 || (e.opcode >>> 16) === 0xF2) has_f2.push(e);
        else if((e.opcode >>> 8 & 0xFF) === 0xF3 || (e.opcode >>> 16) === 0xF3) has_f3.push(e);
        else no_prefix.push(e);
    }

    const code = [];

    if(encoding.e)
    {
        code.push("let modrm_byte = ctx.cpu.read_imm8();");
    }

    if(has_66.length || has_f2.length || has_f3.length)
    {
        const if_blocks = [];

        if(has_66.length) {
            const body = gen_instruction_body_after_prefix(has_66, size);
            if_blocks.push({ condition: "ctx.cpu.prefixes & prefix::PREFIX_66 != 0", body, });
        }
        if(has_f2.length) {
            const body = gen_instruction_body_after_prefix(has_f2, size);
            if_blocks.push({ condition: "ctx.cpu.prefixes & prefix::PREFIX_F2 != 0", body, });
        }
        if(has_f3.length) {
            const body = gen_instruction_body_after_prefix(has_f3, size);
            if_blocks.push({ condition: "ctx.cpu.prefixes & prefix::PREFIX_F3 != 0", body, });
        }

        const else_block = {
            body: gen_instruction_body_after_prefix(no_prefix, size),
        };

        return [].concat(code, { type: "if-else", if_blocks, else_block });
    }
    else
    {
        return [].concat(code, gen_instruction_body_after_prefix(encodings, size));
    }
}

function gen_instruction_body_after_prefix(encodings, size)
{
    const encoding = encodings[0];

    if(encoding.fixed_g !== undefined)
    {
        assert(encoding.e);

        let cases = encodings.reduce((cases_by_opcode, case_) => {
            assert(typeof case_.fixed_g === "number");
            cases_by_opcode[case_.opcode & 0xFFFF | case_.fixed_g << 16] = case_;
            return cases_by_opcode;
        }, Object.create(null));
        cases = Object.values(cases).sort((e1, e2) => e1.fixed_g - e2.fixed_g);

        return [
            {
                type: "switch",
                condition: "modrm_byte >> 3 & 7",
                cases: cases.map(case_ => ({
                    conditions: [case_.fixed_g],
                    body: gen_instruction_body_after_fixed_g(case_, size),
                })),
                default_case: {
                    // #UD (raised at the start of the instruction). The memory operand isn't
                    // decoded: the analyzer (shared with the 32-bit jit) ends the instruction
                    // after the modrm byte, and the lengths have to agree
                    body: [
                        `jit64::gen_generic(ctx, "jit64_trigger_ud", &[]);`,
                        "*instr_flags |= jit::JIT_INSTR_BLOCK_BOUNDARY_FLAG;",
                    ],
                },
            },
        ];
    }
    else
    {
        assert(encodings.length === 1);
        return gen_instruction_body_after_fixed_g(encodings[0], size);
    }
}

function gen_instruction_body_after_fixed_g(encoding, size)
{
    const name = make_instruction_name(encoding, size);
    const postfix = [];

    if(encoding.block_boundary && !encoding.not_block_boundary_in_64 && !(!encoding.map && JUMP_OPCODES.has(encoding.opcode) && size !== 16))
    {
        postfix.push("*instr_flags |= jit::JIT_INSTR_BLOCK_BOUNDARY_FLAG;");
    }

    if(encoding.prefix)
    {
        // prefixes and 0x0F
        return [`jit64::${name}_jit64(ctx, instr_flags);`];
    }

    const imm = gen_read_imm(encoding, size);

    // relative jumps with 32/64-bit operand size: generated by the block glue in jit.rs
    if(!encoding.map && JUMP_OPCODES.has(encoding.opcode) && size !== 16)
    {
        assert(imm);
        return [`let _ = ${imm[0]};`];
    }
    if(encoding.opcode === 0xE8 && size === 64)
    {
        return [`let imm = ${imm[0]};`, "jit64::instr64_E8_jit64(ctx, imm);"].concat(postfix);
    }
    if(encoding.opcode === 0xFB)
    {
        return ["jit64::instr_FB_jit64(ctx);"];
    }
    if(encoding.opcode === 0xFA && !process.env.DISABLE_NATIVE_CLI)
    {
        return ["jit64::gen_cli(ctx);"];
    }
    if(!process.env.DISABLE_NATIVE_STRING &&
        ([0xA4, 0xAA, 0xAC].includes(encoding.opcode) || encoding.opcode === 0xA5 && size !== 16 ||
        encoding.opcode === 0xAB && size !== 16 || encoding.opcode === 0xAD && size !== 16))
    {
        // movs/stos/lods without rep: one flagless access plus rsi/rdi updates
        const op = encoding.opcode;
        const bits = op === 0xA4 || op === 0xAA || op === 0xAC ? 8 : size;
        const kind = op === 0xA4 || op === 0xA5 ? "movs" : op === 0xAA || op === 0xAB ? "stos" : "lods";
        const fallback = register_wrapper(encoding, name, []);
        return [`jit64::gen_string_${kind}(ctx, ${bits}, "${fallback}");`];
    }
    if(encoding.opcode === 0x8D)
    {
        return [`jit64::instr_8D_jit64(ctx, modrm_byte, ${size});`];
    }
    if(encoding.opcode === 0x8F)
    {
        return ["jit64::instr_8F_jit64(ctx, modrm_byte, instr_flags);"];
    }
    assert(!encoding.custom_modrm_resolve, "unhandled custom_modrm_resolve: " + name);

    const native = gen_native(encoding, size, imm, () => gen_generic(encoding, size, imm, name, []));
    if(native)
    {
        return native.concat(postfix);
    }

    return gen_generic(encoding, size, imm, name, postfix);
}

// generic: call the interpreter's handler through a wrapper
function gen_generic(encoding, size, imm, name, postfix)
{
    if(encoding.e)
    {
        // fpu register stack indices (st0-st7) are not extended by REX.B
        const reg_args = [encoding.is_fpu ?
            "jit64::A::I32((modrm_byte & 7) as i32)" :
            "jit64::A::I32((modrm_byte & 7) as i32 | ctx.cpu.rex_b() as i32)"];
        const reg_params = ["i32"];
        const mem_args = [];
        const mem_params = ["u64"];

        if(encoding.ignore_mod)
        {
            assert(!imm);
            const wrapper = register_wrapper(encoding, name, ["i32", "i32"]);
            return [].concat(
                `jit64::gen_generic(ctx, "${wrapper}", &[` +
                    "jit64::A::I32((modrm_byte & 7) as i32 | ctx.cpu.rex_b() as i32), " +
                    "jit64::A::I32((modrm_byte >> 3 & 7) as i32 | ctx.cpu.rex_r() as i32)]);",
                postfix
            );
        }

        if(encoding.fixed_g === undefined)
        {
            const r = "jit64::A::I32((modrm_byte >> 3 & 7) as i32 | ctx.cpu.rex_r() as i32)";
            reg_args.push(r);
            reg_params.push("i32");
            mem_args.push(r);
            mem_params.push("i32");
        }

        const imm_binding = [];
        if(imm)
        {
            imm_binding.push(`let imm = jit64::A::${imm[1]}(${imm[0]});`);
            reg_args.push("imm");
            reg_params.push(RUST_TYPE[imm[1]]);
            mem_args.push("imm");
            mem_params.push(RUST_TYPE[imm[1]]);
        }

        const mem_wrapper = register_wrapper(encoding, name + "_mem", mem_params);
        const reg_wrapper = register_wrapper(encoding, name + "_reg", reg_params);

        return [].concat(
            {
                type: "if-else",
                if_blocks: [{
                    condition: "modrm_byte < 0xC0",
                    body: [].concat(
                        "let addr = jit64::decode_modrm(ctx.cpu, modrm_byte);",
                        imm_binding,
                        `jit64::gen_generic_mem(ctx, "${mem_wrapper}", &addr, &[${mem_args.join(", ")}]);`
                    ),
                }],
                else_block: {
                    body: [].concat(
                        imm_binding,
                        `jit64::gen_generic(ctx, "${reg_wrapper}", &[${reg_args.join(", ")}]);`
                    ),
                },
            },
            postfix
        );
    }
    else
    {
        const args = [];
        const params = [];
        const bindings = [];

        if(imm)
        {
            bindings.push(`let imm = jit64::A::${imm[1]}(${imm[0]});`);
            args.push("imm");
            params.push(RUST_TYPE[imm[1]]);
        }
        if(encoding.extra_imm16)
        {
            assert(imm);
            bindings.push("let imm2 = jit64::A::I32(ctx.cpu.read_imm16() as i32);");
            args.push("imm2");
            params.push("i32");
        }
        else if(encoding.extra_imm8)
        {
            assert(imm);
            bindings.push("let imm2 = jit64::A::I32(ctx.cpu.read_imm8() as i32);");
            args.push("imm2");
            params.push("i32");
        }

        const wrapper = register_wrapper(encoding, name, params);
        return [].concat(
            bindings,
            `jit64::gen_generic(ctx, "${wrapper}", &[${args.join(", ")}]);`,
            postfix
        );
    }
}

// Natively compiled instructions (see the native section of src/rust/jit64.rs).
// Returns the generated code or undefined (then the interpreter's handler is called).
const ALU_OPS = { 0: "OP_ADD", 1: "OP_OR", 4: "OP_AND", 5: "OP_SUB", 6: "OP_XOR", 7: "OP_CMP" };

// For debugging: NATIVE64_SKIP=group,... falls back to the interpreter's handlers for groups
// of natively compiled instructions (see native_group)
const NATIVE64_SKIP = (process.env.NATIVE64_SKIP || "").split(",").filter(x => x);

const SHIFT_OPS = [0xC0, 0xC1, 0xD0, 0xD1, 0xD2, 0xD3];

function native_group(op, encoding)
{
    if(op <= 0x3F && ((op >> 3) === 2 || (op >> 3) === 3)) return "adc";
    if(op <= 0x3F) return "alu";
    if((op === 0x81 || op === 0x83) && (encoding.fixed_g === 2 || encoding.fixed_g === 3)) return "adc";
    if(op === 0x0FA3 || op === 0x0FAB || op === 0x0FB3 || op === 0x0FBB || op === 0x0FBA) return "bt";
    if(op === 0x0FB1 || op === 0x0FC1) return "atomic";
    if(op >= 0x80 && op <= 0x83) return "aluimm";
    if(op === 0x84 || op === 0x85 || op === 0xA8 || op === 0xA9 || op === 0xF6 || op === 0xF7) return "test";
    if(op >= 0x88 && op <= 0x8B || op === 0xC6 || op === 0xC7 || op >= 0xB0 && op <= 0xBF) return "mov";
    if(op >= 0x0FB6 && op <= 0x0FBF || op === 0x63) return "movx";
    if(op === 0xFE || op === 0xFF && encoding.fixed_g < 2) return "incdec";
    if(op === 0xC3) return "ret";
    if(SHIFT_OPS.includes(op)) return "shift";
    if(op === 0x0FAF || op === 0x69 || op === 0x6B) return "imul";
    if((op & 0xFFF0) === 0x0F90 || (op & 0xFFF0) === 0x0F40) return "cc";
    if(op === 0xFF) return "jump";
    if(op === 0x63 || op >= 0x0F18 && op <= 0x0F1F || op >= 0x90 && op <= 0x99 || op === 0x87 || op >= 0x0FC8 && op <= 0x0FCF || op === 0xF7) return "misc";
    return "stack";
}

function gen_native(encoding, size, imm, generic)
{
    const code = gen_native_any(encoding, size, imm, generic);
    if(code && NATIVE64_SKIP.includes(native_group(encoding.opcode, encoding)))
    {
        return undefined;
    }
    return code;
}

function gen_native_any(encoding, size, imm, generic)
{
    const op = encoding.opcode;
    // SSSE3 pshufb/palignr (register form): natively compiled like the other integer
    // SSE instructions; the memory forms keep the interpreter's wrapper
    if(op === 0x660F00 && encoding.map === 0x38)
    {
        const mem_wrapper = register_wrapper(encoding, "instr_660F3800_mem", ["u64", "i32"]);
        return [{
            type: "if-else",
            if_blocks: [{
                condition: "modrm_byte < 0xC0",
                body: [
                    "let addr = jit64::decode_modrm(ctx.cpu, modrm_byte);",
                    `jit64::gen_generic_mem(ctx, "${mem_wrapper}", &addr, &[jit64::A::I32((modrm_byte >> 3 & 7) as i32 | ctx.cpu.rex_r() as i32)]);`,
                ],
            }],
            else_block: {
                body: ["jit64::gen_sse_pshufb(ctx, modrm_byte);"],
            },
        }];
    }
    if(op === 0x660F0F && encoding.map === 0x3A)
    {
        assert(imm && imm[1] === "I32");
        const mem_wrapper = register_wrapper(encoding, "instr_660F3A0F_mem", ["u64", "i32", "i32"]);
        return [{
            type: "if-else",
            if_blocks: [{
                condition: "modrm_byte < 0xC0",
                body: [
                    "let addr = jit64::decode_modrm(ctx.cpu, modrm_byte);",
                    `let imm = ${imm[0]};`,
                    `jit64::gen_generic_mem(ctx, "${mem_wrapper}", &addr, &[jit64::A::I32((modrm_byte >> 3 & 7) as i32 | ctx.cpu.rex_r() as i32), jit64::A::I32(imm)]);`,
                ],
            }],
            else_block: {
                body: [
                    `let imm = ${imm[0]};`,
                    "jit64::gen_sse_palignr(ctx, imm, modrm_byte);",
                ],
            },
        }];
    }
    if(encoding.map)
    {
        // three-byte maps: their opcodes would alias 0F xx in the checks below
        return undefined;
    }
    const bits_of = wide => wide ? size : 8;
    const R = "r";
    const RM_REG = "((modrm_byte & 7) as u32 | ctx.cpu.rex_b())";
    const IMM = imm && imm[1] === "I32" ? `jit64::Opnd::Imm(${imm[0]} as i64)` : undefined;

    // Common 66-prefixed integer SSE instructions, compiled natively: they set no flags and
    // don't touch general-purpose registers, so they skip the interpreter-call ceremony
    // (flag commit, register spill, tlb cache reset). See jit64::gen_sse_binary64 etc.
    // Memory-operand forms that aren't covered fall back to the interpreter's wrapper.
    const sse_nm = `${hex(op >> 16 & 0xFF, 2)}${hex(op >> 8 & 0xFF, 2)}${hex(op & 0xFF, 2)}`;
    if(op === 0x660FEF || op === 0x660FEB || op === 0x660FDB || op === 0x660FD4)
    {
        return [`jit64::gen_sse_binary64(ctx, ${op}, modrm_byte);`];
    }
    if(op === 0x660F6F || op === 0xF30F6F)
    {
        return ["jit64::gen_sse_mov_load(ctx, modrm_byte);"];
    }
    if(op === 0x660F7F || op === 0xF30F7F)
    {
        return ["jit64::gen_sse_mov_store(ctx, modrm_byte);"];
    }
    if(op === 0x0F10 || op === 0x0F28)
    {
        // movups/movaps xmm, xmm/m128 (also unaligned, like the interpreter)
        return ["jit64::gen_sse_mov_load(ctx, modrm_byte);"];
    }
    if(op === 0x0F11 || op === 0x0F29)
    {
        return ["jit64::gen_sse_mov_store(ctx, modrm_byte);"];
    }
    if([0x660FFC, 0x660FFD, 0x660FFE, 0x660FF8, 0x660FF9, 0x660FFA].includes(op))
    {
        // paddb/paddw/paddd/psubb/psubw/psubd: native for the register form
        assert(!imm);
        const mem_wrapper = register_wrapper(encoding, `instr_${sse_nm}_mem`, ["u64", "i32"]);
        return [{
            type: "if-else",
            if_blocks: [{
                condition: "modrm_byte < 0xC0",
                body: [
                    "let addr = jit64::decode_modrm(ctx.cpu, modrm_byte);",
                    `jit64::gen_generic_mem(ctx, "${mem_wrapper}", &addr, &[jit64::A::I32((modrm_byte >> 3 & 7) as i32 | ctx.cpu.rex_r() as i32)]);`,
                ],
            }],
            else_block: {
                body: [`jit64::gen_sse_padd(ctx, ${op}, modrm_byte);`],
            },
        }];
    }
    if(op === 0x660F71 || op === 0x660F72 || op === 0x660F73)
    {
        assert(imm && imm[1] === "I32");
        const kind = encoding.fixed_g;
        // the register form is natively compiled; the (memory) form keeps the wrapper
        const mem_wrapper = register_wrapper(encoding, `instr_${sse_nm}_${kind}_mem`, ["u64", "i32"]);
        return [{
            type: "if-else",
            if_blocks: [{
                condition: "modrm_byte < 0xC0",
                body: [
                    "let addr = jit64::decode_modrm(ctx.cpu, modrm_byte);",
                    `let imm = ${imm[0]};`,
                    `jit64::gen_generic_mem(ctx, "${mem_wrapper}", &addr, &[jit64::A::I32(imm)]);`,
                ],
            }],
            else_block: {
                body: [
                    `let imm = ${imm[0]};`,
                    `jit64::gen_sse_shift_imm(ctx, ${op}, ${kind}, imm, modrm_byte);`,
                ],
            },
        }];
    }
    if(op === 0x660F70)
    {
        assert(imm && imm[1] === "I32");
        const mem_wrapper = register_wrapper(encoding, `instr_${sse_nm}_mem`, ["u64", "i32", "i32"]);
        return [{
            type: "if-else",
            if_blocks: [{
                condition: "modrm_byte < 0xC0",
                body: [
                    "let addr = jit64::decode_modrm(ctx.cpu, modrm_byte);",
                    `let imm = ${imm[0]};`,
                    `jit64::gen_generic_mem(ctx, "${mem_wrapper}", &addr, &[jit64::A::I32((modrm_byte >> 3 & 7) as i32 | ctx.cpu.rex_r() as i32), jit64::A::I32(imm)]);`,
                ],
            }],
            else_block: {
                body: [
                    `let imm = ${imm[0]};`,
                    "jit64::gen_sse_pshufd(ctx, imm, modrm_byte);",
                ],
            },
        }];
    }
    if(op === 0x660F6C || op === 0x660F6D)
    {
        const mem_wrapper = register_wrapper(encoding, `instr_${sse_nm}_mem`, ["u64", "i32"]);
        return [{
            type: "if-else",
            if_blocks: [{
                condition: "modrm_byte < 0xC0",
                body: [
                    "let addr = jit64::decode_modrm(ctx.cpu, modrm_byte);",
                    `jit64::gen_generic_mem(ctx, "${mem_wrapper}", &addr, &[jit64::A::I32((modrm_byte >> 3 & 7) as i32 | ctx.cpu.rex_r() as i32)]);`,
                ],
            }],
            else_block: {
                body: [`jit64::gen_sse_punpckqdq(ctx, ${op}, modrm_byte);`],
            },
        }];
    }

    // dst/src: "rm" (the modrm operand), "r" (the modrm reg field), "imm", "eax"
    function modrm_form(bits, operand_code_fn)
    {
        // operand_code_fn(rm_operand) -> code
        return ["let r = (modrm_byte >> 3 & 7) as u32 | ctx.cpu.rex_r();", {
            type: "if-else",
            if_blocks: [{
                condition: "modrm_byte < 0xC0",
                body: [].concat(
                    "let addr = jit64::decode_modrm(ctx.cpu, modrm_byte);",
                    imm ? [`let imm = ${IMM};`] : [],
                    operand_code_fn("jit64::Opnd::Mem(addr)", "imm")
                ),
            }],
            else_block: {
                body: [].concat(
                    imm ? [`let imm = ${IMM};`] : [],
                    `let rm = ${RM_REG};`,
                    operand_code_fn(`jit64::Opnd::Reg(rm)`, "imm")
                ),
            },
        }];
    }

    if(op <= 0x3F && (op & 7) < 4 && ALU_OPS[op >> 3] && !encoding.prefix)
    {
        const bits = bits_of(op & 1);
        const alu = ALU_OPS[op >> 3];
        if(op & 2)
        {
            // r <op>= r/m
            return modrm_form(bits, rm => [`jit64::gen_alu(ctx, jit64::${alu}, ${bits}, jit64::Opnd::Reg(${R}), ${rm});`]);
        }
        else
        {
            return modrm_form(bits, rm => [`jit64::gen_alu(ctx, jit64::${alu}, ${bits}, ${rm}, jit64::Opnd::Reg(${R}));`]);
        }
    }
    if(op <= 0x3F && ((op & 7) === 4 || (op & 7) === 5) && ALU_OPS[op >> 3] && IMM)
    {
        // al/eax <op>= imm
        const bits = bits_of(op & 1);
        return [`let imm = ${IMM};`, `jit64::gen_alu(ctx, jit64::${ALU_OPS[op >> 3]}, ${bits}, jit64::Opnd::Reg(0), imm);`];
    }
    if((op === 0x80 || op === 0x81 || op === 0x83) && ALU_OPS[encoding.fixed_g] && IMM)
    {
        const bits = op === 0x80 ? 8 : size;
        return modrm_form(bits, (rm, i) => [`jit64::gen_alu(ctx, jit64::${ALU_OPS[encoding.fixed_g]}, ${bits}, ${rm}, ${i});`]);
    }
    if(op === 0x84 || op === 0x85)
    {
        const bits = bits_of(op & 1);
        return modrm_form(bits, rm => [`jit64::gen_alu(ctx, jit64::OP_TEST, ${bits}, ${rm}, jit64::Opnd::Reg(${R}));`]);
    }
    if((op === 0xA8 || op === 0xA9) && IMM)
    {
        return [`let imm = ${IMM};`, `jit64::gen_alu(ctx, jit64::OP_TEST, ${bits_of(op & 1)}, jit64::Opnd::Reg(0), imm);`];
    }
    if((op === 0xF6 || op === 0xF7) && encoding.fixed_g === 0 && IMM)
    {
        const bits = bits_of(op & 1);
        return modrm_form(bits, (rm, i) => [`jit64::gen_alu(ctx, jit64::OP_TEST, ${bits}, ${rm}, ${i});`]);
    }
    if(op >= 0x88 && op <= 0x8B)
    {
        const bits = bits_of(op & 1);
        if(op & 2)
        {
            return modrm_form(bits, rm => [`jit64::gen_alu(ctx, jit64::OP_MOV, ${bits}, jit64::Opnd::Reg(${R}), ${rm});`]);
        }
        return modrm_form(bits, rm => [`jit64::gen_alu(ctx, jit64::OP_MOV, ${bits}, ${rm}, jit64::Opnd::Reg(${R}));`]);
    }
    if((op === 0xC6 || op === 0xC7) && encoding.fixed_g === 0 && IMM)
    {
        const bits = bits_of(op & 1);
        return modrm_form(bits, (rm, i) => [`jit64::gen_alu(ctx, jit64::OP_MOV, ${bits}, ${rm}, ${i});`]);
    }
    if(op < 0x100 && (op & 0xF8) === 0xB0 && IMM)
    {
        return [`let imm = ${IMM};`, `let r = ${op & 7} | ctx.cpu.rex_b();`, `jit64::gen_alu(ctx, jit64::OP_MOV, 8, jit64::Opnd::Reg(r), imm);`];
    }
    if(op < 0x100 && (op & 0xF8) === 0xB8 && imm)
    {
        const value = imm[1] === "I64" ? `${imm[0]} as i64` : `${imm[0]} as i64`;
        return [`let imm = jit64::Opnd::Imm(${value});`, `let r = ${op & 7} | ctx.cpu.rex_b();`, `jit64::gen_alu(ctx, jit64::OP_MOV, ${size}, jit64::Opnd::Reg(r), imm);`];
    }
    if(op === 0x0FB6 || op === 0x0FB7 || op === 0x0FBE || op === 0x0FBF)
    {
        const src_bits = (op & 1) ? 16 : 8;
        const signed = (op & 8) !== 0;
        return modrm_form(src_bits, rm => [`jit64::gen_movx(ctx, ${signed}, ${src_bits}, ${size}, ${R}, ${rm});`]);
    }
    if((op === 0xFF && size !== 16 || op === 0xFE) && (encoding.fixed_g === 0 || encoding.fixed_g === 1))
    {
        // note: 0xFF is a default-64 group, inc/dec are 64-bit only with REX.W
        const bits = op === 0xFE ? 8 : size === 64 ? "if ctx.cpu.rex_w() { 64 } else { 32 }" : size;
        return [`let bits = ${bits};`].concat(modrm_form(8, rm => [`jit64::gen_incdec(ctx, ${encoding.fixed_g === 1}, bits, ${rm});`]));
    }
    if(size === 64 && op < 0x100 && (op & 0xF8) === 0x50)
    {
        return [`let r = ${op & 7} | ctx.cpu.rex_b();`, `jit64::gen_push64(ctx, jit64::Opnd::Reg(r));`];
    }
    if(size === 64 && op < 0x100 && (op & 0xF8) === 0x58)
    {
        return [`let r = ${op & 7} | ctx.cpu.rex_b();`, `jit64::gen_pop64(ctx, r);`];
    }
    if(size === 64 && (op === 0x68 || op === 0x6A) && IMM)
    {
        return [`let imm = ${IMM};`, `jit64::gen_push64(ctx, imm);`];
    }
    if(size === 64 && op === 0xFF && encoding.fixed_g === 6)
    {
        return modrm_form(64, rm => [`jit64::gen_push64(ctx, ${rm});`]);
    }
    if(size === 64 && op === 0xC3)
    {
        return ["jit64::gen_ret64(ctx, 0);"];
    }
    if(!process.env.DISABLE_NATIVE_SHIFT && SHIFT_OPS.includes(op) && [0, 1, 4, 5, 7].includes(encoding.fixed_g))
    {
        const bits = (op === 0xC1 || op === 0xD1 || op === 0xD3) ? size : 8;
        const kind = encoding.fixed_g;
        // rotates are only native for 32/64-bit operands: for 8/16-bit rotates the
        // effective count wraps at the operand width and the flags differ
        if(kind < 4 && bits < 32)
        {
            return undefined;
        }
        if(op === 0xD2 || op === 0xD3)
        {
            return modrm_form(bits, rm => [`jit64::gen_shift(ctx, ${kind}, ${bits}, ${rm}, jit64::ShiftCount::Cl);`]);
        }
        if(op === 0xD0 || op === 0xD1)
        {
            return modrm_form(bits, rm => [`jit64::gen_shift(ctx, ${kind}, ${bits}, ${rm}, jit64::ShiftCount::Imm(1));`]);
        }
        // shifts by a constant (the count byte follows the modrm operand); counts are
        // masked to 5 bits (6 for 64-bit operands), like the interpreter's handlers
        const count_mask = bits === 64 ? 63 : 31;
        return modrm_form(bits, (rm, i) => [].concat(
            [`let count = match ${i} { jit64::Opnd::Imm(i) => i as u32, _ => 0 } & ${count_mask};`],
            {
                type: "if-else",
                if_blocks: [{ condition: "count != 0", body: [`jit64::gen_shift(ctx, ${kind}, ${bits}, ${rm}, jit64::ShiftCount::Imm(count));`] }],
                // note: the interpreter writes the (unchanged) register, which zero-extends
                // 32-bit operands; for 8/16 bits the register write is a no-op and the
                // memory operand is still read and written back (gen_shift_by_zero)
                else_block: { body: [`jit64::gen_shift_by_zero(ctx, ${bits}, ${rm});`] },
            }
        ));
    }
    if(op === 0x0FAF && size !== 16)
    {
        return modrm_form(size, rm => [`jit64::gen_imul(ctx, ${size}, r, ${rm}, None);`]);
    }
    if((op === 0x69 || op === 0x6B) && size !== 16)
    {
        return modrm_form(size, (rm, i) => [`let i = match ${i} { jit64::Opnd::Imm(i) => i, _ => 0 };`, `jit64::gen_imul(ctx, ${size}, r, ${rm}, Some(i));`]);
    }
    if((op & 0xFFF0) === 0x0F90)
    {
        return modrm_form(8, rm => [`jit64::gen_setcc(ctx, ${op & 0xF}, ${rm});`]);
    }
    if((op & 0xFFF0) === 0x0F40 && size !== 16)
    {
        return modrm_form(size, rm => [`jit64::gen_cmovcc(ctx, ${op & 0xF}, ${size}, r, ${rm});`]);
    }
    const if_else = (condition, body, else_body) => [{
        type: "if-else",
        if_blocks: [{ condition, body }],
        else_block: { body: else_body },
    }];
    const NO_66 = "ctx.cpu.prefixes & prefix::PREFIX_66 == 0";
    if(op === 0x63)
    {
        // movsxd (REX.W) or mov r32, r/m32 (0x66: interpreter)
        return if_else(NO_66,
            modrm_form(32, rm => if_else("ctx.cpu.rex_w()",
                [`jit64::gen_movx(ctx, true, 32, 64, r, ${rm});`],
                [`jit64::gen_alu(ctx, jit64::OP_MOV, 32, jit64::Opnd::Reg(r), ${rm});`])),
            generic());
    }
    if([0x0F18, 0x0F19, 0x0F1C, 0x0F1D, 0x0F1E, 0x0F1F].includes(op))
    {
        // prefetch and hint nops (the memory operand isn't accessed)
        return ["jit64::skip_modrm(ctx.cpu, modrm_byte);"];
    }
    if(op === 0x90)
    {
        // nop, or xchg r8, rax with REX.B
        return if_else("ctx.cpu.rex_b() == 0", [], generic());
    }
    if(op >= 0x91 && op <= 0x97 && size !== 16)
    {
        return [`let r = ${op & 7} | ctx.cpu.rex_b();`, `jit64::gen_xchg(ctx, ${size}, 0, jit64::Opnd::Reg(r));`];
    }
    if(op === 0x87 && size !== 16)
    {
        return modrm_form(size, rm => [`jit64::gen_xchg(ctx, ${size}, r, ${rm});`]);
    }
    if(op === 0x98)
    {
        return [`jit64::gen_cbw(ctx, ${size});`];
    }
    if(op === 0x99)
    {
        return [`jit64::gen_cwd(ctx, ${size});`];
    }
    const ADC_SBB = { 2: false, 3: true };
    if(op <= 0x3F && (op >> 3) in ADC_SBB && (op & 7) < 6 && size !== 16 && (op & 1))
    {
        // adc/sbb (32/64 bits)
        const is_sbb = ADC_SBB[op >> 3];
        if((op & 7) === 5)
        {
            return IMM ? [`let imm = ${IMM};`, `jit64::gen_adc_sbb(ctx, ${is_sbb}, ${size}, jit64::Opnd::Reg(0), imm);`] : undefined;
        }
        if(op & 2)
        {
            return modrm_form(size, rm => [`jit64::gen_adc_sbb(ctx, ${is_sbb}, ${size}, jit64::Opnd::Reg(r), ${rm});`]);
        }
        return modrm_form(size, rm => [`jit64::gen_adc_sbb(ctx, ${is_sbb}, ${size}, ${rm}, jit64::Opnd::Reg(r));`]);
    }
    if((op === 0x81 || op === 0x83) && (encoding.fixed_g === 2 || encoding.fixed_g === 3) && size !== 16 && IMM)
    {
        return modrm_form(size, (rm, i) => [`jit64::gen_adc_sbb(ctx, ${encoding.fixed_g === 3}, ${size}, ${rm}, ${i});`]);
    }
    const BT = { 0x0FA3: 4, 0x0FAB: 5, 0x0FB3: 6, 0x0FBB: 7 };
    if(op in BT && size !== 16)
    {
        // bt r, r (with a memory operand the offset addresses a bit string: interpreter)
        return if_else("modrm_byte >= 0xC0", [
            "let r = (modrm_byte >> 3 & 7) as u32 | ctx.cpu.rex_r();",
            "let rm = (modrm_byte & 7) as u32 | ctx.cpu.rex_b();",
            `jit64::gen_bt(ctx, ${BT[op]}, ${size}, jit64::Opnd::Reg(rm), jit64::Opnd::Reg(r));`,
        ], generic());
    }
    if(op === 0x0FBA && encoding.fixed_g >= 4 && size !== 16 && IMM)
    {
        return modrm_form(size, (rm, i) => [`jit64::gen_bt(ctx, ${encoding.fixed_g}, ${size}, ${rm}, ${i});`]);
    }
    if(op === 0x0FB1 && size !== 16)
    {
        return modrm_form(size, rm => [`jit64::gen_cmpxchg(ctx, ${size}, ${rm}, r);`]);
    }
    if(op === 0x0FC1 && size !== 16)
    {
        return modrm_form(size, rm => [`jit64::gen_xadd(ctx, ${size}, ${rm}, r);`]);
    }
    if(op === 0x9C && size === 64)
    {
        return ["jit64::gen_pushf64(ctx);"];
    }
    if(op >= 0x0FC8 && op <= 0x0FCF)
    {
        return if_else(NO_66, [
            "let bits = if ctx.cpu.rex_w() { 64 } else { 32 };",
            `let r = ${op & 7} | ctx.cpu.rex_b();`,
            "jit64::gen_bswap(ctx, bits, r);",
        ], generic());
    }
    if((op === 0xF7) && (encoding.fixed_g === 2 || encoding.fixed_g === 3) && size !== 16)
    {
        return modrm_form(size, rm => [`jit64::gen_not_neg(ctx, ${encoding.fixed_g === 3}, ${size}, ${rm});`]);
    }
    if(op === 0xFF && (encoding.fixed_g === 2 || encoding.fixed_g === 4) && size === 64)
    {
        return modrm_form(64, rm => [`jit64::gen_indirect_jump64(ctx, ${encoding.fixed_g === 2}, ${rm});`]);
    }
    return undefined;
}

function gen_cases(by_opcode)
{
    const cases = [];
    for(let opcode = 0; opcode < 0x100; opcode++)
    {
        const encoding = by_opcode[opcode];
        assert(encoding && encoding.length);

        if(encoding[0].os)
        {
            for(const [tier, size] of [[0, 16], [1, 32], [2, 64]])
            {
                cases.push({
                    conditions: [`0x${hex(opcode | tier << 8, 2)}`],
                    body: gen_instruction_body(encoding, size),
                });
            }
        }
        else
        {
            cases.push({
                conditions: [0, 1, 2].map(tier => `0x${hex(opcode | tier << 8, 2)}`),
                body: gen_instruction_body(encoding, undefined),
            });
        }
    }
    return {
        type: "switch",
        condition: "opcode",
        cases,
        default_case: {
            body: ["assert!(false);"]
        },
    };
}

function gen_wrappers()
{
    const code = [
        "#![allow(unused_variables)]",
        "#![allow(non_snake_case)]",
        "use crate::cpu::cpu::{task_switch_test, task_switch_test_mmx, trigger_ud};",
        "use crate::cpu::instructions;",
        "use crate::cpu::instructions_0f;",
        "use crate::jit64::{jit64_enter, jit64_leave};",
        "",
        "#[no_mangle]",
        "pub unsafe fn jit64_trigger_ud(ips: i32, prefixes: i32) -> i32 { jit64_enter(ips, prefixes); trigger_ud(); jit64_leave() }",
    ];

    const names = Array.from(wrappers.keys()).sort();
    for(const name of names)
    {
        const w = wrappers.get(name);
        const params = w.params.map((t, i) => `a${i}: ${t}`);
        const args = w.params.map((t, i) => `a${i}`);
        const check = w.sse ? "if !task_switch_test_mmx() { return jit64_leave(); } " :
            w.task_switch_test ? "if !task_switch_test() { return jit64_leave(); } " : "";
        code.push("#[no_mangle]");
        code.push(
            `pub unsafe fn ${name}(${["ips: i32", "prefixes: i32"].concat(params).join(", ")}) -> i32 { ` +
            `jit64_enter(ips, prefixes); ${check}${w.handler}(${args.join(", ")}); jit64_leave() }`
        );
    }
    return code.join("\n") + "\n";
}

function gen_table()
{
    const by_map = group_by_opcode_map(x86_table);

    const header = [
        "#![allow(unused)]",
        "#[cfg_attr(rustfmt, rustfmt_skip)]",
        "use crate::prefix;",
        "use crate::jit;",
        "use crate::jit64;",
        "pub fn jit(opcode: u32, ctx: &mut jit::JitContext, instr_flags: &mut u32) {",
    ];

    // all tables are always generated, as the wrappers are collected from them
    for(const map of OPCODE_MAPS)
    {
        const table = gen_cases(by_map[map]);
        const name = "jit64" + (map ? "_" + map : "");
        if(to_generate[name])
        {
            finalize_table_rust(
                OUT_DIR,
                name + ".rs",
                rust_ast.print_syntax_tree([].concat(header, table, "}")).join("\n") + "\n"
            );
        }
    }
    if(to_generate.jit64_wrappers)
    {
        finalize_table_rust(OUT_DIR, "jit64_wrappers.rs", gen_wrappers());
    }
}

gen_table();
