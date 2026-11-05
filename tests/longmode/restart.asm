; Restart audit: every memory-writing instruction must be restartable.
;
; Each case (an instruction with the memory operand [rdi], optionally preceded
; by a setup instruction) runs several times from identical register, flag,
; x87, SSE and memory state:
;
;   A  the page is writable: the reference result
;   B  the page is present and read-only (CR0.WP=1) and the instruction has
;      already read it, so the TLB holds a read-only translation. The #PF
;      handler makes the page writable and iretq restarts the instruction.
;
; and the final state of B must equal A's, byte for byte. The operand is also
; run straddling a page boundary (4 bytes before the end of page 0), with
; either of the two pages read-only. A register, flag or x87 change made
; before the write that can fault shows up as a difference.
;
; Results: one qword per case at RESULTS, see restart.js for the bits.
; This file is assembled by restart.js. CASE lines are parsed by it too, so
; keep the `CASE name, {pre}, {instruction}` form. CASEA is for instructions
; that need an aligned operand (they only run at offset 0).

org 0xF0000

%define RESULTS   0x90000
%define S         0x91000      ; snapshot scratch (the stack for it is S+0x400)
%define VARS      0x92000
%define CASEIDX   (VARS+0)
%define CFGIDX    (VARS+8)
%define CURSTUB   (VARS+16)
%define RETP      (VARS+24)
%define FCNT      (VARS+32)
%define RES       (VARS+40)
%define HSAVE     (VARS+48)
%define FMASK     (VARS+56)
%define BUFS      0x93000      ; 5 snapshots of 0x100 bytes
%define DIFF      0x94000      ; first difference per case: (cfg<<8 | slot) + 1
%define RUNSTACK  0x7E000
%define SAFE      0x7F000      ; the stack the instruction under test runs on
%define PAGE      0x40400000   ; 4 KiB pages 0,1 under test, page 2 = movs source
%define PT        0x8000
%define NSLOT     23           ; snapshot qwords compared
%define FLAGMASK  0xCD5        ; CF PF AF ZF SF DF OF

[bits 16]
start16:
    cli
    mov edi, 0x1000
    mov ecx, 0x2000          ; clear 0x1000..0x8FFF
zero_tables:
    mov dword [edi], 0
    add edi, 4
    dec ecx
    jnz zero_tables

    mov dword [0x1000], 0x2007   ; PML4[0] -> PDPT
    mov dword [0x2000], 0x3007   ; PDPT[0] -> PD (identity, 2 MiB pages)
    mov dword [0x2008], 0x5007   ; PDPT[1] -> PD2
    mov dword [0x5010], 0x8007   ; PD2[2] (0x40400000) -> PT @0x8000

    xor eax, eax
    mov edi, 0x3000
    mov ecx, 128
fill_pd:
    mov edx, eax
    shl edx, 21
    or  edx, 0x87
    mov [edi], edx
    mov dword [edi+4], 0
    inc eax
    add edi, 8
    dec ecx
    jnz fill_pd

    xor eax, eax
    mov edi, 0x800
    mov ecx, 6
zero_gdt:
    mov dword [edi], 0
    add edi, 4
    dec ecx
    jnz zero_gdt
    mov dword [0x808], 0x0000FFFF
    mov dword [0x80C], 0x00AF9A00    ; 64-bit code
    mov dword [0x810], 0x0000FFFF
    mov dword [0x814], 0x00CF9200    ; data

    a32 lgdt [gdt_ptr]
    mov eax, cr4
    or  eax, (1 << 5) | (1 << 9) | (1 << 10)   ; PAE, OSFXSR, OSXMMEXCPT
    mov cr4, eax
    mov ecx, 0xC0000080
    rdmsr
    or  eax, 0x100 | 0x800
    wrmsr
    mov eax, 0x1000
    mov cr3, eax
    mov eax, cr0
    or  eax, 0x80000001
    mov cr0, eax
    db 0x66, 0xEA                ; jmp ptr16:32 (spelled out: some nasm versions drop o32)
    dd longmode
    dw 0x08

[bits 64]
longmode:
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov rsp, RUNSTACK

    ; IDT: vector 14 (#PF), 64-bit interrupt gate
    mov rdi, 0x6000
    mov rax, pf_handler
    mov word [rdi + 14*16], ax
    mov word [rdi + 14*16 + 2], 0x08
    mov byte [rdi + 14*16 + 5], 0x8E
    shr rax, 16
    mov word [rdi + 14*16 + 6], ax
    shr rax, 16
    mov qword [rdi + 14*16 + 8], rax
    lidt [rel idt_ptr]

    ; 4 pages at phys 0xC00000: pages 0,1 under test, 2 = source, 3 spare
    mov dword [abs PT + 0*8], 0xC00007
    mov dword [abs PT + 1*8], 0xC01007
    mov dword [abs PT + 2*8], 0xC02007
    mov dword [abs PT + 3*8], 0xC03007

    mov rax, cr0
    or  rax, 1 << 16             ; CR0.WP: ring 0 honours read-only pages
    mov cr0, rax
    jmp main

pf_handler:
    ; may not touch any register: the state is compared
    mov [abs HSAVE], rax
    mov rax, cr2
    shr eax, 12
    and eax, 3
    or  dword [rax*8 + PT], 2    ; the page becomes writable
    mov rax, cr2
    invlpg [rax]
    inc qword [abs FCNT]
    mov rax, [abs HSAVE]
    add rsp, 8                   ; error code
    iretq

%macro PRE_EQ8 0
    mov eax, [rdi]
    mov edx, [rdi + 4]
%endmacro
%macro PRE_EQ16 0
    mov rax, [rdi]
    mov rdx, [rdi + 8]
%endmacro

%assign ncases 0
%assign cmask 0
%macro CASE_ 4                   ; name, aligned-only, pre, instruction
    align 64
%%stub:
    %3
    %4
    jmp qword [abs RETP]
    times 62 - ($ - %%stub) db 0xCC
    db cmask
    db %2
%assign ncases ncases + 1
%endmacro
%macro CASE 3
    CASE_ %1, 0, {%2}, {%3}
%endmacro
%macro CASEU 3                   ; AF and OF are undefined (shifts, rotates)
%assign cmask 1
    CASE_ %1, 0, {%2}, {%3}
%assign cmask 0
%endmacro
%macro CASEB 3                   ; only CF is defined (bit test and modify)
%assign cmask 2
    CASE_ %1, 0, {%2}, {%3}
%assign cmask 0
%endmacro
%macro CASEA 3
    CASE_ %1, 1, {%2}, {%3}
%endmacro

align 64
stubs:
CASE add_q_r, {nop}, {add qword [rdi], rax}
CASE add_q_i, {nop}, {add qword [rdi], 0x12345}
CASE or_q_r, {nop}, {or qword [rdi], rax}
CASE or_q_i, {nop}, {or qword [rdi], 0x12345}
CASE adc_q_r, {nop}, {adc qword [rdi], rax}
CASE adc_q_i, {nop}, {adc qword [rdi], 0x12345}
CASE sbb_q_r, {nop}, {sbb qword [rdi], rax}
CASE sbb_q_i, {nop}, {sbb qword [rdi], 0x12345}
CASE and_q_r, {nop}, {and qword [rdi], rax}
CASE and_q_i, {nop}, {and qword [rdi], 0x12345}
CASE sub_q_r, {nop}, {sub qword [rdi], rax}
CASE sub_q_i, {nop}, {sub qword [rdi], 0x12345}
CASE xor_q_r, {nop}, {xor qword [rdi], rax}
CASE xor_q_i, {nop}, {xor qword [rdi], 0x12345}
CASE inc_q, {nop}, {inc qword [rdi]}
CASE dec_q, {nop}, {dec qword [rdi]}
CASE neg_q, {nop}, {neg qword [rdi]}
CASE not_q, {nop}, {not qword [rdi]}
CASEU rol_q_1, {nop}, {rol qword [rdi], 1}
CASEU rol_q_cl, {nop}, {rol qword [rdi], cl}
CASEU rol_q_i, {nop}, {rol qword [rdi], 3}
CASEU ror_q_1, {nop}, {ror qword [rdi], 1}
CASEU ror_q_cl, {nop}, {ror qword [rdi], cl}
CASEU ror_q_i, {nop}, {ror qword [rdi], 3}
CASEU rcl_q_1, {nop}, {rcl qword [rdi], 1}
CASEU rcl_q_cl, {nop}, {rcl qword [rdi], cl}
CASEU rcl_q_i, {nop}, {rcl qword [rdi], 3}
CASEU rcr_q_1, {nop}, {rcr qword [rdi], 1}
CASEU rcr_q_cl, {nop}, {rcr qword [rdi], cl}
CASEU rcr_q_i, {nop}, {rcr qword [rdi], 3}
CASEU shl_q_1, {nop}, {shl qword [rdi], 1}
CASEU shl_q_cl, {nop}, {shl qword [rdi], cl}
CASEU shl_q_i, {nop}, {shl qword [rdi], 3}
CASEU shr_q_1, {nop}, {shr qword [rdi], 1}
CASEU shr_q_cl, {nop}, {shr qword [rdi], cl}
CASEU shr_q_i, {nop}, {shr qword [rdi], 3}
CASEU sar_q_1, {nop}, {sar qword [rdi], 1}
CASEU sar_q_cl, {nop}, {sar qword [rdi], cl}
CASEU sar_q_i, {nop}, {sar qword [rdi], 3}
CASE xadd_q, {nop}, {lock xadd qword [rdi], rax}
CASE cmpxchg_q_ne, {nop}, {lock cmpxchg qword [rdi], rcx}
CASE cmpxchg_q_eq, {mov rax, [rdi]}, {lock cmpxchg qword [rdi], rcx}
CASE xchg_q, {nop}, {xchg qword [rdi], rax}
CASE mov_q_r, {nop}, {mov qword [rdi], rax}
CASE mov_q_i, {nop}, {mov qword [rdi], 0x1234}
CASE lock_add_q, {nop}, {lock add qword [rdi], rax}
CASE lock_dec_q, {nop}, {lock dec qword [rdi]}
CASE lock_or_q, {nop}, {lock or qword [rdi], 1}
CASE add_d_r, {nop}, {add dword [rdi], eax}
CASE add_d_i, {nop}, {add dword [rdi], 0x12345}
CASE or_d_r, {nop}, {or dword [rdi], eax}
CASE or_d_i, {nop}, {or dword [rdi], 0x12345}
CASE adc_d_r, {nop}, {adc dword [rdi], eax}
CASE adc_d_i, {nop}, {adc dword [rdi], 0x12345}
CASE sbb_d_r, {nop}, {sbb dword [rdi], eax}
CASE sbb_d_i, {nop}, {sbb dword [rdi], 0x12345}
CASE and_d_r, {nop}, {and dword [rdi], eax}
CASE and_d_i, {nop}, {and dword [rdi], 0x12345}
CASE sub_d_r, {nop}, {sub dword [rdi], eax}
CASE sub_d_i, {nop}, {sub dword [rdi], 0x12345}
CASE xor_d_r, {nop}, {xor dword [rdi], eax}
CASE xor_d_i, {nop}, {xor dword [rdi], 0x12345}
CASE inc_d, {nop}, {inc dword [rdi]}
CASE dec_d, {nop}, {dec dword [rdi]}
CASE neg_d, {nop}, {neg dword [rdi]}
CASE not_d, {nop}, {not dword [rdi]}
CASEU rol_d_1, {nop}, {rol dword [rdi], 1}
CASEU rol_d_cl, {nop}, {rol dword [rdi], cl}
CASEU rol_d_i, {nop}, {rol dword [rdi], 3}
CASEU ror_d_1, {nop}, {ror dword [rdi], 1}
CASEU ror_d_cl, {nop}, {ror dword [rdi], cl}
CASEU ror_d_i, {nop}, {ror dword [rdi], 3}
CASEU rcl_d_1, {nop}, {rcl dword [rdi], 1}
CASEU rcl_d_cl, {nop}, {rcl dword [rdi], cl}
CASEU rcl_d_i, {nop}, {rcl dword [rdi], 3}
CASEU rcr_d_1, {nop}, {rcr dword [rdi], 1}
CASEU rcr_d_cl, {nop}, {rcr dword [rdi], cl}
CASEU rcr_d_i, {nop}, {rcr dword [rdi], 3}
CASEU shl_d_1, {nop}, {shl dword [rdi], 1}
CASEU shl_d_cl, {nop}, {shl dword [rdi], cl}
CASEU shl_d_i, {nop}, {shl dword [rdi], 3}
CASEU shr_d_1, {nop}, {shr dword [rdi], 1}
CASEU shr_d_cl, {nop}, {shr dword [rdi], cl}
CASEU shr_d_i, {nop}, {shr dword [rdi], 3}
CASEU sar_d_1, {nop}, {sar dword [rdi], 1}
CASEU sar_d_cl, {nop}, {sar dword [rdi], cl}
CASEU sar_d_i, {nop}, {sar dword [rdi], 3}
CASE xadd_d, {nop}, {lock xadd dword [rdi], eax}
CASE cmpxchg_d_ne, {nop}, {lock cmpxchg dword [rdi], ecx}
CASE cmpxchg_d_eq, {mov eax, [rdi]}, {lock cmpxchg dword [rdi], ecx}
CASE xchg_d, {nop}, {xchg dword [rdi], eax}
CASE mov_d_r, {nop}, {mov dword [rdi], eax}
CASE mov_d_i, {nop}, {mov dword [rdi], 0x1234}
CASE lock_add_d, {nop}, {lock add dword [rdi], eax}
CASE lock_dec_d, {nop}, {lock dec dword [rdi]}
CASE lock_or_d, {nop}, {lock or dword [rdi], 1}
CASE add_w_r, {nop}, {add word [rdi], ax}
CASE add_w_i, {nop}, {add word [rdi], 0x1234}
CASE or_w_r, {nop}, {or word [rdi], ax}
CASE or_w_i, {nop}, {or word [rdi], 0x1234}
CASE adc_w_r, {nop}, {adc word [rdi], ax}
CASE adc_w_i, {nop}, {adc word [rdi], 0x1234}
CASE sbb_w_r, {nop}, {sbb word [rdi], ax}
CASE sbb_w_i, {nop}, {sbb word [rdi], 0x1234}
CASE and_w_r, {nop}, {and word [rdi], ax}
CASE and_w_i, {nop}, {and word [rdi], 0x1234}
CASE sub_w_r, {nop}, {sub word [rdi], ax}
CASE sub_w_i, {nop}, {sub word [rdi], 0x1234}
CASE xor_w_r, {nop}, {xor word [rdi], ax}
CASE xor_w_i, {nop}, {xor word [rdi], 0x1234}
CASE inc_w, {nop}, {inc word [rdi]}
CASE dec_w, {nop}, {dec word [rdi]}
CASE neg_w, {nop}, {neg word [rdi]}
CASE not_w, {nop}, {not word [rdi]}
CASEU rol_w_1, {nop}, {rol word [rdi], 1}
CASEU rol_w_cl, {nop}, {rol word [rdi], cl}
CASEU rol_w_i, {nop}, {rol word [rdi], 3}
CASEU ror_w_1, {nop}, {ror word [rdi], 1}
CASEU ror_w_cl, {nop}, {ror word [rdi], cl}
CASEU ror_w_i, {nop}, {ror word [rdi], 3}
CASEU rcl_w_1, {nop}, {rcl word [rdi], 1}
CASEU rcl_w_cl, {nop}, {rcl word [rdi], cl}
CASEU rcl_w_i, {nop}, {rcl word [rdi], 3}
CASEU rcr_w_1, {nop}, {rcr word [rdi], 1}
CASEU rcr_w_cl, {nop}, {rcr word [rdi], cl}
CASEU rcr_w_i, {nop}, {rcr word [rdi], 3}
CASEU shl_w_1, {nop}, {shl word [rdi], 1}
CASEU shl_w_cl, {nop}, {shl word [rdi], cl}
CASEU shl_w_i, {nop}, {shl word [rdi], 3}
CASEU shr_w_1, {nop}, {shr word [rdi], 1}
CASEU shr_w_cl, {nop}, {shr word [rdi], cl}
CASEU shr_w_i, {nop}, {shr word [rdi], 3}
CASEU sar_w_1, {nop}, {sar word [rdi], 1}
CASEU sar_w_cl, {nop}, {sar word [rdi], cl}
CASEU sar_w_i, {nop}, {sar word [rdi], 3}
CASE xadd_w, {nop}, {lock xadd word [rdi], ax}
CASE cmpxchg_w_ne, {nop}, {lock cmpxchg word [rdi], cx}
CASE cmpxchg_w_eq, {mov ax, [rdi]}, {lock cmpxchg word [rdi], cx}
CASE xchg_w, {nop}, {xchg word [rdi], ax}
CASE mov_w_r, {nop}, {mov word [rdi], ax}
CASE mov_w_i, {nop}, {mov word [rdi], 0x1234}
CASE lock_add_w, {nop}, {lock add word [rdi], ax}
CASE lock_dec_w, {nop}, {lock dec word [rdi]}
CASE lock_or_w, {nop}, {lock or word [rdi], 1}
CASE add_b_r, {nop}, {add byte [rdi], al}
CASE add_b_i, {nop}, {add byte [rdi], 5}
CASE or_b_r, {nop}, {or byte [rdi], al}
CASE or_b_i, {nop}, {or byte [rdi], 5}
CASE adc_b_r, {nop}, {adc byte [rdi], al}
CASE adc_b_i, {nop}, {adc byte [rdi], 5}
CASE sbb_b_r, {nop}, {sbb byte [rdi], al}
CASE sbb_b_i, {nop}, {sbb byte [rdi], 5}
CASE and_b_r, {nop}, {and byte [rdi], al}
CASE and_b_i, {nop}, {and byte [rdi], 5}
CASE sub_b_r, {nop}, {sub byte [rdi], al}
CASE sub_b_i, {nop}, {sub byte [rdi], 5}
CASE xor_b_r, {nop}, {xor byte [rdi], al}
CASE xor_b_i, {nop}, {xor byte [rdi], 5}
CASE inc_b, {nop}, {inc byte [rdi]}
CASE dec_b, {nop}, {dec byte [rdi]}
CASE neg_b, {nop}, {neg byte [rdi]}
CASE not_b, {nop}, {not byte [rdi]}
CASEU rol_b_1, {nop}, {rol byte [rdi], 1}
CASEU rol_b_cl, {nop}, {rol byte [rdi], cl}
CASEU rol_b_i, {nop}, {rol byte [rdi], 3}
CASEU ror_b_1, {nop}, {ror byte [rdi], 1}
CASEU ror_b_cl, {nop}, {ror byte [rdi], cl}
CASEU ror_b_i, {nop}, {ror byte [rdi], 3}
CASEU rcl_b_1, {nop}, {rcl byte [rdi], 1}
CASEU rcl_b_cl, {nop}, {rcl byte [rdi], cl}
CASEU rcl_b_i, {nop}, {rcl byte [rdi], 3}
CASEU rcr_b_1, {nop}, {rcr byte [rdi], 1}
CASEU rcr_b_cl, {nop}, {rcr byte [rdi], cl}
CASEU rcr_b_i, {nop}, {rcr byte [rdi], 3}
CASEU shl_b_1, {nop}, {shl byte [rdi], 1}
CASEU shl_b_cl, {nop}, {shl byte [rdi], cl}
CASEU shl_b_i, {nop}, {shl byte [rdi], 3}
CASEU shr_b_1, {nop}, {shr byte [rdi], 1}
CASEU shr_b_cl, {nop}, {shr byte [rdi], cl}
CASEU shr_b_i, {nop}, {shr byte [rdi], 3}
CASEU sar_b_1, {nop}, {sar byte [rdi], 1}
CASEU sar_b_cl, {nop}, {sar byte [rdi], cl}
CASEU sar_b_i, {nop}, {sar byte [rdi], 3}
CASE xadd_b, {nop}, {lock xadd byte [rdi], al}
CASE cmpxchg_b_ne, {nop}, {lock cmpxchg byte [rdi], cl}
CASE cmpxchg_b_eq, {mov al, [rdi]}, {lock cmpxchg byte [rdi], cl}
CASE xchg_b, {nop}, {xchg byte [rdi], al}
CASE mov_b_r, {nop}, {mov byte [rdi], al}
CASE mov_b_i, {nop}, {mov byte [rdi], 5}
CASE lock_add_b, {nop}, {lock add byte [rdi], al}
CASE lock_dec_b, {nop}, {lock dec byte [rdi]}
CASE lock_or_b, {nop}, {lock or byte [rdi], 1}
CASEU shld_q_i, {nop}, {shld qword [rdi], rdx, 5}
CASEU shld_q_cl, {nop}, {shld qword [rdi], rdx, cl}
CASEU shrd_q_i, {nop}, {shrd qword [rdi], rdx, 5}
CASEU shrd_q_cl, {nop}, {shrd qword [rdi], rdx, cl}
CASEB bts_q_i, {nop}, {bts qword [rdi], 5}
CASEB bts_q_r, {mov edx, 9}, {bts qword [rdi], rdx}
CASEB lock_bts_q_r, {mov edx, 9}, {lock bts qword [rdi], rdx}
CASEB btr_q_i, {nop}, {btr qword [rdi], 5}
CASEB btr_q_r, {mov edx, 9}, {btr qword [rdi], rdx}
CASEB lock_btr_q_r, {mov edx, 9}, {lock btr qword [rdi], rdx}
CASEB btc_q_i, {nop}, {btc qword [rdi], 5}
CASEB btc_q_r, {mov edx, 9}, {btc qword [rdi], rdx}
CASEB lock_btc_q_r, {mov edx, 9}, {lock btc qword [rdi], rdx}
CASEU shld_d_i, {nop}, {shld dword [rdi], edx, 5}
CASEU shld_d_cl, {nop}, {shld dword [rdi], edx, cl}
CASEU shrd_d_i, {nop}, {shrd dword [rdi], edx, 5}
CASEU shrd_d_cl, {nop}, {shrd dword [rdi], edx, cl}
CASEB bts_d_i, {nop}, {bts dword [rdi], 5}
CASEB bts_d_r, {mov edx, 9}, {bts dword [rdi], edx}
CASEB lock_bts_d_r, {mov edx, 9}, {lock bts dword [rdi], edx}
CASEB btr_d_i, {nop}, {btr dword [rdi], 5}
CASEB btr_d_r, {mov edx, 9}, {btr dword [rdi], edx}
CASEB lock_btr_d_r, {mov edx, 9}, {lock btr dword [rdi], edx}
CASEB btc_d_i, {nop}, {btc dword [rdi], 5}
CASEB btc_d_r, {mov edx, 9}, {btc dword [rdi], edx}
CASEB lock_btc_d_r, {mov edx, 9}, {lock btc dword [rdi], edx}
CASEU shld_w_i, {nop}, {shld word [rdi], dx, 5}
CASEU shld_w_cl, {nop}, {shld word [rdi], dx, cl}
CASEU shrd_w_i, {nop}, {shrd word [rdi], dx, 5}
CASEU shrd_w_cl, {nop}, {shrd word [rdi], dx, cl}
CASEB bts_w_i, {nop}, {bts word [rdi], 5}
CASEB bts_w_r, {mov edx, 9}, {bts word [rdi], dx}
CASEB lock_bts_w_r, {mov edx, 9}, {lock bts word [rdi], dx}
CASEB btr_w_i, {nop}, {btr word [rdi], 5}
CASEB btr_w_r, {mov edx, 9}, {btr word [rdi], dx}
CASEB lock_btr_w_r, {mov edx, 9}, {lock btr word [rdi], dx}
CASEB btc_w_i, {nop}, {btc word [rdi], 5}
CASEB btc_w_r, {mov edx, 9}, {btc word [rdi], dx}
CASEB lock_btc_w_r, {mov edx, 9}, {lock btc word [rdi], dx}
CASE setz, {nop}, {setz [rdi]}
CASE setc, {nop}, {setc [rdi]}
CASE setl, {nop}, {setl [rdi]}
CASE setne, {nop}, {setne [rdi]}
CASE pop_q, {nop}, {pop qword [rdi]}
CASE pop_w, {nop}, {pop word [rdi]}
CASE cmpxchg8b_ne, {nop}, {lock cmpxchg8b [rdi]}
CASE cmpxchg8b_eq, {PRE_EQ8}, {lock cmpxchg8b [rdi]}
CASEA cmpxchg16b_ne, {nop}, {lock cmpxchg16b [rdi]}
CASEA cmpxchg16b_eq, {PRE_EQ16}, {lock cmpxchg16b [rdi]}
CASE mov_sreg_ds, {nop}, {mov [rdi], ds}
CASE mov_sreg_ss, {nop}, {mov [rdi], ss}
CASE sgdt, {nop}, {sgdt [rdi]}
CASE sidt, {nop}, {sidt [rdi]}
CASE sldt, {nop}, {sldt [rdi]}
CASE smsw, {nop}, {smsw [rdi]}
CASE movnti_q, {nop}, {movnti [rdi], rax}
CASE movnti_d, {nop}, {movnti [rdi], eax}
CASE movups, {nop}, {movups [rdi], xmm0}
CASE movdqu, {nop}, {movdqu [rdi], xmm1}
CASE movq_x, {nop}, {movq [rdi], xmm0}
CASE movd_x, {nop}, {movd [rdi], xmm0}
CASE movss, {nop}, {movss [rdi], xmm0}
CASE movsd_x, {nop}, {movsd [rdi], xmm0}
CASE movlps, {nop}, {movlps [rdi], xmm0}
CASE movhps, {nop}, {movhps [rdi], xmm0}
CASE movlpd, {nop}, {movlpd [rdi], xmm0}
CASE movhpd, {nop}, {movhpd [rdi], xmm0}
CASE stmxcsr, {nop}, {stmxcsr [rdi]}
CASE pextrb, {nop}, {pextrb [rdi], xmm0, 3}
CASE pextrw, {nop}, {pextrw [rdi], xmm0, 2}
CASE pextrd, {nop}, {pextrd [rdi], xmm0, 1}
CASE pextrq, {nop}, {pextrq [rdi], xmm0, 1}
CASE extractps, {nop}, {extractps [rdi], xmm0, 1}
CASE maskmovdqu, {nop}, {maskmovdqu xmm0, xmm1}
CASE fst_d, {nop}, {fst dword [rdi]}
CASE fst_q, {nop}, {fst qword [rdi]}
CASE fstp_d, {nop}, {fstp dword [rdi]}
CASE fstp_q, {nop}, {fstp qword [rdi]}
CASE fstp_t, {nop}, {fstp tword [rdi]}
CASE fist_w, {nop}, {fist word [rdi]}
CASE fist_d, {nop}, {fist dword [rdi]}
CASE fistp_w, {nop}, {fistp word [rdi]}
CASE fistp_d, {nop}, {fistp dword [rdi]}
CASE fistp_q, {nop}, {fistp qword [rdi]}
CASE fisttp_w, {nop}, {fisttp word [rdi]}
CASE fisttp_d, {nop}, {fisttp dword [rdi]}
CASE fisttp_q, {nop}, {fisttp qword [rdi]}
CASE fbstp, {nop}, {fbstp [rdi]}
CASE fnstcw, {nop}, {fnstcw [rdi]}
CASE fnstsw_m, {nop}, {fnstsw [rdi]}
CASE fnstenv, {nop}, {fnstenv [rdi]}
CASE fnsave, {nop}, {fnsave [rdi]}
CASE stosb, {nop}, {stosb}
CASE stosw, {nop}, {stosw}
CASE stosd, {nop}, {stosd}
CASE stosq, {nop}, {stosq}
CASE rep_stosq, {nop}, {rep stosq}
CASE rep_stosb, {nop}, {rep stosb}
CASE rep_stosd, {nop}, {rep stosd}
CASE movsb, {nop}, {movsb}
CASE movsw, {nop}, {movsw}
CASE movsd_s, {nop}, {movsd}
CASE movsq, {nop}, {movsq}
CASE rep_movsq, {nop}, {rep movsq}
CASE rep_movsb, {nop}, {rep movsb}
CASEA movdqa, {nop}, {movdqa [rdi], xmm0}
CASEA movaps, {nop}, {movaps [rdi], xmm1}
CASEA movapd, {nop}, {movapd [rdi], xmm0}
CASEA movntdq, {nop}, {movntdq [rdi], xmm0}
CASEA movntps, {nop}, {movntps [rdi], xmm1}
CASEA fxsave, {nop}, {fxsave [rdi]}
CASEA fxsave64, {nop}, {fxsave64 [rdi]}

align 64
main:
    mov dword [abs CASEIDX], 0
.case_loop:
    mov dword [abs CFGIDX], 0
    mov qword [abs RES], 0
.cfg_loop:
    mov rsp, RUNSTACK
    call do_run
    inc dword [abs CFGIDX]
    cmp dword [abs CFGIDX], 5
    jb .cfg_loop
    call fold
    inc dword [abs CASEIDX]
    cmp dword [abs CASEIDX], ncases
    jb .case_loop
    mov al, 'K'
    mov dx, 0x3F8
    out dx, al
.hang:
    hlt
    jmp .hang

; stub address of the current case in rsi, aligned-only flag in al
get_stub:
    mov esi, [abs CASEIDX]
    shl esi, 6
    add esi, stubs
    movzx eax, byte [rsi + 63]
    ret

do_run:
    call get_stub
    mov [abs CURSTUB], rsi
    movzx edx, byte [rsi + 62]
    mov edx, [rdx*4 + masktab]    ; flags compared for this case
    mov [abs FMASK], rdx
    mov ebx, [abs CFGIDX]
    cmp ebx, 2
    jb .go
    test eax, eax
    jnz .skip                     ; aligned-only cases only run configs 0 and 1
.go:
    shl ebx, 5
    add ebx, cfgtab               ; rbx = this configuration's row
    ; 1. both pages writable, fill with a pattern
    mov dword [abs PT + 0*8], 0xC00007
    mov dword [abs PT + 1*8], 0xC01007
    invlpg [abs PAGE]
    invlpg [abs PAGE + 0x1000]
    mov rdi, PAGE
    mov ecx, 3 * 512
    mov rax, 0x0123456789ABCDEF
    mov rdx, 0x1111111111111111
.fill:
    mov [rdi], rax
    add rax, rdx
    add rdi, 8
    dec ecx
    jnz .fill
    mov rax, 0x7777777777777777
    mov [abs SAFE], rax
    ; 2. this run's page permissions, and cache the read translations
    mov eax, [rbx]
    or  eax, 0xC00000
    mov [abs PT + 0*8], eax
    mov eax, [rbx + 8]
    or  eax, 0xC01000
    mov [abs PT + 1*8], eax
    invlpg [abs PAGE]
    invlpg [abs PAGE + 0x1000]
    mov rax, [abs PAGE]
    mov rax, [abs PAGE + 0x1000]
    mov qword [abs FCNT], 0
    mov qword [abs RETP], .cont
    ; 3. identical architectural state for every run
    fninit
    fld qword [rel fconst_a]
    fld qword [rel fconst_b]
    movdqu xmm0, [rel xconst_0]
    movdqu xmm1, [rel xconst_1]
    mov rdi, [rbx + 16]
    add rdi, PAGE
    mov rax, 0x000000000004000A
    mov rcx, 3
    mov rdx, 0xD1D2D3D4D5D6D7D8
    mov rbx, 0xB1B2B3B4B5B6B7B8
    mov rbp, 0xBBBBBBBBBBBBBB01
    mov rsi, PAGE + 0x2000 + 0x40
    mov r8,  0x8888888888888801
    mov r9,  0x9999999999999902
    mov r10, 0xAAAAAAAAAAAAAA03
    mov r11, 0xBBBBBBBBBBBBBB04
    mov r12, 0xCCCCCCCCCCCCCC05
    mov r13, 0xDDDDDDDDDDDDDD06
    mov r14, 0xEEEEEEEEEEEEEE07
    mov r15, 0xFFFFFFFFFFFFFF08
    mov rsp, SAFE
    push 3                        ; CF set
    popfq
    jmp qword [abs CURSTUB]
.cont:
    ; the instruction under test is done: snapshot everything
    mov [abs S + 0*8], rax
    mov [abs S + 1*8], rcx
    mov [abs S + 2*8], rdx
    mov [abs S + 3*8], rbx
    mov [abs S + 4*8], rsp
    mov [abs S + 5*8], rbp
    mov [abs S + 6*8], rsi
    mov [abs S + 7*8], rdi
    mov [abs S + 8*8], r8
    mov [abs S + 9*8], r9
    mov [abs S + 10*8], r10
    mov [abs S + 11*8], r11
    mov [abs S + 12*8], r12
    mov [abs S + 13*8], r13
    mov [abs S + 14*8], r14
    mov [abs S + 15*8], r15
    mov rsp, S + 0x400
    pushfq
    pop rax
    and rax, [abs FMASK]
    mov [abs S + 16*8], rax
    movdqu [abs S + 17*8], xmm0
    movdqu [abs S + 19*8], xmm1
    fnstsw ax
    movzx eax, ax
    mov [abs S + 21*8], rax
    mov rsi, PAGE
    mov ecx, 1024
    xor eax, eax
.sum:
    rol rax, 5
    xor rax, [rsi]
    add rsi, 8
    dec ecx
    jnz .sum
    mov [abs S + 22*8], rax
    mov rax, [abs FCNT]
    mov [abs S + 23*8], rax
    ; copy to this configuration's buffer
    mov ebx, [abs CFGIDX]
    shl ebx, 8
    lea rdi, [rbx + BUFS]
    mov esi, S
    mov ecx, 24
    cld
    rep movsq
.skip:
    mov rsp, RUNSTACK - 8
    ret

; compare buffer %1 with %2 over the NSLOT state qwords; sets bit %3 if equal,
; and records where the first difference is
%macro EQBIT 3
    mov esi, BUFS + %1 * 0x100
    mov edi, BUFS + %2 * 0x100
    mov ecx, NSLOT
    cld
    repe cmpsq
    je %%eq
    mov eax, NSLOT - 1
    sub eax, ecx
    add eax, 1 + (%1 << 8)
    mov ebx, [abs CASEIDX]
    cmp qword [abs DIFF + rbx*8], 0
    jne %%eq
    mov [abs DIFF + rbx*8], rax
    jmp %%done
%%eq:
    or dword [abs RES], 1 << %3
%%done:
%endmacro
%macro FAULTBIT 2                ; buffer, bit
    cmp qword [abs BUFS + %1 * 0x100 + NSLOT * 8], 0
    je %%no
    or dword [abs RES], 1 << %2
%%no:
%endmacro

fold:
    call get_stub
    mov r8d, eax                  ; aligned-only
    EQBIT 1, 0, 0
    FAULTBIT 1, 3
    ; bit 6: the writable reference runs didn't fault
    cmp qword [abs BUFS + 0 * 0x100 + NSLOT * 8], 0
    jne .nobit6
    test r8d, r8d
    jnz .bit6
    cmp qword [abs BUFS + 2 * 0x100 + NSLOT * 8], 0
    jne .nobit6
.bit6:
    or dword [abs RES], 1 << 6
.nobit6:
    test r8d, r8d
    jnz .store
    EQBIT 3, 2, 1
    FAULTBIT 3, 4
    EQBIT 4, 2, 2
    FAULTBIT 4, 5
.store:
    mov eax, [abs CASEIDX]
    mov rdx, [abs RES]
    mov [abs RESULTS + rax*8], rdx
    ret

; rows of 32 bytes: page 0 pte flags, page 1 pte flags, operand offset, unused
; 0: reference, all writable        1: page 0 and 1 read-only
; 2: reference at the page boundary 3: page 1 read-only  4: page 0 read-only
align 16
cfgtab:
    dq 7, 7, 0,      0
    dq 5, 5, 0,      0
    dq 7, 7, 0xFFC,  0
    dq 7, 5, 0xFFC,  0
    dq 5, 7, 0xFFC,  0
masktab: dd FLAGMASK, FLAGMASK & ~0x810, 1   ; default; no AF/OF; CF only
fconst_a: dq 1.5
fconst_b: dq -2.75
align 16
xconst_0: dq 0x1020304050607080, 0x1121314151617181
xconst_1: dq 0x8070605040302010, 0x8171615141312111

idt_ptr:
    dw 0xFF
    dq 0x6000
gdt_ptr:
    dw 0x17
    dd 0x800

[bits 16]
    times 0xFFF0 - ($ - $$) db 0x90
reset_vector:
    jmp 0xF000:start16 - 0xF0000
    times 16 - 5 db 0x90
