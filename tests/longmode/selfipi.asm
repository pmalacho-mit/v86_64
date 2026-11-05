; APIC self-IPI from compiled 64-bit code (see selfipi.js).
;
; A hot loop (copied to RAM so that it gets compiled) sends itself an IPI by
; writing the ICR, with stack operations around the write, and waits for the
; handler. The interrupt must be taken on an instruction boundary: delivering it
; in the middle of the ICR write pushes a stale rip and switches stacks under the
; compiled code, which then continues with its cached registers.
;
; Results are stored as qwords at 0x90000; 'K' is written to the serial port when done.

org 0xF0000

%define RESULTS 0x90000
%define COUNT 0x9000
%define ITERATIONS 20000
%define IF_LOST 0x9008

[bits 16]
start16:
    cli

    ; page tables: PML4 @0x1000, PDPT @0x2000, PD @0x3000 (0..256 MiB), PD @0x4000 (3..4 GiB)
    mov edi, 0x1000
    mov ecx, 0x1000
zero_tables:
    mov dword [edi], 0
    add edi, 4
    dec ecx
    jnz zero_tables

    mov dword [0x1000], 0x2007
    mov dword [0x2000], 0x3007
    mov dword [0x2018], 0x4007

    xor eax, eax
    mov edi, 0x3000
    mov ecx, 128
fill_pd:
    mov edx, eax
    shl edx, 21
    or  edx, 0x87
    mov [edi], edx
    inc eax
    add edi, 8
    dec ecx
    jnz fill_pd

    ; the APIC page: 2 MiB page at 0xFEE00000, uncached
    mov dword [0x4000 + 0x1F7*8], 0xFEE0009F

    ; GDT at 0x800: null, 64-bit code (0x08), data (0x10)
    xor eax, eax
    mov edi, 0x800
    mov ecx, 6
zero_gdt:
    mov dword [edi], 0
    add edi, 4
    dec ecx
    jnz zero_gdt
    mov dword [0x808], 0x0000FFFF
    mov dword [0x80C], 0x00AF9A00
    mov dword [0x810], 0x0000FFFF
    mov dword [0x814], 0x00CF9200

    a32 lgdt [gdt_ptr]

    mov eax, cr4
    or  eax, 1 << 5              ; CR4.PAE
    mov cr4, eax

    mov ecx, 0xC0000080          ; IA32_EFER
    rdmsr
    or  eax, 0x100               ; EFER.LME
    wrmsr

    mov eax, 0x1000
    mov cr3, eax

    mov eax, cr0
    or  eax, 0x80000001          ; CR0.PE|CR0.PG
    mov cr0, eax

    db 0x66, 0xEA
    dd longmode
    dw 0x08

[bits 64]
longmode:
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov rsp, 0x80000
    mov r15, RESULTS

    ; IDT at 0x6000 with one entry: vector 0xF6 -> ipi_handler (interrupt gate)
    mov rdi, 0x6000
    mov ecx, 0x200
    xor eax, eax
.idt_zero:
    mov [rdi], rax
    add rdi, 8
    dec ecx
    jnz .idt_zero
    mov rax, ipi_handler
    mov rdi, 0x6000 + 0xF6*16
    mov word [rdi], ax
    mov word [rdi + 2], 0x08
    mov byte [rdi + 4], 0
    mov byte [rdi + 5], 0x8E
    shr rax, 16
    mov word [rdi + 6], ax
    shr rax, 16
    mov qword [rdi + 8], rax
    lidt [rel idt_ptr]

    ; software-enable the APIC, accept all priorities
    mov rdi, 0xFEE00000
    mov dword [rdi + 0xF0], 0x1FF
    mov dword [rdi + 0x80], 0

    ; copy the hot loop to RAM
    lea rsi, [rel hot_start]
    mov rdi, 0x10000
    mov rcx, hot_end - hot_start
    rep movsb

    mov qword [abs COUNT], 0
    mov qword [abs IF_LOST], 0
    mov rbx, 0x1122334455667788
    mov r12, rsp
    sti
    mov rax, 0x10000
    call rax

    mov rax, [abs COUNT]
    mov [r15 + 0*8], rax         ; ITERATIONS: every IPI handled
    mov rax, rsp
    sub rax, r12
    mov [r15 + 1*8], rax         ; 0: stack pointer intact
    mov [r15 + 2*8], rbx         ; 0x1122334455667788: rbx restored by pop
    pushfq
    pop rax
    and rax, 0x200
    mov [r15 + 3*8], rax         ; 0x200: IF still set
    mov [r15 + 4*8], r13         ; 0: no IPI timed out
    mov rax, [abs IF_LOST]
    mov [r15 + 5*8], rax         ; 0: IF never found clear after sending

    mov dx, 0x3F8
    mov al, 'K'
    out dx, al
.done:
    cli
    hlt
    jmp .done

; position independent: only absolute data addresses
hot_start:
    mov ecx, ITERATIONS
    xor r13d, r13d
    xor r14d, r14d
    mov rsi, 0xFEE00300          ; ICR low
.loop:
    ; like Linux's default_send_IPI_self(IRQ_WORK_VECTOR)
    mov edi, 0xF6
    call .send_self
    pushfq
    pop rax
    test eax, 0x200
    jnz .if_ok
    inc qword [abs IF_LOST]
    sti
.if_ok:
    inc r14
    mov edx, 1000000
.wait:
    cmp [abs COUNT], r14
    je .handled
    dec edx
    jnz .wait
    inc r13
    mov [abs COUNT], r14
.handled:
    dec ecx
    jnz .loop
    ret
.send_self:
    push rbx
    mov rbx, 0x5555
.busy:
    mov eax, [rsi]
    test ah, 0x10
    jnz .busy
    or edi, 0x40000
    mov [rsi], edi
    pop rbx
    ret
hot_end:

ipi_handler:
    push rax
    inc qword [abs COUNT]
    mov rax, 0xFEE000B0
    mov dword [rax], 0           ; EOI
    pop rax
    iretq

idt_ptr:
    dw 0xFFF
    dq 0x6000

gdt_ptr:
    dw 0x17
    dd 0x800

[bits 16]
    times 0xFFF0 - ($ - $$) db 0x90
reset_vector:
    jmp 0xF000:start16 - 0xF0000
    times 16 - 5 db 0x90
