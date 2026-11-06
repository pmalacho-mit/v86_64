// Native x86-64 Linux runner for tests/fuzz 64-bit cases: executes each case's
// 64-bit body on the host CPU and reports the final machine state, so that v86
// can be compared against real hardware (tests/fuzz/hw.js).
//
// Freestanding (no libc): builds with any x86-64 gcc/clang, including a cross
// compiler without an x86 libc:
//     x86_64-linux-gnu-gcc -O2 -static -nostdlib -ffreestanding -fno-stack-protector
//         -fno-pie -no-pie -o runner tests/fuzz/hw/runner.c
//
// stdin: a batch written by `tests/fuzz/hw.js export`:
//     "V86FUZZ1" u32 count, then per case:
//     u16 name_len, name, u64 entry, u64 done_addr, u32 n_regions,
//     n_regions * (u32 addr, u32 len, bytes)
// stdout, per case:
//     "CASE" u16 name_len, name, u32 status (0: signal, 1: timeout/crash),
//     u32 signo, u32 si_code, u32 0, u64 trapno, u64 err, u64 cr2,
//     u64 gregs[23] (glibc's REG_* order), u8 fxsave[512], u8 scratch[SCRATCH_SIZE]
//
// Each case runs in a forked child. The child maps the low memory that v86's
// long-mode identity map covers (everything from 0x10000 up to 2 MiB; Linux
// can't map below mmap_min_addr), writes the case's memory regions, installs
// handlers for the synchronous signals on an alternate stack and jumps to the
// 64-bit body. The body ends with `hlt`, which raises #GP in ring 3: a signal
// at done_addr with trapno 13 means "completed". Any other signal is the
// exception the case raised. The handler copies the signal context and the
// scratch region to the parent and exits.

typedef unsigned long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
typedef long i64;

#define LOW_START 0x10000UL
#define LOW_END 0x200000UL
#define SCRATCH_ADDR 0x100000UL
#define SCRATCH_SIZE 0x2000UL

#define SYS_read 0
#define SYS_write 1
#define SYS_close 3
#define SYS_mmap 9
#define SYS_rt_sigaction 13
#define SYS_pipe 22
#define SYS_alarm 37
#define SYS_fork 57
#define SYS_wait4 61
#define SYS_sigaltstack 131
#define SYS_exit_group 231

static i64 sys6(i64 n, i64 a, i64 b, i64 c, i64 d, i64 e, i64 f)
{
    i64 ret;
    register i64 r10 __asm__("r10") = d;
    register i64 r8 __asm__("r8") = e;
    register i64 r9 __asm__("r9") = f;
    __asm__ volatile("syscall" : "=a"(ret) : "a"(n), "D"(a), "S"(b), "d"(c), "r"(r10), "r"(r8), "r"(r9)
                     : "rcx", "r11", "memory");
    return ret;
}
#define sys3(n, a, b, c) sys6(n, (i64)(a), (i64)(b), (i64)(c), 0, 0, 0)

static void die(const char *msg)
{
    u64 n = 0;
    while(msg[n]) n++;
    sys3(SYS_write, 2, msg, n);
    sys3(SYS_exit_group, 2, 0, 0);
}

static void write_all(int fd, const void *buf, u64 n)
{
    const u8 *p = buf;
    while(n)
    {
        i64 r = sys3(SYS_write, fd, p, n);
        if(r <= 0) die("runner: write failed\n");
        p += r;
        n -= r;
    }
}

static int read_all(int fd, void *buf, u64 n)
{
    u8 *p = buf;
    while(n)
    {
        i64 r = sys3(SYS_read, fd, p, n);
        if(r <= 0) return -1;
        p += r;
        n -= r;
    }
    return 0;
}

static void *memcpy_(void *d, const void *s, u64 n)
{
    u8 *dp = d;
    const u8 *sp = s;
    while(n--) *dp++ = *sp++;
    return d;
}
// gcc may emit calls to these even when freestanding
void *memcpy(void *d, const void *s, u64 n) { return memcpy_(d, s, n); }
void *memset(void *d, int c, u64 n)
{
    u8 *p = d;
    while(n--) *p++ = (u8)c;
    return d;
}

// --- kernel ABI structures (x86-64) ---
struct kernel_sigaction
{
    void (*handler)(int, void *, void *);
    u64 flags;
    void (*restorer)(void);
    u64 mask;
};
#define SA_SIGINFO 4
#define SA_ONSTACK 0x08000000
#define SA_NODEFER 0x40000000
#define SA_RESTORER 0x04000000 // required on x86-64: without it the kernel can't build the frame

struct stack_t_
{
    void *sp;
    int flags;
    u64 size;
};

// ucontext_t: uc_flags(8) uc_link(8) uc_stack(24) then mcontext: gregs[23], fpregs*
struct sigcontext_
{
    u64 gregs[23];
    u8 *fpregs; // fxsave image (512 bytes)
};
struct ucontext_
{
    u64 flags;
    void *link;
    struct stack_t_ stack;
    struct sigcontext_ mc;
};

// glibc REG_* indices
#define REG_ERR 19
#define REG_TRAPNO 20
#define REG_CR2 22

static int out_fd;

struct result
{
    u32 status, signo, si_code, pad;
    u64 trapno, err, cr2;
    u64 gregs[23];
    u8 fxsave[512];
    u8 scratch[SCRATCH_SIZE];
};
static struct result result_buf;

static void handler(int sig, void *info, void *uc_)
{
    struct ucontext_ *uc = uc_;
    struct result *r = &result_buf;
    r->status = 0;
    r->signo = sig;
    r->si_code = ((int *)info)[2]; // siginfo_t: si_signo, si_errno, si_code
    r->trapno = uc->mc.gregs[REG_TRAPNO];
    r->err = uc->mc.gregs[REG_ERR];
    r->cr2 = uc->mc.gregs[REG_CR2];
    memcpy_(r->gregs, uc->mc.gregs, sizeof r->gregs);
    if(uc->mc.fpregs) memcpy_(r->fxsave, uc->mc.fpregs, 512);
    memcpy_(r->scratch, (void *)SCRATCH_ADDR, SCRATCH_SIZE);
    write_all(out_fd, r, sizeof *r);
    sys3(SYS_exit_group, 0, 0, 0);
}

void restore_rt(void);
__asm__(
    ".globl restore_rt\n"
    "restore_rt:\n"
    "    mov $15, %eax\n" // rt_sigreturn
    "    syscall\n");

static u8 altstack[64 * 1024] __attribute__((aligned(16)));

static void run_child(u64 entry, int fd)
{
    out_fd = fd;
    struct stack_t_ ss = { altstack, 0, sizeof altstack };
    if(sys3(SYS_sigaltstack, &ss, 0, 0) != 0) die("runner: sigaltstack\n");

    struct kernel_sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.handler = handler;
    sa.flags = SA_SIGINFO | SA_ONSTACK | SA_NODEFER | SA_RESTORER;
    sa.restorer = restore_rt; // never used: the handler exits
    const int sigs[] = { 4 /*ILL*/, 5 /*TRAP*/, 7 /*BUS*/, 8 /*FPE*/, 11 /*SEGV*/ };
    for(u64 i = 0; i < sizeof sigs / sizeof sigs[0]; i++)
    {
        if(sys6(SYS_rt_sigaction, sigs[i], (i64)&sa, 0, 8, 0, 0) != 0) die("runner: sigaction\n");
    }
    sys3(SYS_alarm, 2, 0, 0); // a case that loops is killed by SIGALRM

    // Jump to the body. It sets all GPRs (including rsp), the FPU/SSE state
    // and the flags itself.
    __asm__ volatile("jmp *%0" : : "r"(entry));
    __builtin_unreachable();
}

static u8 region_buf[LOW_END];

void _start_c(void)
{
    char magic[8];
    u32 count;
    if(read_all(0, magic, 8) || read_all(0, &count, 4)) die("runner: bad header\n");
    const char *want = "V86FUZZ1";
    for(int i = 0; i < 8; i++) if(magic[i] != want[i]) die("runner: bad magic\n");

    for(u32 c = 0; c < count; c++)
    {
        u16 name_len;
        char name[256];
        u64 entry, done_addr;
        u32 n_regions;
        if(read_all(0, &name_len, 2) || name_len > sizeof name || read_all(0, name, name_len) ||
           read_all(0, &entry, 8) || read_all(0, &done_addr, 8) || read_all(0, &n_regions, 4))
        {
            die("runner: truncated case\n");
        }
        (void)done_addr;

        // stage the memory image: regions are written in the child
        static u32 reg_addr[16], reg_len[16];
        u64 used = 0;
        if(n_regions > 16) die("runner: too many regions\n");
        for(u32 i = 0; i < n_regions; i++)
        {
            if(read_all(0, &reg_addr[i], 4) || read_all(0, &reg_len[i], 4)) die("runner: truncated region\n");
            if(reg_addr[i] < LOW_START || reg_addr[i] + (u64)reg_len[i] > LOW_END || used + reg_len[i] > sizeof region_buf)
            {
                die("runner: region out of range\n");
            }
            if(read_all(0, region_buf + used, reg_len[i])) die("runner: truncated region data\n");
            used += reg_len[i];
        }

        int fds[2];
        if(sys3(SYS_pipe, fds, 0, 0) != 0) die("runner: pipe\n");
        i64 pid = sys3(SYS_fork, 0, 0, 0);
        if(pid < 0) die("runner: fork\n");
        if(pid == 0)
        {
            sys3(SYS_close, fds[0], 0, 0);
            // MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED, rwx
            i64 m = sys6(SYS_mmap, LOW_START, LOW_END - LOW_START, 7, 0x32, -1, 0);
            if(m != (i64)LOW_START) die("runner: mmap of low memory failed\n");
            u64 off = 0;
            for(u32 i = 0; i < n_regions; i++)
            {
                memcpy_((void *)(u64)reg_addr[i], region_buf + off, reg_len[i]);
                off += reg_len[i];
            }
            run_child(entry, fds[1]);
        }
        sys3(SYS_close, fds[1], 0, 0);

        // the result fits in the pipe buffer, but read before waiting anyway
        static struct result r;
        int ok = read_all(fds[0], &r, sizeof r) == 0;
        sys3(SYS_close, fds[0], 0, 0);
        int wstatus = 0;
        sys6(SYS_wait4, pid, (i64)&wstatus, 0, 0, 0, 0);
        if(!ok)
        {
            memset(&r, 0, sizeof r);
            r.status = 1;
            r.signo = wstatus & 0x7F; // the signal that killed the child, if any
        }

        write_all(1, "CASE", 4);
        write_all(1, &name_len, 2);
        write_all(1, name, name_len);
        write_all(1, &r, sizeof r);
    }
    sys3(SYS_exit_group, 0, 0, 0);
}

__asm__(
    ".globl _start\n"
    "_start:\n"
    "    xor %rbp, %rbp\n"
    "    and $-16, %rsp\n"
    "    call _start_c\n"
    "    hlt\n");
