# gdb helpers for gen_fixtures_qemu.sh (provisional fixtures from qemu-user).
#
# qemu's gdb stub reports the x87 tag word as 0 ("all valid") and keeps
# whatever it has in the st registers, so gdb-extract-def can't tell which
# registers real hardware would call "special" (tag 2) and aborts with
# "Invalid floating value found in program" when it prints an unsupported
# 80-bit encoding. These functions classify the registers from their raw
# contents instead, which is what a real CPU's full tag word does for
# non-empty registers. Empty registers can't be recognised.
#
# $qemu_st_special(i)  1 if st(i) is NaN/infinity/denormal/unnormal, else 0
# $qemu_ftag()         0xAAAA if the registers look like MMX state (any
#                      pseudo-denormal: exponent 0 with the integer bit set,
#                      which x87 arithmetic never produces but random MMX data
#                      does), which run.js expects as "compare mm registers";
#                      else gdb's own $ftag (0 under qemu)

import re
import gdb


def _raw_st(i):
    text = gdb.execute("info registers st%d" % i, to_string=True)
    m = re.search(r"\(raw 0x([0-9a-fA-F]+)\)", text)
    return int(m.group(1), 16)


def _special(raw):
    exponent = (raw >> 64) & 0x7FFF
    mantissa = raw & 0xFFFFFFFFFFFFFFFF
    if exponent == 0x7FFF:
        return True
    if exponent == 0:
        return mantissa != 0
    return (mantissa >> 63) == 0


def _mmx_state():
    for i in range(8):
        raw = _raw_st(i)
        if (raw >> 64) & 0x7FFF == 0 and (raw >> 63) & 1:
            return True
    return False


class StSpecial(gdb.Function):
    def __init__(self):
        super().__init__("qemu_st_special")

    def invoke(self, i):
        return 1 if _special(_raw_st(int(i))) else 0


class Ftag(gdb.Function):
    def __init__(self):
        super().__init__("qemu_ftag")

    def invoke(self):
        if _mmx_state():
            return 0xAAAA
        return int(gdb.parse_and_eval("$ftag"))


StSpecial()
Ftag()
