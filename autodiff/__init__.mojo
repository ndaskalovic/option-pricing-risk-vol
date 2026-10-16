"""Compile-time AD: write kernels once against `ADNum`; the compiler emits straight-line forward and reverse code."""
from .ops import Node
from .adnum import ADNum, maximum, minimum
from .trace import Kernel, TraceNum, trace
from .compiled import Compiled
from .scan import Scan
from .real import Real, Vec, Dual
