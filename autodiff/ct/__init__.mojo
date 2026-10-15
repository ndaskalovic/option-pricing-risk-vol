"""Compile-time AAD: write kernels once against `ADNum`, compile forward + reverse code."""
from .ops import Node
from .adnum import ADNum, maximum, minimum, absolute, smooth_relu
from .trace import Kernel, TraceNum, trace
from .compiled import Compiled
from .tapenum import TapeNum
from .scan import Scan
from .mc import McModel, McResult, McSecond, Hessian, mc_compiled, mc_hvp, mc_hessian, mc_hessian_bumped, mc_tape
from .fd import FdModel, FdResult, FdLanes, FdSecond, fd_compiled, fd_batch, fd_price, fd_hvp, fd_hessian, fd_hessian_bumped, fd_tape
from .real import Real, Vec, Dual
