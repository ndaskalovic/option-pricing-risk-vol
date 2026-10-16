"""Graph node and opcodes shared by the tracer and the compile-time differentiator."""

comptime INPUT = 0   # a = input slot
comptime DATA = 1    # a = data slot (non-differentiable input, e.g. a normal draw)
comptime CONST = 2   # k = value
comptime ADD = 3
comptime SUB = 4
comptime MUL = 5
comptime DIV = 6
comptime NEG = 7
comptime EXP = 8
comptime LOG = 9
comptime SQRT = 10
comptime POWK = 11   # a ** k
comptime RELU = 12   # max(a, 0)
comptime SELGT = 13  # a > b ? c : d
comptime OUT = 14    # output slot b <- node a
comptime STOP = 15   # value of a, zero derivative (stop_gradient)


@fieldwise_init
struct Node(ImplicitlyCopyable, Movable, Writable):
    var op: Int
    var a: Int
    var b: Int
    var c: Int
    var d: Int
    var k: Float64

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Node(op=", self.op, ", a=", self.a, ", b=", self.b, ", c=", self.c, ", d=", self.d, ", k=", self.k, ")")
