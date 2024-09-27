@register_passable("trivial")
struct Node[width: Int = 1, type: DType = DType.float32](Stringable):
    var numArg: Int
    var idx1: Int
    var idx2: Int
    var der1: SIMD[type, width]
    var der2: SIMD[type, width]

    @always_inline("nodebug")
    fn __init__(
        inout self, n: Int, i1: Int, i2: Int, d1: SIMD[type, width], d2: SIMD[type, width]
    ):
        self.numArg = n
        self.idx1 = i1
        self.idx2 = i2
        self.der1 = d1
        self.der2 = d2

    fn print(self):
        print(
            "<Node["
            + str(width)
            + ", "
            + str(type)
            + "]: args:"
            + str(self.numArg)
            + ", index1:"
            + str(self.idx1)
            + ", index2:"
            + str(self.idx2)
            + ", der1:"
            + str(self.der1)
            + ", der2:"
            + str(self.der2)
            + ">"
        )
    fn __str__(self) -> String:
        return(
            "<Node["
            + str(width)
            + ", "
            + str(type)
            + "]: args:"
            + str(self.numArg)
            + ", index1:"
            + str(self.idx1)
            + ", index2:"
            + str(self.idx2)
            + ", der1:"
            + str(self.der1)
            + ", der2:"
            + str(self.der2)
            + ">"
        )