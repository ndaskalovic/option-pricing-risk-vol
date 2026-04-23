struct Node[width: Int = 1, type: DType = DType.float32](TrivialRegisterPassable, Writable):
    var numArg: Int
    var idx1: Int
    var idx2: Int
    var der1: SIMD[Self.type, Self.width]
    var der2: SIMD[Self.type, Self.width]

    @always_inline("nodebug")
    def __init__(
        out self, n: Int, i1: Int, i2: Int, d1: SIMD[Self.type, Self.width], d2: SIMD[Self.type, Self.width]
    ):
        self.numArg = n
        self.idx1 = i1
        self.idx2 = i2
        self.der1 = d1
        self.der2 = d2

    def print(self):
        print(
            "<Node["
            + String.write(Self.width)
            + ", "
            + String.write(Self.type)
            + "]: args:"
            + String.write(self.numArg)
            + ", index1:"
            + String.write(self.idx1)
            + ", index2:"
            + String.write(self.idx2)
            + ", der1:"
            + String.write(self.der1)
            + ", der2:"
            + String.write(self.der2)
            + ">"
        )

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "<Node[",
            Self.width,
            ", ",
            Self.type,
            "]: args:",
            self.numArg,
            ", index1:",
            self.idx1,
            ", index2:",
            self.idx2,
            ", der1:",
            self.der1,
            ", der2:",
            self.der2,
            ">",
        )
