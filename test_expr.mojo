"""
Test expression templates for operation fusion in AAD.

Verifies that fused expression nodes produce correct gradients by
comparing against the standard (non-fused) Number system.
"""

from autodiff import (
    Tape, Number, Leaf, AddExpr, MulExpr, SubExpr, ExprNumber, make_leaf,
    MulAddExpr, MulMulAddExpr, MulSubExpr, ExpMulExpr,
)

comptime width = 1
comptime type = DType.float32


def test_add_mul() raises:
    """Test (a + b) * c using MulAddExpr.

    Expected gradients: d/da = c, d/db = c, d/dc = a + b.
    """
    print("=== Test: (a + b) * c ===")
    var tape = Tape[width, type]()
    var a = Number[width, type](3.0, tape)
    var b = Number[width, type](5.0, tape)
    var c = Number[width, type](7.0, tape)
    tape.set_mark()

    var la = make_leaf[width, type](a.value, a.idx)
    var lb = make_leaf[width, type](b.value, b.idx)
    var lc = make_leaf[width, type](c.value, c.idx)

    var expr = MulAddExpr[width, type](la, lb, lc)
    var result = expr.record(tape)

    print("  value:", result.value, " (expected: 56.0)")

    result.propagate_adjoints(SIMD[type, width](1.0))

    var da = tape.adjoint(a.idx)
    var db = tape.adjoint(b.idx)
    var dc = tape.adjoint(c.idx)
    print("  d/da:", da, " (expected: 7.0)")
    print("  d/db:", db, " (expected: 7.0)")
    print("  d/dc:", dc, " (expected: 8.0)")

    var ok = True
    if da[0] != 7.0:
        print("  FAIL: d/da")
        ok = False
    if db[0] != 7.0:
        print("  FAIL: d/db")
        ok = False
    if dc[0] != 8.0:
        print("  FAIL: d/dc")
        ok = False
    if ok:
        print("  PASS")
    tape.free()


def test_add_mul_standard() raises:
    """Reference: compute (a + b) * c using standard Number ops (no fusion)."""
    print("=== Reference: (a + b) * c with standard Number ===")
    var tape = Tape[width, type]()
    var a = Number[width, type](3.0, tape)
    var b = Number[width, type](5.0, tape)
    var c = Number[width, type](7.0, tape)
    tape.set_mark()

    var result = (a + b) * c
    print("  value:", result.value, " (expected: 56.0)")
    print("  tape nodes after mark:", tape.get_tail() - 2)

    result.propagate_adjoints(SIMD[type, width](1.0))
    print("  d/da:", tape.adjoint(a.idx))
    print("  d/db:", tape.adjoint(b.idx))
    print("  d/dc:", tape.adjoint(c.idx))
    tape.free()


def test_mul_add_mul() raises:
    """Test (a * b + c) * d using MulMulAddExpr.

    Expected gradients:
      d/da = b * d = 15
      d/db = a * d = 10
      d/dc = d = 5
      d/dd = a * b + c = 10
    """
    print("=== Test: (a * b + c) * d ===")
    var tape = Tape[width, type]()
    var a = Number[width, type](2.0, tape)
    var b = Number[width, type](3.0, tape)
    var c = Number[width, type](4.0, tape)
    var d = Number[width, type](5.0, tape)
    tape.set_mark()

    var la = make_leaf[width, type](a.value, a.idx)
    var lb = make_leaf[width, type](b.value, b.idx)
    var lc = make_leaf[width, type](c.value, c.idx)
    var ld = make_leaf[width, type](d.value, d.idx)

    var expr = MulMulAddExpr[width, type](la, lb, lc, ld)
    var result = expr.record(tape)

    # Expected: (2*3 + 4) * 5 = 50
    print("  value:", result.value, " (expected: 50.0)")

    result.propagate_adjoints(SIMD[type, width](1.0))

    var da = tape.adjoint(a.idx)
    var db = tape.adjoint(b.idx)
    var dc = tape.adjoint(c.idx)
    var dd = tape.adjoint(d.idx)
    print("  d/da:", da, " (expected: 15.0)")
    print("  d/db:", db, " (expected: 10.0)")
    print("  d/dc:", dc, " (expected: 5.0)")
    print("  d/dd:", dd, " (expected: 10.0)")

    var ok = True
    if da[0] != 15.0:
        print("  FAIL: d/da")
        ok = False
    if db[0] != 10.0:
        print("  FAIL: d/db")
        ok = False
    if dc[0] != 5.0:
        print("  FAIL: d/dc")
        ok = False
    if dd[0] != 10.0:
        print("  FAIL: d/dd")
        ok = False
    if ok:
        print("  PASS")
    tape.free()


def test_tape_size_reduction() raises:
    """Demonstrate tape size reduction: fused vs standard for (a + b) * c."""
    print("=== Tape size comparison ===")

    # Standard: 3 leaves + 2 ops = 5 nodes total
    var tape1 = Tape[width, type]()
    var a1 = Number[width, type](3.0, tape1)
    var b1 = Number[width, type](5.0, tape1)
    var c1 = Number[width, type](7.0, tape1)
    tape1.set_mark()
    _ = (a1 + b1) * c1
    var standard_nodes = tape1.get_tail() + 1
    print("  Standard: total nodes =", standard_nodes, "(3 leaves + 2 ops)")

    # Fused: 3 leaves + 1 fused op = 4 nodes total
    var tape2 = Tape[width, type]()
    var a2 = Number[width, type](3.0, tape2)
    var b2 = Number[width, type](5.0, tape2)
    var c2 = Number[width, type](7.0, tape2)
    tape2.set_mark()
    var la = make_leaf[width, type](a2.value, a2.idx)
    var lb = make_leaf[width, type](b2.value, b2.idx)
    var lc = make_leaf[width, type](c2.value, c2.idx)
    var expr = MulAddExpr[width, type](la, lb, lc)
    _ = expr.record(tape2)
    var fused_nodes = tape2.get_tail() + 1
    print("  Fused:    total nodes =", fused_nodes, "(3 leaves + 1 fused op)")
    print("  Reduction:", standard_nodes - fused_nodes, "fewer nodes")

    tape1.free()
    tape2.free()


def test_sub_mul() raises:
    """Test (a - b) * c using MulSubExpr.

    Expected: d/da = c, d/db = -c, d/dc = a - b.
    """
    print("=== Test: (a - b) * c ===")
    var tape = Tape[width, type]()
    var a = Number[width, type](10.0, tape)
    var b = Number[width, type](3.0, tape)
    var c = Number[width, type](4.0, tape)
    tape.set_mark()

    var la = make_leaf[width, type](a.value, a.idx)
    var lb = make_leaf[width, type](b.value, b.idx)
    var lc = make_leaf[width, type](c.value, c.idx)

    var expr = MulSubExpr[width, type](la, lb, lc)
    var result = expr.record(tape)

    # (10 - 3) * 4 = 28
    print("  value:", result.value, " (expected: 28.0)")

    result.propagate_adjoints(SIMD[type, width](1.0))

    var da = tape.adjoint(a.idx)
    var db = tape.adjoint(b.idx)
    var dc = tape.adjoint(c.idx)
    print("  d/da:", da, " (expected: 4.0)")
    print("  d/db:", db, " (expected: -4.0)")
    print("  d/dc:", dc, " (expected: 7.0)")

    var ok = True
    if da[0] != 4.0:
        print("  FAIL: d/da")
        ok = False
    if db[0] != -4.0:
        print("  FAIL: d/db")
        ok = False
    if dc[0] != 7.0:
        print("  FAIL: d/dc")
        ok = False
    if ok:
        print("  PASS")
    tape.free()


def test_exp_mul() raises:
    """Test exp(a + b) * c using ExpMulExpr.

    Let f = exp(a+b) * c. Then:
      d/da = exp(a+b) * c
      d/db = exp(a+b) * c
      d/dc = exp(a+b)
    """
    print("=== Test: exp(a + b) * c ===")
    var tape = Tape[width, type]()
    var a = Number[width, type](1.0, tape)
    var b = Number[width, type](2.0, tape)
    var c = Number[width, type](0.5, tape)
    tape.set_mark()

    var la = make_leaf[width, type](a.value, a.idx)
    var lb = make_leaf[width, type](b.value, b.idx)
    var lc = make_leaf[width, type](c.value, c.idx)

    var expr = ExpMulExpr[width, type](la, lb, lc)
    var result = expr.record(tape)

    from std.math import exp
    var exp3 = exp(SIMD[type, width](3.0))
    var expected_val = exp3 * SIMD[type, width](0.5)
    print("  value:", result.value, " (expected:", expected_val, ")")

    result.propagate_adjoints(SIMD[type, width](1.0))

    var da = tape.adjoint(a.idx)
    var db = tape.adjoint(b.idx)
    var dc = tape.adjoint(c.idx)
    # d/da = exp(3)*0.5, d/db = exp(3)*0.5, d/dc = exp(3)
    var expected_dadb = exp3 * SIMD[type, width](0.5)
    var expected_dc = exp3
    print("  d/da:", da, " (expected:", expected_dadb, ")")
    print("  d/db:", db, " (expected:", expected_dadb, ")")
    print("  d/dc:", dc, " (expected:", expected_dc, ")")

    # Use tolerance for floating point
    var tol = SIMD[type, width](0.01)
    var ok = True
    var err_a = da - expected_dadb
    if err_a[0] > tol[0] or err_a[0] < -tol[0]:
        print("  FAIL: d/da, error =", err_a)
        ok = False
    var err_b = db - expected_dadb
    if err_b[0] > tol[0] or err_b[0] < -tol[0]:
        print("  FAIL: d/db, error =", err_b)
        ok = False
    var err_c = dc - expected_dc
    if err_c[0] > tol[0] or err_c[0] < -tol[0]:
        print("  FAIL: d/dc, error =", err_c)
        ok = False
    if ok:
        print("  PASS")
    tape.free()


def test_diamond_pattern() raises:
    """Test a * a (same variable used twice) using MulExpr.

    d(a^2)/da = 2a. The diamond pattern means both children of MulExpr
    are the same leaf — the propagation must correctly accumulate.
    """
    print("=== Test: a * a (diamond) ===")
    var tape = Tape[width, type]()
    var a = Number[width, type](4.0, tape)
    tape.set_mark()

    var la = make_leaf[width, type](a.value, a.idx)

    var expr = MulExpr[width, type](la, la)
    var result = expr.record(tape)

    print("  value:", result.value, " (expected: 16.0)")

    result.propagate_adjoints(SIMD[type, width](1.0))
    var da = tape.adjoint(a.idx)
    print("  d/da:", da, " (expected: 8.0)")

    var ok = True
    if da[0] != 8.0:
        print("  FAIL: d/da")
        ok = False
    if ok:
        print("  PASS")
    tape.free()


def test_simple_add() raises:
    """Test a + b using AddExpr.

    d/da = 1, d/db = 1.
    """
    print("=== Test: a + b ===")
    var tape = Tape[width, type]()
    var a = Number[width, type](3.0, tape)
    var b = Number[width, type](5.0, tape)
    tape.set_mark()

    var la = make_leaf[width, type](a.value, a.idx)
    var lb = make_leaf[width, type](b.value, b.idx)

    var expr = AddExpr[width, type](la, lb)
    var result = expr.record(tape)

    print("  value:", result.value, " (expected: 8.0)")

    result.propagate_adjoints(SIMD[type, width](1.0))
    var da = tape.adjoint(a.idx)
    var db = tape.adjoint(b.idx)
    print("  d/da:", da, " (expected: 1.0)")
    print("  d/db:", db, " (expected: 1.0)")

    var ok = True
    if da[0] != 1.0:
        print("  FAIL: d/da")
        ok = False
    if db[0] != 1.0:
        print("  FAIL: d/db")
        ok = False
    if ok:
        print("  PASS")
    tape.free()


def test_simple_mul() raises:
    """Test a * b using MulExpr.

    d/da = b, d/db = a.
    """
    print("=== Test: a * b ===")
    var tape = Tape[width, type]()
    var a = Number[width, type](3.0, tape)
    var b = Number[width, type](5.0, tape)
    tape.set_mark()

    var la = make_leaf[width, type](a.value, a.idx)
    var lb = make_leaf[width, type](b.value, b.idx)

    var expr = MulExpr[width, type](la, lb)
    var result = expr.record(tape)

    print("  value:", result.value, " (expected: 15.0)")

    result.propagate_adjoints(SIMD[type, width](1.0))
    var da = tape.adjoint(a.idx)
    var db = tape.adjoint(b.idx)
    print("  d/da:", da, " (expected: 5.0)")
    print("  d/db:", db, " (expected: 3.0)")

    var ok = True
    if da[0] != 5.0:
        print("  FAIL: d/da")
        ok = False
    if db[0] != 3.0:
        print("  FAIL: d/db")
        ok = False
    if ok:
        print("  PASS")
    tape.free()


def test_simd_width() raises:
    """Test expression templates with SIMD width > 1."""
    print("=== Test: (a + b) * c with SIMD width=4 ===")
    comptime W = 4
    var tape = Tape[W, type]()
    var a = Number[W, type](3.0, tape)
    var b = Number[W, type](5.0, tape)
    var c = Number[W, type](7.0, tape)
    tape.set_mark()

    var la = make_leaf[W, type](a.value, a.idx)
    var lb = make_leaf[W, type](b.value, b.idx)
    var lc = make_leaf[W, type](c.value, c.idx)

    var expr = MulAddExpr[W, type](la, lb, lc)
    var result = expr.record(tape)

    print("  value:", result.value)
    result.propagate_adjoints(SIMD[type, W](1.0))

    var da = tape.adjoint(a.idx)
    var db = tape.adjoint(b.idx)
    var dc = tape.adjoint(c.idx)
    print("  d/da:", da, " (expected: all 7.0)")
    print("  d/db:", db, " (expected: all 7.0)")
    print("  d/dc:", dc, " (expected: all 8.0)")

    # Check all lanes
    var ok = True
    for i in range(W):
        if da[i] != 7.0 or db[i] != 7.0 or dc[i] != 8.0:
            ok = False
    if ok:
        print("  PASS")
    else:
        print("  FAIL")
    tape.free()


def test_chained_exprs() raises:
    """Test chaining multiple expression recordings via to_leaf().

    Compute z = (a + b) * c, then w = z * d.
    Full expression: w = (a + b) * c * d.
    d/da = c*d = 28, d/db = c*d = 28, d/dc = (a+b)*d = 32, d/dd = (a+b)*c = 56
    """
    print("=== Test: chained: z = (a+b)*c, then w = z*d ===")
    var tape = Tape[width, type]()
    var a = Number[width, type](3.0, tape)
    var b = Number[width, type](5.0, tape)
    var c = Number[width, type](7.0, tape)
    var d = Number[width, type](4.0, tape)
    tape.set_mark()

    # First expression: z = (a + b) * c
    var la = make_leaf[width, type](a.value, a.idx)
    var lb = make_leaf[width, type](b.value, b.idx)
    var lc = make_leaf[width, type](c.value, c.idx)
    var expr1 = MulAddExpr[width, type](la, lb, lc)
    var z = expr1.record(tape)

    # Second expression: w = z * d
    var lz = z.to_leaf()
    var ld = make_leaf[width, type](d.value, d.idx)
    var expr2 = MulExpr[width, type](lz, ld)
    var w = expr2.record(tape)

    # (3+5)*7*4 = 224
    print("  value:", w.value, " (expected: 224.0)")

    w.propagate_adjoints(SIMD[type, width](1.0))

    var da = tape.adjoint(a.idx)
    var db = tape.adjoint(b.idx)
    var dc = tape.adjoint(c.idx)
    var dd = tape.adjoint(d.idx)
    print("  d/da:", da, " (expected: 28.0)")
    print("  d/db:", db, " (expected: 28.0)")
    print("  d/dc:", dc, " (expected: 32.0)")
    print("  d/dd:", dd, " (expected: 56.0)")

    var ok = True
    if da[0] != 28.0:
        print("  FAIL: d/da")
        ok = False
    if db[0] != 28.0:
        print("  FAIL: d/db")
        ok = False
    if dc[0] != 32.0:
        print("  FAIL: d/dc")
        ok = False
    if dd[0] != 56.0:
        print("  FAIL: d/dd")
        ok = False
    if ok:
        print("  PASS")
    tape.free()


def test_mc_step_pattern() raises:
    """Test the MC inner-loop pattern: path * exp(drift + vsdt * rand).

    This is the key use case. With expression templates, the entire
    timestep becomes 1 fused node instead of ~5 standard nodes.

    Using ExpMulExpr: exp(drift + vsdt_rand) * path
    where drift and vsdt_rand are pre-computed leaves.
    """
    print("=== Test: MC step pattern: path * exp(drift + vsdt_rand) ===")

    # --- Standard approach on tape1 ---
    var tape1 = Tape[width, type]()
    var path1 = Number[width, type](100.0, tape1)
    var drift1 = Number[width, type](0.01, tape1)
    var vsdt_rand1 = Number[width, type](0.05, tape1)
    tape1.set_mark()

    var std_result = path1 * (drift1 + vsdt_rand1).expon()
    var std_nodes = tape1.get_tail() + 1

    std_result.propagate_adjoints(SIMD[type, width](1.0))
    var std_dpath = tape1.adjoint(path1.idx)
    var std_ddrift = tape1.adjoint(drift1.idx)
    var std_dvsdt = tape1.adjoint(vsdt_rand1.idx)

    print("  Standard nodes:", std_nodes, "(3 leaves +", std_nodes - 3, "ops)")
    print("  Standard value:", std_result.value)
    print("  Standard d/dpath:", std_dpath)
    print("  Standard d/ddrift:", std_ddrift)
    print("  Standard d/dvsdt_rand:", std_dvsdt)

    # --- Fused approach on tape2 ---
    var tape2 = Tape[width, type]()
    var path2 = Number[width, type](100.0, tape2)
    var drift2 = Number[width, type](0.01, tape2)
    var vsdt_rand2 = Number[width, type](0.05, tape2)
    tape2.set_mark()

    var l_drift = make_leaf[width, type](drift2.value, drift2.idx)
    var l_vsdt = make_leaf[width, type](vsdt_rand2.value, vsdt_rand2.idx)
    var l_path = make_leaf[width, type](path2.value, path2.idx)

    # exp(drift + vsdt_rand) * path
    var expr = ExpMulExpr[width, type](l_drift, l_vsdt, l_path)
    var fused_result = expr.record(tape2)
    var fused_nodes = tape2.get_tail() + 1

    fused_result.propagate_adjoints(SIMD[type, width](1.0))
    var fused_dpath = tape2.adjoint(path2.idx)
    var fused_ddrift = tape2.adjoint(drift2.idx)
    var fused_dvsdt = tape2.adjoint(vsdt_rand2.idx)

    print("  Fused nodes:", fused_nodes, "(3 leaves + 1 fused op)")
    print("  Fused value:", fused_result.value)
    print("  Fused d/dpath:", fused_dpath)
    print("  Fused d/ddrift:", fused_ddrift)
    print("  Fused d/dvsdt_rand:", fused_dvsdt)
    print("  Node reduction:", std_nodes - fused_nodes, "fewer nodes per timestep")

    # Verify they match
    var tol = SIMD[type, width](0.01)
    var ok = True
    var val_err = fused_result.value - std_result.value
    if val_err[0] > tol[0] or val_err[0] < -tol[0]:
        print("  FAIL: value mismatch")
        ok = False
    var dp_err = fused_dpath - std_dpath
    if dp_err[0] > tol[0] or dp_err[0] < -tol[0]:
        print("  FAIL: d/dpath mismatch")
        ok = False
    var dd_err = fused_ddrift - std_ddrift
    if dd_err[0] > tol[0] or dd_err[0] < -tol[0]:
        print("  FAIL: d/ddrift mismatch")
        ok = False
    var dv_err = fused_dvsdt - std_dvsdt
    if dv_err[0] > tol[0] or dv_err[0] < -tol[0]:
        print("  FAIL: d/dvsdt_rand mismatch")
        ok = False
    if ok:
        print("  PASS")
    tape1.free()
    tape2.free()


def main() raises:
    print("Expression Template Tests")
    print("=========================")
    print()
    test_add_mul_standard()
    print()
    test_simple_add()
    print()
    test_simple_mul()
    print()
    test_add_mul()
    print()
    test_mul_add_mul()
    print()
    test_sub_mul()
    print()
    test_exp_mul()
    print()
    test_diamond_pattern()
    print()
    test_simd_width()
    print()
    test_chained_exprs()
    print()
    test_mc_step_pattern()
    print()
    test_tape_size_reduction()
