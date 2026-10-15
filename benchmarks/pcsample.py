"""Poor-man's PC sampler for macOS (no Xcode needed): attach with LLDB, interrupt N times,
histogram the main binary's program counter by source line.

Build the target with line tables and run it in a long loop first:
    .venv/bin/mojo build -I . --debug-level line-tables <file>.mojo -o /tmp/prof && /tmp/prof &
    /usr/bin/python3 benchmarks/pcsample.py /tmp/prof $(pgrep -f /tmp/prof) 1500 /tmp/pcs.txt

Must run under /usr/bin/python3 (matches LLDB's bundled Python). Note that
@always_inline("nodebug") code is attributed to its call site.
"""
import os, sys, time, collections, random
sys.path.insert(0, "/Library/Developer/CommandLineTools/Library/PrivateFrameworks/LLDB.framework/Resources/Python")
import lldb
exe, pid, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
dbg = lldb.SBDebugger.Create(); dbg.SetAsync(False)
target = dbg.CreateTarget(exe)
err = lldb.SBError()
proc = target.AttachToProcessWithID(dbg.GetListener(), pid, err)
assert err.Success(), err
mainthread = None
lines = collections.Counter(); pcs = collections.Counter()
listener = dbg.GetListener()
dbg.SetAsync(True)
def wait_state(want):
    ev = lldb.SBEvent()
    for _ in range(100):
        if proc.GetState() == want: return True
        listener.WaitForEvent(1, ev)
    return proc.GetState() == want
proc.Continue(); wait_state(lldb.eStateRunning)
for k in range(n):
    time.sleep(0.003 + random.random() * 0.007)
    proc.SendAsyncInterrupt() if False else proc.Stop()
    if not wait_state(lldb.eStateStopped): continue
    for t in proc.threads:
        f = t.GetFrameAtIndex(0)
        if f.GetModule().GetFileSpec().GetFilename() != os.path.basename(exe): continue
        addr = f.GetPCAddress()
        off = addr.GetFileAddress()
        le = addr.GetLineEntry()
        key = f"{le.GetFileSpec().GetFilename()}:{le.GetLine()}" if le.IsValid() else "?"
        lines[key] += 1; pcs[off] += 1
    proc.Continue(); wait_state(lldb.eStateRunning)
proc.Stop(); wait_state(lldb.eStateStopped)
proc.Detach()
tot = sum(lines.values())
print("samples", tot)
for k, v in lines.most_common(25): print(f"{100*v/tot:5.1f}%  {k}")
open(sys.argv[4], "w").write("\n".join(f"{hex(a)} {c}" for a, c in pcs.most_common()))
