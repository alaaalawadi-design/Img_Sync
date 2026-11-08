#!/usr/bin/env python3
"""Download every run folder from servers .22, .16 and .12 into outputs/data/<run>/.
File by file, each with retries and a fresh connection after any network error; the whole pass repeats until
nothing is missing. Skips files already present with the same size and server-only checkpoints."""
import os, time, stat
import sync_outputs as so

SKIP = ("resume_state.pt", ".part.mp4", ".tmp", "segments.txt", ".silent.mp4")

def listing(server):
    c, s = so.connect(server)
    out = []
    def walk(p):
        for e in s.listdir_attr(p):
            q = f"{p}/{e.filename}"
            if stat.S_ISDIR(e.st_mode): walk(q)
            elif not e.filename.endswith(SKIP): out.append((q, e.st_size))
    for r in sorted(s.listdir(so.REMOTE)):
        walk(f"{so.REMOTE}/{r}")
    c.close()
    return out

def missing(files):
    m = []
    for rp, size in files:
        lp = os.path.join(so.LOCAL, os.path.relpath(rp, so.REMOTE))
        if not (os.path.exists(lp) and os.path.getsize(lp) == size): m.append((rp, size, lp))
    return m

for rnd in range(1, 50):
    todo_total = 0
    for server in ("22", "16", "12"):
        try:
            files = listing(server)
        except Exception as e:
            print(f"[{server}] listing failed: {e}", flush=True); todo_total += 1; continue
        todo = missing(files)
        todo_total += len(todo)
        print(f"round {rnd} server {server}: {len(files)} files, {len(todo)} missing ({sum(z for _, z, _ in todo)/1e9:.2f} GB)", flush=True)
        c = s = None
        for i, (rp, size, lp) in enumerate(todo, 1):
            for attempt in range(4):
                try:
                    if s is None: c, s = so.connect(server)
                    os.makedirs(os.path.dirname(lp), exist_ok=True)
                    s.get(rp, lp + ".part"); os.replace(lp + ".part", lp)
                    break
                except Exception as e:
                    try: c.close()
                    except Exception: pass
                    c = s = None; time.sleep(15)
            if i % 50 == 0 or i == len(todo): print(f"   [{server}] {i}/{len(todo)} done", flush=True)
        try: c.close()
        except Exception: pass
    if todo_total == 0:
        print("DOWNLOAD_ALL_DONE", time.strftime("%H:%M:%S"), flush=True); break
    time.sleep(30)
