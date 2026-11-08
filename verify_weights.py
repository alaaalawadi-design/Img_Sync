import json, hashlib, os, sys, urllib.request
SRC = "/home/administrator/alaa-workspace/LongCat-Video/weights"
REPOS = {"LongCat-Video": "meituan-longcat/LongCat-Video", "LongCat-Video-Avatar-1.5": "meituan-longcat/LongCat-Video-Avatar-1.5"}
bad = []; ok = 0
for local, rid in REPOS.items():
    tree = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{rid}/tree/main?recursive=true"))
    for e in tree:
        if e["type"] != "file": continue
        p = os.path.join(SRC, local, e["path"])
        if not os.path.exists(p):
            bad.append(("MISSING", local, e["path"], e["size"])); print("MISSING", p, flush=True); continue
        if os.path.getsize(p) != e["size"]:
            bad.append(("SIZE", local, e["path"], e["size"])); print("SIZE", p, flush=True); continue
        if "lfs" in e:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
            good = h.hexdigest() == e["lfs"]["oid"]
        else:
            data = open(p, "rb").read()
            good = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == e["oid"]
        if good: ok += 1; print("OK", local, e["path"], flush=True)
        else: bad.append(("HASH", local, e["path"], e["size"])); print("HASH_MISMATCH", p, flush=True)
json.dump(bad, open("/home/administrator/imgsync/logs/verify_bad.json", "w"), indent=1)
print(f"DONE ok={ok} bad={len(bad)}", flush=True)
