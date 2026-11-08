
"""Worker server -> evaluation server (22): send finished runs (final mp4 + input image + job_config + DONE.json) so the
eval watcher there evaluates them.  Credentials come from the environment:  EVAL_HOST (default 57.217.140.22),
SRV_USER (default administrator), SRV_PASS (required)."""
import os, time, json, paramiko
EVAL_HOST = os.environ.get("EVAL_HOST", "57.217.140.22")
SRV_USER = os.environ.get("SRV_USER", "administrator")
SRV_PASS = os.environ["SRV_PASS"]
L = "/home/administrator/imgsync/outputs"; R = "/home/administrator/imgsync/outputs"
done_file = "/home/administrator/imgsync/logs/pushed_to_22.json"
pushed = set(json.load(open(done_file))) if os.path.exists(done_file) else set()
while True:
    try:
        for j in sorted(os.listdir(L)):
            d = f"{L}/{j}"
            if j in pushed or not os.path.exists(f"{d}/DONE.json") or not os.path.exists(f"{d}/{j}.mp4"):
                continue
            cfg = json.load(open(f"{d}/job_config.json")); img = cfg["image"]; ext = os.path.splitext(img)[1]
            cfg["image"] = f"{R}/{j}/input_image{ext}"          # path of the image on server 22
            t = paramiko.Transport((EVAL_HOST, 22)); t.connect(username=SRV_USER, password=SRV_PASS)
            s = paramiko.SFTPClient.from_transport(t)
            try: s.mkdir(f"{R}/{j}")
            except IOError: pass
            s.put(img, f"{R}/{j}/input_image{ext}")
            with s.open(f"{R}/{j}/job_config.json", "w") as f: f.write(json.dumps(cfg, indent=2))
            for f in (f"{j}.mp4", "DONE.json"):  # DONE.json last: it triggers the evaluation
                s.put(f"{d}/{f}", f"{R}/{j}/{f}.part"); s.posix_rename(f"{R}/{j}/{f}.part", f"{R}/{j}/{f}")
            t.close(); pushed.add(j); json.dump(sorted(pushed), open(done_file, "w"))
            print(time.strftime("%F %T"), "pushed", j, flush=True)
    except Exception as e:
        print(time.strftime("%F %T"), "push error", e, flush=True)
    time.sleep(60)
