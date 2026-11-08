#!/usr/bin/env python3
"""Pull finished segments / final videos of LongCat jobs from server .22 to this machine.

  python3 sync_outputs.py job_a job_b 16:job_c        (prefix "16:" = job on server .16, default server .22)

Polls every 60 s, downloads every new or changed file under
/home/administrator/imgsync/outputs/<job>/ (segments/*.mp4, final mp4, json logs), skipping
partial files, and exits once every listed job has DONE.json + its eval/report.md and is fully downloaded.
"""
import os
import sys
import time
import stat
import paramiko

HOSTS = {"22": "57.217.140.22", "16": "57.217.140.16", "12": "57.217.140.12"}
USER, PASSWORD = os.environ.get("SRV_USER", "administrator"), os.environ["SRV_PASS"]  # export SRV_PASS=...
REMOTE = "/home/administrator/imgsync/outputs"
LOCAL = os.environ.get("LOCAL_OUT", os.path.join(os.getcwd(), "outputs", "data"))
SKIP = (".part.mp4", ".tmp", "resume_state.pt", "segments.txt", "per_frame.json")


def connect(server="22"):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOSTS[server], username=USER, password=PASSWORD, timeout=30, banner_timeout=30, auth_timeout=30)
    c.get_transport().set_keepalive(15)          # detect dead links
    sftp = c.open_sftp()
    sftp.get_channel().settimeout(120)           # a stalled read raises instead of hanging forever
    return c, sftp


def walk(sftp, path):
    for e in sftp.listdir_attr(path):
        p = f"{path}/{e.filename}"
        if stat.S_ISDIR(e.st_mode):
            yield from walk(sftp, p)
        else:
            yield p, e


def sync_job(sftp, job):
    rroot = f"{REMOTE}/{job}"
    try:
        sftp.stat(rroot)
    except FileNotFoundError:
        return False
    done = False
    for rp, e in walk(sftp, rroot):
        if rp.endswith(SKIP):
            continue
        lp = os.path.join(LOCAL, job, os.path.relpath(rp, rroot))
        if os.path.exists(lp) and os.path.getsize(lp) == e.st_size:
            continue
        os.makedirs(os.path.dirname(lp), exist_ok=True)
        sftp.get(rp, lp + ".part")
        os.replace(lp + ".part", lp)
        print(time.strftime("%H:%M:%S"), "downloaded", os.path.relpath(lp, LOCAL), f"{e.st_size / 1e6:.1f} MB", flush=True)
    # complete = video finished (DONE.json) AND its evaluation report downloaded (or evaluation failed)
    loc = os.path.join(LOCAL, job)
    return os.path.exists(os.path.join(loc, "DONE.json")) and (
        os.path.exists(os.path.join(loc, "eval", "report.md")) or os.path.exists(os.path.join(loc, "eval", ".failed")))


def main(jobs):
    specs = [(j.split(":", 1) if ":" in j else ("22", j)) for j in jobs]
    finished = set()
    while len(finished) < len(specs):
        for server in sorted({sv for sv, _ in specs}):
            try:
                c, sftp = connect(server)
                for sv, j in specs:
                    if sv != server or j in finished:
                        continue
                    if sync_job(sftp, j):
                        sync_job(sftp, j)  # one more pass so the final files are surely complete
                        finished.add(j)
                        print(time.strftime("%H:%M:%S"), "JOB COMPLETE", j, flush=True)
                c.close()
            except Exception as ex:  # network hiccup: retry next round
                print("sync error:", server, ex, flush=True)
        if len(finished) < len(specs):
            time.sleep(60)


if __name__ == "__main__":
    main(sys.argv[1:])
