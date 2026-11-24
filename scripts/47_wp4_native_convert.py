"""WP4: convert ALL usable NLST series (72,545) from raw DICOM at detector spacing (prespec Section 3: native input chosen
by the label-free check on 17 Nov 2026, agreement 0.815 < 0.891).

    LUNGCAD_NLST_NATIVE=$LUNGCAD_NLST_HDD/nlst_native_detector python scripts/47_wp4_native_convert.py [--procs 4]

Same converter as the robustness-study sample (lungcad.data.nlst.convert_native, unchanged); only the store differs
(raw-data disk, ~4.2 TB).
Resumable: finished series are skipped. Log: $LUNGCAD_NLST_NATIVE/convert.log, one line per series ("ok", "skip" or "FAIL").
A series counts as converted only once it appears in that log as ok/skip (the log line is written after both files exist),
which is what scripts/41_wp4_infer.py --converted-only reads.
"""
from __future__ import annotations
import argparse, os, sys, time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungcad.data.nlst import usable_series, convert_native, NATIVE


def job(a):
    stem, lp = a
    try:
        return stem, convert_native(stem, lp)
    except Exception as e:
        return stem, f"FAIL {type(e).__name__}: {e}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=4); a = ap.parse_args()
    if "LUNGCAD_NLST_NATIVE" not in os.environ:
        sys.exit("set LUNGCAD_NLST_NATIVE (the full native set, ~4.2 TB, goes on the raw-data disk, not the working data root)")
    u = usable_series()
    names = [Path(s).name for s in u.stem]
    assert len(set(names)) == len(names), "stem names are not unique"
    jobs = list(zip(u.stem, u.local_path))
    NATIVE.mkdir(parents=True, exist_ok=True); log = open(NATIVE / "convert.log", "a"); t0 = time.time(); nf = 0
    print(f"{len(jobs)} series -> {NATIVE}, {a.procs} procs", flush=True)
    with Pool(a.procs, maxtasksperchild=200) as pool:
        for n, (stem, msg) in enumerate(pool.imap_unordered(job, jobs, chunksize=4), 1):
            nf += msg.startswith("FAIL")
            line = f"{time.strftime('%F %T')} {n}/{len(jobs)} {Path(stem).name} {msg}"; log.write(line + "\n"); log.flush()
            if n % 500 == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"{line}  fails {nf}  {el / 3600:.1f} h  ETA {el / n * (len(jobs) - n) / 3600:.1f} h", flush=True)
    print(f"done, fails {nf}", flush=True); sys.exit(1 if nf else 0)
