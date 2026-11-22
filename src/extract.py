"""
extract.py — Parse real Bernese GNSS Software 5.4 output files into the
             plain-dict format that the monitors and agent use.

This file is for real Bernese outputs only.  The handmade test fixtures live
under data/dummy/ and are structured identically but were created by hand.

Three parsers, one writer:

  parse_prc(path)   — Bernese .PRC summary file (the per-PID status table that
                      Bernese writes after each BPE run, e.g. RNX2SNX.PRC).
                      Extracts: PID statuses, per-station epoch counts and
                      residual RMS where present.

  parse_bpe_log(path) — Bernese BPE .LOG file.
                       Extracts: the same PID/status/message triples as the
                       hand-made prc_summary.txt, so it can substitute for it
                       or supplement it.

  parse_crd(path)   — Bernese .CRD coordinate file (the ADDNEQ2 / CC_SAVE output).
                      Extracts: station XYZ in metres and the formal sigma in mm.

  write_run(out_dir, prc_data, crd_data) — Writes prc_summary.txt and
                      coordinates.json into out_dir so the rest of the pipeline
                      can consume them without change.

Usage (Hicham runs this after each Bernese session):

    python src/extract.py \\
        --prc   path/to/RNX2SNX.PRC \\
        --log   path/to/RNX2SNX.LOG \\
        --crd   path/to/CAMPAIGN/STA/RNX2SNX.CRD \\
        --out   data/runs/F1

The only output is prc_summary.txt and coordinates.json; no Bernese files are
modified and nothing internal is copied to the repo.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# parse_prc: Bernese .PRC summary file
# ---------------------------------------------------------------------------

def parse_prc(path: str | Path) -> dict:
    """
    Parse a Bernese BPE PRC summary file.

    Returns:
        {
          "pids": {
              "013": {"status": "OK", "program": "COOVEL", "message": "..."},
              ...
          },
          "stations": {
              "RABT": {"epochs": 2880, "residual_rms_mm": 1.8},
              ...
          },
          "failed_pids": ["311"]     # PIDs whose status is FAIL or ERROR
        }

    The PRC file format varies by PCF; this parser covers the standard
    RNX2SNX-style layout where each PID block starts with a line like
    '  PID 013 OK  COOVEL ...' and station statistics appear as indented lines
    below the PID header.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    pids: dict[str, dict] = {}
    stations: dict[str, dict] = {}
    failed_pids: list[str] = []
    current_pid: str | None = None

    # Pattern: optional whitespace, PID NNN, status word, program name, rest
    pid_line = re.compile(
        r"^\s*PID\s+(\d+)\s+(OK|FAIL|WARN|WARNING|ERROR|SKIP)\s+(\S+)(.*)",
        re.IGNORECASE,
    )
    # Station epoch line: "  XXXX  epochs=NNN" or "  XXXX  2880 epochs"
    sta_epochs = re.compile(r"([A-Z0-9]{3,5})\s+epochs=(\d+)", re.IGNORECASE)
    sta_epochs2 = re.compile(r"([A-Z0-9]{3,5})\s+(\d+)\s+epochs", re.IGNORECASE)
    # Station residual line: "  XXXX  residual_rms=N.N mm" or "  XXXX  RMS N.N mm"
    sta_rms = re.compile(r"([A-Z0-9]{3,5})\s+(?:residual_rms|RMS)[\s=]+([\d.]+)\s*mm", re.IGNORECASE)

    for line in text.splitlines():
        m = pid_line.match(line)
        if m:
            current_pid = m.group(1).zfill(3)
            status = m.group(2).upper()
            if status in ("FAIL", "ERROR"):
                failed_pids.append(current_pid)
                status = "FAIL"
            elif status in ("WARN", "WARNING"):
                status = "WARN"
            pids[current_pid] = {
                "status": status,
                "program": m.group(3),
                "message": m.group(4).strip(),
            }
            continue

        # Station epoch counts
        for pat in (sta_epochs, sta_epochs2):
            m2 = pat.search(line)
            if m2:
                name = m2.group(1).upper()
                stations.setdefault(name, {})["epochs"] = int(m2.group(2))
                break

        # Station residual RMS
        m3 = sta_rms.search(line)
        if m3:
            name = m3.group(1).upper()
            stations.setdefault(name, {})["residual_rms_mm"] = float(m3.group(2))

    return {"pids": pids, "stations": stations, "failed_pids": failed_pids}


# ---------------------------------------------------------------------------
# parse_bpe_log: Bernese BPE .LOG file
# ---------------------------------------------------------------------------

def parse_bpe_log(path: str | Path) -> dict:
    """
    Parse a Bernese BPE log file.

    The BPE log records which scripts and programs ran, with their exit status.
    This parser extracts PID-level status lines and passes them through the same
    structure as parse_prc so the two can be merged or used interchangeably.

    Returns the same dict structure as parse_prc.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    pids: dict[str, dict] = {}
    stations: dict[str, dict] = {}
    failed_pids: list[str] = []

    # BPE log line example:  "  013  COOVEL  OK  A priori coordinates loaded"
    # or the hand-made format: "PID=013 STATUS=OK PROGRAM=COOVEL MSG=..."
    bpe_line = re.compile(
        r"(?:PID[=\s]+)?(\d{3})\s+(?:STATUS[=\s]+)?(\w+)\s+(?:PROGRAM[=\s]+)?(\S+)\s+(?:MSG[=\s]+)?(.*)",
        re.IGNORECASE,
    )
    sta_epochs = re.compile(r"([A-Z0-9]{3,5})\s+epochs=(\d+)", re.IGNORECASE)
    sta_rms = re.compile(r"([A-Z0-9]{3,5})\s+(?:residual_rms|RMS)[\s=]+([\d.]+)\s*mm", re.IGNORECASE)

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        m = bpe_line.match(line)
        if m:
            pid = m.group(1).zfill(3)
            status = m.group(2).upper()
            if status in ("FAIL", "ERROR"):
                failed_pids.append(pid)
                status = "FAIL"
            pids[pid] = {
                "status": status,
                "program": m.group(3),
                "message": m.group(4).strip(),
            }
            continue

        m2 = sta_epochs.search(line)
        if m2:
            stations.setdefault(m2.group(1).upper(), {})["epochs"] = int(m2.group(2))

        m3 = sta_rms.search(line)
        if m3:
            stations.setdefault(m3.group(1).upper(), {})["residual_rms_mm"] = float(m3.group(2))

    return {"pids": pids, "stations": stations, "failed_pids": failed_pids}


# ---------------------------------------------------------------------------
# parse_crd: Bernese .CRD coordinate file
# ---------------------------------------------------------------------------

def parse_crd(path: str | Path) -> dict:
    """
    Parse a Bernese .CRD coordinate file.

    Standard CRD format (ADDNEQ2 / CC_SAVE output):

        STATION NAME      NUM   X (M)             Y (M)             Z (M)        FLAG
        ----------------  ---   ----------------  ----------------  ------------ ----
        RABT                1   5255766.741234    -554955.082341    3444655.1234  A

    Formal sigmas (mm) may appear in an optional SIGMAS block later in the file.

    Returns:
        {
          "session": "<extracted from header or None>",
          "stations": {
              "RABT": {"X_m": 5255766.741, "Y_m": -554955.082, "Z_m": 3444655.123,
                       "sX_mm": null, "sY_mm": null, "sZ_mm": null},
              ...
          }
        }
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    session: str | None = None
    stations: dict[str, dict] = {}

    # Try to extract session label from the header
    for line in text.splitlines()[:10]:
        m = re.search(r"(\d{4}[-_]\d{3})", line)
        if m:
            session = m.group(1)
            break

    # CRD data lines: NAME  NUM  X  Y  Z  FLAG
    # Columns are fixed-width but vary slightly across Bernese versions; use regex.
    crd_line = re.compile(
        r"^([A-Z0-9]{3,10})\s+\d+\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
    )
    # Sigma lines (optional): NAME  sX  sY  sZ  (in mm)
    sig_line = re.compile(
        r"^([A-Z0-9]{3,10})\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"
    )

    in_sigma_block = False
    for line in text.splitlines():
        if "SIGMAS" in line.upper() or "SIGMA" in line.upper():
            in_sigma_block = True
            continue
        if line.strip().startswith("---") or not line.strip():
            in_sigma_block = False
            continue

        if not in_sigma_block:
            m = crd_line.match(line.strip())
            if m:
                name = m.group(1).upper()
                stations[name] = {
                    "X_m": float(m.group(2)),
                    "Y_m": float(m.group(3)),
                    "Z_m": float(m.group(4)),
                    "sX_mm": None,
                    "sY_mm": None,
                    "sZ_mm": None,
                }
        else:
            m = sig_line.match(line.strip())
            if m:
                name = m.group(1).upper()
                if name in stations:
                    stations[name]["sX_mm"] = float(m.group(2))
                    stations[name]["sY_mm"] = float(m.group(3))
                    stations[name]["sZ_mm"] = float(m.group(4))

    return {"session": session, "stations": stations}


# ---------------------------------------------------------------------------
# write_run: write monitor-ready files into a run folder
# ---------------------------------------------------------------------------

def write_run(
    out_dir: str | Path,
    prc_data: dict,
    crd_data: dict,
    run_id: str = "?",
) -> None:
    """
    Write prc_summary.txt and coordinates.json into out_dir.

    These files have the same structure as the hand-made fixtures in data/dummy/
    so the agent and monitors can consume them without any change.

    Does NOT write truth.json — that must be filled in by Hicham after each run.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- prc_summary.txt ---
    lines = [f"BPE SESSION SUMMARY - {run_id} (extracted by extract.py)"]
    for pid, info in sorted(prc_data.get("pids", {}).items()):
        lines.append(
            f"PID {pid} {info['program']:<10} {info['status']:<6} {info['message']}"
        )
        # Append per-station statistics under the relevant PIDs
        for sta, stats in prc_data.get("stations", {}).items():
            sta_lines = []
            if "epochs" in stats:
                flag = "WARNING below threshold 2000" if stats["epochs"] < 2000 else "ok"
                sta_lines.append(f"  {sta}  epochs={stats['epochs']}  {flag}")
            if "residual_rms_mm" in stats:
                flag = "FAIL exceeds 3 mm threshold" if stats["residual_rms_mm"] > 3.0 else "ok"
                sta_lines.append(f"  {sta}  residual_rms={stats['residual_rms_mm']} mm  {flag}")
            if sta_lines and pid in ("221", "311"):
                lines.extend(sta_lines)

    (out_dir / "prc_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    # --- coordinates.json ---
    coord_out = {
        "session": crd_data.get("session", run_id),
        "stations": crd_data.get("stations", {}),
    }
    (out_dir / "coordinates.json").write_text(
        json.dumps(coord_out, indent=2), encoding="utf-8"
    )

    print(f"Wrote {out_dir / 'prc_summary.txt'} and {out_dir / 'coordinates.json'}")
    print(f"Next: fill in {out_dir / 'truth.json'} with the correct action and your reasoning.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Bernese output files into monitor-ready format."
    )
    parser.add_argument("--prc", help="Path to Bernese .PRC summary file")
    parser.add_argument("--log", help="Path to Bernese BPE .LOG file (alternative to --prc)")
    parser.add_argument("--crd", required=True, help="Path to Bernese .CRD coordinate file")
    parser.add_argument("--out", required=True, help="Output directory (e.g. data/runs/F1)")
    parser.add_argument("--run-id", default="?", help="Run label (e.g. F1)")
    args = parser.parse_args()

    if not args.prc and not args.log:
        parser.error("Provide at least one of --prc or --log.")

    if args.prc:
        prc_data = parse_prc(args.prc)
    else:
        prc_data = parse_bpe_log(args.log)

    crd_data = parse_crd(args.crd)
    write_run(args.out, prc_data, crd_data, run_id=args.run_id)


if __name__ == "__main__":
    main()
