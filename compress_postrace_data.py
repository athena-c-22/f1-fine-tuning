"""
Compress raw F1 telemetry JSON inputs into compact summaries for fine-tuning.

Raw input:  ~2800 tokens
Compressed: ~600 tokens  (car_data resampled to 1 row per 5s; lap times kept in full)

Usage:
  # Compress training dataset
  python compress_race_data.py \
    --input post_race_combined.jsonl \
    --output post_race_compressed.jsonl

  # Compress a single race for inference (pipe-friendly)
  python compress_race_data.py --single race.json
"""

import json
import argparse
import statistics
from pathlib import Path


def compress_race_data(raw_input: dict, metadata: dict = None) -> dict:
    """
    Convert raw telemetry JSON into a compact summary.

    Args:
        raw_input: parsed dict with keys: laps, positions, pit_stops, stints, car_data
        metadata:  optional dict with year, gp, driver_name, driver_abbr

    Returns:
        Compact dict suitable as model input (~500 tokens vs ~1950 raw)
    """
    laps = raw_input.get("laps", [])
    stints = raw_input.get("stints", [])
    pit_stops = raw_input.get("pit_stops", [])
    positions = raw_input.get("positions", [])
    car_data = raw_input.get("car_data", [])

    # --- Lap times ---
    # Store as plain ordered array (index = lap_number - 1) to minimise tokens.
    # {"lap":1,"time":99.019} → 99.019  saves ~16 chars × 57 laps ≈ 300 tokens.
    lap_time_values = [l["time"] for l in sorted(laps, key=lambda l: l.get("lap", 0)) if l.get("time")]
    lap_times = [round(t, 2) for t in lap_time_values]   # 2 dp is enough precision

    if lap_time_values:
        fastest_idx = lap_time_values.index(min(lap_time_values))
        slowest_idx = lap_time_values.index(max(lap_time_values))
        lap_summary = {
            "total": len(lap_time_values),
            "avg": round(statistics.mean(lap_time_values), 2),
            "fastest": {"lap": fastest_idx + 1, "time": lap_time_values[fastest_idx]},
            "slowest": {"lap": slowest_idx + 1, "time": lap_time_values[slowest_idx]},
        }
    else:
        lap_summary = {"total": len(laps)}

    # --- Stints (already compact, keep as-is) ---
    stints_clean = [
        {
            "stint": s["stint"],
            "compound": s.get("compound", "UNKNOWN"),
            "laps": f"{s['lap_start']}-{s['lap_end']}",
            "length": s["lap_end"] - s["lap_start"] + 1,
        }
        for s in stints
    ]

    # --- Per-stint average lap time (helps model see pace evolution) ---
    if lap_times and stints:
        lap_map = {l["lap"]: l["time"] for l in laps if l.get("time")}
        for stint in stints_clean:
            start, end = stints[stints_clean.index(stint)]["lap_start"], stints[stints_clean.index(stint)]["lap_end"]
            stint_laps = [lap_map[i] for i in range(start, end + 1) if i in lap_map]
            if stint_laps:
                stint["avg_time"] = round(statistics.mean(stint_laps), 3)
                stint["fastest_time"] = round(min(stint_laps), 3)

    # --- Pit stops ---
    pit_clean = [
        {
            "lap": p.get("lap", p.get("lap_number")),
            "duration": round(p["duration"], 1) if p.get("duration") else None,
        }
        for p in pit_stops
    ]
    # If pit_stops empty but stints imply them, infer pit laps
    if not pit_clean and len(stints) > 1:
        pit_clean = [
            {"lap": stints[i]["lap_end"], "duration": None, "inferred": True}
            for i in range(len(stints) - 1)
        ]

    # --- Position summary ---
    start_pos = positions[0]["position"] if positions else None
    end_pos = positions[-1]["position"] if positions else None
    pos_summary = {
        "start": start_pos,
        "finish": end_pos,
        "changes": len(set(p["position"] for p in positions)) - 1 if positions else 0,
    }

    # --- Car data: resample to one row per 5 seconds ---
    # Original data is sampled at ~0.7-2.4s intervals; 5s spacing gives ~15-20 rows
    # per lap which captures straights, braking zones, and corners for debrief analysis.
    def parse_seconds(date_str):
        """Parse '0 days 00:00:04.630000' → float seconds."""
        try:
            time_part = date_str.split(" ")[-1]
            h, m, rest = time_part.split(":")
            return int(h) * 3600 + int(m) * 60 + float(rest)
        except Exception:
            return None

    car_resampled = []
    if car_data:
        # Sort by time (data can be unsorted)
        timed = [(parse_seconds(r["date"]), r) for r in car_data]
        timed = [(t, r) for t, r in timed if t is not None and t >= 0]
        timed.sort(key=lambda x: x[0])

        # Store as arrays [t, spd, rpm, gear, thr, brk, drs] — no keys repeated per row.
        # {"t":0.0,"spd":0,"rpm":10076,"gear":1,"thr":16,"brk":0,"drs":1} → [0.0,0,10076,1,16,0,1]
        # Saves ~40 chars × 20 rows ≈ 200 tokens.
        next_threshold = 0.0
        for t, row in timed:
            if t >= next_threshold:
                car_resampled.append([
                    round(t, 1),
                    row.get("speed"),
                    row.get("rpm"),
                    row.get("n_gear"),
                    row.get("throttle"),
                    row.get("brake"),
                    row.get("drs"),
                ])
                next_threshold = t + 5.0

    # --- Assemble compressed summary ---
    summary = {}

    summary["result"] = pos_summary
    summary["lap_summary"] = lap_summary
    summary["laps"] = lap_times          # ordered array: index 0 = lap 1
    summary["stints"] = stints_clean
    summary["pit_stops"] = pit_clean
    if car_resampled:
        summary["car_data_cols"] = ["t_sec", "speed_kmh", "rpm", "gear", "throttle_pct", "brake", "drs"]
        summary["car_data"] = car_resampled   # array-of-arrays at 5s intervals

    return summary


def main():
    parser = argparse.ArgumentParser(description="Compress F1 race data for fine-tuning")
    parser.add_argument("--input", type=str, help="Input JSONL file (post_race_combined.jsonl)")
    parser.add_argument("--output", type=str, help="Output JSONL file")
    parser.add_argument("--single", type=str, help="Compress a single JSON file and print result")
    parser.add_argument("--stats", action="store_true", help="Print token count comparison stats")
    args = parser.parse_args()

    if args.single:
        with open(args.single) as f:
            raw = json.load(f)
        compressed = compress_race_data(raw)
        print(json.dumps(compressed, indent=2))
        return

    if not args.input:
        parser.error("--input required unless using --single")

    output_path = args.output or args.input.replace(".jsonl", "_compressed.jsonl")

    print(f"Reading:  {args.input}")
    print(f"Writing:  {output_path}")

    examples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"Examples: {len(examples)}")

    raw_token_total = 0
    compressed_token_total = 0
    written = 0

    with open(output_path, "w") as out:
        for ex in examples:
            raw_input_str = ex.get("input", "")
            raw_input = json.loads(raw_input_str) if isinstance(raw_input_str, str) else raw_input_str
            metadata = ex.get("metadata", {})

            compressed = compress_race_data(raw_input, metadata)
            compressed_str = json.dumps(compressed, separators=(",", ":"))

            # Rough token estimate: ~4 chars per token
            raw_tokens = len(raw_input_str) // 4
            comp_tokens = len(compressed_str) // 4
            raw_token_total += raw_tokens
            compressed_token_total += comp_tokens

            out.write(json.dumps({
                "input": compressed_str,
                "output": ex["output"],
                "metadata": metadata,
            }) + "\n")
            written += 1

    print(f"Written:  {written} examples")
    if args.stats or True:
        avg_raw = raw_token_total // written
        avg_comp = compressed_token_total // written
        reduction = round(100 * (1 - avg_comp / avg_raw))
        print(f"\nToken stats (estimated):")
        print(f"  Avg raw input:        ~{avg_raw} tokens")
        print(f"  Avg compressed input: ~{avg_comp} tokens")
        print(f"  Reduction:            {reduction}%")
        print(f"\nDone. Use {output_path} for fine-tuning.")


if __name__ == "__main__":
    main()
