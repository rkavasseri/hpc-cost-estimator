#!/usr/bin/env python3
"""
hpc_cost_estimator.py — GPU/HPC compute cost estimator for ML research pipelines.

Zero external dependencies (stdlib only). Two ways to use it:

  1) QUICK ESTIMATE — one phase, straight from flags:

       python3 hpc_cost_estimator.py --params 200M --tokens 50B --gpu h100 --num-gpus 4

  2) FULL PIPELINE — a multi-phase research pipeline described in a JSON file:

       python3 hpc_cost_estimator.py --config pipeline.json
       python3 hpc_cost_estimator.py --config pipeline.json --compare-commercial
       python3 hpc_cost_estimator.py --config pipeline.json --csv out.csv

See pipeline_example_vlm.json and pipeline_example_sabha.json for config format.
Run `python3 hpc_cost_estimator.py --help` for all options.
"""

import argparse
import json
import sys
import csv as csv_module

# ── Hardware & rate defaults (edit these to match your cluster / cloud) ─────

GPU_DEFAULTS = {
    "h100": {"peak_tflops_bf16": 989.0, "su_rate": 1.00},
    "a100": {"peak_tflops_bf16": 312.0, "su_rate": 0.80},
}

# Rough on-demand commercial rates ($/GPU-hr), mid-2026 market, for --compare-commercial.
# These drift monthly — treat as ballpark, not quote.
COMMERCIAL_RATES = {
    "h100": {"specialized": 3.00, "hyperscaler": 6.00},
    "a100": {"specialized": 1.70, "hyperscaler": 3.30},
}

# Default Model FLOPs Utilization for the params+tokens auto-estimator.
# Published research pipelines commonly land in the 20-40% range; 30% is a
# reasonable non-optimistic default. Override with --mfu or a per-phase "mfu" field.
DEFAULT_MFU = 0.30

# Default grant-budget contingency layers, applied multiplicatively to the base total.
DEFAULT_CONTINGENCY = [
    ("Training restarts & instability", 0.10),
    ("Exploratory runs & debugging", 0.15),
    ("Peer review / revision experiments", 0.10),
]


def parse_size(value):
    """Parse '200M', '7B', '1.5B', '900K', or a plain number into a float."""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().upper()
    mult = 1.0
    if s.endswith("B"):
        mult, s = 1e9, s[:-1]
    elif s.endswith("M"):
        mult, s = 1e6, s[:-1]
    elif s.endswith("K"):
        mult, s = 1e3, s[:-1]
    return float(s) * mult


def fmt_size(n):
    """Inverse of parse_size, for display."""
    if n >= 1e9:
        return f"{n/1e9:g}B"
    if n >= 1e6:
        return f"{n/1e6:g}M"
    if n >= 1e3:
        return f"{n/1e3:g}K"
    return f"{n:g}"


def compute_phase_gpu_hours(phase, default_mfu):
    """
    Returns (gpu_hours, method_string).

    Priority order:
      1. Explicit "gpu_hours" field.
      2. "wall_clock_hours" * "num_gpus".
      3. FLOPs-based estimate from "params" + "tokens" (LLM/text) or
         "params" + "images" (vision), using peak GPU FLOPs * MFU.
    """
    if "gpu_hours" in phase:
        return float(phase["gpu_hours"]), "explicit"

    if "wall_clock_hours" in phase and "num_gpus" in phase:
        gh = float(phase["wall_clock_hours"]) * float(phase["num_gpus"])
        return gh, "wall_clock x GPUs"

    gpu = phase.get("gpu", "h100").lower()
    if gpu not in GPU_DEFAULTS:
        raise ValueError(f"Unknown GPU type '{gpu}' in phase '{phase.get('name', '?')}'")
    peak_flops = GPU_DEFAULTS[gpu]["peak_tflops_bf16"] * 1e12
    mfu = float(phase.get("mfu", default_mfu))
    effective_flops_per_sec = peak_flops * mfu

    params = parse_size(phase["params"]) if "params" in phase else None
    if params is None:
        raise ValueError(
            f"Phase '{phase.get('name', '?')}' needs one of: gpu_hours | "
            f"wall_clock_hours+num_gpus | params+tokens | params+images"
        )

    if "tokens" in phase:
        tokens = parse_size(phase["tokens"])
        flops = 6.0 * params * tokens  # standard training-compute approximation
        method = f"estimated: 6*params*tokens, mfu={mfu:.2f}"
    elif "images" in phase:
        images = parse_size(phase["images"])
        # Rough heuristic: ~4x params FLOPs per image (fwd+bwd), absent a better number.
        # Override with an explicit "flops_per_image" field when you have one from a paper.
        flops_per_image = float(phase.get("flops_per_image", 4.0 * params))
        flops = flops_per_image * images
        method = f"estimated: heuristic flops/image, mfu={mfu:.2f}"
    else:
        raise ValueError(
            f"Phase '{phase.get('name', '?')}' has 'params' but neither 'tokens' nor 'images'"
        )

    gpu_seconds = flops / effective_flops_per_sec
    gpu_hours = gpu_seconds / 3600.0
    return gpu_hours, method


def process_phase(phase, default_mfu):
    name = phase.get("name", "Unnamed phase")

    if "flat_cost" in phase:
        # Non-GPU line item (storage, egress, etc.) — cost given directly.
        return {
            "name": name,
            "gpu": "-",
            "gpu_hours": None,
            "su_rate": None,
            "cost": float(phase["flat_cost"]),
            "method": "flat cost",
            "notes": phase.get("notes", ""),
        }

    gpu = phase.get("gpu", "h100").lower()
    gpu_hours, method = compute_phase_gpu_hours(phase, default_mfu)
    su_rate = float(phase.get("su_rate", GPU_DEFAULTS[gpu]["su_rate"]))
    cost = gpu_hours * su_rate

    return {
        "name": name,
        "gpu": gpu,
        "gpu_hours": gpu_hours,
        "su_rate": su_rate,
        "cost": cost,
        "method": method,
        "notes": phase.get("notes", ""),
    }


def commercial_cost(rows, tier):
    """Recompute total cost for all GPU rows at a commercial rate tier ('specialized'|'hyperscaler')."""
    total = 0.0
    for r in rows:
        if r["gpu_hours"] is None:
            total += r["cost"]  # flat costs unchanged across tiers
            continue
        rate = COMMERCIAL_RATES.get(r["gpu"], {}).get(tier)
        if rate is None:
            total += r["cost"]  # unknown gpu type: fall back to SU cost
            continue
        total += r["gpu_hours"] * rate
    return total


# ── Table printing (stdlib only) ─────────────────────────────────────────────

def print_table(headers, rows, aligns=None):
    aligns = aligns or ["l"] * len(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(cells):
        parts = []
        for cell, w, a in zip(cells, widths, aligns):
            s = str(cell)
            parts.append(s.rjust(w) if a == "r" else s.ljust(w))
        return "  ".join(parts)

    print(fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def money(x):
    return f"${x:,.2f}"


def hours(x):
    return f"{x:,.1f}"


# ── Main pipeline runner ─────────────────────────────────────────────────────

def run_pipeline(config, default_mfu, compare_commercial=False, contingency_override=None):
    phases = config["phases"]
    rows = [process_phase(p, default_mfu) for p in phases]
    base_total = sum(r["cost"] for r in rows)

    print()
    if "name" in config:
        print(config["name"])
        print("=" * len(config["name"]))
        print()

    table_rows = []
    for r in rows:
        gh = hours(r["gpu_hours"]) if r["gpu_hours"] is not None else "-"
        rate = money(r["su_rate"]) if r["su_rate"] is not None else "-"
        table_rows.append([r["name"], r["gpu"], gh, rate, money(r["cost"]), r["method"]])
    table_rows.append(["TOTAL", "", "", "", money(base_total), ""])

    print_table(
        ["Phase", "GPU", "GPU-hrs", "SU Rate", "Cost", "Method"],
        table_rows,
        aligns=["l", "l", "r", "r", "r", "l"],
    )

    # Contingency
    contingency = contingency_override if contingency_override is not None else DEFAULT_CONTINGENCY
    print()
    print("Contingency")
    print("-----------")
    cont_rows = []
    running_total = base_total
    for label, pct in contingency:
        amt = base_total * pct
        running_total += amt
        cont_rows.append([label, f"{pct*100:.0f}%", money(amt)])
    cont_rows.append(["TOTAL REQUESTED", "", money(running_total)])
    print_table(["Item", "Rate", "Amount"], cont_rows, aligns=["l", "r", "r"])

    if compare_commercial:
        print()
        print("Commercial cloud comparison (on-demand, approximate)")
        print("------------------------------------------------------")
        spec_total = commercial_cost(rows, "specialized")
        hyper_total = commercial_cost(rows, "hyperscaler")
        comp_rows = [
            ["Institutional SU rate (this cluster)", money(base_total), "1.0x"],
            ["Specialized cloud (RunPod/Lambda/CoreWeave-class)", money(spec_total), f"{spec_total/base_total:.1f}x"],
            ["Hyperscaler on-demand (AWS/GCP/Azure-class)", money(hyper_total), f"{hyper_total/base_total:.1f}x"],
        ]
        print_table(["Tier", "Base pipeline cost", "Multiplier"], comp_rows, aligns=["l", "r", "r"])

    print()
    return rows, base_total, running_total


def write_csv(path, rows, base_total, contingency, final_total):
    with open(path, "w", newline="") as f:
        w = csv_module.writer(f)
        w.writerow(["Phase", "GPU", "GPU-hrs", "SU Rate", "Cost", "Method", "Notes"])
        for r in rows:
            w.writerow([
                r["name"], r["gpu"],
                f"{r['gpu_hours']:.2f}" if r["gpu_hours"] is not None else "",
                f"{r['su_rate']:.2f}" if r["su_rate"] is not None else "",
                f"{r['cost']:.2f}", r["method"], r["notes"],
            ])
        w.writerow([])
        w.writerow(["", "", "", "", f"{base_total:.2f}", "BASE TOTAL", ""])
        for label, pct in contingency:
            w.writerow(["", "", "", "", f"{base_total*pct:.2f}", label, f"{pct*100:.0f}%"])
        w.writerow(["", "", "", "", f"{final_total:.2f}", "TOTAL REQUESTED", ""])
    print(f"Wrote {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="GPU/HPC compute cost estimator for ML research pipelines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--config", help="Path to a pipeline JSON config (multi-phase mode)")
    ap.add_argument("--params", help="Model parameter count, e.g. 200M, 7B (quick mode)")
    ap.add_argument("--tokens", help="Training tokens, e.g. 50B (quick mode, LLM)")
    ap.add_argument("--images", help="Training images, e.g. 10M (quick mode, vision)")
    ap.add_argument("--gpu", default="h100", choices=list(GPU_DEFAULTS.keys()))
    ap.add_argument("--num-gpus", type=int, default=1, help="For wall-clock display only")
    ap.add_argument("--mfu", type=float, default=DEFAULT_MFU, help=f"Model FLOPs Utilization (default {DEFAULT_MFU})")
    ap.add_argument("--su-rate", type=float, help="Override SU rate ($/GPU-hr)")
    ap.add_argument("--compare-commercial", action="store_true", help="Add commercial cloud comparison")
    ap.add_argument("--csv", help="Also write results to this CSV path")
    ap.add_argument("--no-contingency", action="store_true", help="Skip contingency section")
    args = ap.parse_args()

    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        contingency = None
        if args.no_contingency:
            contingency = []
        elif "contingency" in config:
            contingency = [(c["label"], c["pct"]) for c in config["contingency"]]

        rows, base_total, final_total = run_pipeline(
            config, args.mfu, compare_commercial=args.compare_commercial,
            contingency_override=contingency,
        )
        if args.csv:
            cont = contingency if contingency is not None else DEFAULT_CONTINGENCY
            write_csv(args.csv, rows, base_total, cont, final_total)
        return

    # Quick single-phase mode
    if not args.params:
        ap.error("Provide --config for pipeline mode, or --params (+--tokens/--images) for quick mode")

    phase = {"name": "Quick estimate", "params": args.params, "gpu": args.gpu}
    if args.tokens:
        phase["tokens"] = args.tokens
    if args.images:
        phase["images"] = args.images
    if args.su_rate:
        phase["su_rate"] = args.su_rate

    row = process_phase(phase, args.mfu)
    print()
    print_table(
        ["Phase", "GPU", "GPU-hrs", "SU Rate", "Cost", "Method"],
        [[row["name"], row["gpu"], hours(row["gpu_hours"]), money(row["su_rate"]), money(row["cost"]), row["method"]]],
        aligns=["l", "l", "r", "r", "r", "l"],
    )
    if args.num_gpus > 1:
        print(f"\nAt {args.num_gpus} GPUs in parallel: ~{row['gpu_hours']/args.num_gpus:,.1f} hrs wall-clock")
    if args.compare_commercial:
        spec = commercial_cost([row], "specialized")
        hyper = commercial_cost([row], "hyperscaler")
        print(f"\nSpecialized cloud: {money(spec)} ({spec/row['cost']:.1f}x)")
        print(f"Hyperscaler on-demand: {money(hyper)} ({hyper/row['cost']:.1f}x)")
    print()


if __name__ == "__main__":
    main()
