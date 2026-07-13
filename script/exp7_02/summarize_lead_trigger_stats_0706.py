import json
from pathlib import Path
from statistics import mean, median

BASE = Path(
    "output/experiments/20260705_integrated_cot_lead_baselines/"
    "integrated_repo_cot_lead_baselines/r1_onevision_7b"
)


def pct(n, d):
    return None if d == 0 else n / d


def q(values, quantile):
    if not values:
        return None
    values = sorted(values)
    idx = int(round((len(values) - 1) * quantile))
    return values[idx]


def summarize_run(run_dir: Path):
    full = run_dir / "token_entropy_full.jsonl"
    brief = run_dir / "token_entropy.jsonl"
    results = run_dir / "results.jsonl"
    if not full.exists():
        return None

    rows = []
    with full.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            tokens = obj.get("tokens") or []
            token_n = len(tokens)
            soft = sum(1 for t in tokens if t.get("mode") == "soft")
            to_soft = sum(1 for t in tokens if t.get("to_soft"))
            to_normal = sum(1 for t in tokens if t.get("to_normal"))
            step0_soft = bool(tokens and tokens[0].get("mode") == "soft")
            step1_to_normal = bool(len(tokens) > 1 and tokens[1].get("to_normal"))
            later_to_soft = sum(
                1
                for t in tokens
                if t.get("to_soft") and int(t.get("step", -1)) > 1
            )
            first_later_to_soft_step = None
            for t in tokens:
                if t.get("to_soft") and int(t.get("step", -1)) > 1:
                    first_later_to_soft_step = int(t.get("step", -1))
                    break
            rows.append(
                {
                    "token_n": token_n,
                    "soft": soft,
                    "soft_ratio": soft / token_n if token_n else 0.0,
                    "to_soft": to_soft,
                    "to_normal": to_normal,
                    "switch": to_soft + to_normal,
                    "step0_soft": step0_soft,
                    "step1_to_normal": step1_to_normal,
                    "later_to_soft": later_to_soft,
                    "first_later_to_soft_step": first_later_to_soft_step,
                    "only_initial_transition": step0_soft
                    and step1_to_normal
                    and later_to_soft == 0,
                }
            )

    n = len(rows)
    result_n = sum(1 for _ in results.open()) if results.exists() else None
    return {
        "samples": n,
        "results_rows": result_n,
        "brief_trace_rows": sum(1 for _ in brief.open()) if brief.exists() else None,
        "mean_output_tokens": mean([r["token_n"] for r in rows]) if rows else None,
        "mean_soft_tokens": mean([r["soft"] for r in rows]) if rows else None,
        "mean_soft_ratio": mean([r["soft_ratio"] for r in rows]) if rows else None,
        "median_soft_ratio": median([r["soft_ratio"] for r in rows]) if rows else None,
        "mean_to_soft": mean([r["to_soft"] for r in rows]) if rows else None,
        "mean_to_normal": mean([r["to_normal"] for r in rows]) if rows else None,
        "mean_switch": mean([r["switch"] for r in rows]) if rows else None,
        "p50_switch": q([r["switch"] for r in rows], 0.50),
        "p90_switch": q([r["switch"] for r in rows], 0.90),
        "max_switch": max([r["switch"] for r in rows], default=None),
        "step0_soft_rate": pct(sum(r["step0_soft"] for r in rows), n),
        "step1_to_normal_rate": pct(sum(r["step1_to_normal"] for r in rows), n),
        "later_to_soft_sample_rate": pct(sum(r["later_to_soft"] > 0 for r in rows), n),
        "only_initial_transition_rate": pct(
            sum(r["only_initial_transition"] for r in rows), n
        ),
        "mean_later_to_soft": mean([r["later_to_soft"] for r in rows]) if rows else None,
        "first_later_to_soft_step_median": median(
            [
                r["first_later_to_soft_step"]
                for r in rows
                if r["first_later_to_soft_step"] is not None
            ]
        )
        if any(r["first_later_to_soft_step"] is not None for r in rows)
        else None,
    }


def fmt_pct(x):
    return "NA" if x is None else f"{x*100:.1f}%"


def main():
    out = {}
    for run in sorted(BASE.glob("*/lead_gpu1")):
        dataset = run.parent.name
        stats = summarize_run(run)
        if stats:
            out[dataset] = stats

    out_path = BASE.parent / "lead_trigger_stats.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    lines = [
        "| dataset | samples | result rows | mean switch | mean to_soft | mean to_normal | later to_soft sample | only initial transition | mean soft ratio | mean soft tokens | p90 switch | max switch | first later to_soft step median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, s in out.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    dataset,
                    str(s["samples"]),
                    str(s["results_rows"]),
                    f"{s['mean_switch']:.2f}",
                    f"{s['mean_to_soft']:.2f}",
                    f"{s['mean_to_normal']:.2f}",
                    fmt_pct(s["later_to_soft_sample_rate"]),
                    fmt_pct(s["only_initial_transition_rate"]),
                    f"{s['mean_soft_ratio']:.4f}",
                    f"{s['mean_soft_tokens']:.2f}",
                    str(s["p90_switch"]),
                    str(s["max_switch"]),
                    str(s["first_later_to_soft_step_median"]),
                ]
            )
            + " |"
        )
    md_path = BASE.parent / "lead_trigger_stats.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(md_path)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
