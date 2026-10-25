"""Markdown fairness report from one or more eval_external FairFace JSONs (raw report or Modal wrapper).

    python scripts/fairness_report.py --inputs runs/modal/b1/eval_fairface.json runs/modal/b3/eval_fairface.json \
        --labels B1,B3 --output runs/fairness_b1_b3.md

Per input: race and race|gender tables (FPR = verdict 'fake' on real faces, review = verdict 'abstain', 95% Wilson
CIs), disparity summaries per attribute, then a side-by-side comparison of worst-group FPR, gap and overall FPR.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_report(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    report = data["report"] if isinstance(data.get("report"), dict) else data
    if "fairness" not in report:
        raise ValueError(f"{path}: no 'fairness' block (re-run eval_external.py --dataset fairface)")
    return report


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.2f}%"


def _cell(x: str) -> str:
    return str(x).replace("|", "\\|")


def _rate_ci(r: dict) -> str:
    return f"{_pct(r['rate'])} [{_pct(r['ci_low'])}, {_pct(r['ci_high'])}]"


def _table(attr_block: dict, mean_p: dict | None) -> list[str]:
    fpr, review = attr_block["fpr"]["table"], attr_block["review"]["table"]
    head = "| group | n | FPR [95% CI] | review [95% CI] |" + (" mean_p_fake |" if mean_p else "")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for g in fpr:
        row = f"| {_cell(g)} | {fpr[g]['n']} | {_rate_ci(fpr[g])} | {_rate_ci(review[g])} |"
        if mean_p:
            row += f" {mean_p[g]['mean_p_fake']:.4f} |" if g in mean_p else " - |"
        lines.append(row)
    return lines


def _summary_line(name: str, s: dict, target: float | None) -> str:
    if s["max_group"] is None:
        return f"- **{name}**: no powered groups; underpowered: {', '.join(s['underpowered']) or 'none'}"
    ratio = "n/a (min 0)" if s["ratio"] is None else f"{s['ratio']:.2f}x"
    gap_ci = s.get("gap_ci")
    gap = _pct(s["gap"]) + (f" [{_pct(gap_ci[0])}, {_pct(gap_ci[1])}]" if gap_ci else "")
    line = (f"- **{name}**: max {s['max_group']} {_pct(s['max_rate'])}, min {s['min_group']} {_pct(s['min_rate'])}, "
            f"gap {gap}, ratio {ratio}")
    if target is not None:
        line += (f"; above target {_pct(target)} (point): {', '.join(s['above_target_point']) or 'none'}"
                 f"; (Wilson lower bound): {', '.join(s['above_target_ci']) or 'none'}")
        if s.get("above_target_holm") is not None:
            line += f"; (Holm-corrected exact test): {', '.join(s['above_target_holm']) or 'none'}"
    return line + f"; underpowered: {', '.join(s['underpowered']) or 'none'}"


def render_input(label: str, report: dict) -> list[str]:
    fair = report["fairness"]
    lines = [f"## {label}", "", f"- checkpoint: `{report.get('checkpoint', '?')}`",
             f"- calibration: {report.get('calibration', {}).get('status', '?')}, target FPR {_pct(fair['target_fpr'])}, "
             f"min_n {fair['min_n']}", f"- overall FPR {_pct(report['overall'].get('fpr_fake_verdict'))}, "
             f"review {_pct(report['overall'].get('abstain_rate'))}, n {report['overall'].get('n')}"]
    if not fair["verdicts_valid"]:
        lines.append("- **WARNING: verdicts not valid (calibration not matched; all forced to abstain). "
                     "FPR numbers are not meaningful.**")
    attrs = fair["attributes"]
    lines += ["", "### Race", ""] + _table(attrs["race"], report.get("by_race"))
    if "race|gender" in attrs:
        lines += ["", "### Race x gender", ""] + _table(attrs["race|gender"], None)
    lines += ["", "### Summary", ""]
    for attr, block in attrs.items():
        lines.append(_summary_line(f"{attr} FPR", block["fpr"]["summary"], fair["target_fpr"]))
        lines.append(_summary_line(f"{attr} review", block["review"]["summary"], None))
    return lines + [""]


def render_comparison(labels: list[str], reports: list[dict]) -> list[str]:
    def worst(rep, attr):
        s = rep["fairness"]["attributes"].get(attr, {}).get("fpr", {}).get("summary", {})
        return "-" if not s.get("max_group") else f"{s['max_group']} {_pct(s['max_rate'])}"

    def gap(rep, attr):
        s = rep["fairness"]["attributes"].get(attr, {}).get("fpr", {}).get("summary", {})
        ci = s.get("gap_ci")
        return _pct(s.get("gap")) + (f" [{_pct(ci[0])}, {_pct(ci[1])}]" if ci else "")

    def holm(rep, attr):
        s = rep["fairness"]["attributes"].get(attr, {}).get("fpr", {}).get("summary", {})
        return "-" if s.get("above_target_holm") is None else (", ".join(s["above_target_holm"]) or "none")

    rows = [("overall FPR", lambda r: _pct(r["overall"].get("fpr_fake_verdict"))),
            ("overall review", lambda r: _pct(r["overall"].get("abstain_rate"))),
            ("worst race FPR", lambda r: worst(r, "race")), ("race FPR gap", lambda r: gap(r, "race")),
            ("worst race|gender FPR", lambda r: worst(r, "race|gender")),
            ("race|gender FPR gap", lambda r: gap(r, "race|gender")),
            ("race above target (Holm)", lambda r: holm(r, "race")),
            ("verdicts valid", lambda r: str(r["fairness"]["verdicts_valid"]))]
    lines = ["## Comparison", "", "| metric | " + " | ".join(map(_cell, labels)) + " |",
             "|---|" + "---|" * len(labels)]
    lines += [f"| {_cell(name)} | " + " | ".join(_cell(fn(r)) for r in reports) + " |" for name, fn in rows]
    return lines + [""]


def render(labels: list[str], reports: list[dict]) -> str:
    lines = ["# FairFace fairness report", "",
             "FPR = share of real faces with verdict 'fake'; review = share with verdict 'abstain'. "
             "Brackets are 95% Wilson intervals (gap: parametric bootstrap); gap/ratio use only groups with n >= min_n. "
             "'Holm' = exact one-sided binomial test of rate > target, Holm-corrected within each attribute.", ""]
    for label, rep in zip(labels, reports):
        lines += render_input(label, rep)
    if len(reports) > 1:
        lines += render_comparison(labels, reports)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> str:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--labels", default=None, help="comma-separated, one per input (default: file stems)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)
    labels = args.labels.split(",") if args.labels else [Path(p).stem for p in args.inputs]
    if len(labels) != len(args.inputs):
        raise SystemExit(f"--labels has {len(labels)} entries for {len(args.inputs)} inputs")
    md = render(labels, [load_report(p) for p in args.inputs])
    if args.output:
        Path(args.output).write_text(md)
    print(md)
    return md


if __name__ == "__main__":
    main()
