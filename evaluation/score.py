"""Aggregate case-level DepGuide judgments offline. Run with --help for input formats."""

import argparse
from collections import Counter
import json
import random
from statistics import mean

if __package__:
    from .prompts import DIALOGUE_DIMENSIONS, REVISION_DIMENSIONS, SIDES
else:
    from prompts import DIALOGUE_DIMENSIONS, REVISION_DIMENSIONS, SIDES


def checked_scores(scores, dimensions):
    if not isinstance(scores, dict) or set(scores) != set(dimensions):
        raise ValueError(f"Expected score fields {dimensions}")
    if any(type(n) is not int or n not in (0, 1, 2) for n in scores.values()):
        raise ValueError("Each dimension requires an integer from 0 to 2")
    return {**scores, "total": sum(scores.values())}


def summarize(values, seed=42, samples=2000):
    """Resample source groups with replacement, retaining every case in each sampled group."""
    if not values:
        return {"n": 0, "source_groups": 0, "mean": None, "ci95": None}
    groups = {}
    for group, value in values:
        groups.setdefault(group, []).append(value)
    aggregates = [(sum(v), len(v)) for v in groups.values()]
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(samples):
        draw = [aggregates[rng.randrange(len(aggregates))] for _ in aggregates]
        bootstrap.append(sum(v for v, _ in draw) / sum(n for _, n in draw))
    bootstrap.sort()
    def percentile(q):
        pos = (samples - 1) * q
        lo = int(pos)
        hi = min(lo + 1, samples - 1)
        return bootstrap[lo] + (bootstrap[hi] - bootstrap[lo]) * (pos - lo)
    return {"n": len(values), "source_groups": len(groups), "mean": mean(v for _, v in values),
            "ci95": [percentile(0.025), percentile(0.975)]}


def dialogue(rows, baseline, method):
    """Means use the entire input cohort; also return the paired gain for completed pairs only."""
    if baseline == method:
        raise ValueError("Choose two distinct systems")
    systems = (baseline, method)
    metrics = (*DIALOGUE_DIMENSIONS, "total")
    scores = {system: [] for system in systems}
    scored = Counter()
    paired = []
    completed = []
    for row in rows:
        group = row["source_group_id"]
        current = {}
        for system in systems:
            raw = row["scores"][system]
            if raw is None:
                current[system] = {d: 0 for d in metrics}
            else:
                current[system] = checked_scores(raw, DIALOGUE_DIMENSIONS)
                scored[system] += 1
            scores[system].append((group, current[system]))
        item = (group, current[method]["total"] - current[baseline]["total"])
        paired.append(item)
        if all(row["scores"][s] is not None for s in systems):
            completed.append(item)
    result = {}
    for system, entries in scores.items():
        result[system] = {
            "scored": scored[system],
            "all_case_means": {d: mean(v[d] for _, v in entries) for d in metrics},
            "total_ci95": summarize([(g, v["total"]) for g, v in entries])["ci95"],
        }
    return {"eligible": len(rows), "systems": result, "gain_method_minus_baseline": {
        "all_cases": summarize(paired), "completed_pairs": summarize(completed)}}


def revision(rows):
    """Average each side across orders, then compute revised minus original within each case."""
    metrics = (*REVISION_DIMENSIONS, "total")
    diffs = {d: [] for d in metrics}
    levels = {s: {d: [] for d in metrics} for s in SIDES}
    counts = Counter()
    preference = Counter()
    for row in sorted(rows, key=lambda r: r["case_id"]):
        group = row["source_group_id"]
        orders = row["orders"]
        identical = row["identical_replies"]
        if type(identical) is not bool:
            raise ValueError("identical_replies must be true or false")
        if identical:
            if orders != []:
                raise ValueError("Identical replies take an empty order list")
            counts["identical"] += 1
            preference["tie"] += 1
            for d in metrics:
                diffs[d].append((group, 0.0))
            continue
        if not isinstance(orders, list) or len(orders) > 2:
            raise ValueError("Expected a list of up to two judgments")
        if len(orders) < 2 or any(order is None for order in orders):
            counts["incomplete"] += 1
            continue
        if {order["x_arm"] for order in orders} != set(SIDES):
            raise ValueError("Each reply must be shown once in each position")
        judgments = {s: [] for s in SIDES}
        votes = []
        for order in orders:
            x = order["x_arm"]
            y = "revised" if x == "original" else "original"
            answer = order["judgment"]
            judgments[x].append(checked_scores(answer["X"], REVISION_DIMENSIONS))
            judgments[y].append(checked_scores(answer["Y"], REVISION_DIMENSIONS))
            votes.append({"X": x, "Y": y, "tie": "tie"}[answer["overall_preference"]])
        preference[votes[0] if votes[0] == votes[1] else "order_inconsistent"] += 1
        counts["double_scored"] += 1
        for d in metrics:
            for side in SIDES:
                levels[side][d].append(mean(v[d] for v in judgments[side]))
            diffs[d].append((group, levels["revised"][d][-1] - levels["original"][d][-1]))
    return {
        "eligible": len(rows), "double_scored": counts["double_scored"],
        "identical": counts["identical"], "incomplete": counts["incomplete"],
        "preference": {k: preference[k] for k in (*SIDES, "tie", "order_inconsistent")},
        "nonidentical_means": {s: {d: mean(v) if v else None for d, v in m.items()}
                               for s, m in levels.items()},
        "gain_revised_minus_original": {d: summarize(v) for d, v in diffs.items()},
    }


def read_rows(path):
    rows = []
    seen = set()
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not all(
                isinstance(row.get(key), str) and row[key].strip()
                for key in ("case_id", "source_group_id")
            ):
                raise ValueError(f"Missing case_id or source_group_id at line {line_number}")
            if row["case_id"] in seen:
                raise ValueError(f"Duplicate case_id at line {line_number}")
            seen.add(row["case_id"])
            rows.append(row)
    if not rows:
        raise ValueError("Expected at least one eligible case")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="task", required=True)
    dialogue_cli = commands.add_parser("dialogue", description=(
        "One JSON object per eligible case in the benchmark's original order: case_id, "
        "source_group_id and scores keyed by system name. Each score contains goal_fit, "
        "concrete_execution and feedback_adaptation (0/1/2); use null for an unscored system. "
        "All-case means give unscored systems zero on that case."
    ), epilog='Row example: {"case_id":"a","source_group_id":"g","scores":{"backbone":null,'
              '"depreplan":{"goal_fit":2,"concrete_execution":1,"feedback_adaptation":2}}}')
    dialogue_cli.add_argument("judgments", help="UTF-8 JSONL with all eligible cases")
    dialogue_cli.add_argument("--baseline", required=True)
    dialogue_cli.add_argument("--method", required=True)
    revision_cli = commands.add_parser("revision", description=(
        "One JSON object per eligible case: case_id, source_group_id, identical_replies, "
        "orders. Each order has x_arm (original/revised) and judgment (the judge's raw X, Y "
        "and overall_preference fields). Use [] for identical replies; an incomplete pair "
        "with identical_replies=false is excluded from paired gains."
    ), epilog='Identical-reply row: {"case_id":"a","source_group_id":"g",'
              '"identical_replies":true,"orders":[]}. For distinct replies store both '
              '{"x_arm":side,"judgment":answer} objects from revision_prompts in orders.')
    revision_cli.add_argument("judgments", help="UTF-8 JSONL with all eligible cases")
    args = parser.parse_args()
    try:
        rows = read_rows(args.judgments)
        result = dialogue(rows, args.baseline, args.method) if args.task == "dialogue" else revision(rows)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.exit(2, f"Invalid input: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
