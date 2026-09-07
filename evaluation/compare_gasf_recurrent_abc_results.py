"""Compare feed-forward and previous-ABC recurrent GASF evaluations."""

import argparse
import json
from pathlib import Path


TRACKING_METRICS = (
    "position_prediction_rmse",
    "position_update_rmse",
    "velocity_prediction_rmse",
    "velocity_update_rmse",
)


def relative_change(baseline, recurrent):
    return 100.0 * (recurrent - baseline) / max(abs(baseline), 1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--recurrent", required=True)
    parser.add_argument("--output", required=True)
    cli = parser.parse_args()
    baseline = json.loads(Path(cli.baseline).read_text(encoding="utf-8"))
    recurrent = json.loads(Path(cli.recurrent).read_text(encoding="utf-8"))

    comparison = {
        "baseline_report": str(Path(cli.baseline).resolve()),
        "recurrent_report": str(Path(cli.recurrent).resolve()),
        "single_factor": "previous per-scale ABC parameter recurrence",
        "tracking": {},
        "filter_dynamics": {},
        "timing": {},
    }
    for event in ("all", "steady", "switch", "strong"):
        baseline_event = baseline.get("events", {}).get(event)
        recurrent_event = recurrent.get("events", {}).get(event)
        if not baseline_event or not recurrent_event:
            continue
        comparison["tracking"][event] = {
            metric: {
                "baseline": baseline_event[metric],
                "recurrent": recurrent_event[metric],
                "recurrent_minus_baseline": (
                    recurrent_event[metric] - baseline_event[metric]
                ),
                "relative_change_percent": relative_change(
                    baseline_event[metric], recurrent_event[metric]
                ),
            }
            for metric in TRACKING_METRICS
        }
        dynamics = {}
        for metric in (
            "gasf_parameter_update_gate_mean",
            "gasf_parameter_step_norm_mean",
            "gasf_parameter_step_norm_rms",
            "gasf_filter_response_step_rms_mean",
            "gasf_filter_response_step_rms_rms",
        ):
            if metric in baseline_event and metric in recurrent_event:
                dynamics[metric] = {
                    "baseline": baseline_event[metric],
                    "recurrent": recurrent_event[metric],
                    "ratio": (
                        recurrent_event[metric]
                        / max(abs(baseline_event[metric]), 1e-12)
                    ),
                }
        comparison["filter_dynamics"][event] = dynamics

    for name, report in (("baseline", baseline), ("recurrent", recurrent)):
        comparison["timing"][name] = report.get("timing", {})
    recurrent_events = recurrent.get("events", {})
    if "steady" in recurrent_events and "strong" in recurrent_events:
        steady = recurrent_events["steady"]
        strong = recurrent_events["strong"]
        comparison["filter_dynamics"]["recurrent_strong_to_steady"] = {
            metric: strong[metric] / max(abs(steady[metric]), 1e-12)
            for metric in (
                "gasf_parameter_update_gate_mean",
                "gasf_parameter_step_norm_mean",
                "gasf_filter_response_step_rms_mean",
            )
            if metric in steady and metric in strong
        }

    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
