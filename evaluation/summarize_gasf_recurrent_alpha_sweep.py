"""Summarize a fixed-alpha previous-ABC recurrence sweep."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    cli = parser.parse_args()
    rows = []
    for path in sorted(Path(cli.input_dir).glob("alpha_*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        alpha = report["protocol"]["gasf_recurrent_alpha_override"]
        all_event = report["events"]["all"]
        steady = report["events"].get("steady", {})
        strong = report["events"].get("strong", {})
        rows.append({
            "alpha": float(alpha),
            "position_prediction_rmse": all_event["position_prediction_rmse"],
            "position_update_rmse": all_event["position_update_rmse"],
            "velocity_prediction_rmse": all_event["velocity_prediction_rmse"],
            "velocity_update_rmse": all_event["velocity_update_rmse"],
            "steady_position_prediction_rmse": steady.get(
                "position_prediction_rmse"
            ),
            "strong_position_prediction_rmse": strong.get(
                "position_prediction_rmse"
            ),
            "filter_response_step_rms_mean": all_event.get(
                "gasf_filter_response_step_rms_mean"
            ),
            "strong_to_steady_response_step_ratio": (
                strong["gasf_filter_response_step_rms_mean"]
                / max(steady["gasf_filter_response_step_rms_mean"], 1e-12)
                if steady and strong else None
            ),
            "milliseconds_per_valid_node_step": report["timing"].get(
                "milliseconds_per_valid_node_step"
            ),
            "source": str(path.resolve()),
        })
    if not rows:
        raise FileNotFoundError(f"No alpha_*.json files in {cli.input_dir}")
    rows.sort(key=lambda row: row["alpha"])
    best = min(rows, key=lambda row: row["position_prediction_rmse"])
    payload = {
        "selection_metric": "position_prediction_rmse",
        "best_alpha": best["alpha"],
        "best_position_prediction_rmse": best["position_prediction_rmse"],
        "rows": rows,
    }
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
