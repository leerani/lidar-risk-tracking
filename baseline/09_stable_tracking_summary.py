import json
from pathlib import Path

import pandas as pd


CSV_PATH = Path("outputs/logs/tracking_results_stable.csv")

OUTPUT_DIR = Path("outputs/logs")
OUTPUT_TRACK_SUMMARY_CSV = OUTPUT_DIR / "stable_track_summary.csv"
OUTPUT_FRAME_SUMMARY_CSV = OUTPUT_DIR / "stable_frame_summary.csv"
OUTPUT_SUMMARY_JSON = OUTPUT_DIR / "stable_tracking_summary.json"

CONFIRM_MIN_HITS = 3
APPROACH_DISTANCE_DELTA_THRESHOLD = 0.5


RISK_PRIORITY = {
    "SAFE": 0,
    "CAUTION": 1,
    "WARNING": 2,
    "DANGER": 3,
}


def get_max_risk_level(risk_levels):
    if not risk_levels:
        return "SAFE"
    return max(risk_levels, key=lambda x: RISK_PRIORITY.get(x, 0))


def summarize_tracks(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for track_id, group in df.groupby("track_id"):
        group = group.sort_values("frame_idx")

        confirmed_group = group[group["is_confirmed"] == True]

        start_frame = int(group["frame_idx"].iloc[0])
        end_frame = int(group["frame_idx"].iloc[-1])
        num_frames = int(group["frame_idx"].nunique())

        confirmed_frames = int(confirmed_group["frame_idx"].nunique())

        start_distance = float(group["distance_m"].iloc[0])
        end_distance = float(group["distance_m"].iloc[-1])
        distance_delta = start_distance - end_distance

        start_smooth_distance = float(group["smoothed_distance_m"].iloc[0])
        end_smooth_distance = float(group["smoothed_distance_m"].iloc[-1])
        smooth_distance_delta = start_smooth_distance - end_smooth_distance

        min_distance = float(group["distance_m"].min())
        min_smooth_distance = float(group["smoothed_distance_m"].min())

        max_hits = int(group["track_hits"].max())
        is_confirmed_ever = bool(group["is_confirmed"].any())

        approaching_frame_count = int(group["is_approaching"].sum())
        approaching_ratio = approaching_frame_count / max(num_frames, 1)

        is_stable_approaching = (
            is_confirmed_ever and
            smooth_distance_delta >= APPROACH_DISTANCE_DELTA_THRESHOLD
        )

        rows.append({
            "track_id": int(track_id),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "num_frames": num_frames,
            "confirmed_frames": confirmed_frames,
            "max_hits": max_hits,
            "is_confirmed_ever": is_confirmed_ever,
            "start_distance_m": round(start_distance, 3),
            "end_distance_m": round(end_distance, 3),
            "distance_delta_m": round(distance_delta, 3),
            "start_smoothed_distance_m": round(start_smooth_distance, 3),
            "end_smoothed_distance_m": round(end_smooth_distance, 3),
            "smoothed_distance_delta_m": round(smooth_distance_delta, 3),
            "min_distance_m": round(min_distance, 3),
            "min_smoothed_distance_m": round(min_smooth_distance, 3),
            "approaching_frame_count": approaching_frame_count,
            "approaching_ratio": round(approaching_ratio, 3),
            "is_stable_approaching": is_stable_approaching,
            "max_risk": get_max_risk_level(group["risk_level"].tolist()),
            "final_risk": str(group["risk_level"].iloc[-1]),
        })

    summary = pd.DataFrame(rows)

    if not summary.empty:
        summary = summary.sort_values(
            by=[
                "is_confirmed_ever",
                "is_stable_approaching",
                "smoothed_distance_delta_m",
                "confirmed_frames",
                "num_frames",
            ],
            ascending=[False, False, False, False, False],
        )

    return summary


def summarize_frames(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for frame_idx, group in df.groupby("frame_idx"):
        confirmed_group = group[group["is_confirmed"] == True]

        rows.append({
            "frame_idx": int(frame_idx),
            "object_count": int(len(group)),
            "confirmed_object_count": int(len(confirmed_group)),
            "tentative_object_count": int((group["is_confirmed"] == False).sum()),
            "min_distance_m": round(float(group["distance_m"].min()), 3) if len(group) else None,
            "min_smoothed_distance_m": round(float(group["smoothed_distance_m"].min()), 3) if len(group) else None,
            "approaching_object_count": int(group["is_approaching"].sum()),
            "confirmed_approaching_count": int(confirmed_group["is_approaching"].sum()) if len(confirmed_group) else 0,
            "frame_max_risk": get_max_risk_level(group["risk_level"].tolist()),
            "confirmed_frame_max_risk": get_max_risk_level(confirmed_group["risk_level"].tolist()) if len(confirmed_group) else "SAFE",
            "safe_count": int((group["risk_level"] == "SAFE").sum()),
            "caution_count": int((group["risk_level"] == "CAUTION").sum()),
            "warning_count": int((group["risk_level"] == "WARNING").sum()),
            "danger_count": int((group["risk_level"] == "DANGER").sum()),
        })

    return pd.DataFrame(rows).sort_values("frame_idx")


def build_summary_dict(df, track_summary, frame_summary):
    confirmed_df = df[df["is_confirmed"] == True]

    risk_count_all = df["risk_level"].value_counts().to_dict()
    risk_count_confirmed = confirmed_df["risk_level"].value_counts().to_dict()

    frame_risk_counts = frame_summary["frame_max_risk"].value_counts().to_dict()
    confirmed_frame_risk_counts = frame_summary["confirmed_frame_max_risk"].value_counts().to_dict()

    confirmed_tracks = track_summary[track_summary["is_confirmed_ever"] == True]
    stable_approaching_tracks = track_summary[
        track_summary["is_stable_approaching"] == True
    ]

    top_stable_approaching = stable_approaching_tracks.head(10).to_dict(orient="records")

    return {
        "input_csv": str(CSV_PATH),
        "total_frames": int(df["frame_idx"].nunique()),
        "total_detections": int(len(df)),
        "total_tracks": int(df["track_id"].nunique()),
        "confirmed_track_count": int(len(confirmed_tracks)),
        "stable_approaching_track_count": int(len(stable_approaching_tracks)),
        "confirm_min_hits": CONFIRM_MIN_HITS,
        "approach_distance_delta_threshold_m": APPROACH_DISTANCE_DELTA_THRESHOLD,
        "confirmed_detection_count": int(len(confirmed_df)),
        "tentative_detection_count": int((df["is_confirmed"] == False).sum()),
        "minimum_distance_m": round(float(df["distance_m"].min()), 3),
        "minimum_smoothed_distance_m": round(float(df["smoothed_distance_m"].min()), 3),
        "risk_count_all_objects": {
            "SAFE": int(risk_count_all.get("SAFE", 0)),
            "CAUTION": int(risk_count_all.get("CAUTION", 0)),
            "WARNING": int(risk_count_all.get("WARNING", 0)),
            "DANGER": int(risk_count_all.get("DANGER", 0)),
        },
        "risk_count_confirmed_objects": {
            "SAFE": int(risk_count_confirmed.get("SAFE", 0)),
            "CAUTION": int(risk_count_confirmed.get("CAUTION", 0)),
            "WARNING": int(risk_count_confirmed.get("WARNING", 0)),
            "DANGER": int(risk_count_confirmed.get("DANGER", 0)),
        },
        "frame_max_risk_counts": {
            "SAFE": int(frame_risk_counts.get("SAFE", 0)),
            "CAUTION": int(frame_risk_counts.get("CAUTION", 0)),
            "WARNING": int(frame_risk_counts.get("WARNING", 0)),
            "DANGER": int(frame_risk_counts.get("DANGER", 0)),
        },
        "confirmed_frame_max_risk_counts": {
            "SAFE": int(confirmed_frame_risk_counts.get("SAFE", 0)),
            "CAUTION": int(confirmed_frame_risk_counts.get("CAUTION", 0)),
            "WARNING": int(confirmed_frame_risk_counts.get("WARNING", 0)),
            "DANGER": int(confirmed_frame_risk_counts.get("DANGER", 0)),
        },
        "top_stable_approaching_tracks": top_stable_approaching,
    }


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"{CSV_PATH} 파일이 없습니다. 먼저 src/10_tracking_stable.py를 실행해 주세요."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    required_columns = {
        "frame_idx",
        "track_id",
        "distance_m",
        "smoothed_distance_m",
        "is_approaching",
        "is_confirmed",
        "track_hits",
        "risk_level",
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {missing_columns}")

    track_summary = summarize_tracks(df)
    frame_summary = summarize_frames(df)
    summary = build_summary_dict(df, track_summary, frame_summary)

    track_summary.to_csv(OUTPUT_TRACK_SUMMARY_CSV, index=False)
    frame_summary.to_csv(OUTPUT_FRAME_SUMMARY_CSV, index=False)

    with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nStable Tracking Summary")
    print("=" * 80)
    print(f"Total frames: {summary['total_frames']}")
    print(f"Total detections: {summary['total_detections']}")
    print(f"Total tracks: {summary['total_tracks']}")
    print(f"Confirmed tracks: {summary['confirmed_track_count']}")
    print(f"Stable approaching tracks: {summary['stable_approaching_track_count']}")
    print(f"Confirmed detections: {summary['confirmed_detection_count']}")
    print(f"Tentative detections: {summary['tentative_detection_count']}")
    print(f"Minimum raw distance: {summary['minimum_distance_m']} m")
    print(f"Minimum smoothed distance: {summary['minimum_smoothed_distance_m']} m")

    print("\nRisk count - all objects")
    print(summary["risk_count_all_objects"])

    print("\nRisk count - confirmed objects")
    print(summary["risk_count_confirmed_objects"])

    print("\nFrame max risk counts")
    print(summary["frame_max_risk_counts"])

    print("\nConfirmed frame max risk counts")
    print(summary["confirmed_frame_max_risk_counts"])

    print("\nTop stable approaching tracks")
    print("-" * 80)

    if not summary["top_stable_approaching_tracks"]:
        print("No stable approaching tracks found.")
    else:
        for track in summary["top_stable_approaching_tracks"]:
            print(
                f"track={track['track_id']:>2} | "
                f"frames={track['num_frames']} | "
                f"confirmed_frames={track['confirmed_frames']} | "
                f"smooth_distance={track['start_smoothed_distance_m']}m -> "
                f"{track['end_smoothed_distance_m']}m | "
                f"delta={track['smoothed_distance_delta_m']}m | "
                f"max_risk={track['max_risk']}"
            )

    print("\nSaved:")
    print(OUTPUT_TRACK_SUMMARY_CSV)
    print(OUTPUT_FRAME_SUMMARY_CSV)
    print(OUTPUT_SUMMARY_JSON)


if __name__ == "__main__":
    main()