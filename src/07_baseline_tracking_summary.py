import json
from pathlib import Path

import pandas as pd


CSV_PATH = Path("outputs/logs/tracking_results.csv")

OUTPUT_DIR = Path("outputs/logs")
OUTPUT_SUMMARY_JSON = OUTPUT_DIR / "tracking_summary.json"
OUTPUT_TRACK_SUMMARY_CSV = OUTPUT_DIR / "track_summary.csv"
OUTPUT_FRAME_SUMMARY_CSV = OUTPUT_DIR / "frame_summary.csv"

STABLE_TRACK_MIN_FRAMES = 3
APPROACH_DISTANCE_DELTA_THRESHOLD = 0.5


def summarize_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """
    track_id별 유지 길이, 시작/끝 거리, 거리 감소량, 위험도 등을 요약한다.
    """

    track_rows = []

    for track_id, group in df.groupby("track_id"):
        group = group.sort_values("frame_idx")

        start_frame = int(group["frame_idx"].iloc[0])
        end_frame = int(group["frame_idx"].iloc[-1])
        num_frames = int(group["frame_idx"].nunique())

        start_distance = float(group["distance_m"].iloc[0])
        end_distance = float(group["distance_m"].iloc[-1])
        min_distance = float(group["distance_m"].min())
        max_distance = float(group["distance_m"].max())

        # 양수면 가까워진 것
        distance_delta = start_distance - end_distance

        avg_speed = float(group["speed_per_frame"].mean())
        max_speed = float(group["speed_per_frame"].max())

        approaching_frame_count = int(group["is_approaching"].sum())
        approaching_ratio = approaching_frame_count / max(num_frames, 1)

        risk_levels = group["risk_level"].tolist()
        final_risk = str(group["risk_level"].iloc[-1])
        max_risk = get_max_risk_level(risk_levels)

        is_stable = num_frames >= STABLE_TRACK_MIN_FRAMES
        is_approaching_track = distance_delta >= APPROACH_DISTANCE_DELTA_THRESHOLD

        track_rows.append({
            "track_id": int(track_id),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "num_frames": num_frames,
            "start_distance_m": round(start_distance, 3),
            "end_distance_m": round(end_distance, 3),
            "distance_delta_m": round(distance_delta, 3),
            "min_distance_m": round(min_distance, 3),
            "max_distance_m": round(max_distance, 3),
            "avg_speed_per_frame": round(avg_speed, 3),
            "max_speed_per_frame": round(max_speed, 3),
            "approaching_frame_count": approaching_frame_count,
            "approaching_ratio": round(approaching_ratio, 3),
            "is_stable": bool(is_stable),
            "is_approaching_track": bool(is_approaching_track),
            "final_risk": final_risk,
            "max_risk": max_risk,
        })

    track_summary = pd.DataFrame(track_rows)

    if not track_summary.empty:
        track_summary = track_summary.sort_values(
            by=["is_stable", "is_approaching_track", "distance_delta_m", "num_frames"],
            ascending=[False, False, False, False]
        )

    return track_summary


def summarize_frames(df: pd.DataFrame) -> pd.DataFrame:
    """
    frame별 객체 수, 최소 거리, 위험도 분포를 요약한다.
    """

    frame_rows = []

    for frame_idx, group in df.groupby("frame_idx"):
        risk_levels = group["risk_level"].tolist()

        frame_rows.append({
            "frame_idx": int(frame_idx),
            "object_count": int(len(group)),
            "min_distance_m": round(float(group["distance_m"].min()), 3),
            "mean_distance_m": round(float(group["distance_m"].mean()), 3),
            "approaching_object_count": int(group["is_approaching"].sum()),
            "frame_max_risk": get_max_risk_level(risk_levels),
            "safe_count": int((group["risk_level"] == "SAFE").sum()),
            "caution_count": int((group["risk_level"] == "CAUTION").sum()),
            "warning_count": int((group["risk_level"] == "WARNING").sum()),
            "danger_count": int((group["risk_level"] == "DANGER").sum()),
        })

    return pd.DataFrame(frame_rows).sort_values("frame_idx")


def get_max_risk_level(risk_levels):
    """
    여러 risk 중 가장 높은 위험도를 반환한다.
    """
    priority = {
        "SAFE": 0,
        "CAUTION": 1,
        "WARNING": 2,
        "DANGER": 3,
    }

    if not risk_levels:
        return "SAFE"

    return max(risk_levels, key=lambda x: priority.get(x, 0))


def build_summary_dict(df: pd.DataFrame, track_summary: pd.DataFrame, frame_summary: pd.DataFrame):
    total_frames = int(df["frame_idx"].nunique())
    total_tracks = int(df["track_id"].nunique())

    stable_tracks = track_summary[track_summary["is_stable"] == True]
    approaching_tracks = track_summary[track_summary["is_approaching_track"] == True]
    stable_approaching_tracks = track_summary[
        (track_summary["is_stable"] == True) &
        (track_summary["is_approaching_track"] == True)
    ]

    risk_count_by_object = df["risk_level"].value_counts().to_dict()

    frame_risk_counts = frame_summary["frame_max_risk"].value_counts().to_dict()

    top_approaching_tracks = stable_approaching_tracks.head(10).to_dict(orient="records")

    summary = {
        "input_csv": str(CSV_PATH),
        "total_frames": total_frames,
        "total_detections": int(len(df)),
        "total_tracks": total_tracks,
        "stable_track_min_frames": STABLE_TRACK_MIN_FRAMES,
        "stable_track_count": int(len(stable_tracks)),
        "approach_distance_delta_threshold_m": APPROACH_DISTANCE_DELTA_THRESHOLD,
        "approaching_track_count": int(len(approaching_tracks)),
        "stable_approaching_track_count": int(len(stable_approaching_tracks)),
        "risk_count_by_object": {
            "SAFE": int(risk_count_by_object.get("SAFE", 0)),
            "CAUTION": int(risk_count_by_object.get("CAUTION", 0)),
            "WARNING": int(risk_count_by_object.get("WARNING", 0)),
            "DANGER": int(risk_count_by_object.get("DANGER", 0)),
        },
        "frame_max_risk_counts": {
            "SAFE": int(frame_risk_counts.get("SAFE", 0)),
            "CAUTION": int(frame_risk_counts.get("CAUTION", 0)),
            "WARNING": int(frame_risk_counts.get("WARNING", 0)),
            "DANGER": int(frame_risk_counts.get("DANGER", 0)),
        },
        "minimum_distance_m": round(float(df["distance_m"].min()), 3),
        "top_stable_approaching_tracks": top_approaching_tracks,
    }

    return summary


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"{CSV_PATH} 파일이 없습니다. 먼저 src/06_tracking.py를 실행해 주세요."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    required_columns = {
        "frame_idx",
        "track_id",
        "distance_m",
        "speed_per_frame",
        "is_approaching",
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

    print("\nTracking Summary")
    print("=" * 80)
    print(f"Total frames: {summary['total_frames']}")
    print(f"Total detections: {summary['total_detections']}")
    print(f"Total tracks: {summary['total_tracks']}")
    print(f"Stable tracks (>= {STABLE_TRACK_MIN_FRAMES} frames): {summary['stable_track_count']}")
    print(f"Approaching tracks: {summary['approaching_track_count']}")
    print(f"Stable approaching tracks: {summary['stable_approaching_track_count']}")
    print(f"Minimum distance: {summary['minimum_distance_m']} m")

    print("\nRisk count by object")
    print(summary["risk_count_by_object"])

    print("\nFrame max risk counts")
    print(summary["frame_max_risk_counts"])

    print("\nTop stable approaching tracks")
    print("-" * 80)

    if len(summary["top_stable_approaching_tracks"]) == 0:
        print("No stable approaching tracks found.")
    else:
        for track in summary["top_stable_approaching_tracks"]:
            print(
                f"track={track['track_id']:>2} | "
                f"frames={track['num_frames']} | "
                f"distance={track['start_distance_m']}m -> {track['end_distance_m']}m | "
                f"delta={track['distance_delta_m']}m | "
                f"max_risk={track['max_risk']}"
            )

    print("\nSaved:")
    print(OUTPUT_TRACK_SUMMARY_CSV)
    print(OUTPUT_FRAME_SUMMARY_CSV)
    print(OUTPUT_SUMMARY_JSON)


if __name__ == "__main__":
    main()