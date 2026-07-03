from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = Path("outputs/logs/tracking_results_stable.csv")
OUTPUT_DIR = Path("outputs/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TRACK_IDS = [31, 32]


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"{CSV_PATH} 파일이 없습니다. 먼저 src/10_tracking_stable.py를 실행해 주세요."
        )

    df = pd.read_csv(CSV_PATH)
    target_df = df[df["track_id"].isin(TARGET_TRACK_IDS)].copy()

    if target_df.empty:
        print("No target track IDs found.")
        print("Available track IDs:", sorted(df["track_id"].unique()))
        return

    plt.figure(figsize=(8, 5))

    for track_id in TARGET_TRACK_IDS:
        track_df = target_df[target_df["track_id"] == track_id].sort_values("frame_idx")

        if track_df.empty:
            continue

        plt.plot(
            track_df["frame_idx"],
            track_df["smoothed_distance_m"],
            marker="o",
            linewidth=2.5,
            label=f"Track {track_id}"
        )

        confirmed_df = track_df[track_df["is_confirmed"] == True]

        for _, row in confirmed_df.iterrows():
            plt.scatter(
                row["frame_idx"],
                row["smoothed_distance_m"],
                s=120,
                marker="o"
            )

    plt.xlabel("Frame Index")
    plt.ylabel("Smoothed Distance from LiDAR (m)")
    plt.title("Stable Tracking Distance Change")
    plt.legend()
    plt.grid(True)

    output_path = OUTPUT_DIR / "stable_tracking_distance_change_clean.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)

    print("Saved:", output_path)


if __name__ == "__main__":
    main()