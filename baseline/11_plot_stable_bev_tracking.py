from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = Path("outputs/logs/tracking_results_stable.csv")
OUTPUT_DIR = Path("outputs/images")
ASSETS_DIR = Path("assets")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TRACK_IDS = [31, 32]

OUTPUT_IMAGE = OUTPUT_DIR / "bev_tracking_stable_tracks_31_32.png"
ASSET_IMAGE = ASSETS_DIR / "bev_tracking_stable_tracks_31_32.png"


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

    print("Target BEV tracking data:")
    print(
        target_df[
            [
                "frame_idx",
                "track_id",
                "center_x",
                "center_y",
                "distance_m",
                "smoothed_distance_m",
                "is_confirmed",
                "track_hits",
                "risk_level",
            ]
        ].sort_values(["track_id", "frame_idx"])
    )

    plt.figure(figsize=(9, 7))

    # ego vehicle 기준선
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.axvline(0, linestyle="--", linewidth=1)

    for track_id in TARGET_TRACK_IDS:
        track_df = target_df[target_df["track_id"] == track_id].sort_values("frame_idx")

        if track_df.empty:
            continue

        plt.plot(
            track_df["center_x"],
            track_df["center_y"],
            marker="o",
            linewidth=2.5,
            label=f"Track {track_id}",
        )

        # frame index와 confirmed 여부 표시
        for _, row in track_df.iterrows():
            confirmed_mark = "C" if bool(row["is_confirmed"]) else "T"

            plt.text(
                row["center_x"],
                row["center_y"],
                f"F{int(row['frame_idx'])}\n{confirmed_mark}",
                fontsize=9,
                ha="center",
                va="bottom",
            )

        # 시작점 / 끝점 강조
        start_row = track_df.iloc[0]
        end_row = track_df.iloc[-1]

        plt.scatter(
            start_row["center_x"],
            start_row["center_y"],
            s=100,
            marker="s",
        )
        plt.scatter(
            end_row["center_x"],
            end_row["center_y"],
            s=160,
            marker="*",
        )

    plt.xlabel("Forward Distance X (m)")
    plt.ylabel("Lateral Position Y (m)")
    plt.title("BEV Trajectory of Stable Approaching Tracks")
    plt.xlim(0, 30)
    plt.ylim(-10, 10)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=200)
    plt.savefig(ASSET_IMAGE, dpi=200)

    print("\nSaved:")
    print(OUTPUT_IMAGE)
    print(ASSET_IMAGE)


if __name__ == "__main__":
    main()