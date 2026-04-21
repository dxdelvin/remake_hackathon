import pandas as pd
import sys
sys.path.insert(0, "c:/Users/dxdel/Downloads/zeis")

# 1. Check test_segments.csv
df = pd.read_csv("c:/Users/dxdel/Downloads/zeis/data/training/test_segments.csv")
print("test_segments.csv shape:", df.shape)
print("Scenario codes:", df["scenario_code"].unique() if "scenario_code" in df.columns else "NO scenario_code")
print()

# 2. Simulate loading S1 raw CSV through aggregate_to_segments
from app.data_processor import aggregate_to_segments, load_training_data
s1_raw = load_training_data("c:/Users/dxdel/Downloads/zeis/data/training", include_patterns=["S1_"])
print("S1 raw shape:", s1_raw.shape)
s1_segs = aggregate_to_segments(s1_raw, has_label=False)
print("S1 segments:", s1_segs.shape)
print()

# 3. Show first segment details
seg1 = s1_segs.iloc[0]
print("Segment 1 phase:", seg1["phase_name"])
print("Segment 1 duration:", seg1["duration_sec"])
print("Segment 1 avg_power:", seg1["estimated_system_power_w_mean"])
print("Segment 1 energy:", seg1["estimated_energy_wh_interval_sum"])
print("Segment 1 power_vs_baseline:", seg1["power_vs_baseline"])
print("Segment 1 live_view:", seg1["live_view_enabled_share"])
print("Segment 1 user_interacting:", seg1["user_interacting_share"])
print("Segment 1 monitoring:", seg1["monitoring_required_share"])
print()

# 4. Find any segment with power_vs_baseline > 1000 W
high_power = s1_segs[s1_segs["power_vs_baseline"] > 1000]
print("Segments with power_vs_baseline > 1000:", len(high_power))
if len(high_power) > 0:
    print(high_power[["phase_name","estimated_system_power_w_mean","power_vs_baseline","estimated_energy_wh_interval_sum"]].head())
