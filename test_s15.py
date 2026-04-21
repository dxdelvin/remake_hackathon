"""
test_s15.py
End-to-end test of S15_percentage_case_without_phase_assigned.csv through
the full hybrid engine (train MLP on S1-S13, infer phases, run analysis).
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
from app.data_processor import aggregate_to_segments, load_training_data
from app.hybrid_engine import analyze_segment
from app.ml_model import model as mlp_model

S15_PATH = r"C:\Users\dxdel\Downloads\S15_percentage_case_without_phase_assigned.csv"
TRAINING_DIR = "data/training"

# ── 1. Train MLP on S1-S13 ────────────────────────────────────────────────────
print("=" * 70)
print("Training MLP on S1-S13 data (S15 is inference-only, no labels)...")
raw_df = load_training_data(TRAINING_DIR)
train_segs = aggregate_to_segments(raw_df, has_label=True)
stats = mlp_model.train(train_segs)
print(f"  Accuracy : {stats['accuracy']:.2%}")
print(f"  F1-macro : {stats['f1_macro']:.2%}")
print(f"  Train/val: {stats['n_train']} / {stats['n_val']} segments")
print(f"  Split    : {stats['split_method']}")

# ── 2. Load + aggregate S15 (no labels — phase inferred from experiment_type) ──
print()
print("=" * 70)
print("Loading S15 CSV and inferring phases from experiment_type...")
df15 = pd.read_csv(S15_PATH)
segs15 = aggregate_to_segments(df15, has_label=False)

print(f"  Total segments : {len(segs15)}")
print(f"  Phase breakdown:")
for phase, cnt in segs15["phase_name"].value_counts().items():
    print(f"    {phase:<28} {cnt}")

# ── 3. Run hybrid engine on each segment ─────────────────────────────────────
print()
print("=" * 70)
print("Running hybrid engine (rules + MLP) on all 18 segments...")
print()

results = [analyze_segment(row.to_dict()) for _, row in segs15.iterrows()]

# ── 4. Print results table ───────────────────────────────────────────────────
COL = {
    "seg":    14,
    "phase":  26,
    "action": 36,
    "conf":    6,
    "save":    9,
    "rule":    5,
}
header = (
    f"{'Block':<{COL['seg']}} {'Phase':<{COL['phase']}} "
    f"{'Recommended Action':<{COL['action']}} {'Conf':>{COL['conf']}} "
    f"{'Save Wh':>{COL['save']}} {'Rule':<{COL['rule']}}"
)
print(header)
print("-" * len(header))

total_savings = 0.0
action_counts = {}
for r in results:
    total_savings += r["estimated_savings_wh"]
    action_counts[r["recommended_action"]] = action_counts.get(r["recommended_action"], 0) + 1
    rule_tag = r["rule_id"] or "ML"
    spike_tag = " ⚡" if r["is_energy_spike"] else ""
    print(
        f"{r['segment_id']:<{COL['seg']}} {r['phase']:<{COL['phase']}} "
        f"{r['recommended_action']:<{COL['action']}} {r['confidence']:>{COL['conf']}.2f} "
        f"{r['estimated_savings_wh']:>{COL['save']}.4f} {rule_tag:<{COL['rule']}}{spike_tag}"
    )
    if r["action_reason"]:
        print(f"  > {r['action_reason']}")

# ── 5. Summary ────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("SUMMARY")
print(f"  Total estimated savings : {total_savings:.4f} Wh  ({total_savings*1000:.1f} mWh)")
print()
print("  Action distribution:")
for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
    print(f"    {action:<36} {count} segment(s)")

# ── 6. MLP probability deep-dive for non-no_action blocks ────────────────────
print()
print("=" * 70)
print("MLP probability breakdown (blocks with action ≠ no_action):")
for r in results:
    if r["recommended_action"] != "no_action":
        probs = r["mlp_probabilities"]
        print(
            f"  Block {r['segment_id']:>3} | {r['phase']:<26} | "
            f"no_action={probs.get('no_action', 0):.3f}  "
            f"pause_lv={probs.get('pause_live_view', 0):.3f}  "
            f"opt_tile={probs.get('optimize_tile_scan_settings', 0):.3f}"
        )
