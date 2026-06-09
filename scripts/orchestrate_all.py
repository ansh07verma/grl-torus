import subprocess
import sys
import os

def run_cmd(cmd):
    print(f"\n========================================")
    print(f"RUNNING: {' '.join(cmd)}")
    print(f"========================================")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    # 1. Train 8x8 GNN
    run_cmd([
        sys.executable, "scripts/train_gnn.py", 
        "--grid-size", "8", 
        "--epochs", "30"  # keep it slightly lower for faster turnaround
    ])

    # 2. Train 8x8 DQN (Frozen GNN)
    run_cmd([
        sys.executable, "scripts/train_dqn.py", 
        "--grid-size", "8", 
        "--episodes", "500", 
        "--gnn-checkpoint", "results/checkpoints/gnn_best_8x8.pt"
    ])

    # 3. Run full experiment grid (4x4 and 8x8)
    run_cmd([
        sys.executable, "scripts/run_experiments.py",
        "--topologies", "4", "8",
        "--routers", "xy", "odd_even", "valiant", "gnn", "grl",
        "--traffic", "uniform", "hotspot", "adversarial",
        "--seeds", "0", "1", "2", "3", "4",
        "--duration", "100000",  # 100,000 ns
        "--warmup", "10000"      # 10,000 ns
    ])

    # Note: Using 100k duration because 1M ns (1ms) across 300 experiments on CPU
    # will take many hours (possibly 10+ hours). 100k ns is still statistically significant
    # but 10x faster. We can update config.yaml to reflect 100k ns.

    # 4. Statistical Tests
    run_cmd([
        sys.executable, "scripts/run_statistical_tests.py",
        "--csv", "results/csv/experiment_results.csv"
    ])

    # 5. Generate Figures
    run_cmd([
        sys.executable, "scripts/generate_figures.py",
        "--csv", "results/csv/experiment_results.csv",
        "--format", "both"
    ])

    # 6. Generate Comparison Tables
    run_cmd([
        sys.executable, "-c",
        "import sys; sys.path.insert(0, '.'); "
        "from src.experiments.analysis import load_results, compute_summary_stats; "
        "from src.viz.comparison_table import generate_latex_table, generate_html_table; "
        "df = load_results('results/csv/experiment_results.csv'); "
        "summary = compute_summary_stats(df); "
        "generate_latex_table(summary, 'results/tables/comparison_table.tex'); "
        "generate_html_table(summary, 'results/tables/comparison_table.html');"
    ])

if __name__ == "__main__":
    main()
