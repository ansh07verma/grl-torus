"""
Generate LaTeX and HTML comparison tables for the manuscript and dashboard.
"""

import os
import pandas as pd
from src.utils.logging import get_logger

logger = get_logger("viz.comparison_table")

ROUTER_LABELS = {
    "xy": "XY",
    "odd_even": "Odd-Even",
    "valiant": "Valiant",
    "gnn": "Supervised GNN",
    "grl": "GRL (Ours)",
}


def generate_latex_table(summary_df: pd.DataFrame, output_path: str) -> str:
    """Generate a LaTeX table grouped by Traffic Pattern and Router."""
    # We want rows: Traffic Pattern -> Router
    # Columns: Latency (Healthy/Failed), Drop Rate (Healthy/Failed)

    # Make sure we have the expected routers
    routers = [r for r in ["xy", "odd_even", "valiant", "gnn", "grl"] if r in summary_df["router"].unique()]
    traffics = sorted(summary_df["traffic_pattern"].unique())

    latex = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Performance Comparison across Routing Algorithms on 4$\\times$4 Torus}",
        "\\label{tab:performance_comparison}",
        "\\begin{tabular}{ll|cc|cc}",
        "\\toprule",
        " & & \\multicolumn{2}{c|}{\\textbf{Healthy (0\\% Failures)}} & \\multicolumn{2}{c}{\\textbf{Degraded (10\\% Failures)}} \\\\",
        "\\textbf{Traffic} & \\textbf{Algorithm} & \\textbf{Latency (ns)} & \\textbf{Drop Rate (\\%)} & \\textbf{Latency (ns)} & \\textbf{Drop Rate (\\%)} \\\\",
        "\\midrule"
    ]

    for i, traffic in enumerate(traffics):
        t_df = summary_df[summary_df["traffic_pattern"] == traffic]
        
        for j, router in enumerate(routers):
            r_df = t_df[t_df["router"] == router]
            
            # Healthy
            h_df = r_df[r_df["failure_rate"] == 0.0]
            if len(h_df) > 0:
                h_lat = f"{h_df['avg_latency_ns_mean'].iloc[0]:.1f} \\pm {h_df['avg_latency_ns_std'].iloc[0]:.1f}" if "avg_latency_ns_std" in h_df.columns else f"{h_df['avg_latency_ns_mean'].iloc[0]:.1f}"
                h_drop = f"{h_df['drop_rate_mean'].iloc[0]*100:.1f}"
            else:
                h_lat, h_drop = "-", "-"
                
            # Degraded
            d_df = r_df[r_df["failure_rate"] > 0.0]
            if len(d_df) > 0:
                d_lat = f"{d_df['avg_latency_ns_mean'].iloc[0]:.1f} \\pm {d_df['avg_latency_ns_std'].iloc[0]:.1f}" if "avg_latency_ns_std" in d_df.columns else f"{d_df['avg_latency_ns_mean'].iloc[0]:.1f}"
                d_drop = f"{d_df['drop_rate_mean'].iloc[0]*100:.1f}"
            else:
                d_lat, d_drop = "-", "-"
            
            traffic_label = traffic.capitalize() if j == 0 else ""
            latex.append(f"{traffic_label} & {ROUTER_LABELS.get(router, router)} & {h_lat} & {h_drop} & {d_lat} & {d_drop} \\\\")
            
        if i < len(traffics) - 1:
            latex.append("\\midrule")

    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}"
    ])

    result = "\n".join(latex)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(result)
        
    logger.info(f"Saved LaTeX table: {output_path}")
    return result


def generate_html_table(summary_df: pd.DataFrame, output_path: str) -> str:
    """Generate an HTML table for the demo dashboard."""
    routers = [r for r in ["xy", "odd_even", "valiant", "gnn", "grl"] if r in summary_df["router"].unique()]
    traffics = sorted(summary_df["traffic_pattern"].unique())

    html = [
        "<table class='comparison-table'>",
        "  <thead>",
        "    <tr>",
        "      <th rowspan='2'>Traffic</th>",
        "      <th rowspan='2'>Algorithm</th>",
        "      <th colspan='2'>Healthy (0% Failures)</th>",
        "      <th colspan='2'>Degraded (10% Failures)</th>",
        "    </tr>",
        "    <tr>",
        "      <th>Latency (ns)</th>",
        "      <th>Drop Rate (%)</th>",
        "      <th>Latency (ns)</th>",
        "      <th>Drop Rate (%)</th>",
        "    </tr>",
        "  </thead>",
        "  <tbody>"
    ]

    for traffic in traffics:
        t_df = summary_df[summary_df["traffic_pattern"] == traffic]
        for j, router in enumerate(routers):
            r_df = t_df[t_df["router"] == router]
            
            h_df = r_df[r_df["failure_rate"] == 0.0]
            if len(h_df) > 0:
                h_lat = f"{h_df['avg_latency_ns_mean'].iloc[0]:.1f}"
                h_drop = f"{h_df['drop_rate_mean'].iloc[0]*100:.1f}"
            else:
                h_lat, h_drop = "-", "-"
                
            d_df = r_df[r_df["failure_rate"] > 0.0]
            if len(d_df) > 0:
                d_lat = f"{d_df['avg_latency_ns_mean'].iloc[0]:.1f}"
                d_drop = f"{d_df['drop_rate_mean'].iloc[0]*100:.1f}"
            else:
                d_lat, d_drop = "-", "-"
            
            html.append("    <tr>")
            if j == 0:
                html.append(f"      <td rowspan='{len(routers)}'>{traffic.capitalize()}</td>")
            html.append(f"      <td>{ROUTER_LABELS.get(router, router)}</td>")
            html.append(f"      <td>{h_lat}</td>")
            html.append(f"      <td>{h_drop}</td>")
            html.append(f"      <td>{d_lat}</td>")
            html.append(f"      <td>{d_drop}</td>")
            html.append("    </tr>")

    html.extend([
        "  </tbody>",
        "</table>"
    ])

    result = "\n".join(html)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(result)
        
    logger.info(f"Saved HTML table: {output_path}")
    return result
