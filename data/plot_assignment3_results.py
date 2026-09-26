import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def paired_run_plot(df, metric, ylabel, output_path):
    runs = df["run_number"]
    plt.figure(figsize=(8, 4.5))
    plt.plot(
        runs,
        df[f"baseline_{metric}"],
        marker="o",
        markersize=2,
        linewidth=0.8,
        label="Baseline",
    )
    plt.plot(
        runs,
        df[f"incremental_{metric}"],
        marker="o",
        markersize=2,
        linewidth=0.8,
        label="Incremental",
    )
    plt.xlabel("Run")
    plt.ylabel(ylabel)
    plt.legend()
    save_figure(output_path)


def improvement_plot(df, column, ylabel, output_path):
    plt.figure(figsize=(8, 4.5))
    plt.bar(df["run_number"], df[column])
    plt.axhline(0, linewidth=0.8)
    plt.xlabel("Run")
    plt.ylabel(ylabel)
    save_figure(output_path)


def run_directory(root, run_number, seed):
    return os.path.join(root, f"run_{int(run_number):04d}_seed_{int(seed)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../report/outputFiles/")
    parser.add_argument("--output", default="../report/outputFiles/out_figures/")
    args = parser.parse_args()
    root = args.input
    out = args.output
    os.makedirs(out, exist_ok=True)
    summary = pd.read_csv(os.path.join(root, "all_runs_summary.csv")).sort_values(
        "run_number"
    )
    distribution = pd.read_csv(os.path.join(root, "class_distribution.csv"))
    plt.figure(figsize=(7, 4.5))
    plt.bar(distribution["class"].astype(str), distribution["full"])
    plt.yscale("log")
    plt.xlabel("Class")
    plt.ylabel("Number of observations (log scale)")
    save_figure(os.path.join(out, "class_distribution.png"))
    paired_run_plot(
        summary,
        "macro_recall",
        "Macro recall",
        os.path.join(out, "macro_recall_runs.png"),
    )
    paired_run_plot(
        summary, "macro_f1", "Macro F1", os.path.join(out, "macro_f1_runs.png")
    )
    paired_run_plot(
        summary, "accuracy", "Accuracy", os.path.join(out, "accuracy_runs.png")
    )
    paired_run_plot(
        summary,
        "macro_precision",
        "Macro precision",
        os.path.join(out, "macro_precision_runs.png"),
    )
    paired_run_plot(
        summary, "weighted_f1", "Weighted F1", os.path.join(out, "weighted_f1_runs.png")
    )
    improvement_plot(
        summary,
        "macro_recall_improvement",
        "Incremental - baseline macro recall",
        os.path.join(out, "macro_recall_improvement_runs.png"),
    )
    improvement_plot(
        summary,
        "macro_f1_improvement",
        "Incremental - baseline macro F1",
        os.path.join(out, "macro_f1_improvement_runs.png"),
    )
    improvement_plot(
        summary,
        "accuracy_improvement",
        "Incremental - baseline accuracy",
        os.path.join(out, "accuracy_improvement_runs.png"),
    )
    plt.figure(figsize=(8, 4.5))
    plt.plot(
        summary["run_number"],
        summary["baseline_hidden_units"],
        marker="o",
        markersize=2,
        linewidth=0.8,
        label="Baseline selected",
    )
    plt.plot(
        summary["run_number"],
        summary["incremental_hidden_units"],
        marker="o",
        markersize=2,
        linewidth=0.8,
        label="Incremental final",
    )
    plt.xlabel("Run")
    plt.ylabel("Hidden units")
    plt.legend()
    save_figure(os.path.join(out, "final_hidden_units_runs.png"))
    plt.figure(figsize=(8, 4.5))
    plt.plot(
        summary["run_number"],
        summary["baseline_time_seconds"] / 60.0,
        marker="o",
        markersize=2,
        linewidth=0.8,
        label="Baseline procedure",
    )
    plt.plot(
        summary["run_number"],
        summary["incremental_time_seconds"] / 60.0,
        marker="o",
        markersize=2,
        linewidth=0.8,
        label="Incremental procedure",
    )
    plt.xlabel("Run")
    plt.ylabel("Time (minutes)")
    plt.legend()
    save_figure(os.path.join(out, "runtime_runs.png"))
    plt.figure(figsize=(7, 4.5))
    for _, row in summary.iterrows():
        path = os.path.join(
            run_directory(root, row["run_number"], row["seed"]), "baseline_search.csv"
        )
        if not os.path.exists(path):
            continue
        search = pd.read_csv(path).sort_values("hidden_units")
        plt.plot(
            search["hidden_units"],
            search["validation_macro_recall"],
            marker="o",
            markersize=1.5,
            linewidth=0.6,
            alpha=0.22,
        )
    plt.xlabel("Hidden units")
    plt.ylabel("Validation macro recall")
    save_figure(os.path.join(out, "baseline_hidden_search.png"))
    stage_rows = []
    plt.figure(figsize=(7, 4.5))
    for _, row in summary.iterrows():
        path = os.path.join(
            run_directory(root, row["run_number"], row["seed"]),
            "incremental_stages.csv",
        )
        if not os.path.exists(path):
            continue
        stages = pd.read_csv(path)
        selected = stages.loc[
            stages.groupby("num_classes")["validation_macro_recall"].idxmax()
        ].sort_values("num_classes")
        selected = selected[
            ["num_classes", "hidden_units", "validation_macro_recall"]
        ].copy()
        selected["run_number"] = row["run_number"]
        stage_rows.append(selected)
        plt.plot(
            selected["num_classes"],
            selected["hidden_units"],
            marker="o",
            markersize=1.5,
            linewidth=0.6,
            alpha=0.22,
        )
    plt.xlabel("Number of active classes")
    plt.ylabel("Best hidden units for stage")
    save_figure(os.path.join(out, "incremental_hidden_growth.png"))
    if stage_rows:
        stage_data = pd.concat(stage_rows, ignore_index=True)
        plt.figure(figsize=(7, 4.5))
        for run_number, group in stage_data.groupby("run_number"):
            group = group.sort_values("num_classes")
            plt.plot(
                group["num_classes"],
                group["validation_macro_recall"],
                marker="o",
                markersize=1.5,
                linewidth=0.6,
                alpha=0.22,
            )
        plt.xlabel("Number of active classes")
        plt.ylabel("Best validation macro recall for stage")
        save_figure(os.path.join(out, "incremental_stage_recall.png"))
        stage_summary = (
            stage_data.groupby("num_classes")
            .agg(
                mean_hidden_units=("hidden_units", "mean"),
                std_hidden_units=("hidden_units", "std"),
                mean_validation_macro_recall=("validation_macro_recall", "mean"),
                std_validation_macro_recall=("validation_macro_recall", "std"),
                min_hidden_units=("hidden_units", "min"),
                max_hidden_units=("hidden_units", "max"),
            )
            .reset_index()
        )
        stage_summary.to_csv(
            os.path.join(out, "incremental_stage_summary.csv"), index=False
        )
    class_differences = np.column_stack(
        [
            summary[f"incremental_recall_class_{label}"].to_numpy()
            - summary[f"baseline_recall_class_{label}"].to_numpy()
            for label in range(10)
        ]
    )
    plt.figure(figsize=(8, 6))
    image = plt.imshow(class_differences, aspect="auto")
    plt.colorbar(image, label="Incremental - baseline recall")
    plt.xlabel("Class")
    plt.ylabel("Run")
    plt.xticks(np.arange(10), np.arange(10))
    plt.yticks(np.arange(0, len(summary), 5), summary["run_number"].iloc[::5])
    save_figure(os.path.join(out, "class_recall_difference_heatmap.png"))
    plt.figure(figsize=(8, 4.5))
    offsets = np.linspace(-0.16, 0.16, len(summary))
    for label in range(10):
        difference = class_differences[:, label]
        plt.scatter(np.full(len(summary), label) + offsets, difference, s=8, alpha=0.55)
    plt.axhline(0, linewidth=0.8)
    plt.xlabel("Class")
    plt.ylabel("Incremental - baseline recall")
    plt.xticks(range(10))
    save_figure(os.path.join(out, "class_recall_difference_runs.png"))
    for label in range(10):
        plt.figure(figsize=(8, 4.5))
        plt.plot(
            summary["run_number"],
            summary[f"baseline_recall_class_{label}"],
            marker="o",
            markersize=2,
            linewidth=0.8,
            label="Baseline",
        )
        plt.plot(
            summary["run_number"],
            summary[f"incremental_recall_class_{label}"],
            marker="o",
            markersize=2,
            linewidth=0.8,
            label="Incremental",
        )
        plt.xlabel("Run")
        plt.ylabel(f"Recall for class {label}")
        plt.legend()
        save_figure(os.path.join(out, f"class_{label}_recall_runs.png"))
    metric_rows = []
    for metric in [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ]:
        baseline = summary[f"baseline_{metric}"].to_numpy()
        incremental = summary[f"incremental_{metric}"].to_numpy()
        difference = incremental - baseline
        metric_rows.append(
            {
                "metric": metric,
                "baseline_mean": baseline.mean(),
                "baseline_std": baseline.std(ddof=1),
                "baseline_min": baseline.min(),
                "baseline_max": baseline.max(),
                "incremental_mean": incremental.mean(),
                "incremental_std": incremental.std(ddof=1),
                "incremental_min": incremental.min(),
                "incremental_max": incremental.max(),
                "mean_difference": difference.mean(),
                "difference_std": difference.std(ddof=1),
                "best_difference": difference.max(),
                "worst_difference": difference.min(),
                "improved_runs": int((difference > 0).sum()),
                "equal_runs": int((difference == 0).sum()),
                "worse_runs": int((difference < 0).sum()),
            }
        )
    pd.DataFrame(metric_rows).to_csv(
        os.path.join(out, "summary_statistics.csv"),
        index=False,
    )

    class_rows = []
    for label in range(10):
        baseline = summary[f"baseline_recall_class_{label}"]
        incremental = summary[f"incremental_recall_class_{label}"]
        difference = incremental - baseline
        class_rows.append(
            {
                "class": label,
                "baseline_mean_recall": baseline.mean(),
                "baseline_std_recall": baseline.std(ddof=1),
                "incremental_mean_recall": incremental.mean(),
                "incremental_std_recall": incremental.std(ddof=1),
                "mean_recall_difference": difference.mean(),
                "improved_runs": int((difference > 0).sum()),
                "equal_runs": int((difference == 0).sum()),
                "worse_runs": int((difference < 0).sum()),
            }
        )
    pd.DataFrame(class_rows).to_csv(
        os.path.join(out, "per_class_recall_summary.csv"), index=False
    )


if __name__ == "__main__":
    main()
