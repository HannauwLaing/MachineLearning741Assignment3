import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = "../data/networkTraffic.csv"
OUTPUT_ROOT = "../report/outputFiles/"
OUTPUT_DIR = "../report/outputFiles/out_figures/"
CATEGORY_MAP_FILE = "../data/attack_category_map.csv"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    }
)


def save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def run_directory(root, run_number, seed):
    return os.path.join(root, f"run_{int(run_number):04d}_seed_{int(seed)}")


def load_class_names():
    if os.path.exists(CATEGORY_MAP_FILE):
        mapping = pd.read_csv(CATEGORY_MAP_FILE)
        return {
            int(row["Mapping"]): str(row["Attack Category Name"])
            for _, row in mapping.iterrows()
        }
    return {label: f"Class {label}" for label in range(10)}


def class_label(label, class_names):
    return f"{label} {class_names.get(label, '')}".strip()


def class_recall_plot(summary, prefix, output_path, class_names):
    plt.figure(figsize=(7.2, 4.5))
    for label in range(10):
        plt.plot(
            summary["run_number"],
            summary[f"{prefix}_recall_class_{label}"],
            linewidth=0.9,
            label=class_label(label, class_names),
        )
    plt.xlabel("Run")
    plt.ylabel("Recall")
    plt.ylim(0, 1)
    plt.legend(ncol=2, frameon=False)
    save_figure(output_path)


def average_normalised_confusion(root, summary, prefix):
    matrices = []
    for _, row in summary.iterrows():
        path = os.path.join(
            run_directory(root, row["run_number"], row["seed"]),
            f"{prefix}_confusion_matrix.csv",
        )
        if not os.path.exists(path):
            continue
        matrix = pd.read_csv(path, index_col=0)
        matrix.index = matrix.index.astype(int)
        matrix.columns = matrix.columns.astype(int)
        matrix = matrix.reindex(index=range(10), columns=range(10))
        values = matrix.to_numpy(dtype=float)
        row_totals = values.sum(axis=1, keepdims=True)
        normalised = np.divide(
            values,
            row_totals,
            out=np.zeros_like(values),
            where=row_totals != 0,
        )
        matrices.append(normalised)
    return np.mean(matrices, axis=0)


def confusion_plot(matrix, title, output_path):
    plt.figure(figsize=(5.2, 4.5))
    image = plt.imshow(matrix, vmin=0, vmax=1)
    plt.colorbar(image, label="Mean proportion")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.title(title)
    for row in range(10):
        for col in range(10):
            if matrix[row, col] >= 0.10:
                plt.text(
                    col,
                    row,
                    f"{matrix[row, col]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                )
    save_figure(output_path)


def classification_report_summary(root, summary):
    rows = []
    for prefix in ["baseline", "incremental"]:
        for _, run in summary.iterrows():
            path = os.path.join(
                run_directory(root, run["run_number"], run["seed"]),
                f"{prefix}_classification_report.csv",
            )
            if not os.path.exists(path):
                continue
            report = pd.read_csv(path, index_col=0)
            for label in range(10):
                label_str = str(label)
                rows.append(
                    {
                        "method": prefix,
                        "run_number": int(run["run_number"]),
                        "class": label,
                        "precision": report.loc[label_str, "precision"],
                        "recall": report.loc[label_str, "recall"],
                        "f1": report.loc[label_str, "f1-score"],
                        "support": report.loc[label_str, "support"],
                    }
                )
    values = pd.DataFrame(rows)
    return (
        values.groupby(["class", "method"])
        .agg(
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            support=("support", "mean"),
        )
        .reset_index()
    )


def plot_incremental_validation_recall_by_hidden(
    stages,
    output_path,
    class_names,
    title=None,
):
    plt.figure(figsize=(7.4, 4.8))
    recall_columns = []
    for column in stages.columns:
        if column.startswith("validation_recall_class_"):
            label = int(column.rsplit("_", 1)[1])
            recall_columns.append((label, column))
    recall_columns.sort()

    for label, column in recall_columns:
        x_values = []
        y_values = []
        for _, stage in stages.groupby("num_classes", sort=False):
            values = stage[["hidden_units", column]].dropna()
            if values.empty:
                continue
            x_values.extend(values["hidden_units"].tolist())
            y_values.extend(values[column].tolist())
            x_values.append(np.nan)
            y_values.append(np.nan)

        if not x_values:
            continue

        plt.plot(
            x_values,
            y_values,
            marker="o",
            markersize=2.2,
            linewidth=0.9,
            label=class_label(label, class_names),
        )

    plt.xlabel("Hidden units")
    plt.ylabel("Validation recall")
    plt.ylim(0, 1.02)
    if title:
        plt.title(title)
    plt.legend(ncol=2, frameon=False)
    save_figure(output_path)


def plot_incremental_class_progression(
    stages,
    output_path,
    class_names,
):
    stages = stages.reset_index(drop=True).copy()
    x = np.arange(len(stages))

    plt.figure(figsize=(7.6, 4.8))

    recall_columns = []
    for column in stages.columns:
        if column.startswith("validation_recall_class_"):
            label = int(column.rsplit("_", 1)[1])
            recall_columns.append((label, column))
    recall_columns.sort()

    for label, column in recall_columns:
        values = stages[column].to_numpy(dtype=float)
        plt.plot(
            x,
            values,
            marker="o",
            markersize=2.2,
            linewidth=0.9,
            label=class_label(label, class_names),
        )

    stage_starts = stages.drop_duplicates("num_classes", keep="first")
    for _, row in stage_starts.iterrows():
        position = int(row.name)
        if position > 0:
            plt.axvline(position, linewidth=0.55, linestyle="--", alpha=0.35)
            newest_class = int(row["newest_class"])
            plt.text(
                position + 0.15,
                1.015,
                f"+{newest_class}",
                rotation=90,
                va="bottom",
                ha="left",
                fontsize=6,
            )

    tick_count = min(14, len(stages))
    tick_positions = np.unique(np.linspace(0, len(stages) - 1, tick_count, dtype=int))
    tick_labels = [
        str(int(stages.loc[position, "hidden_units"])) for position in tick_positions
    ]

    plt.xticks(tick_positions, tick_labels)
    plt.xlabel("Training progression (hidden units shown on ticks)")
    plt.ylabel("Validation recall")
    plt.ylim(0, 1.08)
    plt.legend(ncol=2, frameon=False)
    save_figure(output_path)


def add_attack_only_recall(summary):
    baseline_class_columns = [f"baseline_recall_class_{label}" for label in range(1, 9)]
    incremental_class_columns = [
        f"incremental_recall_class_{label}" for label in range(1, 9)
    ]

    if "baseline_attack_macro_recall" not in summary.columns:
        summary["baseline_attack_macro_recall"] = summary[baseline_class_columns].mean(
            axis=1
        )

    if "incremental_attack_macro_recall" not in summary.columns:
        summary["incremental_attack_macro_recall"] = summary[
            incremental_class_columns
        ].mean(axis=1)

    summary["attack_macro_recall_improvement"] = (
        summary["incremental_attack_macro_recall"]
        - summary["baseline_attack_macro_recall"]
    )

    return summary


def attack_only_recall_plot(summary, output_path):
    plt.figure(figsize=(7.2, 4.5))
    plt.plot(
        summary["run_number"],
        summary["baseline_attack_macro_recall"],
        linewidth=0.9,
        label="Baseline",
    )
    plt.plot(
        summary["run_number"],
        summary["incremental_attack_macro_recall"],
        linewidth=0.9,
        label="Incremental",
    )
    plt.xlabel("Run")
    plt.ylabel("Mean recall over classes 1-8")
    plt.ylim(0, 1)
    plt.legend(frameon=False)
    save_figure(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=OUTPUT_ROOT)
    parser.add_argument("--output", default=OUTPUT_DIR)
    args = parser.parse_args()

    root = args.input
    out = args.output
    os.makedirs(out, exist_ok=True)
    class_names = load_class_names()

    summary = pd.read_csv(os.path.join(root, "all_runs_summary.csv")).sort_values(
        "run_number"
    )
    summary = add_attack_only_recall(summary)
    distribution = pd.read_csv(os.path.join(root, "class_distribution.csv"))

    plt.figure(figsize=(6.8, 4.2))
    plt.bar(distribution["class"].astype(str), distribution["full"])
    plt.yscale("log")
    plt.xlabel("Class")
    plt.ylabel("Number of observations")
    save_figure(os.path.join(out, "class_distribution.png"))

    attack_only_recall_plot(
        summary,
        os.path.join(out, "attack_only_recall_runs.png"),
    )

    attack_difference = summary["attack_macro_recall_improvement"]
    pd.DataFrame(
        [
            {
                "baseline_mean": summary["baseline_attack_macro_recall"].mean(),
                "baseline_std": summary["baseline_attack_macro_recall"].std(ddof=1),
                "incremental_mean": summary["incremental_attack_macro_recall"].mean(),
                "incremental_std": summary["incremental_attack_macro_recall"].std(
                    ddof=1
                ),
                "mean_difference": attack_difference.mean(),
                "improved_runs": int((attack_difference > 0).sum()),
                "equal_runs": int((attack_difference == 0).sum()),
                "worse_runs": int((attack_difference < 0).sum()),
            }
        ]
    ).to_csv(
        os.path.join(out, "attack_only_recall_summary.csv"),
        index=False,
    )

    class_recall_plot(
        summary,
        "baseline",
        os.path.join(out, "baseline_class_recall_runs.png"),
        class_names,
    )

    plt.figure(figsize=(6.8, 4.2))
    for _, row in summary.iterrows():
        path = os.path.join(
            run_directory(root, row["run_number"], row["seed"]),
            "baseline_search.csv",
        )
        if not os.path.exists(path):
            continue
        search = pd.read_csv(path).sort_values("hidden_units")
        plt.plot(
            search["hidden_units"],
            search["validation_macro_recall"],
            linewidth=0.55,
            alpha=0.25,
        )
    plt.xlabel("Hidden units")
    plt.ylabel("Validation macro recall")
    save_figure(os.path.join(out, "baseline_hidden_search.png"))

    mean_incremental_recall = summary["incremental_macro_recall"].mean()
    representative_index = (
        (summary["incremental_macro_recall"] - mean_incremental_recall).abs().idxmin()
    )
    representative = summary.loc[representative_index]
    representative_run = int(representative["run_number"])
    representative_seed = int(representative["seed"])
    representative_stages = pd.read_csv(
        os.path.join(
            run_directory(root, representative_run, representative_seed),
            "incremental_stages.csv",
        )
    )
    plot_incremental_validation_recall_by_hidden(
        representative_stages,
        os.path.join(
            out,
            "incremental_validation_recall_by_hidden_representative.png",
        ),
        class_names,
        None,
    )
    plot_incremental_class_progression(
        representative_stages,
        os.path.join(
            out,
            "incremental_class_validation_progression_representative.png",
        ),
        class_names,
    )
    pd.DataFrame(
        [
            {
                "run_number": representative_run,
                "seed": representative_seed,
                "incremental_macro_recall": representative["incremental_macro_recall"],
                "overall_mean_incremental_macro_recall": mean_incremental_recall,
            }
        ]
    ).to_csv(
        os.path.join(out, "representative_incremental_run.csv"),
        index=False,
    )

    for _, row in summary.iterrows():
        run_number = int(row["run_number"])
        seed = int(row["seed"])
        path = os.path.join(
            run_directory(root, run_number, seed),
            "incremental_stages.csv",
        )
        if not os.path.exists(path):
            continue
        stages = pd.read_csv(path)
        plot_incremental_validation_recall_by_hidden(
            stages,
            os.path.join(
                out,
                f"incremental_validation_recall_by_hidden_run_{run_number:04d}.png",
            ),
            class_names,
            f"Incremental validation recall - run {run_number}",
        )

    stage_rows = []
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

    if stage_rows:
        stage_data = pd.concat(stage_rows, ignore_index=True)

        plt.figure(figsize=(6.8, 4.2))
        for _, group in stage_data.groupby("run_number"):
            group = group.sort_values("num_classes")
            plt.plot(
                group["num_classes"],
                group["validation_macro_recall"],
                linewidth=0.55,
                alpha=0.25,
            )
        plt.xlabel("Number of active classes")
        plt.ylabel("Best validation macro recall")
        save_figure(os.path.join(out, "incremental_stage_recall.png"))

        stage_data.groupby("num_classes").agg(
            mean_hidden_units=("hidden_units", "mean"),
            std_hidden_units=("hidden_units", "std"),
            min_hidden_units=("hidden_units", "min"),
            max_hidden_units=("hidden_units", "max"),
            mean_validation_macro_recall=("validation_macro_recall", "mean"),
            std_validation_macro_recall=("validation_macro_recall", "std"),
        ).reset_index().to_csv(
            os.path.join(out, "incremental_stage_summary.csv"),
            index=False,
        )

    plt.figure(figsize=(6.8, 4.2))
    plt.plot(
        summary["run_number"],
        summary["baseline_time_seconds"] / 60,
        linewidth=0.9,
        label="Baseline",
    )
    plt.plot(
        summary["run_number"],
        summary["incremental_time_seconds"] / 60,
        linewidth=0.9,
        label="Incremental",
    )
    plt.xlabel("Run")
    plt.ylabel("Time (minutes)")
    plt.legend(frameon=False)
    save_figure(os.path.join(out, "runtime_runs.png"))

    plt.figure(figsize=(6.8, 4.2))
    plt.plot(
        summary["run_number"],
        summary["baseline_hidden_units"],
        linewidth=0.9,
        label="Baseline selected",
    )
    plt.plot(
        summary["run_number"],
        summary["incremental_hidden_units"],
        linewidth=0.9,
        label="Incremental final",
    )
    plt.xlabel("Run")
    plt.ylabel("Hidden units")
    plt.legend(frameon=False)
    save_figure(os.path.join(out, "final_hidden_units_runs.png"))

    baseline_confusion = average_normalised_confusion(root, summary, "baseline")
    incremental_confusion = average_normalised_confusion(root, summary, "incremental")
    confusion_plot(
        baseline_confusion,
        "Baseline mean normalised confusion matrix",
        os.path.join(out, "baseline_mean_confusion.png"),
    )
    confusion_plot(
        incremental_confusion,
        "Incremental mean normalised confusion matrix",
        os.path.join(out, "incremental_mean_confusion.png"),
    )

    per_class = classification_report_summary(root, summary)
    per_class["category"] = per_class["class"].map(class_names)
    per_class.to_csv(os.path.join(out, "per_class_full_summary.csv"), index=False)

    attack_root = os.path.join(root, "attack_priority")
    attack_summary_path = os.path.join(attack_root, "all_runs_summary.csv")
    if os.path.exists(attack_summary_path):
        attack_summary = pd.read_csv(attack_summary_path).sort_values("run_number")
        matched = summary.merge(
            attack_summary,
            on=["run_number", "seed"],
            suffixes=("_original", "_attack_priority"),
        )

        comparison_rows = []
        for method in ["baseline", "incremental"]:
            original_attack_recall = matched[
                [f"{method}_recall_class_{label}_original" for label in range(1, 9)]
            ].mean(axis=1)
            priority_attack_recall = matched[
                f"{method}_attack_macro_recall_attack_priority"
            ]
            difference = priority_attack_recall - original_attack_recall
            comparison_rows.append(
                {
                    "method": method,
                    "matched_runs": len(matched),
                    "original_attack_recall_mean": original_attack_recall.mean(),
                    "attack_priority_recall_mean": priority_attack_recall.mean(),
                    "mean_change": difference.mean(),
                    "improved_runs": int((difference > 1e-12).sum()),
                    "equal_runs": int((difference.abs() <= 1e-12).sum()),
                    "worse_runs": int((difference < -1e-12).sum()),
                    "architecture_changes": int(
                        (
                            matched[f"{method}_hidden_units_original"]
                            != matched[f"{method}_hidden_units_attack_priority"]
                        ).sum()
                    ),
                }
            )
        pd.DataFrame(comparison_rows).to_csv(
            os.path.join(out, "attack_priority_selection_summary.csv"),
            index=False,
        )

        class_rows = []
        for label in range(10):
            row = {
                "class": label,
                "category": class_names.get(label, f"Class {label}"),
            }
            for method in ["baseline", "incremental"]:
                original = matched[f"{method}_recall_class_{label}_original"].mean()
                priority = matched[
                    f"{method}_recall_class_{label}_attack_priority"
                ].mean()
                row[f"{method}_original_recall"] = original
                row[f"{method}_attack_priority_recall"] = priority
                row[f"{method}_change"] = priority - original
            class_rows.append(row)
        pd.DataFrame(class_rows).to_csv(
            os.path.join(out, "attack_priority_class_comparison.csv"),
            index=False,
        )

    overall_rows = []
    for metric in [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ]:
        baseline = summary[f"baseline_{metric}"]
        incremental = summary[f"incremental_{metric}"]
        difference = incremental - baseline
        overall_rows.append(
            {
                "metric": metric,
                "baseline_mean": baseline.mean(),
                "baseline_std": baseline.std(ddof=1),
                "incremental_mean": incremental.mean(),
                "incremental_std": incremental.std(ddof=1),
                "mean_difference": difference.mean(),
                "improved_runs": int((difference > 0).sum()),
                "equal_runs": int((difference == 0).sum()),
                "worse_runs": int((difference < 0).sum()),
            }
        )
    pd.DataFrame(overall_rows).to_csv(
        os.path.join(out, "summary_statistics.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
