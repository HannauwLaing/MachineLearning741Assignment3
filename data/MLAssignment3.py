import argparse
import copy
import os
import random
import time
import re
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

# make hyperparams global vars
# clean up the outputs and make prints look nice instead of using "Test1: ", "Test2-1: " everywhere
# Handle repeated tests so I can run overnight
#
# Make use of sklearn preprocessor methods instead of my own. (Probably more reliable and faster)
#
# Added the timing to each run

DATA_FILE = "../data/networkTraffic.csv"
OUTPUT_ROOT = "../report/outputFiles"
OUTPUT_DIR = OUTPUT_ROOT
TARGET = "attack_cat"
DROP_FIELDS = ["id", "sbytes", "synack", "ackdat"]
LOG1P_FIELDS = [
    "spkts",
    "dpkts",
    "sload",
    "dload",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "rate",
    "response_body_len",
    "dur",
    "tcprtt",
]
CATEGORICAL_FIELDS = ["proto", "state", "service"]
SEED = 42
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
BATCH_SIZE = 512
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0
PATIENCE = 8
MIN_DELTA = 1e-4
UNDERFIT_TRAIN_RECALL = 0.85
OVERFIT_RECALL_GAP = 0.15
MIN_HIDDEN_IMPROVEMENT = 0.01
HIDDEN_IMPROVEMENT_PATIENCE = 6
BASELINE_HIDDEN_UNITS = [0, 16, 32, 36, 40, 64]
MAX_INCREMENTAL_HIDDEN_UNITS = 64 * 4
TORCH_THREADS = 24
MAX_EPOCHS = 200


def split_data(df):
    X = df.drop(columns=[TARGET], errors="ignore")
    y = df[TARGET]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=1.0 - TRAIN_FRACTION, random_state=SEED, stratify=y
    )
    relative_test = TEST_FRACTION / (VALIDATION_FRACTION + TEST_FRACTION)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, random_state=SEED, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_preprosessor(X_train):
    categorical = [field for field in CATEGORICAL_FIELDS if field in X_train.columns]
    numeric = [
        field
        for field in X_train.columns
        if field not in categorical and field not in DROP_FIELDS
    ]
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ]
    )


def preprocess_data(X_train, X_val, X_test):
    X_train = X_train.drop(columns=DROP_FIELDS, errors="ignore")
    X_val = X_val.drop(columns=DROP_FIELDS, errors="ignore")
    X_test = X_test.drop(columns=DROP_FIELDS, errors="ignore")

    for field in LOG1P_FIELDS:
        if field in X_train.columns:
            X_train[field] = np.log1p(pd.to_numeric(X_train[field], errors="coerce"))
            X_val[field] = np.log1p(pd.to_numeric(X_val[field], errors="coerce"))
            X_test[field] = np.log1p(pd.to_numeric(X_test[field], errors="coerce"))

    preprocessor = make_preprosessor(X_train)
    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_val = X_val.toarray()
        X_test = X_test.toarray()
    return (
        X_train.astype(np.float32),
        X_val.astype(np.float32),
        X_test.astype(np.float32),
        preprocessor,
    )


def encode_labels(y, class_order):
    label_to_index = {label: index for index, label in enumerate(class_order)}
    return np.array([label_to_index[label] for label in y], dtype=np.int64)


def save_class_distributions(y_full, y_train, y_val, y_test):
    classes = sorted(y_full.unique())
    distribution = pd.DataFrame(index=classes)
    distribution.index.name = "class"
    distribution["full"] = y_full.value_counts().reindex(classes, fill_value=0)
    distribution["train"] = y_train.value_counts().reindex(classes, fill_value=0)
    distribution["validation"] = y_val.value_counts().reindex(classes, fill_value=0)
    distribution["test"] = y_test.value_counts().reindex(classes, fill_value=0)
    distribution.to_csv(os.path.join(OUTPUT_DIR, "class_distribution.csv"))


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units=0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        if hidden_units == 0:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.hidden = nn.Linear(input_dim, hidden_units)
            self.output = nn.Linear(hidden_units, output_dim)

    def forward(self, x):
        if self.hidden_units == 0:
            return self.linear(x)
        return self.output(torch.tanh(self.hidden(x)))

    def expand_output(self):
        new_output_dim = self.output_dim + 1
        if self.hidden_units == 0:
            old = self.linear
            new = nn.Linear(self.input_dim, new_output_dim)
            with torch.no_grad():
                new.weight[: self.output_dim] = old.weight
                new.bias[: self.output_dim] = old.bias
                nn.init.xavier_uniform_(
                    new.weight[self.output_dim : self.output_dim + 1]
                )
                new.bias[self.output_dim] = 0.0
            self.linear = new
        else:
            old = self.output
            new = nn.Linear(self.hidden_units, new_output_dim)
            with torch.no_grad():
                new.weight[: self.output_dim] = old.weight
                new.bias[: self.output_dim] = old.bias
                nn.init.xavier_uniform_(
                    new.weight[self.output_dim : self.output_dim + 1]
                )
                new.bias[self.output_dim] = 0.0
            self.output = new
        self.output_dim = new_output_dim
        return self

    def add_hidden_unit(self):
        if self.hidden_units == 0:
            self.hidden_units = 1
            self.hidden = nn.Linear(self.input_dim, 1)
            self.output = nn.Linear(1, self.output_dim)
            del self.linear
            return self
        old_hidden = self.hidden
        old_output = self.output
        new_hidden_units = self.hidden_units + 1
        new_hidden = nn.Linear(self.input_dim, new_hidden_units)
        new_output = nn.Linear(new_hidden_units, self.output_dim)
        with torch.no_grad():
            new_hidden.weight[: self.hidden_units] = old_hidden.weight
            new_hidden.bias[: self.hidden_units] = old_hidden.bias
            new_output.weight[:, : self.hidden_units] = old_output.weight
            new_output.weight[:, self.hidden_units] = 0.0
            new_output.bias[:] = old_output.bias
        self.hidden = new_hidden
        self.output = new_output
        self.hidden_units = new_hidden_units
        return self


def make_subset_loader(dataset, original_labels, active_classes, batch_size, shuffle):
    active_set = set(active_classes)
    indices = np.flatnonzero(np.isin(original_labels, list(active_set))).tolist()
    subset = Subset(dataset, indices)
    return DataLoader(
        subset, batch_size=batch_size, shuffle=shuffle, num_workers=0
    ), len(indices)


def metrics_from_predictions(y_true, y_pred, class_labels):
    recalls = recall_score(
        y_true,
        y_pred,
        labels=np.arange(len(class_labels)),
        average=None,
        zero_division=0,
    )
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }
    for label, recall in zip(class_labels, recalls):
        result[f"recall_class_{label}"] = recall
    return result


def evaluate_loader(model, loader, class_labels):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_count = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            total_count += len(y_batch)
            y_true.extend(y_batch.cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    metrics = metrics_from_predictions(np.array(y_true), np.array(y_pred), class_labels)
    metrics["loss"] = total_loss / max(total_count, 1)
    return metrics, np.array(y_true), np.array(y_pred)


def train_model(model, train_loader, val_loader, class_labels, max_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    start = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
            total_count += len(y_batch)
        val_metrics, _, _ = evaluate_loader(model, val_loader, class_labels)
        train_loss = total_loss / max(total_count, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_metrics["loss"],
                "validation_macro_recall": val_metrics["macro_recall"],
                "validation_macro_f1": val_metrics["macro_f1"],
            }
        )
        if val_metrics["loss"] < best_val_loss - MIN_DELTA:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= PATIENCE:
            break
    model.load_state_dict(best_state)
    train_metrics, _, _ = evaluate_loader(
        model,
        DataLoader(
            train_loader.dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        ),
        class_labels,
    )
    val_metrics, _, _ = evaluate_loader(model, val_loader, class_labels)
    return (
        model,
        train_metrics,
        val_metrics,
        pd.DataFrame(history),
        best_epoch,
        time.time() - start,
    )


def save_final_outputs(prefix, model, loader, class_labels):
    metrics, y_true, y_pred = evaluate_loader(model, loader, class_labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_labels)),
        target_names=[str(label) for label in class_labels],
        zero_division=0,
        output_dict=True,
    )
    pd.DataFrame(report).T.to_csv(
        os.path.join(OUTPUT_DIR, f"{prefix}_classification_report.csv")
    )
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_labels)))
    pd.DataFrame(matrix, index=class_labels, columns=class_labels).to_csv(
        os.path.join(OUTPUT_DIR, f"{prefix}_confusion_matrix.csv")
    )
    return metrics


def run_baseline(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    max_epochs,
    hidden_options,
    run_seed,
):
    baseline_start_time = time.time()
    class_labels = sorted(np.unique(y_train).tolist())
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(encode_labels(y_train, class_labels)),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(encode_labels(y_val, class_labels))
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(encode_labels(y_test, class_labels))
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    results = []
    models = {}
    for hidden_units in hidden_options:

        random.seed(run_seed + hidden_units)
        np.random.seed(run_seed + hidden_units)
        torch.manual_seed(run_seed + hidden_units)

        model = FeedForwardNetwork(X_train.shape[1], len(class_labels), hidden_units)
        model, train_metrics, val_metrics, history, best_epoch, train_time = (
            train_model(model, train_loader, val_loader, class_labels, max_epochs)
        )

        row = {
            "hidden_units": hidden_units,
            "best_epoch": best_epoch,
            "train_time_seconds": train_time,
        }
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"validation_{key}": value for key, value in val_metrics.items()})
        results.append(row)
        models[hidden_units] = model.cpu()
        history.to_csv(
            os.path.join(OUTPUT_DIR, f"baseline_hidden_{hidden_units}_history.csv"),
            index=False,
        )
        print(
            f"Baseline hidden={hidden_units}: val recall={val_metrics['macro_recall']:.4f}, "
            f"val F1={val_metrics['macro_f1']:.4f}, time={train_time:.2f}s"
        )
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "baseline_search.csv"), index=False)
    best_row = max(
        results,
        key=lambda row: (row["validation_macro_recall"], row["validation_macro_f1"]),
    )
    best_hidden = int(best_row["hidden_units"])
    best_model = models[best_hidden]
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_metrics = save_final_outputs("baseline", best_model, test_loader, class_labels)
    baseline_total_time = time.time() - baseline_start_time
    pd.DataFrame(
        [
            {
                **{
                    "selected_hidden_units": best_hidden,
                    "total_time_seconds": baseline_total_time,
                },
                **test_metrics,
            }
        ]
    ).to_csv(os.path.join(OUTPUT_DIR, "baseline_test.csv"), index=False)
    print(
        f"Selected baseline hidden={best_hidden}: test recall={test_metrics['macro_recall']:.4f}, "
        f"test F1={test_metrics['macro_f1']:.4f}, total time={baseline_total_time:.2f}s"
    )
    return best_model, test_metrics, class_labels, baseline_total_time, best_hidden


def run_incremental(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    max_epochs,
    max_hidden_units,
    run_seed,
):
    incremental_start_time = time.time()

    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)

    class_order = pd.Series(y_train).value_counts().sort_values().index.tolist()
    pd.DataFrame(
        {
            "order": range(1, len(class_order) + 1),
            "class": class_order,
            "train_count": [int((y_train == label).sum()) for label in class_order],
        }
    ).to_csv(os.path.join(OUTPUT_DIR, "incremental_class_order.csv"), index=False)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(encode_labels(y_train, class_order))
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(encode_labels(y_val, class_order))
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(encode_labels(y_test, class_order))
    )
    active_classes = class_order[:2]
    model = FeedForwardNetwork(X_train.shape[1], 2, 0)
    stage_rows = []
    history_number = 0
    for stage_size in range(2, len(class_order) + 1):
        if stage_size > 2:
            active_classes = class_order[:stage_size]
            model.expand_output()
        failed_improvements = 0
        best_stage_validation_recall = float("-inf")
        best_stage_model = None
        best_stage_hidden_units = model.hidden_units
        while True:
            train_loader, train_count = make_subset_loader(
                train_dataset, np.asarray(y_train), active_classes, BATCH_SIZE, True
            )
            val_loader, val_count = make_subset_loader(
                val_dataset, np.asarray(y_val), active_classes, BATCH_SIZE, False
            )
            model, train_metrics, val_metrics, history, best_epoch, train_time = (
                train_model(model, train_loader, val_loader, active_classes, max_epochs)
            )
            current_validation_recall = val_metrics["macro_recall"]
            if (
                current_validation_recall
                >= best_stage_validation_recall + MIN_HIDDEN_IMPROVEMENT
            ):
                best_stage_validation_recall = current_validation_recall
                best_stage_model = copy.deepcopy(model).cpu()
                best_stage_hidden_units = model.hidden_units
                failed_improvements = 0
            else:
                if current_validation_recall >= best_stage_validation_recall:
                    best_stage_validation_recall = current_validation_recall
                    best_stage_model = copy.deepcopy(model).cpu()
                    best_stage_hidden_units = model.hidden_units
                failed_improvements += 1
            if (
                train_metrics["macro_recall"] - val_metrics["macro_recall"]
                >= OVERFIT_RECALL_GAP
            ):
                fit_state = "overfit"
            elif train_metrics["macro_recall"] >= UNDERFIT_TRAIN_RECALL:
                fit_state = "acceptable"
            elif failed_improvements >= HIDDEN_IMPROVEMENT_PATIENCE:
                fit_state = "stagnated"
            else:
                fit_state = "underfit"
            history_number += 1
            history.to_csv(
                os.path.join(
                    OUTPUT_DIR,
                    f"incremental_history_{history_number:03d}_classes_{stage_size}_hidden_{model.hidden_units}.csv",
                ),
                index=False,
            )
            row = {
                "stage": stage_size - 1,
                "num_classes": stage_size,
                "classes": ",".join(map(str, active_classes)),
                "newest_class": active_classes[-1],
                "hidden_units": model.hidden_units,
                "train_rows": train_count,
                "validation_rows": val_count,
                "best_epoch": best_epoch,
                "fit_state": fit_state,
                "train_time_seconds": train_time,
            }
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update(
                {f"validation_{key}": value for key, value in val_metrics.items()}
            )
            stage_rows.append(row)
            pd.DataFrame(stage_rows).to_csv(
                os.path.join(OUTPUT_DIR, "incremental_stages.csv"), index=False
            )
            print(
                f"Incremental classes={stage_size}, hidden={model.hidden_units}: "
                f"state={fit_state}, train recall={train_metrics['macro_recall']:.4f}, "
                f"val recall={val_metrics['macro_recall']:.4f}, time={train_time:.2f}s"
            )
            if fit_state == "underfit" and model.hidden_units < max_hidden_units:
                model.add_hidden_unit()
                continue
            if fit_state == "stagnated" and best_stage_model is not None:
                attempted_hidden_units = model.hidden_units
                model = copy.deepcopy(best_stage_model)
                print(
                    f"Stage {stage_size}: stagnated at hidden={attempted_hidden_units}; "
                    f"restoring hidden={best_stage_hidden_units} "
                    f"with val recall={best_stage_validation_recall:.4f}"
                )
            break
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_metrics = save_final_outputs("incremental", model, test_loader, class_order)
    incremental_total_time = time.time() - incremental_start_time
    pd.DataFrame(
        [
            {
                **{
                    "final_hidden_units": model.hidden_units,
                    "total_time_seconds": incremental_total_time,
                },
                **test_metrics,
            }
        ]
    ).to_csv(os.path.join(OUTPUT_DIR, "incremental_test.csv"), index=False)
    print(
        f"Incremental final hidden={model.hidden_units}: test recall={test_metrics['macro_recall']:.4f}, "
        f"test F1={test_metrics['macro_f1']:.4f}, total time={incremental_total_time:.2f}s"
    )
    return model, test_metrics, class_order, incremental_total_time, model.hidden_units


def get_next_run_number(output_root):
    run_numbers = []
    if not os.path.isdir(output_root):
        return 1
    for name in os.listdir(output_root):
        match = re.fullmatch(r"run_(\d{4})_seed_\d+", name)
        if match:
            run_numbers.append(int(match.group(1)))
    if not run_numbers:
        return 1
    return max(run_numbers) + 1


def save_run_config(run_number, run_seed, args):
    config = {
        "run_number": run_number,
        "run_seed": run_seed,
        "fraction": args.fraction,
        "epochs": args.epochs,
        "baseline_hidden": args.baseline_hidden,
        "max_incremental_hidden": args.max_incremental_hidden,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "underfit_train_recall": UNDERFIT_TRAIN_RECALL,
        "overfit_recall_gap": OVERFIT_RECALL_GAP,
        "min_hidden_improvement": MIN_HIDDEN_IMPROVEMENT,
        "hidden_improvement_patience": HIDDEN_IMPROVEMENT_PATIENCE,
        "max_epochs": MAX_EPOCHS,
    }

    pd.DataFrame(
        [{"parameter": key, "value": value} for key, value in config.items()]
    ).to_csv(
        os.path.join(OUTPUT_DIR, "run_config.csv"),
        index=False,
    )


def update_all_runs_summary(summary_row):
    summary_path = os.path.join(OUTPUT_ROOT, "all_runs_summary.csv")
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        summary_df = pd.concat(
            [summary_df, pd.DataFrame([summary_row])], ignore_index=True
        )
    else:
        summary_df = pd.DataFrame([summary_row])

    summary_df.to_csv(summary_path, index=False)


def save_experiment_status(
    experiment_start_time,
    setup_time,
    completed_runs,
    current_run_number,
    interrupted,
):
    status = {
        "completed_runs_this_process": completed_runs,
        "current_run_number": current_run_number,
        "setup_time_seconds": setup_time,
        "process_total_time_seconds": time.time() - experiment_start_time,
        "interrupted": interrupted,
    }

    pd.DataFrame([status]).to_csv(
        os.path.join(OUTPUT_ROOT, "experiment_status.csv"),
        index=False,
    )


def main():
    global OUTPUT_DIR

    experiment_start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument(
        "--baseline-hidden",
        default=",".join(map(str, BASELINE_HIDDEN_UNITS)),
    )
    parser.add_argument(
        "--max-incremental-hidden",
        type=int,
        default=MAX_INCREMENTAL_HIDDEN_UNITS,
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=0,
    )
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-incremental", action="store_true")
    args = parser.parse_args()

    if not 0 < args.fraction <= 1:
        print("--fraction must be in 0 and 1")
        exit()

    if args.runs < 0:
        print("--runs must be >= 0")
        exit()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    OUTPUT_DIR = OUTPUT_ROOT

    torch.set_num_threads(TORCH_THREADS)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    setup_start_time = time.time()

    print("Loading and preprocessing data once for all repeated runs...")

    df = pd.read_csv(DATA_FILE, na_values=["?"], low_memory=False)
    if args.fraction < 1.0:
        _, df = train_test_split(
            df, test_size=args.fraction, random_state=SEED, stratify=df[TARGET]
        )
        df = df.reset_index(drop=True)

    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_data(df)

    save_class_distributions(df[TARGET], y_train, y_val, y_test)
    print(
        "Incremental class order:",
        y_train.value_counts().sort_values().index.tolist(),
    )

    X_train, X_val, X_test, _ = preprocess_data(
        X_train_raw,
        X_val_raw,
        X_test_raw,
    )
    setup_time = time.time() - setup_start_time

    print(f"Processed feature count: {X_train.shape[1]}\n")

    hidden_options = [
        int(value.strip()) for value in args.baseline_hidden.split(",") if value.strip()
    ]

    next_run_number = get_next_run_number(OUTPUT_ROOT)
    completed_runs = 0
    current_run_number = next_run_number

    print(f"Starting at run {next_run_number:04d}\n")

    try:
        while args.runs == 0 or completed_runs < args.runs:
            print("TEST1: ", args.runs)
            current_run_number = next_run_number + completed_runs
            run_seed = SEED + current_run_number - 1
            run_name = f"run_{current_run_number:04d}_seed_{run_seed}"
            OUTPUT_DIR = os.path.join(OUTPUT_ROOT, run_name)
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            run_start_time = time.time()

            print("=" * 80)
            print(
                f"RUN {current_run_number:04d} | "
                f"seed={run_seed} | "
                f"output={OUTPUT_DIR}"
            )
            print("=" * 80)

            random.seed(run_seed)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)
            save_run_config(current_run_number, run_seed, args)
            baseline_metrics = None
            incremental_metrics = None
            baseline_total_time = None
            incremental_total_time = None
            baseline_hidden = None
            incremental_hidden = None
            if not args.skip_incremental:
                (
                    _,
                    incremental_metrics,
                    _,
                    incremental_total_time,
                    incremental_hidden,
                ) = run_incremental(
                    X_train,
                    X_val,
                    X_test,
                    np.asarray(y_train),
                    np.asarray(y_val),
                    np.asarray(y_test),
                    args.epochs,
                    args.max_incremental_hidden,
                    run_seed,
                )
            if not args.skip_baseline:
                (
                    _,
                    baseline_metrics,
                    _,
                    baseline_total_time,
                    baseline_hidden,
                ) = run_baseline(
                    X_train,
                    X_val,
                    X_test,
                    np.asarray(y_train),
                    np.asarray(y_val),
                    np.asarray(y_test),
                    args.epochs,
                    hidden_options,
                    run_seed,
                )
            run_total_time = time.time() - run_start_time
            comparison = []
            if baseline_metrics is not None:
                comparison.append(
                    {
                        "run_number": current_run_number,
                        "seed": run_seed,
                        "method": "baseline",
                        "hidden_units": baseline_hidden,
                        "total_time_seconds": baseline_total_time,
                        **baseline_metrics,
                    }
                )
            if incremental_metrics is not None:
                comparison.append(
                    {
                        "run_number": current_run_number,
                        "seed": run_seed,
                        "method": "incremental",
                        "hidden_units": incremental_hidden,
                        "total_time_seconds": incremental_total_time,
                        **incremental_metrics,
                    }
                )
            if comparison:
                pd.DataFrame(comparison).to_csv(
                    os.path.join(OUTPUT_DIR, "final_comparison.csv"),
                    index=False,
                )
            timing_rows = []
            if baseline_total_time is not None:
                timing_rows.append(
                    {"section": "baseline_total", "time_seconds": baseline_total_time}
                )
            if incremental_total_time is not None:
                timing_rows.append(
                    {
                        "section": "incremental_total",
                        "time_seconds": incremental_total_time,
                    }
                )
            timing_rows.append(
                {
                    "section": "run_total",
                    "time_seconds": run_total_time,
                }
            )

            pd.DataFrame(timing_rows).to_csv(
                os.path.join(OUTPUT_DIR, "timing_summary.csv"),
                index=False,
            )

            summary_row = {
                "run_number": current_run_number,
                "seed": run_seed,
                "run_total_time_seconds": run_total_time,
            }

            if baseline_metrics is not None:
                summary_row["baseline_hidden_units"] = baseline_hidden
                summary_row["baseline_time_seconds"] = baseline_total_time

                for key, value in baseline_metrics.items():
                    summary_row[f"baseline_{key}"] = value

            if incremental_metrics is not None:
                summary_row["incremental_hidden_units"] = incremental_hidden
                summary_row["incremental_time_seconds"] = incremental_total_time

                for key, value in incremental_metrics.items():
                    summary_row[f"incremental_{key}"] = value

            if baseline_metrics is not None and incremental_metrics is not None:
                summary_row["macro_recall_improvement"] = (
                    incremental_metrics["macro_recall"]
                    - baseline_metrics["macro_recall"]
                )
                summary_row["macro_f1_improvement"] = (
                    incremental_metrics["macro_f1"] - baseline_metrics["macro_f1"]
                )
                summary_row["accuracy_improvement"] = (
                    incremental_metrics["accuracy"] - baseline_metrics["accuracy"]
                )

            update_all_runs_summary(summary_row)

            completed_runs += 1

            save_experiment_status(
                experiment_start_time,
                setup_time,
                completed_runs,
                current_run_number,
                False,
            )
            print(
                f"\n\nRun {current_run_number:04d} complete in {run_total_time:.2f}s."
            )

            if baseline_metrics is not None and incremental_metrics is not None:
                difference = (
                    incremental_metrics["macro_recall"]
                    - baseline_metrics["macro_recall"]
                )
                print(
                    f"Macro recall: "
                    f"baseline={baseline_metrics['macro_recall']:.4f}, "
                    f"incremental={incremental_metrics['macro_recall']:.4f}, "
                    f"difference={difference:+.4f}"
                )

            print()

    except KeyboardInterrupt:
        print()
        print("=" * 80)
        print("Interrupted by user.")
        print(f"{completed_runs} complete run(s) from this process were saved.")
        print(
            "The current run folder may contain partial results up to the "
            "point of interruption."
        )
        print(
            "Combined completed-run data is in "
            f"{os.path.join(OUTPUT_ROOT, 'all_runs_summary.csv')}."
        )
        print("=" * 80)

        save_experiment_status(
            experiment_start_time,
            setup_time,
            completed_runs,
            current_run_number,
            True,
        )

        return

    save_experiment_status(
        experiment_start_time,
        setup_time,
        completed_runs,
        current_run_number,
        False,
    )

    total_elapsed = time.time() - experiment_start_time

    print("=" * 80)
    print(f"Finished {completed_runs} repeated run(s) in " f"{total_elapsed:.2f}s.")
    print("Combined summary: " f"{os.path.join(OUTPUT_ROOT, 'all_runs_summary.csv')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
