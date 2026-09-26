import os
import time
from numpy import array
import pandas as pd
from tqdm import tqdm

import optuna

from pandas import DataFrame
from pandas.core.base import np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import traceback
from joblib import Parallel, delayed


def main():
    OUTPUT_DIR = "../report/outputFiles/"

    df = pd.read_csv("../data/networkTraffic.csv", na_values=["?"], low_memory=False)

    df["service"] = df["service"].fillna("unknown")
    # options = getAllDataPreprocesses()
    # print("Test1:", len(options))
    # exit()
    # options = array(np.random.shuffle(np.array(options)))
    # df = df.dropna()

    classification_preprocess(df, "../report/outputFiles/class_preprocess/")

    # classification(df, OUTPUT_DIR + "class_tree_results/")
    # classification_weights_optuna(df, OUTPUT_DIR + "class_weight_optuna_results/")
    # classification_weights(df)
    exit()


def getAllDataPreprocesses():
    options = []
    rm1Opt = [None, "sbytes", "sloss"]
    rm2Opt = [None, "dbytes", "dloss"]
    rm3Opt = [None, "is_ftp_login", "ct_ftp_cmd"]
    rm4Opt = ["id"]  # ,None
    rm1AOpt = [
        [],
        ["synack", "ackdat"],
        ["tcprtt"],
    ]  # ["tcprtt"], ["synack"],["ackdat"],
    misOpt = [None]  # , "median"]
    balOpt = ["under_and_over", None]  # "under", "over",
    outlierFields = [
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
    outlierOpt = {
        # "log1pALL": outlierFields,
        "log1pSome": [
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
        ],
        None: [],
    }

    outliers = lambda method, outlierFlds: (
        lambda X_train, X_test, y_train, y_test: handleOutliers(
            X_train,
            X_test,
            y_train,
            y_test,
            method,
            outlierFlds,
        )
    )

    # rmField = lambda arrFields: lambda df: removeField(
    #     df, [x for x in arrFields if x is not None]
    # )
    # missing = lambda method: lambda df: handleMissing(df, method)
    # oneHot = lambda fields: lambda df: oneHotEncode(df, fields)
    # balance = lambda method: lambda df: balanceData(df, method)

    rmField = lambda arrFields: (
        lambda X_train, X_test, y_train, y_test: removeField(
            X_train, X_test, y_train, y_test, arrFields
        )
    )

    missing = lambda method: (
        lambda X_train, X_test, y_train, y_test: handleMissing(
            X_train, X_test, y_train, y_test, method
        )
    )

    oneHot = lambda fields: (
        lambda X_train, X_test, y_train, y_test: oneHotEncode(
            X_train, X_test, y_train, y_test, fields
        )
    )

    balance = lambda method: (
        lambda X_train, X_test, y_train, y_test: balanceData(
            X_train, X_test, y_train, y_test, method
        )
    )
    for rm1 in rm1Opt:
        for rm2 in rm2Opt:
            for rm3 in rm3Opt:
                for rm4 in rm4Opt:
                    for rmA1 in rm1AOpt:
                        for missingOption in misOpt:
                            for outlierName, outlierFlds in outlierOpt.items():
                                for balanceOption in balOpt:
                                    # for outlierOption in outlierOpt:
                                    curOption = []
                                    curOption.append(rmField([rm1]))
                                    curOption.append(rmField([rm2]))
                                    curOption.append(rmField([rm3]))
                                    curOption.append(rmField([rm4]))
                                    curOption.append(rmField(rmA1))
                                    curOption.append(missing(missingOption))
                                    curOption.append(
                                        outliers(
                                            (
                                                "log1p"
                                                if outlierName is not None
                                                else None
                                            ),
                                            outlierFlds,
                                        )
                                    )
                                    curOption.append(
                                        oneHot(["proto", "state", "service"])
                                    )
                                    curOption.append(balance(balanceOption))
                                    options.append(
                                        {
                                            "steps": curOption,
                                            "removed_fields": [
                                                x
                                                for x in [rm1, rm2, rm3, rm4] + rmA1
                                                if x is not None
                                            ],
                                            "missing": missingOption,
                                            "balance": balanceOption,
                                            "outliers": outlierName,
                                        }
                                    )
    return options


def classification_preprocess(
    df: DataFrame, OUTPUT_DIR=("../report/outputFiles/class_preprocess/")
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    options = getAllDataPreprocesses()

    np.random.shuffle(np.array(options))

    total_tests = len(options)
    print(f"Total preprocessing tests: " f"{total_tests}")
    print()

    results = []
    progress_bar = tqdm(options, total=len(options), ncols=140)

    jobs = (
        delayed(run_preprocess_test)(test_number, option, df, OUTPUT_DIR, total_tests)
        for test_number, option in enumerate(options, start=1)
    )

    parallel_results = Parallel(
        n_jobs=6, prefer="processes", return_as="generator_unordered"
    )(jobs)
    all_start_time = time.time()
    for result, status, test_number in parallel_results:
        total_elapsed = time.time() - all_start_time
        results.append(result)

        average_wall_time = total_elapsed / (test_number + 1)

        remaining_tests = total_tests - (test_number)

        estimated_remaining = average_wall_time * remaining_tests

        status = f"{status} ETA={estimated_remaining / 60:.2f}m"
        # status = f"{status} ETA={estimated_remaining / 60:.2f}m\n"
        progress_bar.set_postfix_str(status, refresh=False)
        print("")
        progress_bar.update(1)
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)
    progress_bar.close()
    return


def classification_weights(df: DataFrame, option):
    y = df["attack_cat"]
    X = df.drop(columns=["id", "attack_cat"], errors="ignore")
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    preprocessing_start = time.time()
    # X_train, X_test = applyPreprocessing(X_train, X_test, option)
    X_train, X_test, y_train, y_test = applyPreprocessing(
        X_train, X_test, y_train, y_test, option
    )
    preprocessing_time = time.time() - preprocessing_start

    # max_depth_values = [12]
    # min_samples_leafs = [3]
    # catagory_values = ["gini"]
    max_depth_values = [9, 12, 14, 17, 20]
    min_samples_leafs = [2, 3, 5, 8]
    catagory_values = ["entropy", "gini"]

    weight_values = [
        None,
        {
            0: 0.03,
            1: 0.06,
            2: 0.07,
            3: 0.11,
            4: 0.08,
            5: 0.14,
            6: 0.06,
            7: 0.14,
            8: 0.19,
            9: 0.12,
        },
        {
            0: 0.08,
            1: 0.08,
            2: 0.07,
            3: 0.07,
            4: 0.03,
            5: 0.16,
            6: 0.12,
            7: 0.12,
            8: 0.18,
            9: 0.09,
        },
        {
            0: 0.03,
            1: 0.13,
            2: 0.08,
            3: 0.11,
            4: 0.09,
            5: 0.08,
            6: 0.05,
            7: 0.14,
            8: 0.15,
            9: 0.15,
        },
    ]

    weight_search_start = time.time()

    weight_results = Parallel(n_jobs=4, prefer="processes")(
        delayed(test_weight)(
            weight_number,
            weights,
            max_depth,
            min_samples_leaf,
            criterion,
            X_train,
            X_test,
            y_train,
            y_test,
        )
        for weight_number, weights in enumerate(weight_values, start=1)
        for max_depth in max_depth_values
        for min_samples_leaf in min_samples_leafs
        for criterion in catagory_values
    )

    weight_search_time = time.time() - weight_search_start

    # best_result = max(
    #     weight_results, key=lambda result: (result["macro_recall"], result["accuracy"])
    # )
    #
    # best_result["weight_search_time"] = weight_search_time
    # best_result["weight_search_time"] = time.time() - weight_search_start
    #
    # best_result["classification_report"] = classification_report(
    #     best_result["y_test"], best_result["y_pred"], zero_division=0
    # )
    #
    # best_result["confusion_matrix"] = confusion_matrix(
    #     best_result["y_test"], best_result["y_pred"]
    # )
    weight_results = sorted(
        weight_results,
        key=lambda result: (
            result["macro_recall"],
            result["accuracy"],
        ),
        reverse=True,
    )

    best_result = weight_results[0]

    best_result["weight_search_time"] = weight_search_time

    return (
        best_result,
        weight_results,
        X_train,
        y_train,
        preprocessing_time,
        weight_search_time,
    )

    return best_result, X_train, y_train


def test_weight(
    weight_number,
    weights,
    max_depth,
    min_samples_leaf,
    criterion,
    X_train,
    X_test,
    y_train,
    y_test,
):
    test_start_time = time.time()

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        class_weight=weights,
        random_state=42,
    )
    fit_start_time = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - fit_start_time
    predict_start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - predict_start_time
    test_time = time.time() - test_start_time
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred)

    return {
        "weight_number": weight_number,
        "weights": weights,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "criterion": criterion,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "fit_time": fit_time,
        "predict_time": predict_time,
        "test_time": test_time,
        "num_features": X_train.shape[1],
        "train_rows": X_train.shape[0],
        "test_rows": X_test.shape[0],
        "model_depth": model.get_depth(),
        "model_leaves": model.get_n_leaves(),
        "classification_report": report,
        "confusion_matrix": matrix,
    }


def run_preprocess_test(test_number, option, df, OUTPUT_DIR, total_tests):
    test_start_time = time.time()
    removed_name = (
        "-".join(option["removed_fields"])
        if len(option["removed_fields"]) > 0
        else "none"
    )
    missing_name = option["missing"] if option["missing"] is not None else "none"
    balance_name = option["balance"] if option["balance"] is not None else "none"
    outlier_name = option["outliers"] if option["outliers"] is not None else "none"
    output_name = (
        f"test_{test_number:04d}"
        f"_remove_{removed_name}"
        f"_missing_{missing_name}"
        f"_outliers_{outlier_name}"
        f"_balance_{balance_name}.txt"
    )
    output_path = os.path.join(OUTPUT_DIR, output_name)
    try:
        (
            best,
            all_results,
            X_train,
            y_train,
            preprocessing_time,
            weight_search_time,
        ) = classification_weights(df, option)
        total_test_time = time.time() - test_start_time
        removed_fields = option["removed_fields"]
        number_weight_sets = len(set(result["weight_number"] for result in all_results))
        with open(output_path, "w") as output_file:
            print("Classification Preprocessing Test", file=output_file)
            print("==============================", file=output_file)
            print(f"Test: {test_number}/{total_tests}", file=output_file)
            print(file=output_file)
            print("Preprocessing:", file=output_file)
            print(f"Removed fields: {removed_fields}", file=output_file)
            print(f"Missing method: {missing_name}", file=output_file)
            print(f"Balance method: {balance_name}", file=output_file)
            print(f"Outlier method: {outlier_name}", file=output_file)
            print(file=output_file)
            print("Dataset:", file=output_file)
            print(f"Original rows: {df.shape[0]}", file=output_file)
            print(f"Original columns: {df.shape[1]}", file=output_file)
            print(f"Processed training rows: {X_train.shape[0]}", file=output_file)
            print(f"Processed columns: {X_train.shape[1]}", file=output_file)
            print(f"Testing rows: {best['test_rows']}", file=output_file)
            print(file=output_file)
            print("BEST TREE / WEIGHT CONFIGURATION", file=output_file)
            print("================================", file=output_file)
            print(
                "Selection method: highest Macro Recall, then highest Accuracy",
                file=output_file,
            )
            print(file=output_file)
            print("Options:", file=output_file)
            print(
                f"Weight set: {best['weight_number']}/{number_weight_sets}",
                file=output_file,
            )
            print(f"Weights: {best['weights']}", file=output_file)
            print(f"max_depth: {best['max_depth']}", file=output_file)
            print(f"min_samples_leaf: {best['min_samples_leaf']}", file=output_file)
            print(f"criterion: {best['criterion']}", file=output_file)
            print("random_state: 42", file=output_file)
            print(f"Actual tree depth: {best['model_depth']}", file=output_file)
            print(f"Number of leaves: {best['model_leaves']}", file=output_file)
            print(file=output_file)
            print("Results:", file=output_file)
            print(f"Accuracy: {best['accuracy'] * 100:.2f}%", file=output_file)
            print(f"Macro F1: {best['macro_f1']:.4f}", file=output_file)
            print(f"Weighted F1: {best['weighted_f1']:.4f}", file=output_file)
            print(f"Macro Precision: {best['macro_precision']:.4f}", file=output_file)
            print(f"Macro Recall: {best['macro_recall']:.4f}", file=output_file)
            print(
                f"Balanced Accuracy: {best['balanced_accuracy']:.4f}", file=output_file
            )
            print(f"MCC: {best['mcc']:.4f}", file=output_file)
            print(file=output_file)
            print("Times:", file=output_file)
            print(
                f"Preprocessing Time: {preprocessing_time:.2f} seconds",
                file=output_file,
            )
            print(
                f"Best Tree Fit Time: {best['fit_time']:.2f} seconds", file=output_file
            )
            print(
                f"Best Tree Prediction Time: {best['predict_time']:.2f} seconds",
                file=output_file,
            )
            print(
                f"All Tree/Weight Tests Time: {weight_search_time:.2f} seconds",
                file=output_file,
            )
            print(f"Total Test Time: {total_test_time:.2f} seconds", file=output_file)
            print(file=output_file)
            print("Processed Class Distribution:", file=output_file)
            print(y_train.value_counts().sort_index().to_string(), file=output_file)
            print(file=output_file)
            print("Best Classification Report:", file=output_file)
            print(best["classification_report"], file=output_file)
            print("Best Confusion Matrix:", file=output_file)
            print(best["confusion_matrix"], file=output_file)
            print(file=output_file)
            print(file=output_file)
            print("ALL TREE / WEIGHT VARIATIONS", file=output_file)
            print("============================", file=output_file)
            print("Ranked by Macro Recall, then Accuracy", file=output_file)
            print(file=output_file)
            header = (
                f"{'Rank':>4} "
                f"{'W':>2} "
                f"{'Depth':>5} "
                f"{'Leaf':>4} "
                f"{'Criterion':>9} "
                f"{'Acc':>7} "
                f"{'MacroF1':>8} "
                f"{'WeightF1':>8} "
                f"{'Prec':>7} "
                f"{'Recall':>7} "
                f"{'BalAcc':>7} "
                f"{'MCC':>7} "
                f"{'ActualD':>7} "
                f"{'Leaves':>7} "
                f"{'Fit(s)':>7}"
            )
            print(header, file=output_file)
            print("-" * len(header), file=output_file)
            for rank, result in enumerate(all_results, start=1):
                print(
                    f"{rank:>4} "
                    f"{result['weight_number']:>2} "
                    f"{result['max_depth']:>5} "
                    f"{result['min_samples_leaf']:>4} "
                    f"{result['criterion']:>9} "
                    f"{result['accuracy'] * 100:>6.2f}% "
                    f"{result['macro_f1']:>8.4f} "
                    f"{result['weighted_f1']:>8.4f} "
                    f"{result['macro_precision']:>7.4f} "
                    f"{result['macro_recall']:>7.4f} "
                    f"{result['balanced_accuracy']:>7.4f} "
                    f"{result['mcc']:>7.4f} "
                    f"{result['model_depth']:>7} "
                    f"{result['model_leaves']:>7} "
                    f"{result['fit_time']:>7.2f}",
                    file=output_file,
                )
            print(file=output_file)
            print(file=output_file)
            print("DETAILED VARIATION RESULTS", file=output_file)
            print("==========================", file=output_file)
            for rank, result in enumerate(all_results, start=1):
                print(file=output_file)
                print(
                    f"Variation {rank}/{len(all_results)}",
                    file=output_file,
                )
                print("--------------------------", file=output_file)
                if rank == 1:
                    print("STATUS: BEST", file=output_file)
                print(
                    f"Weight set: {result['weight_number']}/{number_weight_sets}",
                    file=output_file,
                )
                print(f"Weights: {result['weights']}", file=output_file)
                print(f"max_depth: {result['max_depth']}", file=output_file)
                print(
                    f"min_samples_leaf: {result['min_samples_leaf']}", file=output_file
                )
                print(f"criterion: {result['criterion']}", file=output_file)
                print(f"Actual depth: {result['model_depth']}", file=output_file)
                print(f"Leaves: {result['model_leaves']}", file=output_file)
                print(file=output_file)
                print(f"Accuracy: {result['accuracy'] * 100:.2f}%", file=output_file)
                print(f"Macro F1: {result['macro_f1']:.4f}", file=output_file)
                print(f"Weighted F1: {result['weighted_f1']:.4f}", file=output_file)
                print(
                    f"Macro Precision: {result['macro_precision']:.4f}",
                    file=output_file,
                )
                print(f"Macro Recall: {result['macro_recall']:.4f}", file=output_file)
                print(
                    f"Balanced Accuracy: {result['balanced_accuracy']:.4f}",
                    file=output_file,
                )
                print(f"MCC: {result['mcc']:.4f}", file=output_file)
                print(file=output_file)
                print(f"Fit Time: {result['fit_time']:.2f} seconds", file=output_file)
                print(
                    f"Prediction Time: {result['predict_time']:.2f} seconds",
                    file=output_file,
                )
                print(
                    f"Variation Time: {result['test_time']:.2f} seconds",
                    file=output_file,
                )
                print(file=output_file)
                print("Classification Report:", file=output_file)
                print(result["classification_report"], file=output_file)
                print("Confusion Matrix:", file=output_file)
                print(result["confusion_matrix"], file=output_file)
                print(file=output_file)
        result = {
            "test": test_number,
            "removed_fields": str(option["removed_fields"]),
            "missing": missing_name,
            "balance": balance_name,
            "outlier": outlier_name,
            "rows": X_train.shape[0],
            "columns": X_train.shape[1],
            "weight_set": best["weight_number"],
            "weights": str(best["weights"]),
            "max_depth": best["max_depth"],
            "min_samples_leaf": best["min_samples_leaf"],
            "criterion": best["criterion"],
            "accuracy": best["accuracy"],
            "macro_f1": best["macro_f1"],
            "weighted_f1": best["weighted_f1"],
            "macro_precision": best["macro_precision"],
            "macro_recall": best["macro_recall"],
            "balanced_accuracy": best["balanced_accuracy"],
            "mcc": best["mcc"],
            "preprocess_time_seconds": preprocessing_time,
            "weight_test_time_seconds": weight_search_time,
            "total_time_seconds": total_test_time,
            "tree_depth": best["model_depth"],
            "tree_leaves": best["model_leaves"],
        }
        status = (
            f"F1={best['macro_f1']:.2f} "
            f"Rec={best['macro_recall']:.2f} "
            f"Acc={best['accuracy'] * 100:.2f}% "
            f"Depth={best['max_depth']} "
            f"Leaf={best['min_samples_leaf']} "
            f"{best['criterion']} "
            f"Time={total_test_time:.2f}s"
        )
        return result, status, test_number
    except Exception as e:
        total_test_time = time.time() - test_start_time
        with open(output_path, "w") as output_file:
            print("STATUS: ERROR", file=output_file)
            print(f"Error: {e}", file=output_file)
            print(traceback.format_exc(), file=output_file)
        result = {
            "test": test_number,
            "removed_fields": str(option["removed_fields"]),
            "missing": option["missing"],
            "balance": option["balance"],
            "outlier": option["outliers"],
            "error": str(e),
            "total_time_seconds": total_test_time,
        }
        return result, f"ERROR test {test_number}: {e}", test_number


def applyPreprocessing(
    X_train_data: DataFrame, X_test_data: DataFrame, y_train_data, y_test_data, option
):
    X_train = X_train_data.copy(deep=True)
    X_test = X_test_data.copy(deep=True)
    y_train = y_train_data.copy(deep=True)
    y_test = y_test_data.copy(deep=True)

    for step in option["steps"]:
        X_train, X_test, y_train, y_test = step(X_train, X_test, y_train, y_test)

    return X_train, X_test, y_train, y_test


def removeField(X_train: DataFrame, X_test: DataFrame, y_train, y_test, arrFields):
    arrFields = [x for x in arrFields if x is not None]
    X_train = X_train.drop(columns=arrFields, errors="ignore")
    X_test = X_test.drop(columns=arrFields, errors="ignore")
    return X_train, X_test, y_train, y_test


def handleMissing(X_train: DataFrame, X_test: DataFrame, y_train, y_test, method):
    if method is None or method == "none":
        return X_train, X_test, y_train, y_test

    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train = X_train.replace("?", np.nan)
    X_test = X_test.replace("?", np.nan)

    categorical = ["proto", "state", "service"]

    for field in X_train.columns:

        if field not in categorical:
            X_train[field] = pd.to_numeric(X_train[field], errors="coerce")

            if field in X_test.columns:
                X_test[field] = pd.to_numeric(X_test[field], errors="coerce")

    for field in X_train.columns:

        if field in categorical:
            mode = X_train[field].mode(dropna=True)

            if len(mode) > 0:
                replacement = mode.iloc[0]
            else:
                replacement = "unknown"

        else:
            replacement = X_train[field].median()
            if pd.isna(replacement):
                replacement = 0

        X_train[field] = X_train[field].fillna(replacement)

        if field in X_test.columns:
            X_test[field] = X_test[field].fillna(replacement)

    return X_train, X_test, y_train, y_test


def oneHotEncode(X_train: DataFrame, X_test: DataFrame, y_train, y_test, fields):
    X_train = X_train.copy()
    X_test = X_test.copy()
    fields = [field for field in fields if field in X_train.columns]
    for field in fields:
        categories = X_train[field].dropna().unique().tolist()
        X_train[field] = pd.Categorical(X_train[field], categories=categories)
        X_test[field] = pd.Categorical(X_test[field], categories=categories)
    X_train = pd.get_dummies(X_train, columns=fields, dtype=int)
    X_test = pd.get_dummies(X_test, columns=fields, dtype=int)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    return X_train, X_test, y_train, y_test


def balanceData(X_train: DataFrame, X_test: DataFrame, y_train, y_test, method):
    if method is None or method == "none":
        return X_train, X_test, y_train, y_test
    target_name = y_train.name if y_train.name is not None else "attack_cat"
    train_df = X_train.copy()
    train_df[target_name] = np.asarray(y_train)
    class_counts = train_df[target_name].value_counts()
    largest_class = class_counts.max()
    smallest_class = class_counts.min()
    midpoint = (largest_class + smallest_class) // 2
    balanced_classes = []
    for label, count in class_counts.items():
        df_class = train_df[train_df[target_name] == label]
        if method == "under":
            if count > smallest_class:
                df_class = df_class.sample(
                    n=smallest_class, replace=False, random_state=42
                )
        elif method == "over":
            if count < largest_class:
                df_class = df_class.sample(
                    n=largest_class, replace=True, random_state=42
                )
        elif method == "under_and_over":
            if count > midpoint:
                df_class = df_class.sample(n=midpoint, replace=False, random_state=42)
            elif count < midpoint:
                df_class = df_class.sample(n=midpoint, replace=True, random_state=42)
        balanced_classes.append(df_class)
    train_df = pd.concat(balanced_classes)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    y_train = train_df[target_name].copy()
    X_train = train_df.drop(columns=[target_name])
    return X_train, X_test, y_train, y_test


def handleOutliers(
    X_train: DataFrame, X_test: DataFrame, y_train, y_test, method, fields
):
    if method is None or method == "none":
        return X_train, X_test, y_train, y_test
    X_train = X_train.copy()
    X_test = X_test.copy()
    fields = [field for field in fields if field in X_train.columns]
    if method == "log1p":
        for field in fields:
            X_train[field] = np.log1p(X_train[field])
            X_test[field] = np.log1p(X_test[field])
    return X_train, X_test, y_train, y_test


def classification_weights_optuna(
    df: DataFrame,
    OUTPUT_DIR="../report/outputFiles/class_weight_optuna_results/",
    n_trials=1000,
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    y = df["attack_cat"]
    X = df.drop(columns=["id", "attack_cat"], errors="ignore")

    X = pd.get_dummies(X, drop_first=True)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full,
    )

    trial_results = []

    def objective(trial):
        trial_start_time = time.time()

        max_depth = trial.suggest_int("max_depth", 9, 20)

        min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 8)

        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])

        raw_weights = {
            class_id: trial.suggest_float(f"weight_{class_id}", 0.01, 1.0)
            for class_id in range(10)
        }

        weight_sum = sum(raw_weights.values())

        weights = {
            class_id: weight / weight_sum for class_id, weight in raw_weights.items()
        }

        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            class_weight=weights,
            random_state=42,
        )

        fit_start = time.time()

        model.fit(X_train, y_train)

        fit_time = time.time() - fit_start

        predict_start = time.time()

        y_pred = model.predict(X_val)

        predict_time = time.time() - predict_start

        accuracy = accuracy_score(y_val, y_pred)

        macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)

        weighted_f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

        macro_precision = precision_score(
            y_val, y_pred, average="macro", zero_division=0
        )
        macro_recall = recall_score(y_val, y_pred, average="macro", zero_division=0)

        balanced_accuracy = balanced_accuracy_score(y_val, y_pred)

        mcc = matthews_corrcoef(y_val, y_pred)

        trial_time = time.time() - trial_start_time

        result = {
            "trial": trial.number,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "criterion": criterion,
            "weights": str(weights),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "balanced_accuracy": balanced_accuracy,
            "mcc": mcc,
            "fit_time_seconds": fit_time,
            "prediction_time_seconds": predict_time,
            "total_time_seconds": trial_time,
        }

        trial_results.append(result)
        results_df = pd.DataFrame(trial_results)

        results_df = results_df.sort_values(
            by=["macro_recall", "accuracy"], ascending=False
        )

        results_df.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)

        trial.set_user_attr("accuracy", accuracy)

        trial.set_user_attr("macro_f1", macro_f1)

        trial.set_user_attr("macro_precision", macro_precision)

        trial.set_user_attr("balanced_accuracy", balanced_accuracy)

        trial.set_user_attr("mcc", mcc)

        return macro_recall

    sampler = optuna.samplers.TPESampler(
        seed=42,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="decision_tree_macro_recall",
    )

    all_start_time = time.time()

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=20,
        show_progress_bar=True,
    )

    optimisation_time = time.time() - all_start_time

    best_trial = study.best_trial
    best_params = best_trial.params

    raw_best_weights = {
        class_id: best_params[f"weight_{class_id}"] for class_id in range(10)
    }

    weight_sum = sum(raw_best_weights.values())

    best_weights = {
        class_id: weight / weight_sum for class_id, weight in raw_best_weights.items()
    }

    best_model = DecisionTreeClassifier(
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
        criterion=best_params["criterion"],
        class_weight=best_weights,
        random_state=42,
    )

    final_fit_start = time.time()

    best_model.fit(
        X_train_full,
        y_train_full,
    )

    final_fit_time = time.time() - final_fit_start

    final_predict_start = time.time()

    y_pred = best_model.predict(X_test)

    final_predict_time = time.time() - final_predict_start

    accuracy = accuracy_score(y_test, y_pred)

    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)

    macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred,
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred,
    )

    report = classification_report(y_test, y_pred, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred)

    optuna_df = study.trials_dataframe()

    optuna_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "optuna_trials.csv",
        ),
        index=False,
    )
    output_path = os.path.join(
        OUTPUT_DIR,
        "best_result.txt",
    )

    with open(output_path, "w") as output_file:

        print(
            "Optuna Decision Tree Weight Optimisation",
            file=output_file,
        )

        print(
            "========================================",
            file=output_file,
        )

        print(file=output_file)

        print(
            "Optimisation Objective: Macro Recall",
            file=output_file,
        )

        print(
            f"Number of Trials: {n_trials}",
            file=output_file,
        )

        print(
            f"Best Validation Macro Recall: {study.best_value:.6f}",
            file=output_file,
        )

        print(
            f"Optimisation Time: {optimisation_time:.2f} seconds",
            file=output_file,
        )

        print(file=output_file)

        print(
            "Best Parameters:",
            file=output_file,
        )

        print(
            f"max_depth: {best_params['max_depth']}",
            file=output_file,
        )

        print(
            f"min_samples_leaf: {best_params['min_samples_leaf']}",
            file=output_file,
        )

        print(
            f"criterion: {best_params['criterion']}",
            file=output_file,
        )

        print(file=output_file)

        print(
            "Best Class Weights:",
            file=output_file,
        )

        for class_id in range(10):
            print(
                f"{class_id}: {best_weights[class_id]:.6f}",
                file=output_file,
            )

        print(file=output_file)

        print(
            "Final Held-Out Test Results:",
            file=output_file,
        )

        print(
            f"Accuracy: {accuracy * 100:.2f}%",
            file=output_file,
        )

        print(
            f"Macro F1: {macro_f1:.4f}",
            file=output_file,
        )

        print(
            f"Weighted F1: {weighted_f1:.4f}",
            file=output_file,
        )

        print(
            f"Macro Precision: {macro_precision:.4f}",
            file=output_file,
        )

        print(
            f"Macro Recall: {macro_recall:.4f}",
            file=output_file,
        )

        print(
            f"Balanced Accuracy: {balanced_accuracy:.4f}",
            file=output_file,
        )

        print(
            f"MCC: {mcc:.4f}",
            file=output_file,
        )

        print(file=output_file)

        print(
            f"Final Fit Time: {final_fit_time:.2f} seconds",
            file=output_file,
        )

        print(
            f"Final Prediction Time: {final_predict_time:.2f} seconds",
            file=output_file,
        )

        print(file=output_file)

        print(
            "Classification Report:",
            file=output_file,
        )

        print(
            report,
            file=output_file,
        )

        print(
            "Confusion Matrix:",
            file=output_file,
        )

        print(
            matrix,
            file=output_file,
        )

    print()
    print("========================================")
    print("Optuna optimisation complete")
    print("========================================")
    print(f"Best validation Macro Recall: {study.best_value:.4f}")
    print(f"Final test Macro Recall: {macro_recall:.4f}")
    print(f"Final test Accuracy: {accuracy * 100:.2f}%")
    print(f"Best weights: {best_weights}")
    print()
    print(f"Results saved to: {OUTPUT_DIR}")

    return {
        "study": study,
        "model": best_model,
        "weights": best_weights,
        "max_depth": best_params["max_depth"],
        "min_samples_leaf": best_params["min_samples_leaf"],
        "criterion": best_params["criterion"],
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "classification_report": report,
        "confusion_matrix": matrix,
    }


if __name__ == "__main__":
    main()
