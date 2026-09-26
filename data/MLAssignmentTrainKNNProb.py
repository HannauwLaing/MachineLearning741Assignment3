import os
import time
from numpy import array
import pandas as pd
from tqdm import tqdm

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

    classification_preprocess(df)
    # kNeighbors(df)
    # classification(df, OUTPUT_DIR + "class_tree_results/")
    exit()


def getAllDataPreprocesses():
    options = []
    rm1Opt = ["sbytes", "sloss", None]  # "sloss",
    rm2Opt = ["dbytes", "dloss", None]  # "dloss",
    rm3Opt = ["is_ftp_login", "ct_ftp_cmd", None]  # "ct_ftp_cmd",
    rm4Opt = ["id"]  # ,None
    rm1AOpt = [["synack", "ackdat"]]  # , [], ["tcprtt"]]  # , ["synack"], ["ackdat"]]
    misOpt = [None]  # , "median_mode"]
    balOpt = ["under_and_over"]  # "under", "over",None,
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
        # None: [],
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
    for missingOption in misOpt:
        for outlierName, outlierFlds in outlierOpt.items():
            for balanceOption in balOpt:
                for rm1 in rm1Opt:
                    for rm2 in rm2Opt:
                        for rm3 in rm3Opt:
                            for rm4 in rm4Opt:
                                for rmA1 in rm1AOpt:
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
    df: DataFrame, OUTPUT_DIR=("../report/outputFiles/knn_preprocess/")
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    options = getAllDataPreprocesses()

    # np.random.shuffle(np.array(options))

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
    for result, status, test_index in parallel_results:
        test_number = test_index
        total_elapsed = time.time() - all_start_time
        results.append(result)
        average_wall_time = total_elapsed / (test_number)

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
            model_search_time,
        ) = knn_weights(df, option)
        total_test_time = time.time() - test_start_time
        removed_fields = option["removed_fields"]
        n_neighbors = best.get("n_neighbors", "pending")
        weights = best.get("weights", "pending")
        distance = best.get("distance", "pending")
        p = best.get("p", "pending")
        metric = best.get("metric", "pending")
        algorithm = best.get("algorithm", "pending")
        with open(output_path, "w") as output_file:
            print("KNN Preprocessing Test", file=output_file)
            print("======================", file=output_file)
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
            print("BEST KNN CONFIGURATION", file=output_file)
            print("======================", file=output_file)
            print(
                "Selection method: highest Macro Recall, then highest Accuracy",
                file=output_file,
            )
            print(file=output_file)
            print("Options:", file=output_file)
            print(f"n_neighbors: {n_neighbors}", file=output_file)
            print(f"weights: {weights}", file=output_file)
            print(f"distance: {distance}", file=output_file)
            print(f"p: {p}", file=output_file)
            print(f"metric: {metric}", file=output_file)
            print(f"algorithm: {algorithm}", file=output_file)
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
                f"Best KNN Fit Time: {best['fit_time']:.2f} seconds", file=output_file
            )
            print(
                f"Best KNN Prediction Time: {best['predict_time']:.2f} seconds",
                file=output_file,
            )
            print(
                f"All KNN Tests Time: {model_search_time:.2f} seconds", file=output_file
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
            print("ALL KNN VARIATIONS", file=output_file)
            print("==================", file=output_file)
            print("Ranked by Macro Recall, then Accuracy", file=output_file)
            print(file=output_file)
            header = (
                f"{'Rank':>4} "
                f"{'K':>4} "
                f"{'Weights':>10} "
                f"{'Distance':>10} "
                f"{'P':>3} "
                f"{'Acc':>7} "
                f"{'MacroF1':>8} "
                f"{'WeightF1':>8} "
                f"{'Prec':>7} "
                f"{'Recall':>7} "
                f"{'BalAcc':>7} "
                f"{'MCC':>7} "
                f"{'Fit(s)':>7}"
            )
            print(header, file=output_file)
            print("-" * len(header), file=output_file)
            for rank, result in enumerate(all_results, start=1):
                result_n_neighbors = result.get("n_neighbors", "-")
                result_weights = result.get("weights", "-")
                result_distance = result.get("distance", "-")
                result_p = result.get("p", "-")
                print(
                    f"{rank:>4} "
                    f"{str(result_n_neighbors):>4} "
                    f"{str(result_weights):>10} "
                    f"{str(result_distance):>10} "
                    f"{str(result_p):>3} "
                    f"{result['accuracy'] * 100:>6.2f}% "
                    f"{result['macro_f1']:>8.4f} "
                    f"{result['weighted_f1']:>8.4f} "
                    f"{result['macro_precision']:>7.4f} "
                    f"{result['macro_recall']:>7.4f} "
                    f"{result['balanced_accuracy']:>7.4f} "
                    f"{result['mcc']:>7.4f} "
                    f"{result['fit_time']:>7.2f}",
                    file=output_file,
                )
            print(file=output_file)
            print(file=output_file)
            print("DETAILED KNN VARIATION RESULTS", file=output_file)
            print("==============================", file=output_file)
            for rank, result in enumerate(all_results, start=1):
                print(file=output_file)
                print(f"Variation {rank}/{len(all_results)}", file=output_file)
                print("--------------------------", file=output_file)
                if rank == 1:
                    print("STATUS: BEST", file=output_file)
                print(
                    f"n_neighbors: {result.get('n_neighbors', 'pending')}",
                    file=output_file,
                )
                print(f"weights: {result.get('weights', 'pending')}", file=output_file)
                print(
                    f"distance: {result.get('distance', 'pending')}", file=output_file
                )
                print(f"p: {result.get('p', 'pending')}", file=output_file)
                print(f"metric: {result.get('metric', 'pending')}", file=output_file)
                print(
                    f"algorithm: {result.get('algorithm', 'pending')}", file=output_file
                )
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
            "n_neighbors": best.get("n_neighbors", "pending"),
            "weights": best.get("weights", "pending"),
            "distance": best.get("distance", "pending"),
            "p": best.get("p", "pending"),
            "metric": best.get("metric", "pending"),
            "algorithm": best.get("algorithm", "pending"),
            "accuracy": best["accuracy"],
            "macro_f1": best["macro_f1"],
            "weighted_f1": best["weighted_f1"],
            "macro_precision": best["macro_precision"],
            "macro_recall": best["macro_recall"],
            "balanced_accuracy": best["balanced_accuracy"],
            "mcc": best["mcc"],
            "preprocess_time_seconds": preprocessing_time,
            "model_test_time_seconds": model_search_time,
            "total_time_seconds": total_test_time,
        }
        status = (
            f"F1={best['macro_f1']:.2f} "
            f"Rec={best['macro_recall']:.2f} "
            f"Acc={best['accuracy'] * 100:.2f}% "
            f"K={best.get('n_neighbors', '-')} "
            f"W={best.get('weights', '-')} "
            f"Dist={best.get('distance', '-')} "
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


def knn_weights(df: DataFrame, option):
    y = df["attack_cat"]
    X = df.drop(columns=["id", "attack_cat"], errors="ignore")
    # X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    preprocessing_start = time.time()
    X_train, X_test, y_train, y_test = applyPreprocessing(
        X_train, X_test, y_train, y_test, option
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    preprocessing_time = time.time() - preprocessing_start

    # n_neighbors_values = [3]
    # weights_values = ["distance", "uniform"]
    # p_values = [2]  # ,2]
    n_neighbors_values = [7, 9, 15]
    weights_values = ["distance"]  # , "uniform"]
    p_values = [2, 1]  # ,1]
    # n_neighbors_values = [1, 3, 5]
    # weights_values = ["uniform", "distance"]
    # p_values = [2]

    model_search_start = time.time()
    model_results = Parallel(n_jobs=1, prefer="processes")(
        delayed(test_weight)(
            n_neighbors,
            weights,
            p,
            X_train,
            X_test,
            y_train,
            y_test,
        )
        for weights in weights_values
        for n_neighbors in n_neighbors_values
        for p in p_values
    )
    model_search_time = time.time() - model_search_start
    model_results = sorted(
        model_results,
        key=lambda result: (
            result["macro_recall"],
            result["accuracy"],
        ),
        reverse=True,
    )
    best_result = model_results[0]
    best_result["model_search_time"] = model_search_time
    return (
        best_result,
        model_results,
        X_train,
        y_train,
        preprocessing_time,
        model_search_time,
    )


def test_weight(
    n_neighbors,
    weights,
    p,
    X_train,
    X_test,
    y_train,
    y_test,
):
    test_start_time = time.time()
    distance_name = "manhattan" if p == 1 else "euclidean" if p == 2 else f"p{p}"
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric="minkowski",
        p=p,
        algorithm="auto",
        n_jobs=12,
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
        "n_neighbors": n_neighbors,
        "weights": weights,
        "distance": distance_name,
        "p": p,
        "metric": "minkowski",
        "algorithm": "brute",
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
        "classification_report": report,
        "confusion_matrix": matrix,
    }


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
    X_train = X_train.copy()
    X_test = X_test.copy()
    y_train = y_train.copy()
    y_test = y_test.copy()

    X_train = X_train.replace("?", np.nan)
    X_test = X_test.replace("?", np.nan)

    categorical = ["proto", "state", "service"]

    for field in X_train.columns:
        if field not in categorical:
            X_train[field] = pd.to_numeric(
                X_train[field],
                errors="coerce",
            )

            if field in X_test.columns:
                X_test[field] = pd.to_numeric(
                    X_test[field],
                    errors="coerce",
                )

    if method is None or method == "none":
        train_valid = ~X_train.isna().any(axis=1)
        test_valid = ~X_test.isna().any(axis=1)

        X_train = X_train.loc[train_valid]
        y_train = y_train.loc[train_valid]

        X_test = X_test.loc[test_valid]
        y_test = y_test.loc[test_valid]

        return X_train, X_test, y_train, y_test

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
        X_train[field] = X_train[field].astype(object)
        X_test[field] = X_test[field].astype(object)
        X_test.loc[~X_test[field].isin(categories), field] = np.nan
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


def kNeighbors(df: DataFrame, OUTPUT_DIR="../report/outputFiles/kn_neighbors_results/"):

    os.environ["OMP_NUM_THREADS"] = "24"
    os.environ["OPENBLAS_NUM_THREADS"] = "24"
    os.environ["MKL_NUM_THREADS"] = "24"
    os.environ["NUMEXPR_NUM_THREADS"] = "24"

    ROOT_DIR = "../"
    CUR_DIR = ROOT_DIR + "data/"
    OUTPUT_DIR = ROOT_DIR + ""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(CUR_DIR + "networkTraffic.csv", na_values=["?"], low_memory=False)
    df["service"] = df["service"].fillna("unknown")
    df = df.dropna()

    X = df.drop(columns=["id", "attack_cat"])
    y = df["attack_cat"]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_neighbors_values = [1, 3, 5, 7, 9, 11, 15, 20]

    weights_values = ["uniform", "distance"]

    p_values = [1, 2]
    total_tests = len(n_neighbors_values) * len(weights_values) * len(p_values)

    test_number = 0

    results = []

    all_start_time = time.time()

    for p in p_values:
        for weights in weights_values:
            for n_neighbors in n_neighbors_values:

                test_number += 1

                if p == 1:
                    distance_name = "manhattan"
                elif p == 2:
                    distance_name = "euclidean"
                else:
                    distance_name = f"p{p}"

                output_name = (
                    f"k_{n_neighbors}"
                    f"_weights_{weights}"
                    f"_distance_{distance_name}.txt"
                )

                output_path = OUTPUT_DIR + output_name

                print("========================================")

                print(
                    f"[{test_number}/{total_tests}] "
                    f"k={n_neighbors}, "
                    f"weights={weights}, "
                    f"distance={distance_name}"
                )

                model = KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    metric="minkowski",
                    p=p,
                    algorithm="brute",
                    n_jobs=-1,
                )

                fit_start_time = time.time()

                model.fit(X_train, y_train)

                fit_time = time.time() - fit_start_time

                predict_start_time = time.time()

                y_pred = model.predict(X_test)

                predict_time = time.time() - predict_start_time

                test_time = fit_time + predict_time

                accuracy = accuracy_score(y_test, y_pred)

                macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

                weighted_f1 = f1_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )

                total_elapsed = time.time() - all_start_time

                average_time = total_elapsed / test_number

                remaining_tests = total_tests - test_number

                estimated_remaining = average_time * remaining_tests

                print(f"    Accuracy: " f"{accuracy * 100:.2f}%")

                print(f"    Macro F1: " f"{macro_f1:.4f}")

                print(f"    Weighted F1: " f"{weighted_f1:.4f}")

                print(f"    Fit time: " f"{fit_time:.2f}s")

                print(f"    Prediction time: " f"{predict_time:.2f}s")

                print(f"    Test time: " f"{test_time:.2f}s")

                print(f"    Total elapsed: " f"{total_elapsed / 60:.2f} min")

                print(
                    f"    Estimated remaining: " f"{estimated_remaining / 60:.2f} min"
                )

                print()

                with open(output_path, "w") as output_file:

                    print("K-Nearest Neighbours Test", file=output_file)

                    print("=========================", file=output_file)

                    print(f"Test: " f"{test_number}/{total_tests}", file=output_file)

                    print(file=output_file)

                    print("Parameters:", file=output_file)

                    print(f"n_neighbors: " f"{n_neighbors}", file=output_file)

                    print(f"weights: " f"{weights}", file=output_file)

                    print(f"distance: " f"{distance_name}", file=output_file)

                    print(f"p: " f"{p}", file=output_file)

                    print("metric: minkowski", file=output_file)

                    print("algorithm: brute", file=output_file)

                    print("n_jobs: -1", file=output_file)

                    print(file=output_file)

                    print("Results:", file=output_file)

                    print(f"Accuracy: " f"{accuracy * 100:.2f}%", file=output_file)

                    print(f"Macro F1: " f"{macro_f1:.4f}", file=output_file)

                    print(f"Weighted F1: " f"{weighted_f1:.4f}", file=output_file)

                    print(file=output_file)

                    print("Times:", file=output_file)

                    print(f"Fit Time: " f"{fit_time:.2f} seconds", file=output_file)

                    print(
                        f"Prediction Time: " f"{predict_time:.2f} seconds",
                        file=output_file,
                    )

                    print(
                        f"Total Test Time: " f"{test_time:.2f} seconds",
                        file=output_file,
                    )

                    print(
                        f"Total Elapsed Time: " f"{total_elapsed:.2f} seconds",
                        file=output_file,
                    )

                    print(file=output_file)

                    print("Classification Report:", file=output_file)

                    print(
                        classification_report(y_test, y_pred, zero_division=0),
                        file=output_file,
                    )

                results.append(
                    {
                        "n_neighbors": n_neighbors,
                        "weights": weights,
                        "distance": distance_name,
                        "p": p,
                        "accuracy": accuracy,
                        "macro_f1": macro_f1,
                        "weighted_f1": weighted_f1,
                        "fit_time_seconds": fit_time,
                        "prediction_time_seconds": predict_time,
                        "total_time_seconds": test_time,
                    }
                )

                results_df = pd.DataFrame(results)

                results_df = results_df.sort_values(by="macro_f1", ascending=False)

                results_df.to_csv(OUTPUT_DIR + "summary.csv", index=False)

    total_time = time.time() - all_start_time

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(by="macro_f1", ascending=False)

    results_df.to_csv(OUTPUT_DIR + "summary.csv", index=False)

    print("========================================")

    print("Testing complete")

    print("========================================")

    print(f"Total tests: " f"{total_tests}")

    print(f"Total time: " f"{total_time:.2f} seconds")

    print(f"Total time: " f"{total_time / 60:.2f} minutes")

    print(f"Average time per test: " f"{total_time / total_tests:.2f} seconds")

    print()

    print(f"Results saved to: " f"{OUTPUT_DIR}")

    print()

    print("Best 10 models by Macro F1:")

    print(results_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
