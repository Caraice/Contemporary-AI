"""实验一：比较文本分类方法，选好方案后生成测试集预测。"""
import os

# 让数值计算库使用一个线程，方便比较各方案的耗时。
# 这些设置需要放在 numpy 导入前。
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from preprocessing import clean_metadata

ROOT = Path(__file__).resolve().parent
SEED = 42


def save_json(path, obj):
    """把实验记录存成 JSON，中文直接保留，方便以后查看。"""
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def features(name):
    """按方案创建文本特征工具；词表只从本次训练数据中学习。"""
    settings = dict(max_features=5000, dtype=np.float64)
    if name != "baseline":
        # 扩大词表，同时去掉只出现过一篇的词。
        settings.update(max_features=20000, min_df=2)
    if name in ("sublinear", "bigram", "stopwords", "clean", "hybrid"):
        # 用对数处理词频，减少重复词对结果的影响。
        settings.update(sublinear_tf=True)
    if name in ("bigram", "stopwords", "clean", "hybrid"):
        # 除了单个词，也加入相邻两个词组成的短语。
        settings.update(ngram_range=(1, 2))
    if name == "stopwords":
        settings.update(stop_words="english")
    if name == "clean":
        settings.update(preprocessor=clean_metadata)
    word = TfidfVectorizer(**settings)
    if name == "hybrid":
        # 把词特征和字符片段拼在一起。字符特征乘 0.5，权重没有额外调参。
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                               min_df=3, max_features=30000, sublinear_tf=True)
        return FeatureUnion([("word", word), ("char", char)],
                            transformer_weights={"word": 1.0, "char": 0.5})
    return word


def classifier(name, value):
    """创建分类器。NB 的参数是 alpha，LR 和 SVM 的参数是 C。"""
    if name == "NB":
        return MultinomialNB(alpha=value)
    if name == "LR":
        return LogisticRegression(C=value, solver="lbfgs", max_iter=2000, random_state=SEED)
    if name == "SVM":
        # 使用线性 SVM。它与示例的 SVC(kernel="linear") 实现不同，训练较快。
        return LinearSVC(C=value, dual="auto", max_iter=10000, random_state=SEED)
    raise ValueError(name)


def metrics(y, prediction):
    """同时看整体准确率，以及把十个类别同等对待的宏 F1。"""
    return dict(accuracy=float(accuracy_score(y, prediction)),
                macro_f1=float(f1_score(y, prediction, average="macro")))


def complexity(model):
    """统计分类器保存了多少个数值参数，不包含词表和 IDF。"""
    clf = model.named_steps["classifier"]
    if isinstance(clf, MultinomialNB):
        return int(clf.feature_log_prob_.size + clf.class_log_prior_.size)
    return int(clf.coef_.size + clf.intercept_.size)


def fit_one(feature_name, algorithm, value, train_text, train_y, val_text, val_y):
    """训练一组配置，然后在同一份验证集上比较效果。"""
    # 把特征提取和分类器放在一起，预测时会沿用训练好的词表。
    model = Pipeline([("features", features(feature_name)),
                      ("classifier", classifier(algorithm, value))])
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        # fit 内部先学习 TF-IDF，再训练分类器；这里没有使用验证集。
        model.fit(train_text, train_y)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    prediction = model.predict(val_text)
    predict_seconds = time.perf_counter() - start
    row = dict(feature=feature_name, algorithm=algorithm, parameter=value,
               train_accuracy=accuracy_score(train_y, model.predict(train_text)),
               **metrics(val_y, prediction), fit_seconds=fit_seconds,
               predict_seconds=predict_seconds, parameters=complexity(model),
               n_features=model.named_steps["classifier"].n_features_in_,
               warnings=" | ".join(str(w.message) for w in captured))
    print(f"{feature_name:10s} {algorithm:3s} {value:5g} "
          f"acc={row['accuracy']:.4f} F1={row['macro_f1']:.4f} fit={fit_seconds:.2f}s", flush=True)
    return model, row, prediction


def bootstrap(y, pred):
    """重复抽样估计指标的大致波动范围，这一步不重新训练模型。"""
    generator = np.random.default_rng(SEED)
    accuracies = []
    f1s = []
    for _ in range(1000):
        idx = generator.integers(0, len(y), len(y))
        accuracies.append(accuracy_score(y[idx], pred[idx]))
        f1s.append(f1_score(y[idx], pred[idx], average="macro", labels=np.arange(10), zero_division=0))
    return dict(accuracy_95ci=np.quantile(accuracies, [.025, .975]).tolist(),
                macro_f1_95ci=np.quantile(f1s, [.025, .975]).tolist(),
                method="1000 paired sample-level bootstrap resamples; fixed model; seed 42")


def plot_results(out, results, y, pred, learning):
    """保存四张图，分别看参数、特征、错分类别和训练数据量。"""
    fig, ax = plt.subplots(figsize=(9, 4))
    baseline = results[(results.feature == "baseline") & (results.stage == "algorithm")]
    for name, group in baseline.groupby("algorithm"):
        ax.plot(group.parameter, group.macro_f1, "o-", label=name)
    ax.set(xscale="log", xlabel="alpha (NB) / C (LR, SVM)", ylabel="Validation macro F1",
           title="Hyperparameter comparison: same 5,000-feature TF-IDF")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "figures/parameter_comparison.png", dpi=180)
    plt.close(fig)
    ablation = results[results.stage == "ablation"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(ablation.feature, ablation.macro_f1)
    axes[0].set(ylabel="Validation macro F1", ylim=(max(0, ablation.macro_f1.min()-.04), 1))
    axes[1].bar(ablation.feature, ablation.fit_seconds)
    axes[1].set(ylabel="Fit time (seconds)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=35)
    fig.suptitle("Feature comparison: same classifier and hyperparameter")
    fig.tight_layout()
    fig.savefig(out / "figures/feature_ablation.png", dpi=180)
    plt.close(fig)
    cm = confusion_matrix(y, pred, labels=np.arange(10))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    ax.set(xticks=np.arange(10), yticks=np.arange(10), xlabel="Predicted label",
           ylabel="True label", title="Locked model: held-out confusion matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "figures/confusion_matrix.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(learning.n_train, learning.train_accuracy, "o-", label="Training accuracy")
    ax.plot(learning.n_train, learning.accuracy, "o-", label="Validation accuracy")
    ax.set(xlabel="Training samples", ylabel="Accuracy", title="Learning curve (TF-IDF refitted per subset)")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "figures/learning_curve.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT.parent / "data-Project1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    train_file = args.data_dir / "train_data.csv"
    test_file = args.data_dir / "test_data_unlabeled.csv"
    df, test = pd.read_csv(train_file), pd.read_csv(test_file)
    # 先检查数据，避免列名或缺失值问题影响后面的实验。
    if list(df.columns) != ["text", "target"] or list(test.columns) != ["text"]:
        raise ValueError("Unexpected input columns")
    if df.isna().any().any() or test.isna().any().any():
        raise ValueError("Missing values must be handled explicitly")
    if sorted(df.target.unique().tolist()) != list(range(10)):
        raise ValueError("Expected integer labels 0..9")
    if df.text.duplicated().any():
        raise ValueError("Exact duplicates require a grouped split")
    text, y = df.text.to_numpy(), df.target.to_numpy(dtype=int)
    indices = np.arange(len(df))
    # 按类别比例划分：约 70% 用来训练，15% 调参，15% 留到最后评估。
    development, holdout = train_test_split(indices, test_size=.15, stratify=y, random_state=SEED)
    train, val = train_test_split(development, test_size=.15/.85,
                                  stratify=y[development], random_state=SEED)
    assert not (set(train) & set(val) or set(train) & set(holdout) or set(val) & set(holdout))
    split = np.full(len(df), "", dtype=object)
    split[train], split[val], split[holdout] = "train", "validation", "holdout"
    pd.DataFrame({"row_index": indices, "split": split, "target": y}).to_csv(out / "split_indices.csv", index=False)
    environment = dict(seed=SEED, python=sys.version, platform=platform.platform(),
                       processor=platform.processor(), cpu_count=os.cpu_count(),
                       versions={"numpy": np.__version__, "pandas": pd.__version__,
                                 "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
                                 "matplotlib": matplotlib.__version__, "joblib": joblib.__version__},
                       data_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in [train_file, test_file]},
                       split_counts={"train": len(train), "validation": len(val), "holdout": len(holdout)},
                       selection_rule="highest validation macro F1, then accuracy, then fewer model parameters",
                       class_counts=df.target.value_counts().sort_index().to_dict(),
                       test_samples=len(test), missing_values=0, exact_train_duplicates=0,
                       threads=1)
    save_json(out / "environment.json", environment)
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(train), 1)), y[train])
    save_json(out / "dummy_baseline.json", metrics(y[val], dummy.predict(np.zeros((len(val), 1)))))
    print(f"Split: train={len(train)} validation={len(val)} holdout={len(holdout)}", flush=True)
    rows = []
    candidates = []
    # 第一轮只换算法和参数，所有模型使用相同的 5000 维词特征。
    for algorithm, values in [("NB", [.1, .5, 1]), ("LR", [.1, 1, 10]), ("SVM", [.1, 1, 10])]:
        for value in values:
            model, row, _ = fit_one("baseline", algorithm, value, text[train], y[train], text[val], y[val])
            row["stage"] = "algorithm"
            rows.append(row)
            candidates.append((model, row))
            pd.DataFrame(rows).to_csv(out / "experiments.csv", index=False)
    # 先看宏 F1，再看准确率；相同的时候优先参数少的模型。
    def ranking(item):
        row = item[1]
        return row["macro_f1"], row["accuracy"], -row["parameters"]

    _, base_row = max(candidates, key=ranking)
    algorithm, value = base_row["algorithm"], base_row["parameter"]
    copy = dict(base_row, stage="ablation")
    rows.append(copy)
    # 第二轮固定第一轮的最佳算法和参数，再比较不同文本特征。
    # stopwords、clean 和 hybrid 都从 bigram 方案出发，分别测试。
    for feature_name in ["vocab", "sublinear", "bigram", "stopwords", "clean", "hybrid"]:
        model, row, _ = fit_one(feature_name, algorithm, value, text[train], y[train], text[val], y[val])
        row["stage"] = "ablation"
        rows.append(row)
        candidates.append((model, row))
        pd.DataFrame(rows).to_csv(out / "experiments.csv", index=False)
    winner, selected = max(candidates, key=ranking)
    # 到这里选好方案。之后再看保留集，不根据保留集结果修改参数。
    save_json(out / "selected_config.json", selected)
    print("LOCKED CONFIG: " + json.dumps(selected), flush=True)
    val_pred = winner.predict(text[val])
    hold_pred = winner.predict(text[holdout])
    hold_metrics = dict(**metrics(y[holdout], hold_pred), **bootstrap(y[holdout], hold_pred))
    save_json(out / "holdout_metrics.json", hold_metrics)
    for subset, idx, pred in [("validation", val, val_pred), ("holdout", holdout, hold_pred)]:
        report = classification_report(y[idx], pred, labels=np.arange(10), output_dict=True, zero_division=0)
        save_json(out / f"{subset}_classification_report.json", report)
        pd.DataFrame(report).T.to_csv(out / f"{subset}_classification_report.csv")
        pd.DataFrame({"row_index": idx, "target": y[idx], "prediction": pred}).to_csv(out / f"{subset}_predictions.csv", index=False)
    pd.DataFrame(confusion_matrix(y[holdout], hold_pred)).to_csv(out / "holdout_confusion_matrix.csv", index=False)
    # 保存错分帖子的前 1500 个字符，便于结合内容分析原因。
    errors = []
    for idx, actual, guess in zip(holdout, y[holdout], hold_pred):
        if actual != guess:
            errors.append((int(idx), int(actual), int(guess), text[idx][:1500]))
    pd.DataFrame(errors, columns=["row_index", "target", "prediction", "text_excerpt"]).to_csv(out / "error_examples.csv", index=False)
    joblib.dump(winner, out / "evaluation_model.joblib", compress=3)
    learning_rows = []
    # 使用不同数量的训练样本，检查数据增加后效果是否提高。
    for fraction in [.25, .5, .75, 1.0]:
        if fraction == 1:
            subset = train
        else:
            subset, _ = train_test_split(
                train, train_size=fraction, stratify=y[train], random_state=SEED
            )
        if fraction == 1:
            row = dict(selected)
        else:
            _, row, _ = fit_one(selected["feature"], selected["algorithm"], selected["parameter"],
                                 text[subset], y[subset], text[val], y[val])
        row.update(fraction=fraction, n_train=len(subset))
        learning_rows.append(row)
    learning = pd.DataFrame(learning_rows)
    learning.to_csv(out / "learning_curve.csv", index=False)
    results = pd.DataFrame(rows)
    plot_results(out, results, y[holdout], hold_pred, learning)
    # 评估完成后，用全部带标签数据重新训练，尽量利用已有样本。
    # 最后预测没有标签的测试集；测试集准确率无法直接计算。
    final_model = Pipeline([("features", features(selected["feature"])),
                            ("classifier", classifier(selected["algorithm"], selected["parameter"]))])
    start = time.perf_counter()
    final_model.fit(text, y)
    prediction = final_model.predict(test.text.to_numpy())
    final_seconds = time.perf_counter() - start
    if len(prediction) != len(test) or not np.isin(prediction, np.arange(10)).all():
        raise ValueError("Invalid submission predictions")
    pd.DataFrame(prediction).to_csv(out / "predictions.csv", header=False, index=False)
    joblib.dump(final_model, out / "final_model.joblib", compress=3)
    save_json(out / "final_training.json", dict(seconds=final_seconds, n_train=len(df),
              parameters=complexity(final_model), n_features=final_model.named_steps["classifier"].n_features_in_,
              model_bytes=(out / "final_model.joblib").stat().st_size,
              prediction_counts=pd.Series(prediction).value_counts().sort_index().to_dict(),
              train_test_exact_text_overlap=len(set(df.text) & set(test.text))))
    print("HOLDOUT: " + json.dumps(hold_metrics), flush=True)
    print(f"DONE: {out / 'predictions.csv'} ({len(prediction)} rows)", flush=True)


if __name__ == "__main__":
    main()
