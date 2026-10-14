# 实验一 新闻文本分类

使用 TF-IDF 比较朴素贝叶斯、逻辑回归和线性 SVM，并尝试词表调整、文本清洗和词/字符特征融合。

## 运行

本次使用 Python 3.14.6。将老师提供的数据放在仓库根目录：

```text
data-Project1/
├── train_data.csv
└── test_data_unlabeled.csv
```

在仓库根目录运行：

```powershell
python -m pip install -r requirements.txt
python -u run_experiment.py --data-dir ./data-Project1
```

数据不随仓库上传。请保留 `--data-dir` 参数；默认输出到 `results`，重跑会覆盖同名结果。

## 实验设置与结果

- 随机种子为 42，训练/验证/保留集分别为 5156/1106/1106 条。
- NB 比较 `alpha=0.1/0.5/1`；LR、SVM 比较 `C=0.1/1/10`。
- 按验证宏 F1 选择方案；TF-IDF 只在训练子集拟合，选好方案后再评估保留集。
- 最优方案为词/字符 TF-IDF 融合与 `LinearSVC(C=1)`：验证准确率 **93.76%**，保留集准确率 **93.40%**、宏 F1 **0.9344**。

融合方案只比最佳纯词方案多分对一条验证样本，训练成本更高，不能据此认定它有稳定优势。最终使用全部 7368 条带标签数据重新训练；无标签测试集的真实准确率未知。

## 输出

`results/experiments.csv` 保存参数和特征对比，`holdout_metrics.json` 保存独立评估，`figures/` 保存图表。

预测文件为 `results/predictions.csv`，共2457行，一列整数、无表头和索引。按老师通知，提交时复制并改名为 **`prediction.csv`**。
