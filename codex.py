# codex.py - Baseado EXATAMENTE no code.ipynb original

# Seed value
seed_value = 42

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED'] = str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import auc, roc_curve, accuracy_score, balanced_accuracy_score, f1_score, recall_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
from datetime import datetime

# Obter o diretório onde está o script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Criar pasta reports no diretório do script se não existir
reports_dir = os.path.join(script_dir, "reports")
if not os.path.exists(reports_dir):
    os.makedirs(reports_dir)

# Definir caminhos para os dados
data_dir = os.path.join(script_dir, "data")
models_dir = os.path.join(script_dir, "modelos")

print("📊 Carregando dados...")

# Load Test Data
test = pd.read_parquet(os.path.join(data_dir, "test.parquet"))

y = test["Y"].replace(["Heartbleed", "Infiltration"], "Unknown")
x = test.drop(columns=['Y'])

print("Distribuição y:", y.value_counts())

# Load additional infiltration samples from 2018
infiltration_2018 = pd.read_parquet(os.path.join(data_dir, "infiltration_2018.parquet"))

y_18 = infiltration_2018['y']
x_18 = infiltration_2018.drop(columns=['y'])

print("Distribuição y_18:", y_18.value_counts())

print("🔧 Carregando modelos...")

# Optimized pipelines
with open(os.path.join(models_dir, "stage1_ocsvm.p"), "rb") as f:
    stage1 = pickle.load(f)

with open(os.path.join(models_dir, "stage2_rf.p"), "rb") as f:
    stage2 = pickle.load(f)

# Individual feature scalers and classification models
with open(os.path.join(models_dir, "stage1_ocsvm_model.p"), "rb") as f:
    stage1_model = pickle.load(f)

with open(os.path.join(models_dir, "stage1_ocsvm_scaler.p"), "rb") as f:
    stage1_scaler = pickle.load(f)

with open(os.path.join(models_dir, "stage2_rf_model.p"), "rb") as f:
    stage2_model = pickle.load(f)

with open(os.path.join(models_dir, "stage2_rf_scaler.p"), "rb") as f:
    stage2_scaler = pickle.load(f)

# RF baseline model and feature scaler
with open(os.path.join(models_dir, "baseline_rf.p"), "rb") as f:
    baseline_rf = pickle.load(f)

with open(os.path.join(models_dir, "baseline_rf_scaler.p"), "rb") as f:
    baseline_rf_scaler = pickle.load(f)

print("✅ Modelos carregados")

# Thresholds - EXATAMENTE como no notebook original
tau_b = -0.004199663778210894  # Threshold do paper
tau_m = 0.98
tau_u = 0.007972254569416139

print(f"🎯 Thresholds: tau_b={tau_b}, tau_m={tau_m}, tau_u={tau_u}")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Função hids_predict - EXATAMENTE como no notebook original
def hids_predict(x, tau_b, tau_m, tau_u):
    proba_1 = -stage1.decision_function(x) # invert sign to act as anomaly score 
    pred_1 = np.where(proba_1 < tau_b, "Benign", "Attack").astype(object)
    proba_2 = stage2.predict_proba(x[pred_1 == "Attack"])
    pred_2 = np.where(
        np.max(proba_2, axis=1) > tau_m, 
        stage2.classes_[np.argmax(proba_2, axis=1)], 
        "Unknown")
    proba_3 = proba_1[pred_1 == "Attack"][pred_2 == "Unknown"]
    pred_3 = np.where(proba_3 < tau_u, "Benign", "Unknown")
    pred_1[pred_1 == "Attack"] = pred_2
    pred_1[pred_1 == "Unknown"] = pred_3
    return pred_1

print("⏱️ Executando testes de tempo...")

# Max F-score thresholds
print("Max F-score thresholds...")
start_time = time.time()
for _ in range(3):
    tau_b_temp = -0.0002196942507948895
    tau_m_temp = 0.98
    tau_u_temp = 0.004530129828299084
    y_result = hids_predict(x, tau_b_temp, tau_m_temp, tau_u_temp)
end_time = time.time()
avg_time = (end_time - start_time) / 3

with open(os.path.join(reports_dir, "timing_max_fscore.txt"), "w") as f:
    f.write("=== MÉTRICA DE TEMPO - MAX F-SCORE THRESHOLDS ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tempo médio: {avg_time:.6f} segundos\n")
    f.write(f"Execuções: 3\n")
    f.write(f"Thresholds: tau_b={tau_b_temp}, tau_m={tau_m_temp}, tau_u={tau_u_temp}\n")

print(f"✅ Max F-score: {avg_time:.6f}s")

# Max bACC thresholds
print("Max bACC thresholds...")
start_time = time.time()
for _ in range(3):
    tau_b_temp = -0.004199663778210894
    tau_m_temp = 0.98
    tau_u_temp = 0.007972254569416139
    y_result = hids_predict(x, tau_b_temp, tau_m_temp, tau_u_temp)
end_time = time.time()
avg_time = (end_time - start_time) / 3

with open(os.path.join(reports_dir, "timing_max_bacc.txt"), "w") as f:
    f.write("=== MÉTRICA DE TEMPO - MAX BACC THRESHOLDS ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tempo médio: {avg_time:.6f} segundos\n")
    f.write(f"Execuções: 3\n")
    f.write(f"Thresholds: tau_b={tau_b_temp}, tau_m={tau_m_temp}, tau_u={tau_u_temp}\n")

print(f"✅ Max bACC: {avg_time:.6f}s")

# Best "balanced" thresholds
print("Best balanced thresholds...")
start_time = time.time()
for _ in range(3):
    y_result = hids_predict(x, tau_b, tau_m, tau_u)
end_time = time.time()
avg_time = (end_time - start_time) / 3

with open(os.path.join(reports_dir, "timing_balanced.txt"), "w") as f:
    f.write("=== MÉTRICA DE TEMPO - BALANCED THRESHOLDS ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tempo médio: {avg_time:.6f} segundos\n")
    f.write(f"Execuções: 3\n")
    f.write(f"Thresholds: tau_b={tau_b}, tau_m={tau_m}, tau_u={tau_u}\n")

print(f"✅ Balanced: {avg_time:.6f}s")

# Baseline RF
print("Baseline RF...")
threshold = 0.43
start_time = time.time()
for _ in range(3):
    x_s = baseline_rf_scaler.transform(x)
    y_proba = baseline_rf.predict_proba(x_s)
    y_pred_baseline = np.where(np.max(y_proba, axis=1) > threshold, baseline_rf.classes_[np.argmax(y_proba, axis=1)], 'Unknown')
end_time = time.time()
avg_time = (end_time - start_time) / 3

with open(os.path.join(reports_dir, "timing_baseline_rf.txt"), "w") as f:
    f.write("=== MÉTRICA DE TEMPO - BASELINE RF ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tempo médio: {avg_time:.6f} segundos\n")
    f.write(f"Execuções: 3\n")
    f.write(f"Threshold: {threshold}\n")

print(f"✅ Baseline RF: {avg_time:.6f}s")

print("⏱️ Testando amostras individuais...")

# Single sample timing
sample = np.array(x.values[0]).reshape(1, -1)

# Single sample - Stage1 individual
start_time = time.time()
for _ in range(10):
    x_s = stage1_scaler.transform(sample)
    proba = -stage1_model.decision_function(x_s)
    pred = np.where(proba < tau_b, "Benign", "Attack").astype(object)
end_time = time.time()
avg_time = (end_time - start_time) / 10

with open(os.path.join(reports_dir, "timing_single_stage1_individual.txt"), "w") as f:
    f.write("=== MÉTRICA DE TEMPO - SINGLE SAMPLE STAGE1 INDIVIDUAL ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tempo médio: {avg_time:.6f} segundos\n")
    f.write(f"Execuções: 10\n")

print(f"✅ Single Stage1 Individual: {avg_time:.6f}s")

# Single sample - Stage2 individual
start_time = time.time()
for _ in range(10):
    x_s = stage2_scaler.transform(sample)
    proba = stage2_model.predict_proba(x_s)
    pred_2 = np.where(
        np.max(proba, axis=1) > tau_m, 
        stage2_model.classes_[np.argmax(proba, axis=1)], 
        "Unknown")
end_time = time.time()
avg_time = (end_time - start_time) / 10

with open(os.path.join(reports_dir, "timing_single_stage2_individual.txt"), "w") as f:
    f.write("=== MÉTRICA DE TEMPO - SINGLE SAMPLE STAGE2 INDIVIDUAL ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tempo médio: {avg_time:.6f} segundos\n")
    f.write(f"Execuções: 10\n")

print(f"✅ Single Stage2 Individual: {avg_time:.6f}s")

# Single sample - Pipeline Stage1
start_time = time.time()
for _ in range(10):
    proba_1 = -stage1.decision_function(sample) # invert sign to act as anomaly score 
    pred_1 = np.where(proba_1 < tau_b, "Benign", "Attack").astype(object)
end_time = time.time()
avg_time = (end_time - start_time) / 10

with open(os.path.join(reports_dir, "timing_single_stage1_pipeline.txt"), "w") as f:
    f.write("=== MÉTRICA DE TEMPO - SINGLE SAMPLE STAGE1 PIPELINE ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tempo médio: {avg_time:.6f} segundos\n")
    f.write(f"Execuções: 10\n")

print(f"✅ Single Stage1 Pipeline: {avg_time:.6f}s")

# Single sample - Pipeline Stage2
start_time = time.time()
for _ in range(10):
    proba_2 = stage2.predict_proba(sample)
    pred_2 = np.where(
        np.max(proba_2, axis=1) > tau_m, 
        stage2.classes_[np.argmax(proba_2, axis=1)], 
        "Unknown")
end_time = time.time()
avg_time = (end_time - start_time) / 10

with open(os.path.join(reports_dir, "timing_single_stage2_pipeline.txt"), "w") as f:
    f.write("=== MÉTRICA DE TEMPO - SINGLE SAMPLE STAGE2 PIPELINE ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tempo médio: {avg_time:.6f} segundos\n")
    f.write(f"Execuções: 10\n")

print(f"✅ Single Stage2 Pipeline: {avg_time:.6f}s")

print("🚀 Avaliando Multi-Stage Model...")

# DEBUG: Verificar distribuição dos scores
proba_1 = -stage1.decision_function(x) # invert sign to act as anomaly score 
print(f"🔍 DEBUG Stage 1:")
print(f"   tau_b = {tau_b}")
print(f"   proba_1 min: {np.min(proba_1)}")
print(f"   proba_1 max: {np.max(proba_1)}")
print(f"   proba_1 mean: {np.mean(proba_1)}")
print(f"   proba_1 std: {np.std(proba_1)}")
print(f"   Amostras < tau_b: {np.sum(proba_1 < tau_b)}")
print(f"   Amostras >= tau_b: {np.sum(proba_1 >= tau_b)}")

# Stage 1: Binary Detection
pred_1 = np.where(proba_1 < tau_b, "Benign", "Attack").astype(object)

unique_values, counts = np.unique(pred_1, return_counts=True)
with open(os.path.join(reports_dir, "stage1_binary_detection.txt"), "w") as f:
    f.write("=== STAGE 1: BINARY DETECTION ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("Distribuição de predições:\n")
    for value, count in zip(unique_values, counts):
        f.write(f"{value}: {count}\n")

print(f"✅ Stage 1: {dict(zip(unique_values, counts))}")

# Timing Stage 1
start_time = time.time()
for _ in range(3):
    proba_1 = -stage1.decision_function(x) # invert sign to act as anomaly score 
    pred_1 = np.where(proba_1 < tau_b, "Benign", "Attack").astype(object)
end_time = time.time()
avg_time = (end_time - start_time) / 3

with open(os.path.join(reports_dir, "timing_stage1_binary.txt"), "w") as f:
    f.write("=== MÉTRICA DE TEMPO - STAGE 1 BINARY DETECTION ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tempo médio: {avg_time:.6f} segundos\n")
    f.write(f"Execuções: 3\n")

print(f"✅ Timing Stage 1: {avg_time:.6f}s")

# Stage 2: Multi-Class Classification
proba_2 = stage2.predict_proba(x[pred_1 == "Attack"])
pred_2 = np.where(
    np.max(proba_2, axis=1) > tau_m, 
    stage2.classes_[np.argmax(proba_2, axis=1)], 
    "Unknown")

unique_values, counts = np.unique(pred_2, return_counts=True)
with open(os.path.join(reports_dir, "stage2_multiclass.txt"), "w") as f:
    f.write("=== STAGE 2: MULTI-CLASS CLASSIFICATION ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("Distribuição de predições:\n")
    for value, count in zip(unique_values, counts):
        f.write(f"{value}: {count}\n")

print(f"✅ Stage 2: {dict(zip(unique_values, counts))}")

# Timing Stage 2
start_time = time.time()
for _ in range(3):
    proba_2 = stage2.predict_proba(x[pred_1 == "Attack"])
    pred_2 = np.where(
        np.max(proba_2, axis=1) > tau_m, 
        stage2.classes_[np.argmax(proba_2, axis=1)], 
        "Unknown")
end_time = time.time()
avg_time = (end_time - start_time) / 3

with open(os.path.join(reports_dir, "timing_stage2_multiclass.txt"), "w") as f:
    f.write("=== MÉTRICA DE TEMPO - STAGE 2 MULTI-CLASS ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tempo médio: {avg_time:.6f} segundos\n")
    f.write(f"Execuções: 3\n")

print(f"✅ Timing Stage 2: {avg_time:.6f}s")

# Extension Stage: Zero-Day Detection
proba_3 = proba_1[pred_1 == "Attack"][pred_2 == "Unknown"]
pred_3 = np.where(proba_3 < tau_u, "Benign", "Unknown")

unique_values, counts = np.unique(pred_3, return_counts=True)
with open(os.path.join(reports_dir, "stage3_zeroday.txt"), "w") as f:
    f.write("=== STAGE 3: ZERO-DAY DETECTION ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("Distribuição de predições:\n")
    for value, count in zip(unique_values, counts):
        f.write(f"{value}: {count}\n")

print(f"✅ Stage 3: {dict(zip(unique_values, counts))}")

# Combine stages
y_pred = pred_1.copy()
y_pred[y_pred == "Attack"] = pred_2
y_pred[y_pred == "Unknown"] = pred_3

unique_values, counts = np.unique(y_pred, return_counts=True)
with open(os.path.join(reports_dir, "combined_stages.txt"), "w") as f:
    f.write("=== COMBINED STAGES ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("Distribuição final de predições:\n")
    for value, count in zip(unique_values, counts):
        f.write(f"{value}: {count}\n")

print(f"✅ Combined: {dict(zip(unique_values, counts))}")

print("📊 Gerando visualizações...")

# Statistics and Visualizations
def plot_confusion_matrix(y_true, y_pred, figsize=(7,7), cmap="Blues", values=[-1, 1], labels=["Attack", "Benign"], title="", ax=None, metrics=False):
    cm = confusion_matrix(y_true, y_pred, labels=values)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float)
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            annot[i, j] = '%.1f%%\n%d' % (p * 100, c)
    cm_perc = pd.DataFrame(cm_perc, index=labels, columns=labels)
    cm_perc.index.name = 'Actual'
    cm_perc.columns.name = 'Predicted'
    if ax == None:
        fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_perc, cmap=cmap, annot=annot, fmt='', ax=ax, vmin=0, vmax=1)
    if title != "":
        ax.set_title(title)

classes = ['Benign', '(D)DOS', 'Botnet', 'Brute Force', 'Port Scan', 'Web Attack', 'Unknown']
plot_confusion_matrix(y, y_pred, values=classes, labels=classes, metrics=True)

plt.savefig(os.path.join(reports_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Confusion matrix salva")

# Classification report
report = classification_report(y, y_pred, digits=4)
with open(os.path.join(reports_dir, "classification_report.txt"), "w") as f:
    f.write("=== RELATÓRIO DE CLASSIFICAÇÃO ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(report)

print("\n🔬 Teste de robustez com dados CIC-IDS-2018...")

# Robustness test - Infiltration 2018
y_robust = hids_predict(x_18, tau_b, tau_m, tau_u)

unique_values, counts = np.unique(y_robust, return_counts=True)
with open(os.path.join(reports_dir, "infiltration_2018_hids.txt"), "w") as f:
    f.write("=== INFILTRAÇÃO 2018 - HIDS PRINCIPAL ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Thresholds: tau_b={tau_b}, tau_m={tau_m}, tau_u={tau_u}\n\n")
    f.write("Distribuição de predições:\n")
    for value, count in zip(unique_values, counts):
        f.write(f"{value}: {count}\n")

print(f"✅ Infiltration 2018 HIDS: {dict(zip(unique_values, counts))}")

# Baseline test on infiltration 2018
x_s = baseline_rf_scaler.transform(x_18)
y_proba = baseline_rf.predict_proba(x_s)
y_pred_baseline = np.where(np.max(y_proba, axis=1) > 0.43, baseline_rf.classes_[np.argmax(y_proba, axis=1)], 'Unknown')

unique_values, counts = np.unique(y_pred_baseline, return_counts=True)
with open(os.path.join(reports_dir, "infiltration_2018_baseline.txt"), "w") as f:
    f.write("=== INFILTRAÇÃO 2018 - BASELINE RF ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Threshold: 0.43\n\n")
    f.write("Distribuição de predições:\n")
    for value, count in zip(unique_values, counts):
        f.write(f"{value}: {count}\n")

print(f"✅ Infiltration 2018 Baseline: {dict(zip(unique_values, counts))}")

print(f"\n🎉 Todos os resultados foram salvos na pasta '{reports_dir}'!")
print("✅ Execução completa seguindo exatamente o notebook code.ipynb")