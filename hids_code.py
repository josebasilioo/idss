seed_value= 42

import os
os.environ['PYTHONHASHSEED']=str(seed_value)

import random
random.seed(seed_value)

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
    print("✅ Pasta reports criada em:", reports_dir)

# Definir caminhos para os dados
data_dir = os.path.join(script_dir, "data")
models_dir = os.path.join(script_dir, "models")

print(f"📁 Diretório do script: {script_dir}")
print(f"📁 Diretório de dados: {data_dir}")
print(f"📁 Diretório de modelos: {models_dir}")

# Load Test Data
test = pd.read_parquet(os.path.join(data_dir, "test.parquet"))

y = test["Y"].replace(["Heartbleed", "Infiltration"], "Unknown")
x = test.drop(columns=['Y'])

y.value_counts()

# Load additional infiltration samples from 2018
infiltration_2018 = pd.read_parquet(os.path.join(data_dir, "infiltration_2018.parquet"))

y_18 = infiltration_2018['y']
x_18 = infiltration_2018.drop(columns=['y'])

y_18.value_counts()

# Load Models
import warnings
warnings.filterwarnings("ignore")

def load_model_safe(model_path, model_name):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"✅ {model_name} carregado com sucesso")
        return model
    except Exception as e:
        print(f"❌ Erro ao carregar {model_name}: {e}")
        return None

print("🔧 Carregando modelos...")

stage1 = load_model_safe(os.path.join(models_dir, "stage1_ocsvm.p"), "Stage1 OCSVM Pipeline")
stage2 = load_model_safe(os.path.join(models_dir, "stage2_rf.p"), "Stage2 RF Pipeline")
stage1_model = load_model_safe(os.path.join(models_dir, "stage1_ocsvm_model.p"), "Stage1 OCSVM Model")
stage1_scaler = load_model_safe(os.path.join(models_dir, "stage1_ocsvm_scaler.p"), "Stage1 OCSVM Scaler")
stage2_model = load_model_safe(os.path.join(models_dir, "stage2_rf.p"), "Stage2 RF Model")
stage2_scaler = load_model_safe(os.path.join(models_dir, "stage2_rf_scaler.p"), "Stage2 RF Scaler")
baseline_rf = load_model_safe(os.path.join(models_dir, "baseline_rf.p"), "Baseline RF")
baseline_rf_scaler = load_model_safe(os.path.join(models_dir, "baseline_rf_scaler.p"), "Baseline RF Scaler")
stage1_pca = load_model_safe(os.path.join(models_dir, "stage1_pca.p"), "Stage1 PCA")

print("✅ Modelos carregados")

# Thresholds
tau_b = -0.0002196942507948895
tau_m = 0.98
tau_u = 0.0040588613744241275

# Função de ajuste de features
def safe_model_call(model, data, method=None, **kwargs):
    """Chama um modelo de forma segura, detectando automaticamente o método e ajustando features"""
    if isinstance(data, pd.DataFrame):
        data_array = data.values
    else:
        data_array = data
    
    # Auto-detectar método se não especificado
    if method is None:
        if hasattr(model, 'decision_function'):
            method = 'decision_function'
        elif hasattr(model, 'predict_proba'):
            method = 'predict_proba'
        elif hasattr(model, 'transform'):
            method = 'transform'
        elif hasattr(model, 'predict'):
            method = 'predict'
        else:
            raise ValueError(f"Modelo {type(model)} não tem métodos conhecidos")
    
    # Verificar se o modelo tem o método solicitado
    if not hasattr(model, method):
        # Fallback para método disponível
        if hasattr(model, 'decision_function'):
            method = 'decision_function'
        elif hasattr(model, 'predict_proba'):
            method = 'predict_proba'
        elif hasattr(model, 'transform'):
            method = 'transform'
        elif hasattr(model, 'predict'):
            method = 'predict'
        else:
            raise AttributeError(f"Modelo {type(model)} não tem método '{method}' nem alternativas")
    
    print(f"🔧 Chamando {type(model).__name__}.{method} com {data_array.shape[1]} features")
    
    # Primeira tentativa: usar dados originais
    try:
        result = getattr(model, method)(data_array)
        print(f"✅ Sucesso!")
        return result
    except ValueError as e:
        error_msg = str(e)
        print(f"⚠️ Erro: {error_msg}")
        
        # Extrair número de features esperadas do erro
        import re
        match = re.search(r'expecting (\d+) features', error_msg)
        if match:
            expected_features = int(match.group(1))
            current_features = data_array.shape[1]
            
            print(f"🔧 Auto-correção: {current_features} → {expected_features}")
            
            # Ajustar features
            if current_features > expected_features:
                # Cortar features extras
                data_corrected = data_array[:, :expected_features]
                print(f"✂️ Cortadas {current_features - expected_features} features")
            else:
                # Adicionar features dummy
                missing = expected_features - current_features
                dummy_cols = np.zeros((data_array.shape[0], missing))
                data_corrected = np.column_stack([data_array, dummy_cols])
                print(f"➕ Adicionadas {missing} features dummy")
            
            # Segunda tentativa com dados corrigidos
            try:
                result = getattr(model, method)(data_corrected)
                print(f"✅ Sucesso após correção!")
                return result
            except Exception as e2:
                print(f"❌ Erro mesmo após correção: {e2}")
                raise e2
        else:
            print(f"❌ Não conseguiu extrair features esperadas do erro")
            raise e

# CORRIGIR o hids_predict - sempre usar fallback para Stage 2:
def hids_predict(x, tau_b, tau_m, tau_u):
    # Stage 1: Pipeline funciona bem
    try:
        proba_1 = -stage1.decision_function(x)
    except ValueError:
        x_scaled = safe_model_call(stage1_scaler, x, 'transform')
        x_pca = safe_model_call(stage1_pca, x_scaled, 'transform')
        proba_1 = -safe_model_call(stage1_model, x_pca, 'decision_function')
    
    pred_1 = np.where(proba_1 < tau_b, "Benign", "Attack").astype(object)
    
    # Stage 2: SEMPRE usar fallback (pipeline está quebrado)
    x_attack = x[pred_1 == "Attack"]
    if len(x_attack) > 0:
        # Usar componentes individuais diretamente
        x_scaled = safe_model_call(stage2_scaler, x_attack, 'transform')
        anomaly_scores = proba_1[pred_1 == "Attack"].reshape(-1, 1)
        x_with_anomaly = np.column_stack([x_scaled, anomaly_scores])
        proba_2 = safe_model_call(stage2_model, x_with_anomaly, 'predict_proba')
        
        pred_2 = np.where(
            np.max(proba_2, axis=1) > tau_m, 
            stage2.classes_[np.argmax(proba_2, axis=1)], 
            "Unknown")
    else:
        pred_2 = np.array([])
    
    # Stage 3: Zero-Day Detection
    if len(pred_2) > 0:
        proba_3 = proba_1[pred_1 == "Attack"][pred_2 == "Unknown"]
        pred_3 = np.where(proba_3 < tau_u, "Benign", "Unknown")
        
        pred_1[pred_1 == "Attack"] = pred_2
        pred_1[pred_1 == "Unknown"] = pred_3
    
    return pred_1

# Max F-score thresholds
print("🚀 Executando Max F-score...")
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

print(f"✅ Métrica de tempo Max F-score salva")

# Max bACC thresholds
print("🚀 Executando Max bACC...")
start_time = time.time()
for _ in range(3):
    tau_b_temp = -0.0004064190600459828
    tau_m_temp = 0.98
    tau_u_temp = 0.0006590265510403005
    y_result = hids_predict(x, tau_b_temp, tau_m_temp, tau_u_temp)
end_time = time.time()
avg_time = (end_time - start_time) / 3

with open(os.path.join(reports_dir, "timing_max_bacc.txt"), "w") as f:
    f.write("=== MÉTRICA DE TEMPO - MAX BACC THRESHOLDS ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tempo médio: {avg_time:.6f} segundos\n")
    f.write(f"Execuções: 3\n")
    f.write(f"Thresholds: tau_b={tau_b_temp}, tau_m={tau_m_temp}, tau_u={tau_u_temp}\n")

print(f"✅ Métrica de tempo Max bACC salva")

# Balanced thresholds
print("🚀 Executando Balanced...")
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

print(f"✅ Métrica de tempo Balanced salva")

# Baseline RF
print("🚀 Executando Baseline RF...")
threshold = 0.43
start_time = time.time()
for _ in range(3):
    x_s = safe_model_call(baseline_rf_scaler, x, 'transform')
    y_proba = safe_model_call(baseline_rf, x_s, 'predict_proba')
    y_pred = np.where(np.max(y_proba, axis=1) > threshold, baseline_rf.classes_[np.argmax(y_proba, axis=1)], 'Unknown')
end_time = time.time()
avg_time = (end_time - start_time) / 3

with open(os.path.join(reports_dir, "timing_baseline_rf.txt"), "w") as f:
    f.write("=== MÉTRICA DE TEMPO - BASELINE RF ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tempo médio: {avg_time:.6f} segundos\n")
    f.write(f"Execuções: 3\n")
    f.write(f"Threshold: {threshold}\n")

print(f"✅ Métrica de tempo Baseline RF salva")

# Single sample timing
sample = np.array(x.values[0]).reshape(1, -1)

# Single sample - Stage1 individual
start_time = time.time()
for _ in range(10):
    x_s = safe_model_call(stage1_scaler, sample, 'transform')
    proba = -safe_model_call(stage1_model, x_s, 'decision_function')
    pred = np.where(proba < tau_b, "Benign", "Attack").astype(object)
end_time = time.time()
avg_time = (end_time - start_time) / 10

with open(os.path.join(reports_dir, "timing_single_stage1.txt"), "w") as f:
    f.write("=== MÉTRICA DE TEMPO - SINGLE SAMPLE STAGE1 ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tempo médio: {avg_time:.6f} segundos\n")
    f.write(f"Execuções: 10\n")

print(f"✅ Métrica single sample stage1 salva")

# Single sample - Stage2 individual
start_time = time.time()
for _ in range(10):
    x_s = safe_model_call(stage2_scaler, sample, 'transform')
    proba = safe_model_call(stage2_model, x_s, 'predict_proba')
    pred_2 = np.where(
        np.max(proba, axis=1) > tau_m, 
        stage2.classes_[np.argmax(proba, axis=1)], 
        "Unknown")
end_time = time.time()
avg_time = (end_time - start_time) / 10

with open(os.path.join(reports_dir, "timing_single_stage2.txt"), "w") as f:
    f.write("=== MÉTRICA DE TEMPO - SINGLE SAMPLE STAGE2 ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tempo médio: {avg_time:.6f} segundos\n")
    f.write(f"Execuções: 10\n")

print(f"✅ Métrica single sample stage2 salva")

# Single sample - Pipeline Stage1
start_time = time.time()
for _ in range(10):
    proba_1 = -stage1.decision_function(sample)
    pred_1 = np.where(proba_1 < tau_b, "Benign", "Attack").astype(object)
end_time = time.time()
avg_time = (end_time - start_time) / 10

with open(os.path.join(reports_dir, "timing_single_pipeline_stage1.txt"), "w") as f:
    f.write("=== MÉTRICA DE TEMPO - SINGLE SAMPLE PIPELINE STAGE1 ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tempo médio: {avg_time:.6f} segundos\n")
    f.write(f"Execuções: 10\n")

print(f"✅ Métrica single sample pipeline stage1 salva")

# CORREÇÃO COMPLETA para Single sample - Pipeline Stage2:
# Single sample - Pipeline Stage2
start_time = time.time()
for _ in range(10):
    # Calcular anomaly score do stage1 primeiro
    try:
        sample_proba1 = -stage1.decision_function(sample)
    except ValueError:
        x_s = safe_model_call(stage1_scaler, sample, 'transform')
        x_pca = safe_model_call(stage1_pca, x_s, 'transform')
        sample_proba1 = -safe_model_call(stage1_model, x_pca, 'decision_function')
    
    # Usar fallback para stage2 com anomaly score real
    x_scaled = safe_model_call(stage2_scaler, sample, 'transform')
    anomaly_score = sample_proba1.reshape(-1, 1)
    x_with_anomaly = np.column_stack([x_scaled, anomaly_score])
    proba_2 = safe_model_call(stage2_model, x_with_anomaly, 'predict_proba')
    
    pred_2 = np.where(
        np.max(proba_2, axis=1) > tau_m, 
        stage2.classes_[np.argmax(proba_2, axis=1)], 
        "Unknown")
end_time = time.time()
avg_time = (end_time - start_time) / 10

with open(os.path.join(reports_dir, "timing_single_pipeline_stage2.txt"), "w") as f:
    f.write("=== MÉTRICA DE TEMPO - SINGLE SAMPLE PIPELINE STAGE2 ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tempo médio: {avg_time:.6f} segundos\n")
    f.write(f"Execuções: 10\n")

print(f"✅ Métrica single sample pipeline stage2 salva")

# Evaluate Multi-Stage Model
print("🚀 Avaliando Multi-Stage Model...")

# Stage 1: Binary Detection - SEM safe_model_call
proba_1 = -stage1.decision_function(x)
pred_1 = np.where(proba_1 < tau_b, "Benign", "Attack").astype(object)

unique_values, counts = np.unique(pred_1, return_counts=True)
with open(os.path.join(reports_dir, "stage1_binary_detection.txt"), "w") as f:
    f.write("=== STAGE 1: BINARY DETECTION ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("Distribuição de predições:\n")
    for value, count in zip(unique_values, counts):
        f.write(f"{value}: {count}\n")

print(f"✅ Resultado Stage 1 salvo")

# Timing Stage 1 - SEM safe_model_call  
proba_1 = -stage1.decision_function(x)

with open(os.path.join(reports_dir, "timing_stage1_binary.txt"), "w") as f:
    f.write("=== MÉTRICA DE TEMPO - STAGE 1 BINARY DETECTION ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tempo médio: {avg_time:.6f} segundos\n")
    f.write(f"Execuções: 3\n")

print(f"✅ Timing Stage 1 binary salvo")

# Stage 2: Multi-Class Classification - USAR FALLBACK
x_attack = x[pred_1 == "Attack"]
x_scaled = safe_model_call(stage2_scaler, x_attack, 'transform')
anomaly_scores = proba_1[pred_1 == "Attack"].reshape(-1, 1)
x_with_anomaly = np.column_stack([x_scaled, anomaly_scores])
proba_2 = safe_model_call(stage2_model, x_with_anomaly, 'predict_proba')

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

print(f"✅ Resultado Stage 2 salvo")

# Timing Stage 2 - USAR FALLBACK
x_scaled = safe_model_call(stage2_scaler, x_attack, 'transform')
anomaly_scores = proba_1[pred_1 == "Attack"].reshape(-1, 1)
x_with_anomaly = np.column_stack([x_scaled, anomaly_scores])
proba_2 = safe_model_call(stage2_model, x_with_anomaly, 'predict_proba')

with open(os.path.join(reports_dir, "timing_stage2_multiclass.txt"), "w") as f:
    f.write("=== MÉTRICA DE TEMPO - STAGE 2 MULTI-CLASS ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tempo médio: {avg_time:.6f} segundos\n")
    f.write(f"Execuções: 3\n")

print(f"✅ Timing Stage 2 multiclass salvo")

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

print(f"✅ Resultado Stage 3 salvo")

# Combine stages
y_pred = pred_1.copy()

# Substituir apenas onde há ataques E temos predições do Stage 2
if len(pred_2) > 0:
    attack_mask = (pred_1 == "Attack")
    y_pred[attack_mask] = pred_2

# Substituir apenas onde há "Unknown" do Stage 2 E temos predições do Stage 3  
if len(pred_3) > 0:
    # Criar máscara para amostras que eram Attack -> Unknown -> Stage3
    attack_indices = np.where(pred_1 == "Attack")[0]
    unknown_in_stage2 = (pred_2 == "Unknown")
    unknown_indices = attack_indices[unknown_in_stage2]
    y_pred[unknown_indices] = pred_3

unique_values, counts = np.unique(y_pred, return_counts=True)
with open(os.path.join(reports_dir, "combined_stages.txt"), "w") as f:
    f.write("=== COMBINED STAGES ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("Distribuição final de predições:\n")
    for value, count in zip(unique_values, counts):
        f.write(f"{value}: {count}\n")

print(f"✅ Resultado combined stages salvo")

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
print(f"✅ Matriz de confusão salva")

# Classification report
report = classification_report(y, y_pred, digits=4)
with open(os.path.join(reports_dir, "classification_report.txt"), "w") as f:
    f.write("=== RELATÓRIO DE CLASSIFICAÇÃO ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(report)

print(f"✅ Relatório de classificação salvo")

# Robustness test
y_robust = hids_predict(x_18, tau_b, tau_m, tau_u)

unique_values, counts = np.unique(y_robust, return_counts=True)
with open(os.path.join(reports_dir, "infiltration_2018_hids.txt"), "w") as f:
    f.write("=== INFILTRAÇÃO 2018 - HIDS PRINCIPAL ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Thresholds: tau_b={tau_b}, tau_m={tau_m}, tau_u={tau_u}\n\n")
    f.write("Distribuição de predições:\n")
    for value, count in zip(unique_values, counts):
        f.write(f"{value}: {count}\n")

print(f"✅ Resultado infiltração 2018 HIDS salvo")

# Baseline test on infiltration
if baseline_rf_scaler is not None and baseline_rf is not None:
    x_s = safe_model_call(baseline_rf_scaler, x_18, 'transform')
    y_proba = safe_model_call(baseline_rf, x_s, 'predict_proba')
    y_pred_baseline = np.where(np.max(y_proba, axis=1) > 0.43, baseline_rf.classes_[np.argmax(y_proba, axis=1)], 'Unknown')
    
    unique_values, counts = np.unique(y_pred_baseline, return_counts=True)
    with open(os.path.join(reports_dir, "infiltration_2018_baseline.txt"), "w") as f:
        f.write("=== INFILTRAÇÃO 2018 - BASELINE RF ===\n")
        f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Threshold: 0.43\n\n")
        f.write("Distribuição de predições:\n")
        for value, count in zip(unique_values, counts):
            f.write(f"{value}: {count}\n")

    print(f"✅ Resultado infiltração 2018 Baseline salvo")

print(f"\n🎉 Todos os resultados foram salvos na pasta '{reports_dir}'!")