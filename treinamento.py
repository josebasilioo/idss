#!/usr/bin/env python
# coding: utf-8

# TREINAMENTO.PY - Baseado no Best_Model_Tutorial.ipynb
# Cria todos os modelos necessários para o codex.py

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

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)

import pandas as pd
import util.common as util
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from sklearn.metrics import auc, roc_curve, accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
import pathlib
import pickle
import matplotlib.pyplot as plt
import time  # Adicionado para medir tempo
import sys

print("🚀 Iniciando treinamento dos modelos...")

# Dicionário para armazenar métricas de tempo
timing_metrics = {
    "training_times": {},
    "inference_times": {}
}

# Criar diretórios de modelos e gráficos
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "modelos")
graphics_dir = os.path.join(models_dir, "graphics")

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"📁 Criado diretório: {models_dir}")

if not os.path.exists(graphics_dir):
    os.makedirs(graphics_dir)
    print(f"📁 Criado diretório: {graphics_dir}")

# Estruturas de dados
train = {
    "ocsvm": {},  # 10k samples
    "ae": {},     # 100k samples
    "stage2": {}
}
val = {
    "ocsvm": {},
    "ae": {},
    "stage2": {}
}
test = {}

# Definir diretório dos dados limpos
clean_dir = "/home/CIN/jbsn3/multi-stage-hierarchical-ids/ids-dataset-cleaning/cicids2018/clean"

print("📊 Carregando dados Stage 1 (OCSVM)...")
# Load Data Stage 1 - OCSVM (10k samples)
train["ocsvm"]["x"], train["ocsvm"]["y"], x_benign_val, y_benign_val, _, _, x_malicious_train, y_malicious_train, _, _, _, _, _ = util.load_data(
    clean_dir, 
    sample_size=10000, 
    train_size=10000, 
    val_size=129485, 
    test_size=56468
)

val["ocsvm"]["x"] = np.concatenate((x_benign_val, x_malicious_train))
val["ocsvm"]["y"] = np.concatenate((y_benign_val, np.full(y_malicious_train.shape[0], -1)))

# Load Data Stage 1 - AE (100k samples)
train["ae"]["x"], train["ae"]["y"], x_benign_val, y_benign_val, _, _, x_malicious_train, y_malicious_train, _, _, _, _, _ = util.load_data(
    clean_dir, 
    sample_size=100000, 
    val_size=129485, 
    test_size=56468
)

val["ae"]["x"] = np.concatenate((x_benign_val, x_malicious_train))
val["ae"]["y"] = np.concatenate((y_benign_val, np.full(y_malicious_train.shape[0], -1)))

print("📊 Carregando dados Stage 2...")

# Load Data Stage 2 - MELHORIAS PONTUAIS
n_benign_val = 800

# Agora chamar load_data com sample_size seguro
x_benign_train, _, _, _, x_benign_test, y_benign_test, x_malicious_train, y_malicious_train, x_malicious_test, y_malicious_test, attack_type_train, _, _ = util.load_data(
    clean_dir, 
    sample_size=10000,  # USAR VALOR SEGURO
    train_size=n_benign_val, 
    val_size=6815, 
    test_size=56468
)

attack_series = pd.Series(attack_type_train)
counts = attack_series.value_counts()
keep_labels = counts[counts > 1].index

mask = attack_series.isin(keep_labels).to_numpy()
x_malicious_train = x_malicious_train[mask]
y_malicious_train = y_malicious_train[mask]
attack_type_train = attack_type_train[mask]

# SOLUÇÃO ELEGANTE - Usar attack_type_train para máxima diversidade
train["stage2"]["x"], x_val, train["stage2"]["y"], y_val = train_test_split(
    x_malicious_train, 
    y_malicious_train, 
    stratify=attack_type_train,  # 🎯 GENIAL! Garante todos os subtipos
    test_size=800,              # Maior conjunto de validação
    random_state=42, 
    shuffle=True
)

test['x'] = np.concatenate((x_benign_test, x_malicious_test))
test["y_n"] = np.concatenate((y_benign_test, np.full(y_malicious_test.shape[0], -1)))

val["stage2"]["x"] = np.concatenate((x_val, x_benign_train))
val["stage2"]["y"] = np.concatenate((y_val, np.full(n_benign_val, "Unknown")))

train["stage2"]["y_n"] = pd.get_dummies(train["stage2"]["y"])
val["stage2"]["y_n"] = pd.get_dummies(val["stage2"]["y"])

test["y"] = np.concatenate((np.full(56468, "Benign"), y_malicious_test))
test["y_unknown"] = np.where((test["y"] == "Heartbleed") | (test["y"] == "Infiltration"), "Unknown", test["y"])
test["y_unknown_all"] = np.where(test['y_unknown'] == 'Benign', "Unknown", test['y_unknown'])

print("⚙️ Escalando dados...")

# Scale the data - OCSVM
scaler_ocsvm = QuantileTransformer(output_distribution='normal')
train['ocsvm']['x_s'] = scaler_ocsvm.fit_transform(train['ocsvm']['x'])
val['ocsvm']['x_s'] = scaler_ocsvm.transform(val['ocsvm']['x'])
test['ocsvm_s'] = scaler_ocsvm.transform(test['x'])

# Scale the data - AE (100k)
scaler_ae = QuantileTransformer(output_distribution='normal')
train['ae']['x_s'] = scaler_ae.fit_transform(train['ae']['x'])
val['ae']['x_s'] = scaler_ae.transform(val['ae']['x'])
test['ae_s'] = scaler_ae.transform(test['x'])

# Scale the data - Stage 2
scaler_stage2 = QuantileTransformer(output_distribution='normal')
train['stage2']['x_s'] = scaler_stage2.fit_transform(train['stage2']['x'])
val['stage2']['x_s'] = scaler_stage2.transform(val['stage2']['x'])
test['stage2_s'] = scaler_stage2.transform(test['x'])

# Scale uniform for Stage 2
scaler_stage2_uniform = QuantileTransformer(output_distribution='uniform')
train['stage2']['x_q'] = scaler_stage2_uniform.fit_transform(train['stage2']['x'])
val['stage2']['x_q'] = scaler_stage2_uniform.transform(val['stage2']['x'])
test['stage2_q'] = scaler_stage2_uniform.transform(test['x'])

print("🤖 Treinando modelos...")

# ========== STAGE 1: One-Class SVM ==========
print("  🔹 Treinando One-Class SVM...")

def create_ocsvm(params):
    return Pipeline([
        ("pca", PCA(n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=42)), 
        ("ocsvm", OneClassSVM(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=True, max_iter=-1))
    ]).set_params(**params)

# Treinar OCSVM com 100k dados (como no AE original)
params_ocsvm = {
    "pca__n_components": 66,
    "ocsvm__kernel": "rbf",
    "ocsvm__gamma": 0.001101767345549512,
    "ocsvm__nu": 4.869252853070146e-05
}

ocsvm_model = create_ocsvm(params_ocsvm)

# Medir tempo de treinamento OCSVM
start_time = time.time()
ocsvm_model.fit(train['ae']['x_s'])  # Usar 100k dados
training_time_ocsvm = time.time() - start_time
timing_metrics["training_times"]["ocsvm"] = training_time_ocsvm

print(f"  ✅ One-Class SVM treinado em {training_time_ocsvm:.2f}s")

# Medir tempo de inferência OCSVM
start_time = time.time()
_ = ocsvm_model.decision_function(val['ae']['x_s'][:1000])  # Amostra de 1000 para inferência
inference_time_ocsvm = (time.time() - start_time) / 1000  # Tempo por amostra
timing_metrics["inference_times"]["ocsvm"] = inference_time_ocsvm

# ========== STAGE 2: Random Forest ==========
print("  🔹 Treinando Random Forest...")

def create_rf(params):
    return RandomForestClassifier(random_state=42).set_params(**params)

# MELHORIA 3: Random Forest com class_weight e mais árvores
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(train["stage2"]["y"]), y=train["stage2"]["y"])
class_weight_dict = dict(zip(np.unique(train["stage2"]["y"]), class_weights))

params_rf = {
    "n_estimators": 106,
    "max_samples": 0.9852142919229748,
    "max_features": 0.39279757672456206,
    "min_samples_leaf": 3
}

rf_model = create_rf(params_rf)

# Medir tempo de treinamento RF
start_time = time.time()
rf_model.fit(train['stage2']['x_s'], train["stage2"]["y"])
training_time_rf = time.time() - start_time
timing_metrics["training_times"]["random_forest"] = training_time_rf

print(f"  ✅ Random Forest treinado em {training_time_rf:.2f}s")

# Medir tempo de inferência RF
start_time = time.time()
_ = rf_model.predict_proba(val['stage2']['x_s'][:1000])  # Amostra de 1000 para inferência
inference_time_rf = (time.time() - start_time) / 1000  # Tempo por amostra
timing_metrics["inference_times"]["random_forest"] = inference_time_rf

# ========== MODELO BASELINE ==========
print("  🔹 Treinando modelo baseline...")

baseline_rf = RandomForestClassifier(random_state=42)

# Medir tempo de treinamento Baseline
start_time = time.time()
baseline_rf.fit(train['stage2']['x_s'], train["stage2"]["y"])
training_time_baseline = time.time() - start_time
timing_metrics["training_times"]["baseline_rf"] = training_time_baseline

print(f"  ✅ Modelo baseline treinado em {training_time_baseline:.2f}s")

# Medir tempo de inferência Baseline
start_time = time.time()
_ = baseline_rf.predict_proba(val['stage2']['x_s'][:1000])  # Amostra de 1000 para inferência
inference_time_baseline = (time.time() - start_time) / 1000  # Tempo por amostra
timing_metrics["inference_times"]["baseline_rf"] = inference_time_baseline

# ========== EXTRAIR COMPONENTES INDIVIDUAIS ==========
print("📦 Extraindo componentes individuais...")

# Extrair PCA e OCSVM do pipeline
pca_individual = ocsvm_model.named_steps['pca']
ocsvm_individual = ocsvm_model.named_steps['ocsvm']

# Criar scaler individual para Stage 1 (usando os mesmos dados do pipeline)
scaler_stage1_individual = QuantileTransformer(output_distribution='normal')
scaler_stage1_individual.fit(train['ae']['x'])  # Usar os mesmos dados que o pipeline

print("  ✅ Componentes extraídos")

# ========== VALIDAÇÃO E GRÁFICOS ==========
print("📊 Gerando visualizações...")

# Validação Stage 1
score_val_ocsvm = -ocsvm_model.decision_function(val['ae']['x_s'])
curves_metrics_ocsvm, summary_metrics_ocsvm = util.evaluate_proba(val['ae']['y'], score_val_ocsvm)

# Validação Stage 2
y_proba_val_rf = rf_model.predict_proba(val['stage2']['x_s'])
fmacro, fweight, threshold_fscore, f_best = util.optimal_fscore_multi(val['stage2']['y'], y_proba_val_rf, rf_model.classes_)
y_pred_val_rf = np.where(np.max(y_proba_val_rf, axis=1) > f_best["f1_weighted_threshold"], rf_model.classes_[np.argmax(y_proba_val_rf, axis=1)], 'Unknown')

# Test predictions para confusion matrix
score_test = -ocsvm_model.decision_function(test['ae_s'])
y_proba_test_2 = rf_model.predict_proba(test['stage2_s'])

# Calcular thresholds otimizados baseados nos dados CICIDS2018
from sklearn.metrics import roc_curve

# 1. Threshold para Stage 1 (OCSVM)
score_val_ocsvm = -ocsvm_model.decision_function(val['ae']['x_s'])
fpr, tpr, thresholds_ocsvm = roc_curve(val['ae']['y'], score_val_ocsvm, pos_label=-1)
# Escolher threshold que maximize TPR - FPR
optimal_idx = np.argmax(tpr - fpr)

threshold_b = thresholds_ocsvm[optimal_idx]
threshold_m = f_best["f1_weighted_threshold"]
threshold_u = np.percentile(score_val_ocsvm, 95)

print(f"🎯 Thresholds otimizados:")
print(f"   threshold_b (Stage 1): {threshold_b}")
print(f"   threshold_m (Stage 2): {threshold_m}")  
print(f"   threshold_u (Stage 3): {threshold_u}")

# Simular predição completa da pipeline
y_pred = np.where(score_test < threshold_b, "Benign", "Fraud").astype(object)
y_pred_2 = np.where(np.max(y_proba_test_2[y_pred == "Fraud"], axis=1) > threshold_m, 
                   train["stage2"]["y_n"].columns[np.argmax(y_proba_test_2[y_pred == "Fraud"], axis=1)], 'Unknown')
y_pred[y_pred == "Fraud"] = y_pred_2

y_pred_3 = np.where(score_test[y_pred == "Unknown"] < threshold_u, "Benign", "Unknown")
y_pred[y_pred == "Unknown"] = y_pred_3

# Gerar gráfico completo de comparação (6 subplots)
def generate_complete_confusion_graphs(score_test, y_proba_test_2, threshold_b, threshold_m, threshold_u):
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(wspace=0.3)
    
    classes = ['Benign', '(D)DOS', 'Botnet', 'Brute Force', 'Web Attack', 'Unknown']
    
    # Stage 1 - Binary
    y_pred_1_n = np.where(score_test < threshold_b, 1, -1)
    util.plot_confusion_matrix(test['y_n'], y_pred_1_n, values=[1, -1], labels=["Benign", "Fraud"], title="Stage 1", ax=axs[0, 0])
    y_pred = np.where(score_test < threshold_b, "Benign", "Fraud")
    
    # Stage 2 - Multi-class
    y_pred_2 = np.where(np.max(y_proba_test_2[y_pred == "Fraud"], axis=1) > threshold_m, 
                       train["stage2"]["y_n"].columns[np.argmax(y_proba_test_2[y_pred == "Fraud"], axis=1)], 'Unknown')
    util.plot_confusion_matrix(test['y_unknown'][y_pred == "Fraud"], y_pred_2, values=classes, labels=classes, title="Stage 2", ax=axs[0, 1])

    # Stage 1&2 Combined
    y_pred = y_pred.astype(object)
    y_pred[y_pred == "Fraud"] = y_pred_2
    util.plot_confusion_matrix(test['y_unknown'], y_pred, values=classes, labels=classes, title="Stage 1&2 Combined", ax=axs[0, 2])
    
    # Extension Multi-Class
    mask = ((y_pred == "Unknown") & (test['y_unknown_all'] == "Unknown"))
    y_pred_3 = np.where(score_test[y_pred == "Unknown"] < threshold_u, "Benign", "Unknown")
    util.plot_confusion_matrix(test['y_unknown'][y_pred == "Unknown"], y_pred_3, values=classes, labels=classes, title="Extension Multi-Class", ax=axs[1, 0])
    
    # Extension Binary
    y_pred_3_n = np.where(score_test[mask] < threshold_u, 1, -1)
    util.plot_confusion_matrix(test['y_n'][mask], y_pred_3_n, values=[1, -1], labels=["Benign", "Zero-Day"], title="Extension Binary", ax=axs[1, 1])

    # Stages 1,2 & Extension Combined (Final)
    y_pred[y_pred == "Unknown"] = y_pred_3
    util.plot_confusion_matrix(test['y_unknown'], y_pred, values=classes, labels=classes, title="Stages 1,2 & Extension Combined", ax=axs[1, 2])
    
    return fig, y_pred

try:
    fig, y_pred_final = generate_complete_confusion_graphs(score_test, y_proba_test_2, threshold_b, threshold_m, threshold_u)
    plt.savefig(os.path.join(graphics_dir, "confusion_matrix_complete_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Gráfico completo de comparação salvo")
    
    # Usar y_pred_final para as métricas
    y_pred = y_pred_final
    
except Exception as e:
    print(f"  ⚠️ Erro ao gerar gráfico completo: {e}")
    # Fallback para confusion matrix simples
    plt.figure(figsize=(10, 8))
    classes = ['Benign', '(D)DOS', 'Botnet', 'Brute Force', 'Web Attack', 'Unknown']
    try:
        util.plot_confusion_matrix(test['y_unknown'], y_pred, values=classes, labels=classes)
        plt.title('Confusion Matrix - Pipeline Completa')
        plt.tight_layout()
        plt.savefig(os.path.join(graphics_dir, "confusion_matrix_final.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Confusion matrix simples salva")
    except Exception as e2:
        print(f"  ⚠️ Erro ao gerar confusion matrix simples: {e2}")

# Salvar métricas de performance
try:
    with open(os.path.join(graphics_dir, "performance_metrics.txt"), "w") as f:
        f.write("=== MÉTRICAS DE PERFORMANCE ===\n\n")
        
        # Métricas de tempo
        f.write("TEMPOS DE TREINAMENTO:\n")
        f.write(f"OCSVM: {timing_metrics['training_times']['ocsvm']:.2f}s\n")
        f.write(f"Random Forest: {timing_metrics['training_times']['random_forest']:.2f}s\n")
        f.write(f"Baseline RF: {timing_metrics['training_times']['baseline_rf']:.2f}s\n\n")
        
        f.write("TEMPOS DE INFERÊNCIA (por amostra):\n")
        f.write(f"OCSVM: {timing_metrics['inference_times']['ocsvm']*1000:.4f}ms\n")
        f.write(f"Random Forest: {timing_metrics['inference_times']['random_forest']*1000:.4f}ms\n")
        f.write(f"Baseline RF: {timing_metrics['inference_times']['baseline_rf']*1000:.4f}ms\n\n")
        
        f.write("STAGE 1 (OCSVM):\n")
        f.write(f"Métricas: {summary_metrics_ocsvm}\n\n")
        f.write("STAGE 2 (Random Forest):\n")
        f.write(f"Threshold ótimo F1: {f_best['f1_weighted_threshold']}\n")
        f.write(f"F1 Score: {f1_score(val['stage2']['y'], y_pred_val_rf, average='weighted'):.4f}\n")
        f.write(f"Accuracy: {accuracy_score(val['stage2']['y'], y_pred_val_rf):.4f}\n\n")
        f.write("PIPELINE COMPLETA (Test):\n")
        f.write(f"F1 Macro: {f1_score(test['y_unknown'], y_pred, average='macro'):.4f}\n")
        f.write(f"F1 Weighted: {f1_score(test['y_unknown'], y_pred, average='weighted'):.4f}\n")
        f.write(f"Accuracy: {accuracy_score(test['y_unknown'], y_pred):.4f}\n")
        f.write(f"Balanced Accuracy: {balanced_accuracy_score(test['y_unknown'], y_pred):.4f}\n")
    print("  ✅ Métricas de performance salvas")
except Exception as e:
    print(f"  ⚠️ Erro ao salvar métricas: {e}")

# ========== SALVAR MODELOS ==========
print("💾 Salvando modelos...")

try:
    # 1. Pipeline completo Stage 1
    with open(os.path.join(models_dir, "stage1_ocsvm.p"), "wb") as f:
        pickle.dump(ocsvm_model, f)
    print("  ✅ stage1_ocsvm.p salvo")
    
    # 2. Pipeline completo Stage 2  
    with open(os.path.join(models_dir, "stage2_rf.p"), "wb") as f:
        pickle.dump(rf_model, f)
    print("  ✅ stage2_rf.p salvo")
    
    # 3. OCSVM individual
    with open(os.path.join(models_dir, "stage1_ocsvm_model.p"), "wb") as f:
        pickle.dump(ocsvm_model, f)
    print("  ✅ stage1_ocsvm_model.p salvo")
    
    # 4. Scaler Stage 1
    with open(os.path.join(models_dir, "stage1_ocsvm_scaler.p"), "wb") as f:
        pickle.dump(scaler_stage1_individual, f)
    print("  ✅ stage1_ocsvm_scaler.p salvo")
    
    # 5. PCA individual
    with open(os.path.join(models_dir, "stage1_pca.p"), "wb") as f:
        pickle.dump(pca_individual, f)
    print("  ✅ stage1_pca.p salvo")
    
    # 6. RF individual
    with open(os.path.join(models_dir, "stage2_rf_model.p"), "wb") as f:
        pickle.dump(rf_model, f)
    print("  ✅ stage2_rf_model.p salvo")
    
    # 7. Scaler Stage 2
    with open(os.path.join(models_dir, "stage2_rf_scaler.p"), "wb") as f:
        pickle.dump(scaler_stage2, f)
    print("  ✅ stage2_rf_scaler.p salvo")
    
    # 8. Baseline RF
    with open(os.path.join(models_dir, "baseline_rf.p"), "wb") as f:
        pickle.dump(baseline_rf, f)
    print("  ✅ baseline_rf.p salvo")
    
    # 9. Baseline RF Scaler
    baseline_rf_scaler = QuantileTransformer(output_distribution='normal')
    baseline_rf_scaler.fit(train['stage2']['x'])
    with open(os.path.join(models_dir, "baseline_rf_scaler.p"), "wb") as f:
        pickle.dump(baseline_rf_scaler, f)
    print("  ✅ baseline_rf_scaler.p salvo")

except Exception as e:
    print(f"❌ Erro ao salvar modelos: {e}")
    raise

print("\n🎉 TREINAMENTO CONCLUÍDO!")
print(f"📁 Modelos salvos em: {models_dir}")
print(f"📊 Gráficos salvos em: {graphics_dir}")

# Exibir resumo dos tempos
print("\n⏱️ RESUMO DOS TEMPOS:")
print(f"  🔹 OCSVM - Treino: {timing_metrics['training_times']['ocsvm']:.2f}s | Inferência: {timing_metrics['inference_times']['ocsvm']*1000:.4f}ms/amostra")
print(f"  🔹 Random Forest - Treino: {timing_metrics['training_times']['random_forest']:.2f}s | Inferência: {timing_metrics['inference_times']['random_forest']*1000:.4f}ms/amostra")
print(f"  🔹 Baseline RF - Treino: {timing_metrics['training_times']['baseline_rf']:.2f}s | Inferência: {timing_metrics['inference_times']['baseline_rf']*1000:.4f}ms/amostra")

print("\n📋 Modelos criados:")
print("  1. stage1_ocsvm.p - Pipeline completo Stage 1")
print("  2. stage2_rf.p - Pipeline completo Stage 2")
print("  3. stage1_ocsvm_model.p - OCSVM individual")
print("  4. stage1_ocsvm_scaler.p - Scaler Stage 1")
print("  5. stage1_pca.p - PCA individual")
print("  6. stage2_rf_model.p - RF individual")
print("  7. stage2_rf_scaler.p - Scaler Stage 2")
print("  8. baseline_rf.p - Modelo baseline")
print("  9. baseline_rf_scaler.p - Scaler baseline")

print("\n📊 Visualizações criadas:")
print("  • confusion_matrix_complete_comparison.png - Comparação completa (6 gráficos)")
print("  • performance_metrics.txt - Métricas de performance detalhadas")

print("\n✅ Todos os modelos necessários para codex.py foram criados!")

# TESTE RÁPIDO - Adicionar no codex.py após carregar o modelo
print("🔍 DIAGNÓSTICO DO MODELO:")
print(f"   Classes disponíveis: {rf_model.classes_}")
print(f"   Número de classes: {len(rf_model.classes_)}")

# Se só tem uma classe, o problema está no treinamento!
if len(rf_model.classes_) == 1:
    print("❌ PROBLEMA CRÍTICO: Modelo só tem uma classe!")
    print("   Necessário retreinar o modelo com dados balanceados")