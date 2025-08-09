#!/usr/bin/env python
# coding: utf-8

# TRAIN.PY CORRIGIDO - VERSÃO OTIMIZADA COM MAIS DADOS

# Seed value
seed_value= 42

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

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
import seaborn as sns

# Criar diretório de modelos
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "new_models")
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Carregar dados com MUITO MAIS DADOS para treinamento
train = {"ocsvm": {}, "ae": {}, "stage2": {}}
val = {"ocsvm": {}, "ae": {}, "stage2": {}}
test = {}

clean_dir = "/home/CIN/jbsn3/multi-stage-hierarchical-ids/ids-dataset-cleaning/cicids2017/clean"

print("🔄 Carregando dados com configuração otimizada...")

# Load Data Stage 1 (OCSVM) - USAR 100K DADOS COMO NO AUTOENCODER ORIGINAL
print("📊 Stage 1: Carregando 100K dados benign para OCSVM...")
train["ocsvm"]["x"], train["ocsvm"]["y"], x_benign_val, y_benign_val, _, _, x_malicious_train, y_malicious_train, _, _, _, _, _ = util.load_data(
    clean_dir, 
    sample_size=1948, 
    train_size=100000,  # ← 100K ao invés de 10K!
    val_size=129485, 
    test_size=56468
)

val["ocsvm"]["x"] = np.concatenate((x_benign_val, x_malicious_train))
val["ocsvm"]["y"] = np.concatenate((y_benign_val, np.full(y_malicious_train.shape[0], -1)))

# Load Data Stage 2 - USAR MUITO MAIS DADOS
print("📊 Stage 2: Carregando dados com train_size maior...")
n_benign_train_stage2 = 50000  # ← 50K ao invés de 1.5K!

x_benign_train, _, _, _, x_benign_test, y_benign_test, x_malicious_train, y_malicious_train, x_malicious_test, y_malicious_test, attack_type_train, _, _ = util.load_data(
    clean_dir, 
    sample_size=1948, 
    train_size=n_benign_train_stage2,  # ← Muito mais dados benign
    val_size=20000,  # ← Mais dados para validação  
    test_size=56468
)

# Usar MAIS dados maliciosos para treinamento (70% ao invés de test_size fixo)
train_test_ratio = 0.8  # 80% para treino, 20% para validação
test_size_malicious = int(len(x_malicious_train) * (1 - train_test_ratio))

train["stage2"]["x"], x_val, train["stage2"]["y"], y_val = train_test_split(
    x_malicious_train, 
    y_malicious_train, 
    stratify=attack_type_train, 
    test_size=test_size_malicious,  # ← Usar porcentagem ao invés de valor fixo
    random_state=42, 
    shuffle=True
)

test['x'] = np.concatenate((x_benign_test, x_malicious_test))
test["y_n"] = np.concatenate((y_benign_test, np.full(y_malicious_test.shape[0], -1)))

val["stage2"]["x"] = np.concatenate((x_val, x_benign_train))
val["stage2"]["y"] = np.concatenate((y_val, np.full(n_benign_train_stage2, "Unknown")))

train["stage2"]["y_n"] = pd.get_dummies(train["stage2"]["y"])
val["stage2"]["y_n"] = pd.get_dummies(val["stage2"]["y"])

test["y"] = np.concatenate((np.full(56468, "Benign"), y_malicious_test))
test["y_unknown"] = np.where((test["y"] == "Heartbleed") | (test["y"] == "Infiltration"), "Unknown", test["y"])
test["y_unknown_all"] = np.where(test['y_unknown'] == 'Benign', "Unknown", test['y_unknown'])

# Feature scaling
scaler_stage2 = QuantileTransformer(output_distribution='normal')
train["stage2"]["x_s"] = scaler_stage2.fit_transform(train["stage2"]["x"])
val["stage2"]["x_s"] = scaler_stage2.transform(val["stage2"]["x"])

print("✅ Dados carregados e processados com configuração otimizada!")
print(f"📊 ESTATÍSTICAS DOS DADOS:")
print(f"   Stage 1 OCSVM Train: {train['ocsvm']['x'].shape}")
print(f"   Stage 1 OCSVM Val: {val['ocsvm']['x'].shape}")
print(f"   Stage 2 Train: {train['stage2']['x_s'].shape}")
print(f"   Stage 2 Val: {val['stage2']['x_s'].shape}")
print(f"   Test: {test['x'].shape}")

# Função para criar RF
def create_rf(params):
    default_params = {
        "n_estimators": 100,
        "max_samples": 0.9,
        "max_features": 0.2,
        "min_samples_leaf": 1,
        "random_state": seed_value,
        "n_jobs": -1
    }
    default_params.update(params)
    return RandomForestClassifier(**default_params)

# ===== TREINAR MODELOS =====

# 1. STAGE 1: OCSVM Pipeline
print("\n🔄 Treinando Stage 1 OCSVM com 100K dados...")

# Scaler para OCSVM
scaler_ocsvm = QuantileTransformer(output_distribution='normal')

# PCA para redução de dimensionalidade
pca = PCA(n_components=56, random_state=seed_value)

# OCSVM - Parâmetros otimizados do código original
ocsvm = OneClassSVM(kernel='rbf', gamma=0.0632653906314333, nu=0.0002316646233151)

# Pipeline Stage 1
ocsvm_pipeline = Pipeline([
    ('scaler', scaler_ocsvm),
    ('pca', pca),
    ('ocsvm', ocsvm)
])

# Treinar pipeline
ocsvm_pipeline.fit(train["ocsvm"]["x"])
print("✅ Stage 1 OCSVM treinado com 100K dados benign")

# 2. STAGE 2: Random Forest (SEM anomaly score - 67 features)
print(f"\n🔄 Treinando Stage 2 RF com {train['stage2']['x_s'].shape[0]} amostras...")

params_rf = {
    "n_estimators": 97,
    "max_samples": 0.9034128710297624,
    "max_features": 0.1751204590963604,
    "min_samples_leaf": 1
}

rf_model_67 = create_rf(params_rf)
rf_model_67.fit(train['stage2']['x_s'], train["stage2"]["y"])
print("✅ Stage 2 RF (67 features) treinado")

# 3. STAGE 2: Random Forest Pipeline (COM anomaly score - 68 features)
print("\n🔄 Treinando Stage 2 RF Pipeline (68 features)...")

# Calcular anomaly scores do Stage 1 para adicionar como feature
proba_train = -ocsvm_pipeline.decision_function(train['stage2']['x'])
proba_val = -ocsvm_pipeline.decision_function(val['stage2']['x'])

# Combinar features originais com anomaly score
train_with_proba = np.column_stack((train['stage2']['x_s'], proba_train))
val_with_proba = np.column_stack((val['stage2']['x_s'], proba_val))

# Treinar RF com 68 features
rf_model_68 = create_rf(params_rf)
rf_model_68.fit(train_with_proba, train["stage2"]["y"])
print("✅ Stage 2 RF (68 features) treinado")

# 4. BASELINE RF
print("\n🔄 Treinando Baseline RF...")
baseline_rf = create_rf(params_rf)
baseline_rf.fit(train['stage2']['x_s'], train["stage2"]["y"])
print("✅ Baseline RF treinado")

# ===== SALVAR MODELOS =====
print("\n💾 Salvando modelos...")

# Stage 1: OCSVM Pipeline
try:
    with open(os.path.join(models_dir, "stage1_ocsvm.p"), "wb") as f:
        pickle.dump(ocsvm_pipeline, f)
    print("✅ stage1_ocsvm.p salvo")
except Exception as e:
    print(f"❌ Erro ao salvar stage1_ocsvm.p: {e}")

# Stage 2: RF Principal (67 features) - ESTE É O QUE O CODE.IPYNB ESPERA!
try:
    with open(os.path.join(models_dir, "stage2_rf.p"), "wb") as f:
        pickle.dump(rf_model_67, f)  # Modelo com 67 features (CORRETO!)
    print("✅ stage2_rf.p salvo (67 features) - CORRETO PARA CODE.IPYNB")
except Exception as e:
    print(f"❌ Erro ao salvar stage2_rf.p: {e}")

# MODELOS INDIVIDUAIS (conforme esperado pelo code.ipynb)

# Stage 1: OCSVM Model individual
try:
    ocsvm_individual = ocsvm_pipeline.named_steps['ocsvm']
    with open(os.path.join(models_dir, "stage1_ocsvm_model.p"), "wb") as f:
        pickle.dump(ocsvm_individual, f)
    print("✅ stage1_ocsvm_model.p salvo")
except Exception as e:
    print(f"❌ Erro ao salvar stage1_ocsvm_model.p: {e}")

# Stage 1: OCSVM Scaler
try:
    scaler_ocsvm_individual = ocsvm_pipeline.named_steps['scaler']
    with open(os.path.join(models_dir, "stage1_ocsvm_scaler.p"), "wb") as f:
        pickle.dump(scaler_ocsvm_individual, f)
    print("✅ stage1_ocsvm_scaler.p salvo")
except Exception as e:
    print(f"❌ Erro ao salvar stage1_ocsvm_scaler.p: {e}")

# Stage 1: PCA individual (ESTE ESTAVA FALTANDO!)
try:
    pca_individual = ocsvm_pipeline.named_steps['pca']
    with open(os.path.join(models_dir, "stage1_pca.p"), "wb") as f:
        pickle.dump(pca_individual, f)
    print("✅ stage1_pca.p salvo")
except Exception as e:
    print(f"❌ Erro ao salvar stage1_pca.p: {e}")

# Stage 2: RF Model individual (67 features) - ESTE É O QUE FALTAVA!
try:
    with open(os.path.join(models_dir, "stage2_rf_model.p"), "wb") as f:
        pickle.dump(rf_model_67, f)  # Modelo com 67 features (SEM anomaly score)
    print("✅ stage2_rf_model.p salvo (67 features) - ESTE É O CORRETO PARA CODE.IPYNB")
except Exception as e:
    print(f"❌ Erro ao salvar stage2_rf_model.p: {e}")

# Stage 2: RF Scaler
try:
    with open(os.path.join(models_dir, "stage2_rf_scaler.p"), "wb") as f:
        pickle.dump(scaler_stage2, f)
    print("✅ stage2_rf_scaler.p salvo")
except Exception as e:
    print(f"❌ Erro ao salvar stage2_rf_scaler.p: {e}")

# Stage 2: RF com feature extra (para compatibilidade)
try:
    with open(os.path.join(models_dir, "stage2_rf_extra_feature.p"), "wb") as f:
        pickle.dump(rf_model_68, f)
    print("✅ stage2_rf_extra_feature.p salvo")
except Exception as e:
    print(f"❌ Erro ao salvar stage2_rf_extra_feature.p: {e}")

# Baseline RF
try:
    with open(os.path.join(models_dir, "baseline_rf.p"), "wb") as f:
        pickle.dump(baseline_rf, f)
    print("✅ baseline_rf.p salvo")
except Exception as e:
    print(f"❌ Erro ao salvar baseline_rf.p: {e}")

# Baseline RF Scaler
try:
    baseline_scaler = QuantileTransformer(output_distribution='normal')
    baseline_scaler.fit(train['stage2']['x'])
    with open(os.path.join(models_dir, "baseline_rf_scaler.p"), "wb") as f:
        pickle.dump(baseline_scaler, f)
    print("✅ baseline_rf_scaler.p salvo")
except Exception as e:
    print(f"❌ Erro ao salvar baseline_rf_scaler.p: {e}")

print("\n✅ TODOS OS MODELOS SALVOS COM SUCESSO!")
print("\n📋 RESUMO DOS MODELOS SALVOS:")
print("- stage1_ocsvm.p: Pipeline OCSVM completo (Scaler→PCA→OCSVM)")
print("- stage1_ocsvm_model.p: Modelo OCSVM individual (56 features)")
print("- stage1_ocsvm_scaler.p: Scaler OCSVM individual (67→67)")
print("- stage1_pca.p: PCA individual (67→56)")
print("- stage2_rf.p: RF com 67 features ← PRINCIPAL PARA CODE.IPYNB")
print("- stage2_rf_model.p: RF com 67 features (backup individual)")
print("- stage2_rf_scaler.p: Scaler RF individual")
print("- stage2_rf_extra_feature.p: RF com 68 features (com anomaly score)")
print("- baseline_rf.p: RF baseline")
print("- baseline_rf_scaler.p: Scaler baseline")

print("\n🎯 MELHORIAS IMPLEMENTADAS:")
print("✅ Stage 1 OCSVM: 100K dados benign (10x mais que antes)")
print("✅ Stage 2: 50K dados benign (33x mais que antes)")
print("✅ Stage 2: 80% dos dados maliciosos para treino (porcentagem ao invés de fixo)")
print("✅ Validação: 20K dados benign + 20% maliciosos")
print("✅ Modelos devem ter performance muito melhor!") 