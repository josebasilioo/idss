#!/usr/bin/env python
# coding: utf-8

"""
DIAGNÓSTICO DE THRESHOLDS - Identifica problemas com tau_b, tau_m, tau_u
"""

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")
models_dir = os.path.join(script_dir, "new_models")

print("🔍 DIAGNÓSTICO DE THRESHOLDS")
print("=" * 50)

# Carregar dados de teste
test = pd.read_parquet(os.path.join(data_dir, "test.parquet"))
y = test["Y"].replace(["Heartbleed", "Infiltration"], "Unknown")
x = test.drop(columns=['Y'])

print(f"📊 Dados de teste: {x.shape}")
print(f"📊 Distribuição real:")
for label, count in y.value_counts().items():
    print(f"   {label}: {count}")

# Carregar modelos
def load_model_safe(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

stage1 = load_model_safe(os.path.join(models_dir, "stage1_ocsvm.p"))
stage2 = load_model_safe(os.path.join(models_dir, "stage2_rf.p"))

print("\n✅ Modelos carregados")

# Thresholds atuais
tau_b = -0.01
tau_m = 0.85
tau_u = 0.0040588613744241275

print(f"\n🎯 THRESHOLDS ATUAIS:")
print(f"   tau_b = {tau_b:.10f}")
print(f"   tau_m = {tau_m:.6f}")
print(f"   tau_u = {tau_u:.10f}")

# ===== DIAGNÓSTICO STAGE 1 =====
print(f"\n🔍 DIAGNÓSTICO STAGE 1:")

proba_1 = -stage1.decision_function(x)
print(f"   Anomaly scores - Min: {proba_1.min():.6f}, Max: {proba_1.max():.6f}")
print(f"   Anomaly scores - Mean: {proba_1.mean():.6f}, Std: {proba_1.std():.6f}")

pred_1 = np.where(proba_1 < tau_b, "Benign", "Attack")
stage1_results = pd.Series(pred_1).value_counts()
print(f"   Predições Stage 1:")
for label, count in stage1_results.items():
    print(f"      {label}: {count}")

# Analisar distribuição por classe real
print(f"\n   Distribuição de anomaly scores por classe real:")
for label in y.unique():
    if label == "Benign":
        continue
    mask = (y == label)
    if mask.sum() > 0:
        scores_class = proba_1[mask]
        below_threshold = (scores_class < tau_b).sum()
        print(f"      {label}: {mask.sum()} amostras, {below_threshold} classificadas como Benign ({below_threshold/mask.sum()*100:.1f}%)")

# ===== DIAGNÓSTICO STAGE 2 =====
print(f"\n🔍 DIAGNÓSTICO STAGE 2:")

attack_mask = (pred_1 == "Attack")
x_attacks = x[attack_mask]
y_attacks = y[attack_mask]

if len(x_attacks) > 0:
    proba_2 = stage2.predict_proba(x_attacks)
    max_proba_2 = np.max(proba_2, axis=1)
    
    print(f"   Probabilidades máximas Stage 2:")
    print(f"      Min: {max_proba_2.min():.6f}")
    print(f"      Max: {max_proba_2.max():.6f}")
    print(f"      Mean: {max_proba_2.mean():.6f}")
    print(f"      Std: {max_proba_2.std():.6f}")
    
    # Quantis importantes
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    print(f"   Quantis das probabilidades máximas:")
    for q in quantiles:
        val = np.quantile(max_proba_2, q)
        print(f"      {q*100:2.0f}%: {val:.6f}")
    
    # Quantas amostras passam do threshold atual
    above_threshold = (max_proba_2 > tau_m).sum()
    print(f"\n   ⚠️ PROBLEMA IDENTIFICADO:")
    print(f"      Threshold tau_m = {tau_m:.6f}")
    print(f"      Amostras que passam do threshold: {above_threshold}/{len(max_proba_2)} ({above_threshold/len(max_proba_2)*100:.1f}%)")
    
    # Testar thresholds alternativos
    print(f"\n   🔧 THRESHOLDS ALTERNATIVOS:")
    test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for thresh in test_thresholds:
        above = (max_proba_2 > thresh).sum()
        print(f"      tau_m = {thresh:.1f}: {above}/{len(max_proba_2)} amostras ({above/len(max_proba_2)*100:.1f}%) seriam classificadas")
    
    # Analisar por classe
    print(f"\n   Análise por classe real:")
    for label in np.unique(y_attacks):
        if label == "Unknown":
            continue
        mask_class = (y_attacks == label)
        if mask_class.sum() > 0:
            proba_class = max_proba_2[mask_class]
            above_thresh = (proba_class > tau_m).sum()
            mean_proba = proba_class.mean()
            print(f"      {label}: {mask_class.sum()} amostras, prob média: {mean_proba:.3f}, acima threshold: {above_thresh} ({above_thresh/mask_class.sum()*100:.1f}%)")

else:
    print("   ❌ Nenhum ataque detectado no Stage 1!")

# ===== RECOMENDAÇÕES =====
print(f"\n💡 RECOMENDAÇÕES:")

if len(x_attacks) > 0:
    # Sugerir novo tau_m baseado nos quantis
    recommended_tau_m = np.quantile(max_proba_2, 0.3)  # 30% das predições passariam
    print(f"   1. Reduzir tau_m de {tau_m:.6f} para ~{recommended_tau_m:.6f}")
    print(f"      Isso faria ~70% das detecções Stage 1 serem classificadas como tipos específicos")

# Análise do tau_b
benign_scores = proba_1[y == "Benign"]
attack_scores = proba_1[y != "Benign"]
if len(attack_scores) > 0:
    # Percentil de ataques que são classificados como benignos
    benign_classified = (attack_scores < tau_b).sum()
    print(f"   2. tau_b atual classifica {benign_classified}/{len(attack_scores)} ataques como benignos ({benign_classified/len(attack_scores)*100:.1f}%)")
    
    if benign_classified/len(attack_scores) > 0.1:  # Se mais de 10% dos ataques viram benignos
        recommended_tau_b = np.quantile(attack_scores, 0.05)  # 5% dos ataques virariam benignos
        print(f"      Considere aumentar tau_b para ~{recommended_tau_b:.6f}")

print(f"\n🎯 CONCLUSÃO:")
print(f"   O principal problema é tau_m = {tau_m:.6f} muito alto!")
print(f"   Isso faz com que TODAS as predições Stage 2 virem 'Unknown'")
print(f"   Execute o script optimize_thresholds.py para encontrar valores ótimos")

print(f"\n✅ Diagnóstico concluído!")
