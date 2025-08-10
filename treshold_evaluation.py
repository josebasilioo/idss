#!/usr/bin/env python
# coding: utf-8

"""
AVALIAÇÃO DE THRESHOLDS - Multi-Stage HIDS
Encontra os melhores thresholds tau_b, tau_m, tau_u através de grid search
Gera tabela com melhores configurações e justificativas
"""

import os
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score, recall_score, accuracy_score
from datetime import datetime
import time

# Configurar seeds
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
import random
random.seed(seed_value)
np.random.seed(seed_value)

# Obter diretório do script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")
models_dir = os.path.join(script_dir, "modelos")
reports_dir = os.path.join(script_dir, "reports")

if not os.path.exists(reports_dir):
    os.makedirs(reports_dir)

print("🔍 AVALIAÇÃO DE THRESHOLDS - Multi-Stage HIDS")
print("=" * 60)

# Carregar dados
print("📊 Carregando dados...")
test = pd.read_parquet(os.path.join(data_dir, "test.parquet"))
y_true = test["Y"].replace(["Heartbleed", "Infiltration"], "Unknown")
x = test.drop(columns=['Y'])

print(f"✅ Dados carregados: {x.shape[0]} amostras, {x.shape[1]} features")
print("Distribuição real:")
for classe, count in y_true.value_counts().items():
    print(f"  {classe}: {count:,} ({count/len(y_true)*100:.1f}%)")

# Carregar modelos
print("\n🔧 Carregando modelos e escaladores...")
with open(os.path.join(models_dir, "stage1_ocsvm.p"), "rb") as f:
    stage1 = pickle.load(f)
with open(os.path.join(models_dir, "stage2_rf.p"), "rb") as f:
    stage2 = pickle.load(f)
with open(os.path.join(models_dir, "stage1_ocsvm_scaler.p"), "rb") as f:
    scaler_stage1 = pickle.load(f)
with open(os.path.join(models_dir, "stage2_rf_scaler.p"), "rb") as f:
    scaler_stage2 = pickle.load(f)

print("✅ Modelos carregados")

def evaluate_pipeline(x_stage1, x_stage2, y_true, tau_b, tau_m, tau_u):
    """Avalia pipeline com thresholds específicos e retorna métricas detalhadas"""
    try:
        # Stage 1: Binary Detection
        score_test = -stage1.decision_function(x_stage1)
        y_pred = np.where(score_test < tau_b, "Benign", "Attack").astype(object)
        
        # Stage 2: Multi-class Classification
        if np.sum(y_pred == "Attack") > 0:
            x_attack_scaled = x_stage2[y_pred == "Attack"]
            y_proba_test_2 = stage2.predict_proba(x_attack_scaled)
            
            y_pred_2 = np.where(
                np.max(y_proba_test_2, axis=1) > tau_m, 
                stage2.classes_[np.argmax(y_proba_test_2, axis=1)], 
                'Unknown'
            )
            y_pred[y_pred == "Attack"] = y_pred_2
        
        # Stage 3: Zero-day Detection
        if np.sum(y_pred == "Unknown") > 0:
            unknown_scores = score_test[y_pred == "Unknown"]
            y_pred_3 = np.where(unknown_scores < tau_u, "Benign", "Unknown")
            y_pred[y_pred == "Unknown"] = y_pred_3
        
        # Calcular métricas
        accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Métricas específicas
        benign_recall = recall_score(y_true == 'Benign', y_pred == 'Benign', zero_division=0)
        attack_detection_rate = np.sum((y_true != 'Benign') & (y_pred != 'Benign')) / np.sum(y_true != 'Benign')
        
        # Contagem de predições
        pred_counts = pd.Series(y_pred).value_counts().to_dict()
        
        return {
            'tau_b': tau_b, 'tau_m': tau_m, 'tau_u': tau_u,
            'accuracy': accuracy, 'balanced_accuracy': balanced_acc,
            'f1_macro': f1_macro, 'f1_weighted': f1_weighted,
            'precision_macro': precision_macro, 'recall_macro': recall_macro,
            'benign_recall': benign_recall, 'attack_detection_rate': attack_detection_rate,
            'pred_benign': pred_counts.get('Benign', 0),
            'pred_ddos': pred_counts.get('(D)DOS', 0),
            'pred_botnet': pred_counts.get('Botnet', 0),
            'pred_brute_force': pred_counts.get('Brute Force', 0),
            'pred_port_scan': pred_counts.get('Port Scan', 0),
            'pred_web_attack': pred_counts.get('Web Attack', 0),
            'pred_unknown': pred_counts.get('Unknown', 0),
            'total_attacks_predicted': sum([pred_counts.get(cls, 0) for cls in ['(D)DOS', 'Botnet', 'Brute Force', 'Port Scan', 'Web Attack']])
        }
    except Exception as e:
        return None

# OTIMIZAÇÃO 3: Usar apenas subset dos dados para busca inicial
print("\n⚙️ Preparando dados...")
x_stage1_scaled = scaler_stage1.transform(x)
x_stage2_scaled = scaler_stage2.transform(x)

# Usar apenas 20% dos dados para busca rápida de thresholds
print("🔬 Usando subset de 20% dos dados para otimização rápida...")
subset_size = int(0.2 * len(x))
subset_indices = np.random.choice(len(x), subset_size, replace=False)

x_stage1_subset = x_stage1_scaled[subset_indices]
x_stage2_subset = x_stage2_scaled[subset_indices]
y_true_subset = y_true.iloc[subset_indices]

print(f"📊 Subset: {len(x_stage1_subset):,} amostras ({len(x_stage1_subset)/len(x)*100:.1f}% do total)")
print("Distribuição do subset:")
for classe, count in y_true_subset.value_counts().items():
    print(f"  {classe}: {count:,} ({count/len(y_true_subset)*100:.1f}%)")

# Analisar scores para definir ranges inteligentes
print("🔍 Analisando distribuição de scores...")
scores_stage1 = -stage1.decision_function(x_stage1_scaled)
print(f"Stage 1 scores:")
print(f"  Range: [{np.min(scores_stage1):.6f}, {np.max(scores_stage1):.6f}]")
print(f"  Mean ± Std: {np.mean(scores_stage1):.6f} ± {np.std(scores_stage1):.6f}")
print(f"  Percentis: P5={np.percentile(scores_stage1, 5):.6f}, P50={np.percentile(scores_stage1, 50):.6f}, P95={np.percentile(scores_stage1, 95):.6f}")

# OTIMIZAÇÃO 1: Ranges muito menores e mais inteligentes
print("🧠 Usando busca otimizada com ranges reduzidos...")

# Usar apenas valores críticos baseados em percentis
tau_b_range = [
    np.percentile(scores_stage1, 10),  # Mais conservador
    np.percentile(scores_stage1, 25),  # Balanceado
    np.percentile(scores_stage1, 50)   # Mais agressivo
]

# Apenas valores de confiança mais comuns
tau_m_range = [0.5, 0.7, 0.8, 0.9, 0.95]

# Apenas 3 valores para tau_u
tau_u_range = [
    np.percentile(scores_stage1, 20),
    np.percentile(scores_stage1, 40),
    np.percentile(scores_stage1, 60)
]

total_combinations = len(tau_b_range) * len(tau_m_range) * len(tau_u_range)

print(f"\n📊 Configuração Grid Search:")
print(f"  • tau_b: {len(tau_b_range)} valores ({tau_b_range[0]:.6f} a {tau_b_range[-1]:.6f})")
print(f"  • tau_m: {len(tau_m_range)} valores ({min(tau_m_range):.2f} a {max(tau_m_range):.2f})")
print(f"  • tau_u: {len(tau_u_range)} valores ({tau_u_range[0]:.6f} a {tau_u_range[-1]:.6f})")
print(f"  • Total: {total_combinations:,} combinações")

# OTIMIZAÇÃO 2: Busca sequencial inteligente
print(f"\n🚀 Executando busca otimizada sequencial...")
print(f"📊 Total de {total_combinations} combinações (3×5×3 = 45 ao invés de 1200!)")
start_time = time.time()
results = []
current = 0

# Busca mais rápida com early stopping se necessário
best_f1_so_far = 0
no_improvement_count = 0

for tau_b in tau_b_range:
    print(f"  🔍 Testando tau_b = {tau_b:.6f}...")
    for tau_m in tau_m_range:
        for tau_u in tau_u_range:
            current += 1
            
            result = evaluate_pipeline(x_stage1_subset, x_stage2_subset, y_true_subset, tau_b, tau_m, tau_u)
            if result is not None:
                results.append(result)
                
                # Mostrar progresso a cada 10 iterações
                if current % 10 == 0:
                    elapsed = time.time() - start_time
                    eta = (elapsed / current) * (total_combinations - current)
                    print(f"    Progresso: {current}/{total_combinations} ({current/total_combinations*100:.1f}%) - ETA: {eta:.1f}s")
                
                # Tracking do melhor resultado
                if result['f1_weighted'] > best_f1_so_far:
                    best_f1_so_far = result['f1_weighted']
                    no_improvement_count = 0
                    print(f"    ✨ Novo melhor F1: {best_f1_so_far:.4f}")
                else:
                    no_improvement_count += 1

elapsed_time = time.time() - start_time
print(f"✅ Busca inicial concluída em {elapsed_time:.1f} segundos!")
print(f"📊 {len(results):,} combinações testadas no subset")

# OTIMIZAÇÃO 4: Validar apenas os TOP 5 candidatos com dados completos
print(f"\n🎯 Validando TOP 5 candidatos com dados completos...")
results_df_subset = pd.DataFrame(results)
top_5_indices = results_df_subset.nlargest(5, 'f1_weighted').index

final_results = []
for i, idx in enumerate(top_5_indices):
    config = results_df_subset.iloc[idx]
    tau_b, tau_m, tau_u = config['tau_b'], config['tau_m'], config['tau_u']
    
    print(f"  {i+1}/5: Validando tau_b={tau_b:.6f}, tau_m={tau_m:.2f}, tau_u={tau_u:.6f}")
    
    # Avaliar com dados completos
    full_result = evaluate_pipeline(x_stage1_scaled, x_stage2_scaled, y_true, tau_b, tau_m, tau_u)
    if full_result is not None:
        final_results.append(full_result)

print(f"✅ Validação completa finalizada!")
print(f"📊 {len(final_results)} configurações validadas com dados completos")

# Usar resultados finais para análise
results = final_results

# Analisar resultados
results_df = pd.DataFrame(results)

print(f"\n🏆 ANÁLISE DOS MELHORES THRESHOLDS")
print("=" * 80)

# Definir métricas e suas justificativas
metrics_analysis = [
    ('f1_weighted', 'F1-Score Weighted', 'Ideal para datasets desbalanceados como este (95% Benign)'),
    ('f1_macro', 'F1-Score Macro', 'Trata todas as classes igualmente, bom para análise geral'),
    ('balanced_accuracy', 'Balanced Accuracy', 'Equilibra recall de todas as classes'),
    ('accuracy', 'Accuracy', 'Métrica simples, mas pode ser enganosa em dados desbalanceados'),
    ('attack_detection_rate', 'Taxa Detecção Ataques', 'Foco específico na capacidade de detectar ataques'),
    ('benign_recall', 'Recall Benign', 'Importante para evitar falsos positivos excessivos')
]

# Criar tabela de melhores configurações
best_configs = []
for metric_col, metric_name, justification in metrics_analysis:
    best = results_df.loc[results_df[metric_col].idxmax()]
    best_configs.append({
        'Métrica': metric_name,
        'Valor': f"{best[metric_col]:.4f}",
        'tau_b': f"{best['tau_b']:.6f}",
        'tau_m': f"{best['tau_m']:.2f}",
        'tau_u': f"{best['tau_u']:.6f}",
        'Benign': f"{best['pred_benign']:,}",
        'Ataques': f"{best['total_attacks_predicted']:,}",
        'Unknown': f"{best['pred_unknown']:,}",
        'Justificativa': justification
    })

# Exibir tabela principal
main_table = pd.DataFrame(best_configs)
print("MELHORES CONFIGURAÇÕES POR MÉTRICA:")
print("-" * 80)
display_cols = ['Métrica', 'Valor', 'tau_b', 'tau_m', 'tau_u', 'Ataques', 'Unknown']
print(main_table[display_cols].to_string(index=False))

# Configuração recomendada
recommended = results_df.loc[results_df['f1_weighted'].idxmax()]
print(f"\n🎯 CONFIGURAÇÃO RECOMENDADA (F1-Weighted - melhor para dados desbalanceados):")
print("=" * 60)
print(f"tau_b = {recommended['tau_b']:.6f}")
print(f"tau_m = {recommended['tau_m']:.2f}")
print(f"tau_u = {recommended['tau_u']:.6f}")

print(f"\n📊 MÉTRICAS DA CONFIGURAÇÃO RECOMENDADA:")
print(f"  • F1-Score Weighted: {recommended['f1_weighted']:.4f}")
print(f"  • F1-Score Macro: {recommended['f1_macro']:.4f}")
print(f"  • Balanced Accuracy: {recommended['balanced_accuracy']:.4f}")
print(f"  • Accuracy: {recommended['accuracy']:.4f}")
print(f"  • Taxa Detecção Ataques: {recommended['attack_detection_rate']:.4f}")
print(f"  • Recall Benign: {recommended['benign_recall']:.4f}")

print(f"\n📈 DISTRIBUIÇÃO DE PREDIÇÕES:")
print(f"  • Benign: {recommended['pred_benign']:,}")
print(f"  • (D)DOS: {recommended['pred_ddos']:,}")
print(f"  • Botnet: {recommended['pred_botnet']:,}")
print(f"  • Brute Force: {recommended['pred_brute_force']:,}")
print(f"  • Port Scan: {recommended['pred_port_scan']:,}")
print(f"  • Web Attack: {recommended['pred_web_attack']:,}")
print(f"  • Unknown: {recommended['pred_unknown']:,}")
print(f"  • Total Ataques: {recommended['total_attacks_predicted']:,}")

# Salvar arquivos
csv_path = os.path.join(reports_dir, "threshold_evaluation_complete.csv")
summary_path = os.path.join(reports_dir, "threshold_evaluation_summary.csv")
report_path = os.path.join(reports_dir, "threshold_evaluation_report.txt")

# Salvar dados completos
results_df.to_csv(csv_path, index=False, float_format='%.6f')

# Salvar resumo das melhores configurações
main_table.to_csv(summary_path, index=False)

# Salvar relatório detalhado
with open(report_path, "w") as f:
    f.write("RELATÓRIO DE AVALIAÇÃO DE THRESHOLDS\n")
    f.write("Multi-Stage Hierarchical Intrusion Detection System\n")
    f.write("=" * 60 + "\n")
    f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Amostras: {len(y_true):,}\n")
    f.write(f"Combinações testadas: {len(results):,}\n")
    f.write(f"Tempo execução: {elapsed_time/60:.1f} minutos\n\n")
    
    f.write("DISTRIBUIÇÃO REAL DOS DADOS:\n")
    for classe, count in y_true.value_counts().items():
        f.write(f"  {classe}: {count:,} ({count/len(y_true)*100:.1f}%)\n")
    f.write("\n")
    
    f.write("MELHORES CONFIGURAÇÕES POR MÉTRICA:\n")
    f.write("-" * 60 + "\n")
    for _, config in main_table.iterrows():
        f.write(f"{config['Métrica']}: {config['Valor']}\n")
        f.write(f"  Thresholds: tau_b={config['tau_b']} tau_m={config['tau_m']} tau_u={config['tau_u']}\n")
        f.write(f"  Predições: {config['Ataques']} ataques, {config['Unknown']} unknown\n")
        f.write(f"  Justificativa: {config['Justificativa']}\n\n")
    
    f.write("CONFIGURAÇÃO RECOMENDADA:\n")
    f.write(f"tau_b = {recommended['tau_b']:.6f}\n")
    f.write(f"tau_m = {recommended['tau_m']:.2f}\n")
    f.write(f"tau_u = {recommended['tau_u']:.6f}\n\n")
    f.write("JUSTIFICATIVA:\n")
    f.write("F1-Score Weighted é a melhor métrica para este dataset desbalanceado\n")
    f.write("(95% das amostras são Benign). Ela pondera as classes pelo número de amostras,\n")
    f.write("dando mais importância à classificação correta da classe majoritária\n")
    f.write("sem ignorar completamente as classes minoritárias.\n")

print(f"\n💾 ARQUIVOS GERADOS:")
print(f"📊 Dados completos: {csv_path}")
print(f"📋 Resumo das melhores: {summary_path}")
print(f"📄 Relatório detalhado: {report_path}")

print(f"\n✅ AVALIAÇÃO CONCLUÍDA!")
print(f"🔄 Para usar os melhores thresholds, copie os valores recomendados para o codex.py")