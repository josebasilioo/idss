# codex.py - Versão simplificada da pipeline

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
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

# Criar pasta reports se não existir
if not os.path.exists(reports_dir):
    os.makedirs(reports_dir)

# Dicionário para armazenar tempos
timing_results = {
    'data_loading': 0,
    'model_loading': 0,
    'scaler_loading': 0,
    'data_scaling': 0,
    'stage1_inference': 0,
    'stage2_inference': 0,
    'stage3_inference': 0,
    'total_pipeline': 0,
    'report_generation': 0
}

print("📊 Carregando dados...")
start_time = time.time()

# Carregar dados de teste
test = pd.read_parquet(os.path.join(data_dir, "test.parquet"))
y_true = test["Y"].replace(["Heartbleed", "Infiltration"], "Unknown")
x = test.drop(columns=['Y'])

timing_results['data_loading'] = time.time() - start_time
print(f"✅ Dados carregados: {x.shape[0]} amostras, {x.shape[1]} features")
print(f"⏱️ Tempo carregamento dados: {timing_results['data_loading']:.3f}s")
print("Distribuição real:", y_true.value_counts().to_dict())

print("🔧 Carregando modelos...")
start_time = time.time()

# Carregar modelos treinados
with open(os.path.join(models_dir, "stage1_ocsvm.p"), "rb") as f:
    stage1 = pickle.load(f)

with open(os.path.join(models_dir, "stage2_rf.p"), "rb") as f:
    stage2 = pickle.load(f)

timing_results['model_loading'] = time.time() - start_time
print("✅ Modelos carregados")
print(f"⏱️ Tempo carregamento modelos: {timing_results['model_loading']:.3f}s")

print("🔧 Carregando escaladores...")
start_time = time.time()

# Carregar escaladores necessários
with open(os.path.join(models_dir, "stage1_ocsvm_scaler.p"), "rb") as f:
    scaler_stage1 = pickle.load(f)

with open(os.path.join(models_dir, "stage2_rf_scaler.p"), "rb") as f:
    scaler_stage2 = pickle.load(f)

timing_results['scaler_loading'] = time.time() - start_time
print("✅ Escaladores carregados")
print(f"⏱️ Tempo carregamento escaladores: {timing_results['scaler_loading']:.3f}s")

# Thresholds otimizados
# THRESHOLDS RECOMENDADOS:
threshold_b = -0.005000  # MUITO mais baixo - forçar mais amostras para Stage 2
threshold_m = 0.80       # Mais baixo ainda para aceitar Web Attacks com menor confiança
threshold_u = 0.001000   # Manter ajustado

print(f"🎯 Thresholds: tau_b={threshold_b}, tau_m={threshold_m}, tau_u={threshold_u}")

print("🚀 Executando pipeline...")
pipeline_start = time.time()

# ⚙️ ESCALAR OS DADOS ANTES DE USAR NOS MODELOS
print("⚙️ Escalando dados...")
start_time = time.time()
x_stage1_scaled = scaler_stage1.transform(x)
x_stage2_scaled = scaler_stage2.transform(x)
timing_results['data_scaling'] = time.time() - start_time
print(f"⏱️ Tempo escalamento dados: {timing_results['data_scaling']:.3f}s")

# Stage 1: Detecção binária (Benign vs Attack) - USAR DADOS ESCALADOS
print("🔍 Executando Stage 1 (OCSVM)...")
start_time = time.time()
score_test = -stage1.decision_function(x_stage1_scaled)
y_pred = np.where(score_test < threshold_b, "Benign", "Attack").astype(object)
timing_results['stage1_inference'] = time.time() - start_time

print(f"🔍 DEBUG Stage 1:")
print(f"   Score range: [{np.min(score_test):.6f}, {np.max(score_test):.6f}]")
print(f"   Score mean: {np.mean(score_test):.6f}")
print(f"   Score std: {np.std(score_test):.6f}")
print(f"   Threshold_b: {threshold_b}")
print(f"   Amostras < threshold_b (Benign): {np.sum(score_test < threshold_b)}")
print(f"   Amostras >= threshold_b (Attack): {np.sum(score_test >= threshold_b)}")

print(f"Stage 1 - Benign: {np.sum(y_pred == 'Benign')}, Attack: {np.sum(y_pred == 'Attack')}")
print(f"⏱️ Tempo Stage 1: {timing_results['stage1_inference']:.3f}s")

# Stage 2: Classificação multi-classe (apenas para amostras Attack) - USAR DADOS ESCALADOS
print("🔍 Executando Stage 2 (Random Forest)...")
start_time = time.time()
if np.sum(y_pred == "Attack") > 0:
    # Usar dados escalados para Stage 2
    x_attack_scaled = x_stage2_scaled[y_pred == "Attack"]
    y_proba_test_2 = stage2.predict_proba(x_attack_scaled)
    max_probas = np.max(y_proba_test_2, axis=1)
    
    print(f"🔍 DEBUG Stage 2:")
    print(f"   Max proba range: [{np.min(max_probas):.6f}, {np.max(max_probas):.6f}]")
    print(f"   Max proba mean: {np.mean(max_probas):.6f}")
    print(f"   Threshold_m: {threshold_m}")
    print(f"   Amostras > threshold_m: {np.sum(max_probas > threshold_m)}")
    
    y_pred_2 = np.where(
        np.max(y_proba_test_2, axis=1) > threshold_m, 
        stage2.classes_[np.argmax(y_proba_test_2, axis=1)], 
        'Unknown'
    )
    
    print(f"Stage 2 - Distribuição: {pd.Series(y_pred_2).value_counts().to_dict()}")
    
    # Aplicar predições do Stage 2
    y_pred[y_pred == "Attack"] = y_pred_2
else:
    print("Stage 2 - Nenhuma amostra classificada como Attack")

timing_results['stage2_inference'] = time.time() - start_time
print(f"⏱️ Tempo Stage 2: {timing_results['stage2_inference']:.3f}s")

# Stage 3: Detecção de zero-day (para amostras Unknown)
print("🔍 Executando Stage 3 (Zero-day Detection)...")
start_time = time.time()
if np.sum(y_pred == "Unknown") > 0:
    unknown_scores = score_test[y_pred == "Unknown"]
    y_pred_3 = np.where(unknown_scores < threshold_u, "Benign", "Unknown")
    
    print(f"Stage 3 - Benign: {np.sum(y_pred_3 == 'Benign')}, Unknown: {np.sum(y_pred_3 == 'Unknown')}")
    
    # Aplicar predições do Stage 3
    y_pred[y_pred == "Unknown"] = y_pred_3
else:
    print("Stage 3 - Nenhuma amostra Unknown para processar")

timing_results['stage3_inference'] = time.time() - start_time
print(f"⏱️ Tempo Stage 3: {timing_results['stage3_inference']:.3f}s")

# Calcular tempo total da pipeline
timing_results['total_pipeline'] = time.time() - pipeline_start

print("✅ Pipeline executada")

# Resultado final
final_distribution = pd.Series(y_pred).value_counts()
print(f"\n📊 Distribuição final das predições:")
for classe, count in final_distribution.items():
    print(f"  {classe}: {count}")

print("\n📈 Gerando matriz de confusão...")
start_time = time.time()

# Função para plotar matriz de confusão
def plot_confusion_matrix(y_true, y_pred, title="Matriz de Confusão"):
    # Definir ordem específica das classes
    desired_order = ['Benign', '(D)DOS', 'Botnet', 'Brute Force', 'Port Scan', 'Web Attack', 'Unknown']
    
    # Obter classes únicas presentes nos dados
    unique_classes = set(y_true) | set(y_pred)
    
    # Usar apenas as classes que existem nos dados, na ordem desejada
    classes = [cls for cls in desired_order if cls in unique_classes]
    
    # Adicionar qualquer classe não prevista no final (caso existam)
    for cls in unique_classes:
        if cls not in classes:
            classes.append(cls)
    
    # Calcular matriz de confusão
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Calcular percentuais
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    
    # Criar anotações com percentual e count
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if c == 0:
                annot[i, j] = '0'
            else:
                annot[i, j] = f'{p:.1f}%\n({c})'
    
    # Plotar
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_perc, 
                xticklabels=classes, 
                yticklabels=classes,
                annot=annot, 
                fmt='', 
                cmap='Blues',
                cbar_kws={'label': 'Percentual (%)'})
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Predição', fontsize=12)
    plt.ylabel('Real', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Salvar
    output_path = os.path.join(reports_dir, "confusion_matrix.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

# Gerar matriz de confusão
matrix_path = plot_confusion_matrix(y_true, y_pred, "Matriz de Confusão - Multi-Stage HIDS")
print(f"✅ Matriz de confusão salva em: {matrix_path}")

# Gerar relatório de classificação
report = classification_report(y_true, y_pred, digits=4)
report_path = os.path.join(reports_dir, "classification_report.txt")

with open(report_path, "w") as f:
    f.write("=== RELATÓRIO DE CLASSIFICAÇÃO ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Thresholds: tau_b={threshold_b}, tau_m={threshold_m}, tau_u={threshold_u}\n\n")
    f.write(report)

print(f"✅ Relatório de classificação salvo em: {report_path}")

timing_results['report_generation'] = time.time() - start_time

# Calcular tempo total de execução
total_time = sum(timing_results.values()) - timing_results['total_pipeline']  # Evitar dupla contagem
timing_results['total_execution'] = total_time

# Gerar relatório detalhado de timing
print(f"\n⏱️ RELATÓRIO DE PERFORMANCE:")
print("=" * 50)
print(f"📊 Carregamento dados:     {timing_results['data_loading']:.3f}s")
print(f"🔧 Carregamento modelos:   {timing_results['model_loading']:.3f}s")
print(f"⚙️ Carregamento escaladores: {timing_results['scaler_loading']:.3f}s")
print(f"🔄 Escalamento dados:      {timing_results['data_scaling']:.3f}s")
print("-" * 50)
print(f"🔍 Stage 1 (OCSVM):        {timing_results['stage1_inference']:.3f}s")
print(f"🔍 Stage 2 (Random Forest): {timing_results['stage2_inference']:.3f}s")
print(f"🔍 Stage 3 (Zero-day):     {timing_results['stage3_inference']:.3f}s")
print(f"📊 Pipeline total:         {timing_results['total_pipeline']:.3f}s")
print("-" * 50)
print(f"📈 Geração relatórios:     {timing_results['report_generation']:.3f}s")
print("=" * 50)
print(f"⏱️ TEMPO TOTAL:            {timing_results['total_execution']:.3f}s")

# Calcular throughput
samples_per_second = len(x) / timing_results['total_pipeline']
print(f"🚀 Throughput pipeline:    {samples_per_second:.0f} amostras/segundo")
print(f"🚀 Throughput por estágio:")
print(f"   Stage 1: {len(x) / timing_results['stage1_inference']:.0f} amostras/segundo")
if timing_results['stage2_inference'] > 0:
    stage2_samples = np.sum(y_pred != 'Benign')  # Amostras processadas no Stage 2
    print(f"   Stage 2: {stage2_samples / timing_results['stage2_inference']:.0f} amostras/segundo ({stage2_samples} amostras)")
if timing_results['stage3_inference'] > 0:
    stage3_samples = np.sum(y_pred == 'Unknown')  # Amostras processadas no Stage 3
    print(f"   Stage 3: {stage3_samples / timing_results['stage3_inference']:.0f} amostras/segundo ({stage3_samples} amostras)")

# Salvar relatório de timing
timing_path = os.path.join(reports_dir, "timing_report.txt")
with open(timing_path, "w") as f:
    f.write("=== RELATÓRIO DE PERFORMANCE ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Amostras processadas: {len(x):,}\n\n")
    
    f.write("TEMPOS DE EXECUÇÃO:\n")
    f.write(f"Carregamento dados:      {timing_results['data_loading']:.3f}s\n")
    f.write(f"Carregamento modelos:    {timing_results['model_loading']:.3f}s\n")
    f.write(f"Carregamento escaladores: {timing_results['scaler_loading']:.3f}s\n")
    f.write(f"Escalamento dados:       {timing_results['data_scaling']:.3f}s\n")
    f.write(f"Stage 1 (OCSVM):         {timing_results['stage1_inference']:.3f}s\n")
    f.write(f"Stage 2 (Random Forest): {timing_results['stage2_inference']:.3f}s\n")
    f.write(f"Stage 3 (Zero-day):      {timing_results['stage3_inference']:.3f}s\n")
    f.write(f"Pipeline total:          {timing_results['total_pipeline']:.3f}s\n")
    f.write(f"Geração relatórios:      {timing_results['report_generation']:.3f}s\n")
    f.write(f"Tempo total:             {timing_results['total_execution']:.3f}s\n\n")
    
    f.write("THROUGHPUT:\n")
    f.write(f"Pipeline geral:          {samples_per_second:.0f} amostras/segundo\n")
    f.write(f"Stage 1:                 {len(x) / timing_results['stage1_inference']:.0f} amostras/segundo\n")
    if timing_results['stage2_inference'] > 0:
        stage2_samples = np.sum(y_pred != 'Benign')
        f.write(f"Stage 2:                 {stage2_samples / timing_results['stage2_inference']:.0f} amostras/segundo\n")
    if timing_results['stage3_inference'] > 0:
        stage3_samples = np.sum(y_pred == 'Unknown')
        f.write(f"Stage 3:                 {stage3_samples / timing_results['stage3_inference']:.0f} amostras/segundo\n")

print(f"⏱️ Relatório de timing salvo em: {timing_path}")

print(f"\n🎉 Execução completa! Resultados salvos em '{reports_dir}'")