# coding: utf-8

# In[3]:


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


# In[4]:


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


# In[5]:


# get_ipython().system('pip install pyarrow')


# # Load Test Data
# - Load data from parquet or csv
# - Map 'Heartbleed' and 'Infiltration' attack classes to 'Unknown'

# In[6]:


test = pd.read_parquet(os.path.join(data_dir, "test.parquet"))
# test = pd.read_csv(os.path.join(data_dir, "test.csv"))

y = test["Y"].replace(["Heartbleed", "Infiltration"], "Unknown")
x = test.drop(columns=['Y'])

y.value_counts()


# ### Load additional infiltration samples from 2018

# In[42]:


infiltration_2018 = pd.read_parquet(os.path.join(data_dir, "infiltration_2018.parquet"))

y_18 = infiltration_2018['y']
x_18 = infiltration_2018.drop(columns=['y'])

y_18.value_counts()


# # Load Models
# - the pipelines with feature scaler and optimized model combined for binary detection and multi-class classification
# - the individual feature scalers and optimized models
# - Random Forest (RF) optimized baseline model and feature scaler
# - Optimized models following Bovenzi et al. for comparitative analysis

# In[ ]:


# Optimized pipelines
# f = open(os.path.join(models_dir, "stage1_ocsvm.p"),"rb")
#stage1 = pickle.load(f)
#f.close()
#f = open(os.path.join(models_dir, "stage2_rf.p"),"rb")
#stage2 = pickle.load(f)
#f.close()

# Individual feature scalers and classification models
#f = open(os.path.join(models_dir, "stage1_ocsvm_model.p"),"rb")
#stage1_model = pickle.load(f)
#f.close()
#f = open(os.path.join(models_dir, "stage1_ocsvm_scaler.p"),"rb")
#stage1_scaler = pickle.load(f)
#f.close()
#f = open(os.path.join(models_dir, "stage2_rf_model.p"),"rb")
#stage2_model = pickle.load(f)
#f.close()
#f = open(os.path.join(models_dir, "stage2_rf_scaler.p"),"rb")
#stage2_scaler = pickle.load(f)
#f.close()

# RF baseline model and feature scaler
#f = open(os.path.join(models_dir, "baseline_rf.p"),"rb")
#baseline_rf = pickle.load(f)
#f.close()
#f = open(os.path.join(models_dir, "baseline_rf_scaler.p"),"rb")
#baseline_rf_scaler = pickle.load(f)
#f.close()

# Optimized models for Bovenzi et al.
#from tensorflow import keras
#sota_stage1 = keras.models.load_model(os.path.join(models_dir, "sota_stage1.h5"))
#f = open(os.path.join(models_dir, "sota_stage2.p"),"rb")
#sota_stage2 = pickle.load(f)
#f.close()

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

# ===== CARREGAMENTO DE MODELOS POR GRUPOS =====

print(" Carregando modelos por grupos...")

# GRUPO 1: PIPELINES PRINCIPAIS
print("\n GRUPO 1: PIPELINES PRINCIPAIS")
print("   - stage1_ocsvm.p: Pipeline OCSVM com PCA (detecção binária)")
print("   - stage2_rf.p: Pipeline Random Forest (classificação multiclasse)")

stage1 = load_model_safe(os.path.join(models_dir, "stage1_ocsvm.p"), "Stage1 OCSVM Pipeline")
stage2 = load_model_safe(os.path.join(models_dir, "stage2_rf.p"), "Stage2 RF Pipeline")

# GRUPO 2: MODELOS INDIVIDUAIS E SCALERS
print("\n🔧 GRUPO 2: MODELOS INDIVIDUAIS E SCALERS")
print("   - stage1_ocsvm_model.p: Modelo OCSVM individual")
print("   - stage1_ocsvm_scaler.p: Scaler QuantileTransformer para OCSVM")
print("   - stage2_rf_scaler.p: Scaler QuantileTransformer para RF")
print("   - stage1_pca.p: PCA individual")

stage1_model = load_model_safe(os.path.join(models_dir, "stage1_ocsvm_model.p"), "Stage1 OCSVM Model")
stage1_scaler = load_model_safe(os.path.join(models_dir, "stage1_ocsvm_scaler.p"), "Stage1 OCSVM Scaler")
stage2_scaler = load_model_safe(os.path.join(models_dir, "stage2_rf_scaler.p"), "Stage2 RF Scaler")
stage1_pca = load_model_safe(os.path.join(models_dir, "stage1_pca.p"), "Stage1 PCA")

# GRUPO 3: MODELOS BASELINE E ALTERNATIVOS
print("\n🔧 GRUPO 3: MODELOS BASELINE E ALTERNATIVOS")
print("   - baseline_rf.p: Random Forest baseline (sem feature extra)")
print("   - stage2_rf_extra_feature.p: RF com feature extra")

baseline_rf = load_model_safe(os.path.join(models_dir, "baseline_rf.p"), "Baseline RF")
baseline_rf_scaler = load_model_safe(os.path.join(models_dir, "baseline_rf_scaler.p"), "Baseline RF Scaler")
stage2_rf_extra = load_model_safe(os.path.join(models_dir, "stage2_rf_extra_feature.p"), "Stage2 RF Extra Feature")

# GRUPO 4: CONFIGURAÇÕES E THRESHOLDS
print("\n⚙️ GRUPO 4: CONFIGURAÇÕES E THRESHOLDS")
print("   - thresholds.p: Thresholds otimizados (tau_b, tau_m, tau_u)")
print("   - classes_info.p: Informações das classes")

thresholds = load_model_safe(os.path.join(models_dir, "thresholds.p"), "Thresholds")
classes_info = load_model_safe(os.path.join(models_dir, "classes_info.p"), "Classes Info")

print(f"\n📊 RESUMO: Modelos carregados por grupos")

# Salvar informações dos grupos em arquivo
with open(os.path.join(reports_dir, "modelos_carregados.txt"), "w") as f:
    f.write("=== MODELOS CARREGADOS POR GRUPOS ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write(" GRUPO 1: PIPELINES PRINCIPAIS\n")
    f.write("- stage1_ocsvm.p: Pipeline OCSVM com PCA (detecção binária)\n")
    f.write("- stage2_rf.p: Pipeline Random Forest (classificação multiclasse)\n\n")
    
    f.write("🔧 GRUPO 2: MODELOS INDIVIDUAIS E SCALERS\n")
    f.write("- stage1_ocsvm_model.p: Modelo OCSVM individual\n")
    f.write("- stage1_ocsvm_scaler.p: Scaler QuantileTransformer para OCSVM\n")
    f.write("- stage2_rf_scaler.p: Scaler QuantileTransformer para RF\n")
    f.write("- stage1_pca.p: PCA individual\n\n")
    
    f.write("🔧 GRUPO 3: MODELOS BASELINE E ALTERNATIVOS\n")
    f.write("- baseline_rf.p: Random Forest baseline (sem feature extra)\n")
    f.write("- baseline_rf_scaler.p: Scaler QuantileTransformer para RF baseline\n")
    f.write("- stage2_rf_extra_feature.p: RF com feature extra\n\n")
    
    f.write("⚙️ GRUPO 4: CONFIGURAÇÕES E THRESHOLDS\n")
    f.write("- thresholds.p: Thresholds otimizados (tau_b, tau_m, tau_u)\n")
    f.write("- classes_info.p: Informações das classes\n\n")

print(f"✅ Informações dos modelos salvas em {os.path.join(reports_dir, 'modelos_carregados.txt')}")


# In[8]:


stage1


# In[9]:


stage2


# In[10]:


baseline_rf


# In[11]:


# sota_stage1 - não existe


# In[12]:


# sota_stage2 - não existe


# # Thresholds $\tau_B$, $\tau_M$ and $\tau_U$
# These balanced thresholds are experimentally obtained, see full paper for more details.

# In[13]:


tau_b = -0.0002196942507948895
tau_m = 0.98
tau_u = 0.0040588613744241275


# # Evaluation of Time Complexity

# In[14]:


import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# `hids_predict`: Function to preform classification of all stages combined of the novel hierarchical multi-stage intrusion detection approach by Verkerken et al.
# 
# `hids_sota_predict`: Function to evaluate former SotA approach existing of two stages by Bovenzi et al.

# SOLUÇÃO GERAL: Função que ajusta dados para qualquer modelo
def fix_features_for_model(data, model):
    """Ajusta o número de features dos dados para ser compatível com qualquer modelo"""
    if model is None:
        return data
    
    # Converter para numpy se for DataFrame
    if isinstance(data, pd.DataFrame):
        data_array = data.values
    else:
        data_array = data
    
    # Determinar número de features esperado pelo modelo
    expected_features = None
    
    # Para pipelines
    if hasattr(model, 'steps'):
        # Pegar o primeiro step (scaler)
        first_step = model.steps[0][1]
        if hasattr(first_step, 'n_features_in_'):
            expected_features = first_step.n_features_in_
    
    # Para modelos individuais
    elif hasattr(model, 'n_features_in_'):
        expected_features = model.n_features_in_
    elif hasattr(model, 'n_features_'):
        expected_features = model.n_features_
    
    if expected_features is None:
        return data_array
    
    current_features = data_array.shape[1]
    
    if current_features == expected_features:
        return data_array
    elif current_features < expected_features:
        # Adicionar features dummy
        missing_features = expected_features - current_features
        dummy_cols = np.zeros((data_array.shape[0], missing_features))
        return np.column_stack([data_array, dummy_cols])
    else:
        # Remover features extras
        return data_array[:, :expected_features]


# In[45]:


def hids_predict(x, tau_b, tau_m, tau_u):
    # Aplicar correção de features para ambos os modelos
    x_fixed_stage1 = fix_features_for_model(x, stage1)
    x_fixed_stage2 = fix_features_for_model(x, stage2)
    
    proba_1 = -stage1.decision_function(x_fixed_stage1) # invert sign to act as anomaly score 
    pred_1 = np.where(proba_1 < tau_b, "Benign", "Attack").astype(object)
    proba_2 = stage2.predict_proba(x_fixed_stage2[pred_1 == "Attack"])
    pred_2 = np.where(
        np.max(proba_2, axis=1) > tau_m, 
        stage2.classes_[np.argmax(proba_2, axis=1)], 
        "Unknown")
    proba_3 = proba_1[pred_1 == "Attack"][pred_2 == "Unknown"]
    pred_3 = np.where(proba_3 < tau_u, "Benign", "Unknown")
    pred_1[pred_1 == "Attack"] = pred_2
    pred_1[pred_1 == "Unknown"] = pred_3
    return pred_1


# In[35]:


def hids_sota_predict(x, tau_b, tau_m):
    # Aplicar correção de features para os scalers
    x_fixed_stage1 = fix_features_for_model(x, stage1_scaler)
    x_fixed_stage2 = fix_features_for_model(x, stage2_scaler)
    
    x_s = stage1_scaler.transform(x_fixed_stage1)
    x_pred = stage1_model.predict(x_s) # Assuming stage1_model is the individual model
    proba_1 = np.sum((x_s - x_pred)**2, axis=1)
    pred_1 = np.where(proba_1 < tau_b, "Benign", "Attack").astype(object)
    x_s = stage2_scaler.transform(x_fixed_stage2)
    proba_2 = stage2_model.predict_proba(x_s) # Assuming stage2_model is the individual model
    pred_1[pred_1 == "Attack"] = np.where(
        np.max(proba_2, axis=1) > tau_m, 
        stage2.classes_[np.argmax(proba_2, axis=1)], 
        "Unknown")
    return pred_1


# ### Max F-score thesholds

# In[25]:


# Medir tempo e salvar resultado
start_time = time.time()
for _ in range(3):
    tau_b = -0.0002196942507948895
    tau_m = 0.98
    tau_u = 0.004530129828299084
    y = hids_predict(x, tau_b, tau_m, tau_u)
end_time = time.time()
avg_time = (end_time - start_time) / 3

# Salvar métrica de tempo
with open(os.path.join(reports_dir, "timing_max_fscore.txt"), "w") as f:
    f.write("=== MÉTRICA DE TEMPO - MAX F-SCORE THRESHOLDS ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tempo médio: {avg_time:.6f} segundos\n")
    f.write(f"Execuções: 3\n")
    f.write(f"Thresholds: tau_b={tau_b}, tau_m={tau_m}, tau_u={tau_u}\n")

print(f"✅ Métrica de tempo Max F-score salva em {os.path.join(reports_dir, 'timing_max_fscore.txt')}")


# ### Max bACC thresholds

# In[26]:


# Medir tempo e salvar resultado
start_time = time.time()
for _ in range(3):
    tau_b = -0.0004064190600459828
    tau_m = 0.98
    tau_u = 0.0006590265510403005
    y = hids_predict(x, tau_b, tau_m, tau_u)
end_time = time.time()
avg_time = (end_time - start_time) / 3

# Salvar métrica de tempo
with open(os.path.join(reports_dir, "timing_max_bacc.txt"), "w") as f:
    f.write("=== MÉTRICA DE TEMPO - MAX BACC THRESHOLDS ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tempo médio: {avg_time:.6f} segundos\n")
    f.write(f"Execuções: 3\n")
    f.write(f"Thresholds: tau_b={tau_b}, tau_m={tau_m}, tau_u={tau_u}\n")

print(f"✅ Métrica de tempo Max bACC salva em {os.path.join(reports_dir, 'timing_max_bacc.txt')}")


# ### Best "balanced" thesholds

# In[27]:


# Medir tempo e salvar resultado
start_time = time.time()
for _ in range(3):
    tau_b = -0.0002196942507948895
    tau_m = 0.98
    tau_u = 0.0040588613744241275
    y = hids_predict(x, tau_b, tau_m, tau_u)
end_time = time.time()
avg_time = (end_time - start_time) / 3

# Salvar métrica de tempo
with open(os.path.join(reports_dir, "timing_balanced.txt"), "w") as f:
    f.write("=== MÉTRICA DE TEMPO - BALANCED THRESHOLDS ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tempo médio: {avg_time:.6f} segundos\n")
    f.write(f"Execuções: 3\n")
    f.write(f"Thresholds: tau_b={tau_b}, tau_m={tau_m}, tau_u={tau_u}\n")

print(f"✅ Métrica de tempo Balanced salva em {os.path.join(reports_dir, 'timing_balanced.txt')}")


# ### Baseline RF

# In[31]:


threshold = 0.43


# In[32]:


# Medir tempo e salvar resultado
start_time = time.time()
for _ in range(3):
    x_fixed = fix_features_for_model(x, baseline_rf_scaler)
    x_s = baseline_rf_scaler.transform(x_fixed)
    y_proba = baseline_rf.predict_proba(x_s)
    y_pred = np.where(np.max(y_proba, axis=1) > threshold, baseline_rf.classes_[np.argmax(y_proba, axis=1)], 'Unknown')
end_time = time.time()
avg_time = (end_time - start_time) / 3

# Salvar métrica de tempo
with open(os.path.join(reports_dir, "timing_baseline_rf.txt"), "w") as f:
    f.write("=== MÉTRICA DE TEMPO - BASELINE RF ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tempo médio: {avg_time:.6f} segundos\n")
    f.write(f"Execuções: 3\n")
    f.write(f"Threshold: {threshold}\n")

print(f"✅ Métrica de tempo Baseline RF salva em {os.path.join(reports_dir, 'timing_baseline_rf.txt')}")


# ### Bovenzi et al.

# In[25]:


# Thresholds experimentally optimized
tau_b = 0.7580776764761945
tau_m = 0.98


# In[38]:


# Modelos SOTA não disponíveis - pulando esta avaliação
print("⚠️ Modelos SOTA não disponíveis - pulando avaliação Bovenzi et al.")

# Medir tempo e salvar resultado
start_time = time.time()
for _ in range(3):
    y = hids_sota_predict(x, tau_b, tau_m)
end_time = time.time()
avg_time = (end_time - start_time) / 3

# Salvar métrica de tempo
with open(os.path.join(reports_dir, "timing_bovenzi.txt"), "w") as f:
    f.write("=== MÉTRICA DE TEMPO - BOVENZI ET AL. ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tempo médio: {avg_time:.6f} segundos\n")
    f.write(f"Execuções: 3\n")
    f.write(f"Thresholds: tau_b={tau_b}, tau_m={tau_m}\n")

print(f"✅ Métrica de tempo Bovenzi salva em {os.path.join(reports_dir, 'timing_bovenzi.txt')}")


# ### Single sample
# Inference time for predicting a single flow

# In[10]:


sample = np.array(x.values[0]).reshape(1, -1)
sample


# In[11]:


import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# In[12]:


# Medir tempo e salvar resultado
start_time = time.time()
for _ in range(10):
    x_s = stage1_scaler.transform(sample)
    proba = -stage1_model.decision_function(x_s)
    pred = np.where(proba < tau_b, "Benign", "Attack").astype(object)
end_time = time.time()
avg_time = (end_time - start_time) / 10

# Salvar métrica de tempo
with open(os.path.join(reports_dir, "timing_single_stage1.txt"), "w") as f:
    f.write("=== MÉTRICA DE TEMPO - SINGLE SAMPLE STAGE1 ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tempo médio: {avg_time:.6f} segundos\n")
    f.write(f"Execuções: 10\n")
    f.write(f"Descrição: Stage1 individual (scaler + model)\n")

print(f"✅ Métrica de tempo single sample stage1 salva em {os.path.join(reports_dir, 'timing_single_stage1.txt')}")


# In[13]:


# Medir tempo e salvar resultado
start_time = time.time()
for _ in range(10):
    x_s = stage2_scaler.transform(sample)
    proba = stage2_model.predict_proba(x_s) # Assuming stage2_model is the individual model
    pred_2 = np.where(
        np.max(proba, axis=1) > tau_m, 
        stage2.classes_[np.argmax(proba, axis=1)], 
        "Unknown")
end_time = time.time()
avg_time = (end_time - start_time) / 10

# Salvar métrica de tempo
with open(os.path.join(reports_dir, "timing_single_stage2.txt"), "w") as f:
    f.write("=== MÉTRICA DE TEMPO - SINGLE SAMPLE STAGE2 ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tempo médio: {avg_time:.6f} segundos\n")
    f.write(f"Execuções: 10\n")
    f.write(f"Descrição: Stage2 individual (scaler + model)\n")

print(f"✅ Métrica de tempo single sample stage2 salva em {os.path.join(reports_dir, 'timing_single_stage2.txt')}")


# In[17]:


# Medir tempo e salvar resultado
start_time = time.time()
for _ in range(10):
    proba_1 = -stage1.decision_function(sample) # invert sign to act as anomaly score 
    pred_1 = np.where(proba_1 < tau_b, "Benign", "Attack").astype(object)
end_time = time.time()
avg_time = (end_time - start_time) / 10

# Salvar métrica de tempo
with open(os.path.join(reports_dir, "timing_single_pipeline_stage1.txt"), "w") as f:
    f.write("=== MÉTRICA DE TEMPO - SINGLE SAMPLE PIPELINE STAGE1 ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tempo médio: {avg_time:.6f} segundos\n")
    f.write(f"Execuções: 10\n")
    f.write(f"Descrição: Stage1 pipeline completo\n")

print(f"✅ Métrica de tempo single sample pipeline stage1 salva em {os.path.join(reports_dir, 'timing_single_pipeline_stage1.txt')}")


# In[18]:


# Medir tempo e salvar resultado
start_time = time.time()
for _ in range(10):
    proba_2 = stage2.predict_proba(sample)
    pred_2 = np.where(
        np.max(proba_2, axis=1) > tau_m, 
        stage2.classes_[np.argmax(proba_2, axis=1)], 
        "Unknown")
end_time = time.time()
avg_time = (end_time - start_time) / 10

# Salvar métrica de tempo
with open(os.path.join(reports_dir, "timing_single_pipeline_stage2.txt"), "w") as f:
    f.write("=== MÉTRICA DE TEMPO - SINGLE SAMPLE PIPELINE STAGE2 ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tempo médio: {avg_time:.6f} segundos\n")
    f.write(f"Execuções: 10\n")
    f.write(f"Descrição: Stage2 pipeline completo\n")

print(f"✅ Métrica de tempo single sample pipeline stage2 salva em {os.path.join(reports_dir, 'timing_single_pipeline_stage2.txt')}")


# # Evaluate Multi-Stage Model

# ## Stage 1: Binary Detection

# In[15]:


proba_1 = -stage1.decision_function(x) # invert sign to act as anomaly score 
pred_1 = np.where(proba_1 < tau_b, "Benign", "Attack").astype(object)
np.unique(pred_1, return_counts=True)

# Salvar resultado
unique_values, counts = np.unique(pred_1, return_counts=True)
with open(os.path.join(reports_dir, "stage1_binary_detection.txt"), "w") as f:
    f.write("=== STAGE 1: BINARY DETECTION ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("Distribuição de predições:\n")
    for value, count in zip(unique_values, counts):
        f.write(f"{value}: {count}\n")

print(f"✅ Resultado Stage 1 salvo em {os.path.join(reports_dir, 'stage1_binary_detection.txt')}")


# In[20]:


# Medir tempo e salvar resultado
start_time = time.time()
for _ in range(3):
    proba_1 = -stage1.decision_function(x) # invert sign to act as anomaly score 
    pred_1 = np.where(proba_1 < tau_b, "Benign", "Attack").astype(object)
    np.unique(pred_1, return_counts=True)
end_time = time.time()
avg_time = (end_time - start_time) / 3

# Salvar métrica de tempo
with open(os.path.join(reports_dir, "timing_stage1_binary.txt"), "w") as f:
    f.write("=== MÉTRICA DE TEMPO - STAGE 1 BINARY DETECTION ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tempo médio: {avg_time:.6f} segundos\n")
    f.write(f"Execuções: 3\n")

print(f"✅ Métrica de tempo Stage 1 binary salva em {os.path.join(reports_dir, 'timing_stage1_binary.txt')}")


# ## Stage 2: Multi-Class Classification

# In[9]:


proba_2 = stage2.predict_proba(x[pred_1 == "Attack"])
pred_2 = np.where(
    np.max(proba_2, axis=1) > tau_m, 
    stage2.classes_[np.argmax(proba_2, axis=1)], 
    "Unknown")
np.unique(pred_2, return_counts=True)

# Salvar resultado
unique_values, counts = np.unique(pred_2, return_counts=True)
with open(os.path.join(reports_dir, "stage2_multiclass.txt"), "w") as f:
    f.write("=== STAGE 2: MULTI-CLASS CLASSIFICATION ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("Distribuição de predições:\n")
    for value, count in zip(unique_values, counts):
        f.write(f"{value}: {count}\n")

print(f"✅ Resultado Stage 2 salvo em {os.path.join(reports_dir, 'stage2_multiclass.txt')}")


# In[16]:


# Medir tempo e salvar resultado
start_time = time.time()
for _ in range(3):
    proba_2 = stage2.predict_proba(x[pred_1 == "Attack"])
    pred_2 = np.where(
        np.max(proba_2, axis=1) > tau_m, 
        stage2.classes_[np.argmax(proba_2, axis=1)], 
        "Unknown")
    np.unique(pred_2, return_counts=True)
end_time = time.time()
avg_time = (end_time - start_time) / 3

# Salvar métrica de tempo
with open(os.path.join(reports_dir, "timing_stage2_multiclass.txt"), "w") as f:
    f.write("=== MÉTRICA DE TEMPO - STAGE 2 MULTI-CLASS ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tempo médio: {avg_time:.6f} segundos\n")
    f.write(f"Execuções: 3\n")

print(f"✅ Métrica de tempo Stage 2 multiclass salva em {os.path.join(reports_dir, 'timing_stage2_multiclass.txt')}")


# ## Extension Stage: Zero-Day Detection

# In[10]:


proba_3 = proba_1[pred_1 == "Attack"][pred_2 == "Unknown"]
pred_3 = np.where(proba_3 < tau_u, "Benign", "Unknown")
np.unique(pred_3, return_counts=True)

# Salvar resultado
unique_values, counts = np.unique(pred_3, return_counts=True)
with open(os.path.join(reports_dir, "stage3_zeroday.txt"), "w") as f:
    f.write("=== STAGE 3: ZERO-DAY DETECTION ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("Distribuição de predições:\n")
    for value, count in zip(unique_values, counts):
        f.write(f"{value}: {count}\n")

print(f"✅ Resultado Stage 3 salvo em {os.path.join(reports_dir, 'stage3_zeroday.txt')}")


# ## Combine stages

# In[11]:


y_pred = pred_1.copy()
y_pred[y_pred == "Attack"] = pred_2
y_pred[y_pred == "Unknown"] = pred_3
np.unique(y_pred, return_counts=True)

# Salvar resultado
unique_values, counts = np.unique(y_pred, return_counts=True)
with open(os.path.join(reports_dir, "combined_stages.txt"), "w") as f:
    f.write("=== COMBINED STAGES ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("Distribuição final de predições:\n")
    for value, count in zip(unique_values, counts):
        f.write(f"{value}: {count}\n")

print(f"✅ Resultado combined stages salvo em {os.path.join(reports_dir, 'combined_stages.txt')}")


# # Statistics and Visualizations of the Results

# In[12]:


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


# In[13]:


classes = ['Benign', '(D)DOS', 'Botnet', 'Brute Force', 'Port Scan', 'Web Attack', 'Unknown']
plot_confusion_matrix(y, y_pred, values=classes, labels=classes, metrics=True)

# Salvar matriz de confusão
plt.savefig(os.path.join(reports_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Matriz de confusão salva em {os.path.join(reports_dir, 'confusion_matrix.png')}")


# In[14]:


report = classification_report(y, y_pred, digits=4)
print(report)

# Salvar relatório de classificação
with open(os.path.join(reports_dir, "classification_report.txt"), "w") as f:
    f.write("=== RELATÓRIO DE CLASSIFICAÇÃO ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(report)

print(f"✅ Relatório de classificação salvo em {os.path.join(reports_dir, 'classification_report.txt')}")


# ## Robustness - Preform classification on additional "infiltration" samples from cic-ids-2018

# In[53]:


tau_b = -0.0002196942507948895
tau_m = 0.98
tau_u = 0.0040588613744241275 # balanced threshold -> 29,02% recall on infiltration 2018
# tau_u = 0.0006590265510403005 # bACC threshold -> 78,38% recall on infiltration 2018
y = hids_predict(x_18, tau_b, tau_m, tau_u)

# Salvar resultado
unique_values, counts = np.unique(y, return_counts=True)
with open(os.path.join(reports_dir, "infiltration_2018_hids.txt"), "w") as f:
    f.write("=== INFILTRAÇÃO 2018 - HIDS PRINCIPAL ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Thresholds: tau_b={tau_b}, tau_m={tau_m}, tau_u={tau_u}\n")
    f.write(f"Recall esperado: 29,02% (balanced threshold)\n\n")
    f.write("Distribuição de predições:\n")
    for value, count in zip(unique_values, counts):
        f.write(f"{value}: {count}\n")

print(f"✅ Resultado infiltração 2018 HIDS salvo em {os.path.join(reports_dir, 'infiltration_2018_hids.txt')}")


# In[54]:


np.unique(y, return_counts=True)


# In[50]:


np.unique(y, return_counts=True)


# In[57]:


# Modelos SOTA não disponíveis - pulando esta avaliação
print("⚠️ Modelos SOTA não disponíveis - pulando avaliação infiltração 2018 Bovenzi")

# Medir tempo e salvar resultado
start_time = time.time()
for _ in range(3):
    y = hids_sota_predict(x_18, tau_b, tau_m) # 86.99% recall on infiltration 2018

# Salvar resultado
unique_values, counts = np.unique(y, return_counts=True)
with open(os.path.join(reports_dir, "infiltration_2018_bovenzi.txt"), "w") as f:
    f.write("=== INFILTRAÇÃO 2018 - BOVENZI ET AL. ===\n")
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Thresholds: tau_b={tau_b}, tau_m={tau_m}\n")
    f.write(f"Recall esperado: 86,99%\n\n")
    f.write("Distribuição de predições:\n")
    for value, count in zip(unique_values, counts):
        f.write(f"{value}: {count}\n")

print(f"✅ Resultado infiltração 2018 Bovenzi salvo em {os.path.join(reports_dir, 'infiltration_2018_bovenzi.txt')}")


# In[58]:


np.unique(y, return_counts=True)


# In[59]:


if baseline_rf_scaler is not None and baseline_rf is not None:
    x_fixed = fix_features_for_model(x_18, baseline_rf_scaler)
    x_s = baseline_rf_scaler.transform(x_fixed)
    y_proba = baseline_rf.predict_proba(x_s)
    y_pred = np.where(np.max(y_proba, axis=1) > 0.43, baseline_rf.classes_[np.argmax(y_proba, axis=1)], 'Unknown')
    # 0.06% recall on infiltration 2018

    # Salvar resultado
    unique_values, counts = np.unique(y_pred, return_counts=True)
    with open(os.path.join(reports_dir, "infiltration_2018_baseline.txt"), "w") as f:
        f.write("=== INFILTRAÇÃO 2018 - BASELINE RF ===\n")
        f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Threshold: 0.43\n")
        f.write(f"Recall esperado: 0,06%\n\n")
        f.write("Distribuição de predições:\n")
        for value, count in zip(unique_values, counts):
            f.write(f"{value}: {count}\n")

    print(f"✅ Resultado infiltração 2018 Baseline salvo em {os.path.join(reports_dir, 'infiltration_2018_baseline.txt')}")
else:
    print("⚠️ Baseline RF não disponível - pulando avaliação")


# In[61]:


np.unique(y_pred, return_counts=True)

print(f"\n🎉 Todos os resultados foram salvos na pasta '{reports_dir}'!")

