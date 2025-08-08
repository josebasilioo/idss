#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


# Seed value
# Apparently you may use different seed values at each stage
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
# for later versions: 
# tf.compat.v1.set_random_seed(seed_value)

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

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.regularizers import l2
from util.AUROCEarlyStoppingPruneCallback import AUROCEarlyStoppingPruneCallback


# # Load Data 
# Download data into local folder https://gitlab.ilabt.imec.be/mverkerk/ids-dataset-cleaning

# In[2]:


train = {
    "ocsvm": {}, # 10k samples
    "ae": {}, # 100k samples
    "stage2": {}
}
val = {
    "ocsvm": {},
    "ae": {},
    "stage2": {}
}
test = {
    # "y"
    # "y_binary"
    # "y_unknown"
    # "x"
}


# ## Load Data Stage 1

# In[3]:


clean_dir = "/home/CIN/jbsn3/multi-stage-hierarchical-ids/ids-dataset-cleaning/cicids2017/clean"


train["ocsvm"]["x"], train["ocsvm"]["y"], x_benign_val, y_benign_val, _, _, x_malicious_train, y_malicious_train, _, _, _, _, _ = util.load_data(clean_dir, sample_size=1948, train_size=10000, val_size=129485, test_size=56468)

val["ocsvm"]["x"] = np.concatenate((x_benign_val, x_malicious_train))
val["ocsvm"]["y"] = np.concatenate((y_benign_val, np.full(y_malicious_train.shape[0], -1)))


train["ae"]["x"], train["ae"]["y"], x_benign_val, y_benign_val, _, _, x_malicious_train, y_malicious_train, _, _, _, _, _ = util.load_data(clean_dir, sample_size=1948, val_size=129485, test_size=56468)

val["ae"]["x"] = np.concatenate((x_benign_val, x_malicious_train))
val["ae"]["y"] = np.concatenate((y_benign_val, np.full(y_malicious_train.shape[0], -1)))


# ## Load Data Stage 2

# In[4]:


n_benign_val = 1500

x_benign_train, _, _, _, x_benign_test, y_benign_test, x_malicious_train, y_malicious_train, x_malicious_test, y_malicious_test, attack_type_train, _, _ = util.load_data(clean_dir, sample_size=1948, train_size=n_benign_val, val_size=6815, test_size=56468)
train["stage2"]["x"], x_val, train["stage2"]["y"], y_val = train_test_split(x_malicious_train, y_malicious_train, stratify=attack_type_train, test_size=1500, random_state=42, shuffle=True)

test['x'] = np.concatenate((x_benign_test, x_malicious_test))
test["y_n"] = np.concatenate((y_benign_test, np.full(y_malicious_test.shape[0], -1)))

val["stage2"]["x"] = np.concatenate((x_val, x_benign_train))
val["stage2"]["y"] = np.concatenate((y_val, np.full(n_benign_val, "Unknown")))

train["stage2"]["y_n"] = pd.get_dummies(train["stage2"]["y"])
val["stage2"]["y_n"] = pd.get_dummies(val["stage2"]["y"])

test["y"] = np.concatenate((np.full(56468, "Benign"), y_malicious_test))
test["y_unknown"] = np.where((test["y"] == "Heartbleed") | (test["y"] == "Infiltration"), "Unknown", test["y"])
test["y_unknown_all"] = np.where(test['y_unknown'] == 'Benign', "Unknown", test['y_unknown'])


# ## Scale the data

# In[5]:


scaler = QuantileTransformer(output_distribution='normal')
train['ocsvm']['x_s'] = scaler.fit_transform(train['ocsvm']['x'])
val['ocsvm']['x_s'] = scaler.transform(val['ocsvm']['x'])
test['ocsvm_s'] = scaler.transform(test['x'])

scaler = QuantileTransformer(output_distribution='normal')
train['ae']['x_s'] = scaler.fit_transform(train['ae']['x'])
val['ae']['x_s'] = scaler.transform(val['ae']['x'])
test['ae_s'] = scaler.transform(test['x'])

scaler = QuantileTransformer(output_distribution='normal')
train['stage2']['x_s'] = scaler.fit_transform(train['stage2']['x'])
val['stage2']['x_s'] = scaler.transform(val['stage2']['x'])
test['stage2_s'] = scaler.transform(test['x'])

scaler = QuantileTransformer(output_distribution='uniform')
train['stage2']['x_q'] = scaler.fit_transform(train['stage2']['x'])
val['stage2']['x_q'] = scaler.transform(val['stage2']['x'])
test['stage2_q'] = scaler.transform(test['x'])


# # Fetch Best Hyperparameters 
# see paper for experiment

# In[6]:
#

#f = open("results/predictions.pkl","rb")
#predictions = pickle.load(f)#
#f.close()
#f = open("results/thresholds.pkl","rb")
#thresholds = pickle.load(f)
#f.close()


# ## Best F1 score (macro & micro) and Accuracy

# In[8]:


#results_df = pd.read_csv("results/results_123.csv")
#results_df.sort_values(by="ACC", ascending=False).head(1)


# ## Best Balanced Accuracy

# In[10]:


#results_df[(results_df.index_1 == 3)].sort_values(by="bACC", ascending=False).head(1)


# ## Best overall

# In[9]:


#results_df[results_df.zero_day_recall_extension > 0.7].sort_values(by="f1_weighted", ascending=False).head(1)


# Look up corresponding thresholds from best models in tables below.\
# **Example:** best overall model uses threshold_b corresponding with metric F5 and FPR of 0.312, this is equal to -0.0002196942507948895

# In[10]:


#thresholds["stage1"][3]


# In[11]:


#thresholds["stage2"][4]


# In[12]:


#thresholds["stage2"][9]


# In[13]:


#thresholds["extension"][3]


# # Train Models
# Code below shows training for best overall model

# ## STAGE 1: One-Class SVM

# In[14]:


def create_ocsvm(params):
    return Pipeline(
        [
            ("pca", PCA(n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=42)), 
            ("ocsvm", OneClassSVM(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=True, max_iter=-1))
        ]
    ).set_params(**params)


# ### Train

# In[18]:


params_ocsvm = {
    "pca__n_components": 56,
    "ocsvm__kernel": "rbf",
    "ocsvm__gamma": 0.0632653906314333,
    "ocsvm__nu": 0.0002316646233151
}
ocsvm_model = create_ocsvm(params_ocsvm)
ocsvm_model.fit(train['ocsvm']['x_s'])


# ## Train with equal training size as AE (100k)

# First is always original 10k trainingset, second is new 100k trainingset

# In[15]:


params_ocsvm = {
    "pca__n_components": 56,
    "ocsvm__kernel": "rbf",
    "ocsvm__gamma": 0.0632653906314333,
    "ocsvm__nu": 0.0002316646233151
}
ocsvm_model = create_ocsvm(params_ocsvm)
ocsvm_model.fit(train['ae']['x_s'])


# ### Validation

# In[16]:


score_val = -ocsvm_model.decision_function(val['ocsvm']['x_s'])
curves_metrics, summary_metrics = util.evaluate_proba(val['ocsvm']['y'], score_val)


# In[17]:


score_val = -ocsvm_model.decision_function(val['ae']['x_s'])
curves_metrics, summary_metrics = util.evaluate_proba(val['ae']['y'], score_val)


# In[46]:


summary_metrics


# #### Define Thresholds
# Equal to 14 decimals after the comma as in original experiment

# In[17]:


quantiles = [0.995, 0.99, 0.975, 0.95]
print({(metric, fpr): t for metric, fpr, t in zip(summary_metrics.metric, summary_metrics.FPR, summary_metrics.threshold)})
print({q: np.quantile(score_val[val["ocsvm"]["y"] == 1], q) for q in quantiles})


# In[18]:


quantiles = [0.995, 0.99, 0.975, 0.95]
print({(metric, fpr): t for metric, fpr, t in zip(summary_metrics.metric, summary_metrics.FPR, summary_metrics.threshold)})
print({q: np.quantile(score_val[val["ae"]["y"] == 1], q) for q in quantiles})


# ### Test

# In[18]:


score_test = -ocsvm_model.decision_function(test['ocsvm_s'])
curves_metrics_test, summary_metrics_test = util.evaluate_proba(test["y_n"], score_test)
summary_metrics_test


# In[19]:


score_test = -ocsvm_model.decision_function(test['ae_s'])
curves_metrics_test, summary_metrics_test = util.evaluate_proba(test["y_n"], score_test)
summary_metrics_test


# ## STAGE 2: Random Forest

# In[20]:


def create_rf(params):
    return RandomForestClassifier(random_state=42).set_params(**params)


# ### Train

# In[21]:


params = {
    "n_estimators": 97,
    "max_samples": 0.9034128710297624,
    "max_features": 0.1751204590963604,
    "min_samples_leaf": 1
}
rf_model = create_rf(params)
rf_model.fit(train['stage2']['x_s'], train["stage2"]["y"])


# ### Validation

# In[22]:


y_proba_val_2 = rf_model.predict_proba(val['stage2']['x_s'])


# #### Define Thresholds

# In[23]:


fmacro, fweight, threshold_fscore, f_best = util.optimal_fscore_multi(val['stage2']['y'], y_proba_val_2, rf_model.classes_)
print(f_best["f1_weighted_threshold"])
y_pred_val_2 = np.where(np.max(y_proba_val_2, axis=1) > f_best["f1_weighted_threshold"], rf_model.classes_[np.argmax(y_proba_val_2, axis=1)], 'Unknown')


# ### Test

# In[24]:


y_proba_test_2 = rf_model.predict_proba(test['stage2_s'])
y_pred_test_2 = np.where(np.max(y_proba_test_2, axis=1) > f_best["f1_weighted_threshold"], rf_model.classes_[np.argmax(y_proba_test_2, axis=1)], 'Unknown')
print({
    "f1_macro": f1_score(test["y_unknown_all"], y_pred_test_2, average='macro'),
    "f1_weighted": f1_score(test["y_unknown_all"], y_pred_test_2, average='weighted'),
    'accuracy': accuracy_score(test["y_unknown_all"], y_pred_test_2),
    'balanced_accuracy': balanced_accuracy_score(test["y_unknown_all"], y_pred_test_2)
})


# # Test Multi-Stage Model
# Code belows shows inference for best overall model

# ### First Stage

# In[35]:


# y_proba_1 = predictions["stage1"][3] # Using saved results from initial experiment
y_proba_1 = score_test # See training ocsvm above


# In[83]:


threshold_b = -0.0002196942507948895 # See tables above
y_pred = np.where(y_proba_1 < threshold_b, "Benign", "Fraud").astype(object)
np.unique(y_pred, return_counts=True)


# In[36]:


threshold_b = -0.004199663778210894 # See tables above
y_pred = np.where(y_proba_1 < threshold_b, "Benign", "Fraud").astype(object)
np.unique(y_pred, return_counts=True)


# ### Second Stage

# In[37]:


# y_proba_2 = predictions['stage2'][9] # Using saved results from initial experiment
y_proba_2 = y_proba_test_2 # See training rf above


# In[38]:


threshold_m = 0.98 # See table above
y_pred_2 = np.where(np.max(y_proba_2[y_pred == "Fraud"], axis=1) > threshold_m, train["stage2"]["y_n"].columns[np.argmax(y_proba_2[y_pred == "Fraud"], axis=1)], 'Unknown')
np.unique(y_pred_2, return_counts=True)


# ### Combine first and second stage

# In[28]:
print("\n--- Verificando y_pred IMEDIATAMENTE antes do erro ---")
print(f"Tipo de y_pred: {type(y_pred)}")
print(f"dtype de y_pred: {y_pred.dtype}") # Adicione esta linha
if y_pred.size > 0: # Para evitar erro se y_pred estiver vazio
    print(f"Primeiro elemento de y_pred: {y_pred[0]}") # Adicione esta linha
    print(f"Tipo do primeiro elemento de y_pred: {type(y_pred[0])}") # Adicione esta linha
else:
    print("y_pred está vazio.")

print(f"Valores únicos em y_pred: {np.unique(y_pred)}")
num_frauds = (y_pred == "Fraud").sum()
print(f"Número de ocorrências de 'Fraud' em y_pred: {num_frauds}")

# Verificando a máscara diretamente
mask = (y_pred == "Fraud")
print(f"Soma da máscara (número de True): {mask.sum()}") # Este é o valor que o erro está vendo!
print(f"Primeiros 10 elementos da máscara: {mask[:10]}") # Para ver o conteúdo da máscara
print(f"Tamanho de y_pred_2: {len(y_pred_2)}")
print("---------------------------------------------------\n")

y_pred[y_pred == "Fraud"] = y_pred_2
np.unique(y_pred, return_counts=True)


# In[39]:

print("2222222222222222222222\n")

#y_pred[y_pred == "Fraud"] = y_pred_2
#np.unique(y_pred, return_counts=True)


# ### Extension stage

# In[29]:


threshold_u = 0.0040588613744241275 # See table above
y_pred_3 = np.where(y_proba_1[y_pred == "Unknown"] < threshold_u, "Benign", "Unknown")
np.unique(y_pred_3, return_counts=True)


# In[40]:


threshold_u = 0.007972254569416139 # See table above
y_pred_3 = np.where(y_proba_1[y_pred == "Unknown"] < threshold_u, "Benign", "Unknown")
np.unique(y_pred_3, return_counts=True)


# ### Combine predictions 3 stages

# In[41]:


y_pred[y_pred == "Unknown"] = y_pred_3
np.unique(y_pred, return_counts=True)


# # Final Confusion Matrix

# In[31]:


classes = ['Benign', '(D)DOS', 'Botnet', 'Brute Force', 'Port Scan', 'Web Attack', 'Unknown']
util.plot_confusion_matrix(test['y_unknown'], y_pred, values=classes, labels=classes)


# In[42]:


classes = ['Benign', '(D)DOS', 'Botnet', 'Brute Force', 'Port Scan', 'Web Attack', 'Unknown']
util.plot_confusion_matrix(test['y_unknown'], y_pred, values=classes, labels=classes)


# ## Train second stage with anomaly score stage 1 as extra feature

# Create probability values of train and validation data

# In[38]:


proba_train = -ocsvm_model.decision_function(train['stage2']['x_s'])
proba_val = -ocsvm_model.decision_function(val['stage2']['x_s'])
proba_test = -ocsvm_model.decision_function(test['stage2_s'])


# In[46]:


proba_val.shape


# In[45]:


val['stage2']['x_s'].shape


# In[52]:


test['stage2_s'].shape


# In[49]:


train_with_proba = np.column_stack((train['stage2']['x_s'], proba_train))
val_with_proba = np.column_stack((val['stage2']['x_s'], proba_val))
test_with_proba = np.column_stack((test['stage2_s'], proba_test))


# In[50]:


train_with_proba.shape


# In[51]:


val_with_proba.shape


# In[53]:


test_with_proba.shape


# Use new feature set to train and validate the model

# In[93]:


params = {
    "n_estimators": 97,
    "max_samples": 0.9034128710297624,
    "max_features": 0.1751204590963604,
    "min_samples_leaf": 1
}
rf_model = create_rf({})
rf_model.fit(train_with_proba, train["stage2"]["y"])


# In[94]:


y_proba_val_2_extra_feature = rf_model.predict_proba(val_with_proba)


# In[95]:


fmacro, fweight, threshold_fscore, f_best = util.optimal_fscore_multi(val['stage2']['y'], y_proba_val_2_extra_feature, rf_model.classes_)
print(f_best["f1_weighted_threshold"])
y_pred_val_2_extra_feature = np.where(np.max(y_proba_val_2_extra_feature, axis=1) > f_best["f1_weighted_threshold"], rf_model.classes_[np.argmax(y_proba_val_2_extra_feature, axis=1)], 'Unknown')


# In[97]:


y_proba_test_2_extra_feature = rf_model.predict_proba(test_with_proba)
y_pred_test_2_extra_feature = np.where(np.max(y_proba_test_2_extra_feature, axis=1) > f_best["f1_weighted_threshold"], rf_model.classes_[np.argmax(y_proba_test_2_extra_feature, axis=1)], 'Unknown')
print({
    "f1_macro": f1_score(test["y_unknown_all"], y_pred_test_2_extra_feature, average='macro'),
    "f1_weighted": f1_score(test["y_unknown_all"], y_pred_test_2_extra_feature, average='weighted'),
    'accuracy': accuracy_score(test["y_unknown_all"], y_pred_test_2_extra_feature),
    'balanced_accuracy': balanced_accuracy_score(test["y_unknown_all"], y_pred_test_2_extra_feature)
})


# With default parameters of original training set

# In[74]:


{'f1_macro': 0.8581703603761849, 'f1_weighted': 0.9802877478625802, 'accuracy': 0.9773029359804829, 'balanced_accuracy': 0.9734265804891981}


# ## Full model performance

# In[84]:


# y_proba_2 = predictions['stage2'][9] # Using saved results from initial experiment
y_proba_2 = y_proba_test_2_extra_feature # See training rf above


# In[85]:


threshold_m = 0.94 # See table above
y_pred_2 = np.where(np.max(y_proba_2[y_pred == "Fraud"], axis=1) > threshold_m, train["stage2"]["y_n"].columns[np.argmax(y_proba_2[y_pred == "Fraud"], axis=1)], 'Unknown')
np.unique(y_pred_2, return_counts=True)


# In[86]:

print("3333333333333333333\n")

y_pred[y_pred == "Fraud"] = y_pred_2
np.unique(y_pred, return_counts=True)


# In[87]:


threshold_u = 0.0040588613744241275 # See table above
y_pred_3 = np.where(y_proba_1[y_pred == "Unknown"] < threshold_u, "Benign", "Unknown")
np.unique(y_pred_3, return_counts=True)


# In[88]:


y_pred[y_pred == "Unknown"] = y_pred_3
np.unique(y_pred, return_counts=True)


# In[89]:


classes = ['Benign', '(D)DOS', 'Botnet', 'Brute Force', 'Port Scan', 'Web Attack', 'Unknown']
util.plot_confusion_matrix(test['y_unknown'], y_pred, values=classes, labels=classes)


# # With default params

# In[70]:


classes = ['Benign', '(D)DOS', 'Botnet', 'Brute Force', 'Port Scan', 'Web Attack', 'Unknown']
util.plot_confusion_matrix(test['y_unknown'], y_pred, values=classes, labels=classes)


# In[44]:


def generateConfusionGraphs(y_proba_2_new, threshold_m, include_metrics=False, save_plots=True, experiment_name="default"):
    """
    Gera gráficos de matriz de confusão (heatmaps) e salva em pasta organizada
    
    Args:
        y_proba_2_new: probabilidades do stage 2
        threshold_m: threshold para stage 2
        include_metrics: se deve incluir métricas
        save_plots: se deve salvar os gráficos
        experiment_name: nome do experimento para organização
    """
    fig, axs = plt.subplots(2,3, figsize=(18,12))
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(wspace=0.3)
    y_proba_1 = score_test
    metrics = []
    classes = ['Benign', '(D)DOS', 'Botnet', 'Brute Force', 'Port Scan', 'Web Attack', 'Unknown']
    
    # Stage 1: Binary Classification
    y_pred_1_n = np.where(y_proba_1 < threshold_b, 1, -1)
    confusion_1_binary = util.plot_confusion_matrix(test['y_n'], y_pred_1_n, values=[1, -1], labels=["Benign", "Fraud"], title="Stage 1", ax=axs[0, 0])
    y_pred = np.where(y_proba_1 < threshold_b, "Benign", "Fraud")
    
    # Stage 2: Multi-class Classification
    y_proba_2 = y_proba_2_new
    y_pred_2 = np.where(np.max(y_proba_2[y_pred == "Fraud"], axis=1) > threshold_m, train["stage2"]["y_n"].columns[np.argmax(y_proba_2[y_pred == "Fraud"], axis=1)], 'Unknown')
    confusion_2_multi = util.plot_confusion_matrix(test['y_unknown'][y_pred == "Fraud"], y_pred_2, values=classes, labels=classes, title="Stage 2", ax=axs[0, 1])

    y_pred = y_pred.astype(object)
    y_pred[y_pred == "Fraud"] = y_pred_2
    
    # Stage 1&2 Combined
    if include_metrics:
        result_12 = {
            "threshold_b": threshold_b,
            "threshold_m": threshold_m,
            "threshold_u": "-",
            "bACC": balanced_accuracy_score(test['y_unknown'], y_pred),
            "ACC": accuracy_score(test['y_unknown'], y_pred),
            "f1_micro": f1_score(test['y_unknown'], y_pred, average='micro'),
            "f1_macro": f1_score(test['y_unknown'], y_pred, average='macro'),
            "f1_weighted": f1_score(test['y_unknown'], y_pred, average='weighted'),
            "zero_day_recall_extension": "-",
            "zero_day_recall_total": "-"
        }
        metrics.append(result_12)
    confusion_12_multi = util.plot_confusion_matrix(test['y_unknown'], y_pred, values=classes, labels=classes, title="Stage 1&2 Combined", ax=axs[0, 2])
    
    # Extension Stage
    mask = ((y_pred == "Unknown") & (test['y_unknown_all'] == "Unknown"))
    y_pred_3 = np.where(y_proba_1[y_pred == "Unknown"] < threshold_u, "Benign", "Unknown")
    y_pred_3_n = np.where(y_proba_1[mask] < threshold_u, 1, -1)
    confusion_3_multi = util.plot_confusion_matrix(test['y_unknown'][y_pred == "Unknown"], y_pred_3, values=classes, labels=classes, title="Extension Multi-Class", ax=axs[1, 0])
    confusion_3_binary = util.plot_confusion_matrix(test['y_n'][mask], y_pred_3_n, values=[1, -1], labels=["Benign", "Zero-Day"], title="Extension Binary", ax=axs[1, 1])

    # Final Combined Result
    y_pred[y_pred == "Unknown"] = y_pred_3
    if include_metrics:
        result_123 = {
            "threshold_b": threshold_b,
            "threshold_m": threshold_m,
            "threshold_u": threshold_u,
            "bACC": balanced_accuracy_score(test['y_unknown'], y_pred),
            "ACC": accuracy_score(test['y_unknown'], y_pred),
            "f1_micro": f1_score(test['y_unknown'], y_pred, average='micro'),
            "f1_macro": f1_score(test['y_unknown'], y_pred, average='macro'),
            "f1_weighted": f1_score(test['y_unknown'], y_pred, average='weighted'),
            "zero_day_recall_extension": recall_score(test['y_n'][mask], y_pred_3_n, pos_label=-1),
            "zero_day_recall_total": (y_pred_3_n == -1).sum() / 47
        }
        metrics.append(result_123)
    confusion_123_multi = util.plot_confusion_matrix(test['y_unknown'], y_pred, values=classes, labels=classes, title="Stages 1,2 & Extension Combined", ax=axs[1, 2])
    
    # Salvar gráficos se solicitado
    if save_plots:
        import os
        from datetime import datetime
        
        # CRIAR PASTA graphics DENTRO da pasta atual do projeto
        base_dir = "./graphics"  # Caminho relativo - cria na pasta atual
        os.makedirs(base_dir, exist_ok=True)
        
        # Timestamp para evitar sobrescrita
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Salvar gráfico completo
        full_plot_path = os.path.join(base_dir, f"{experiment_name}_confusion_matrix_complete_{timestamp}.png")
        plt.savefig(full_plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico completo salvo: {full_plot_path}")
        
        # Salvar gráficos individuais - versão corrigida
        plot_info = [
            (axs[0, 0], "stage1_binary", "Stage 1 - Binary Classification"),
            (axs[0, 1], "stage2_multiclass", "Stage 2 - Multi-class Classification"),
            (axs[0, 2], "combined_stages", "Stage 1&2 Combined"),
            (axs[1, 0], "extension_multiclass", "Extension Multi-Class"),
            (axs[1, 1], "extension_binary", "Extension Binary"),
            (axs[1, 2], "final_results", "Final Combined Result")
        ]
        
        for ax, plot_type, title in plot_info:
            # Criar figura individual
            fig_individual, ax_individual = plt.subplots(1, 1, figsize=(8, 6))
            
            # Verificar se o subplot tem um heatmap (imagem)
            if len(ax.images) > 0:
                # Copiar o heatmap
                heatmap_data = ax.images[0].get_array()
                heatmap_cmap = ax.images[0].get_cmap()
                
                # Recriar o heatmap no novo subplot
                sns.heatmap(heatmap_data, cmap=heatmap_cmap, annot=True, fmt='.1%', ax=ax_individual)
                ax_individual.set_title(title)
                
                # Configurar labels se disponíveis
                if hasattr(ax, 'get_xlabel') and ax.get_xlabel():
                    ax_individual.set_xlabel(ax.get_xlabel())
                if hasattr(ax, 'get_ylabel') and ax.get_ylabel():
                    ax_individual.set_ylabel(ax.get_ylabel())
                
                # Salvar gráfico individual
                individual_path = os.path.join(base_dir, f"{experiment_name}_{plot_type}_{timestamp}.png")
                plt.savefig(individual_path, dpi=300, bbox_inches='tight')
                print(f" Gráfico individual salvo: {individual_path}")
            
            plt.close(fig_individual)
        
        # Salvar métricas em arquivo CSV se disponíveis
        if include_metrics and metrics:
            metrics_df = pd.DataFrame(metrics)
            metrics_path = os.path.join(base_dir, f"{experiment_name}_metrics_{timestamp}.csv")
            metrics_df.to_csv(metrics_path, index=False)
            print(f" Métricas salvas: {metrics_path}")
        
        plt.close(fig)  # Fechar figura principal
    
    return pd.DataFrame(metrics)


# In[91]:


generateConfusionGraphs(y_proba_test_2_extra_feature, 0.94, True)


# # Default model params

# In[98]:


generateConfusionGraphs(y_proba_test_2_extra_feature, 0.94, True)


# In[99]:


generateConfusionGraphs(y_proba_test_2_extra_feature, 0.98, True)


# In[92]:


generateConfusionGraphs(y_proba_test_2, 0.98, True)


# In[45]:


generateConfusionGraphs(y_proba_test_2, 0.98, True)


# # Salvar Modelos Treinados
# Salvar os modelos na pasta /models para uso posterior

# In[46]:


import os
import pickle

# CRIAR PASTA models DENTRO da pasta atual do projeto
models_dir = "./models"  # Caminho relativo - cria na pasta atual
os.makedirs(models_dir, exist_ok=True)

print("💾 Salvando modelos treinados...")

# Salvar Stage 1: OCSVM Pipeline (com PCA)
try:
    with open(os.path.join(models_dir, "stage1_ocsvm.p"), "wb") as f:
        pickle.dump(ocsvm_model, f)
    print("✅ Stage1 OCSVM Pipeline salvo: ./models/stage1_ocsvm.p")
except Exception as e:
    print(f"❌ Erro ao salvar Stage1 OCSVM Pipeline: {e}")

# Salvar Stage 2: Random Forest Pipeline
try:
    with open(os.path.join(models_dir, "stage2_rf.p"), "wb") as f:
        pickle.dump(rf_model, f)
    print("✅ Stage2 RF Pipeline salvo: ./models/stage2_rf.p")
except Exception as e:
    print(f"❌ Erro ao salvar Stage2 RF Pipeline: {e}")

# Salvar Random Forest com feature extra (anomaly score do stage 1)
try:
    with open(os.path.join(models_dir, "stage2_rf_extra_feature.p"), "wb") as f:
        pickle.dump(rf_model, f)  # Este é o rf_model treinado com train_with_proba
    print("✅ Stage2 RF com feature extra salvo: ./models/stage2_rf_extra_feature.p")
except Exception as e:
    print(f"❌ Erro ao salvar Stage2 RF com feature extra: {e}")

# Salvar scalers separadamente
try:
    # Scaler para OCSVM (QuantileTransformer)
    scaler_ocsvm = QuantileTransformer(output_distribution='normal')
    scaler_ocsvm.fit(train['ocsvm']['x'])
    with open(os.path.join(models_dir, "stage1_ocsvm_scaler.p"), "wb") as f:
        pickle.dump(scaler_ocsvm, f)
    print("✅ Stage1 OCSVM Scaler salvo: ./models/stage1_ocsvm_scaler.p")
except Exception as e:
    print(f"❌ Erro ao salvar Stage1 OCSVM Scaler: {e}")

try:
    # Scaler para RF (QuantileTransformer)
    scaler_rf = QuantileTransformer(output_distribution='normal')
    scaler_rf.fit(train['stage2']['x'])
    with open(os.path.join(models_dir, "stage2_rf_scaler.p"), "wb") as f:
        pickle.dump(scaler_rf, f)
    print("✅ Stage2 RF Scaler salvo: ./models/stage2_rf_scaler.p")
except Exception as e:
    print(f"❌ Erro ao salvar Stage2 RF Scaler: {e}")

# Salvar thresholds importantes
thresholds_dict = {
    "threshold_b": threshold_b,
    "threshold_m": threshold_m,
    "threshold_u": threshold_u,
    "f1_weighted_threshold": f_best["f1_weighted_threshold"] if 'f_best' in locals() else None
}

try:
    with open(os.path.join(models_dir, "thresholds.p"), "wb") as f:
        pickle.dump(thresholds_dict, f)
    print("✅ Thresholds salvos: ./models/thresholds.p")
except Exception as e:
    print(f"❌ Erro ao salvar thresholds: {e}")

# Salvar informações sobre as classes
classes_info = {
    "stage2_classes": train["stage2"]["y_n"].columns.tolist() if 'train' in locals() else [],
    "all_classes": ['Benign', '(D)DOS', 'Botnet', 'Brute Force', 'Port Scan', 'Web Attack', 'Unknown']
}

try:
    with open(os.path.join(models_dir, "classes_info.p"), "wb") as f:
        pickle.dump(classes_info, f)
    print("✅ Informações das classes salvas: ./models/classes_info.p")
except Exception as e:
    print(f"❌ Erro ao salvar informações das classes: {e}")

print("\n📊 Resumo do salvamento:")
print("Modelos salvos na pasta /models:")
print("- stage1_ocsvm.p: Pipeline OCSVM com PCA")
print("- stage2_rf.p: Pipeline Random Forest")
print("- stage2_rf_extra_feature.p: RF com feature extra")
print("- stage1_ocsvm_scaler.p: Scaler para OCSVM")
print("- stage2_rf_scaler.p: Scaler para RF")
print("- thresholds.p: Thresholds otimizados")
print("- classes_info.p: Informações das classes")

print("\n🎯 Os modelos estão prontos para uso em outros arquivos!")

# Salvar modelos individuais (sem pipeline) se necessário
try:
    # Extrair o modelo OCSVM do pipeline
    ocsvm_individual = ocsvm_model.named_steps['ocsvm']
    with open(os.path.join(models_dir, "stage1_ocsvm_model.p"), "wb") as f:
        pickle.dump(ocsvm_individual, f)
    print("✅ Stage1 OCSVM Model individual salvo: ./models/stage1_ocsvm_model.p")
except Exception as e:
    print(f"❌ Erro ao salvar Stage1 OCSVM Model individual: {e}")

try:
    # Extrair o PCA do pipeline
    pca_individual = ocsvm_model.named_steps['pca']
    with open(os.path.join(models_dir, "stage1_pca.p"), "wb") as f:
        pickle.dump(pca_individual, f)
    print("✅ Stage1 PCA individual salvo: ./models/stage1_pca.p")
except Exception as e:
    print(f"❌ Erro ao salvar Stage1 PCA individual: {e}")

# Salvar baseline RF (modelo treinado sem feature extra)
try:
    baseline_rf_model = create_rf(params)
    baseline_rf_model.fit(train['stage2']['x_s'], train["stage2"]["y"])
    with open(os.path.join(models_dir, "baseline_rf.p"), "wb") as f:
        pickle.dump(baseline_rf_model, f)
    print("✅ Baseline RF salvo: ./models/baseline_rf.p")
except Exception as e:
    print(f"❌ Erro ao salvar Baseline RF: {e}")

# Salvar baseline RF scaler
try:
    baseline_rf_scaler = QuantileTransformer(output_distribution='normal')
    baseline_rf_scaler.fit(train['stage2']['x'])
    with open(os.path.join(models_dir, "baseline_rf_scaler.p"), "wb") as f:
        pickle.dump(baseline_rf_scaler, f)
    print("✅ Baseline RF Scaler salvo: ./models/baseline_rf_scaler.p")
except Exception as e:
    print(f"❌ Erro ao salvar Baseline RF Scaler: {e}")

print("\n✅ Todos os modelos foram salvos com sucesso!")

