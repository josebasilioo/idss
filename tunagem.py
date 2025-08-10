#!/usr/bin/env python
# coding: utf-8

"""
TUNAGEM DE HIPERPARÂMETROS - Multi-Stage HIDS
Otimização eficiente de hiperparâmetros usando Optuna + subsets pequenos
Estratégia: Tunagem rápida com datasets reduzidos para encontrar melhores configs
"""

import os
import numpy as np
import pandas as pd
import pickle
import time
from datetime import datetime
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import warnings
warnings.filterwarnings('ignore')

# Imports dos modelos
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Configurar seeds para reproducibilidade
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
import random
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Importar utilitários
import util.common as util

print("🎯 TUNAGEM DE HIPERPARÂMETROS - Multi-Stage HIDS")
print("=" * 80)
print("🚀 Estratégia: Datasets pequenos + Optuna + Early Stopping")
print("⏱️ Tempo estimado: 30-60 minutos por modelo")

# Configurações
clean_dir = "/home/CIN/jbsn3/multi-stage-hierarchical-ids/ids-dataset-cleaning/cicids2018/clean"
results_dir = "/home/CIN/jbsn3/multi-stage-hierarchical-ids/tunagem_results"
os.makedirs(results_dir, exist_ok=True)

# Configurações de tunagem (datasets MUITO pequenos para velocidade)
TUNE_CONFIG = {
    'ocsvm': {
        'train_size': 5000,    # 5K para treino (muito rápido)
        'val_size': 8000,      # 8K para validação
        'test_size': 5000,     # 5K para teste
        'n_trials': 50,        # 50 tentativas
        'timeout': 1800        # 30 minutos máximo
    },
    'ae': {
        'train_size': 15000,   # 15K para treino
        'val_size': 10000,     # 10K para validação  
        'test_size': 5000,     # 5K para teste
        'n_trials': 30,        # 30 tentativas
        'timeout': 2400        # 40 minutos máximo
    },
    'rf': {
        'train_size': 3000,    # 3K para treino
        'val_size': 5000,      # 5K para validação
        'test_size': 3000,     # 3K para teste
        'n_trials': 100,       # 100 tentativas (RF é rápido)
        'timeout': 1200        # 20 minutos máximo
    }
}

def load_tune_data(model_name):
    """Carregar dados otimizados para tunagem"""
    config = TUNE_CONFIG[model_name]
    
    print(f"📊 Carregando dados para {model_name.upper()}...")
    start_time = time.time()
    
    if model_name == 'rf':
        # Para RF, precisamos de dados malicious também
        _, _, _, _, _, _, x_malicious_train, y_malicious_train, x_malicious_test, y_malicious_test, attack_type_train, attack_type_test, _ = util.load_data(
            clean_dir,
            sample_size=500,  # Amostra pequena de malicious
            train_size=config['train_size'],
            val_size=config['val_size'],
            test_size=config['test_size']
        )
        
        # Split malicious para treino/val
        x_train, x_val, y_train, y_val = train_test_split(
            x_malicious_train, y_malicious_train,
            test_size=0.3, random_state=42, stratify=attack_type_train
        )
        
        data = {
            'x_train': x_train, 'y_train': y_train,
            'x_val': x_val, 'y_val': y_val,
            'x_test': x_malicious_test, 'y_test': y_malicious_test
        }
    else:
        # Para OCSVM e AE, apenas dados benign
        x_benign_train, y_benign_train, x_benign_val, y_benign_val, x_benign_test, y_benign_test, x_malicious_train, y_malicious_train, _, _, _, _, _ = util.load_data(
            clean_dir,
            sample_size=500,
            train_size=config['train_size'],
            val_size=config['val_size'], 
            test_size=config['test_size']
        )
        
        # Para validação: benign + malicious (detecção de anomalias)
        x_val = np.concatenate((x_benign_val, x_malicious_train))
        y_val = np.concatenate((y_benign_val, np.full(x_malicious_train.shape[0], -1)))
        
        data = {
            'x_train': x_benign_train, 'y_train': y_benign_train,
            'x_val': x_val, 'y_val': y_val,
            'x_test': x_benign_test, 'y_test': y_benign_test
        }
    
    load_time = time.time() - start_time
    print(f"✅ Dados carregados em {load_time:.1f}s")
    print(f"   Treino: {data['x_train'].shape}, Val: {data['x_val'].shape}")
    
    return data

def tune_ocsvm(trial, data):
    """Otimizar hiperparâmetros do OneClassSVM - MESMAS VARIÁVEIS do treinamento.py"""
    
    # Tunar exatamente as mesmas variáveis do treinamento.py
    params = {
        "pca__n_components": trial.suggest_int('pca__n_components', 30, 80),  # atual: 56
        "ocsvm__kernel": trial.suggest_categorical('ocsvm__kernel', ['rbf', 'poly', 'sigmoid']),  # atual: 'rbf'
        "ocsvm__gamma": trial.suggest_float('ocsvm__gamma', 0.001, 0.5, log=True),  # atual: 0.0632653906314333
        "ocsvm__nu": trial.suggest_float('ocsvm__nu', 0.00001, 0.01, log=True)  # atual: 0.0002316646233151
    }
    
    try:
        # Escalar dados
        scaler = QuantileTransformer(n_quantiles=1000, random_state=42)
        x_train_scaled = scaler.fit_transform(data['x_train'])
        x_val_scaled = scaler.transform(data['x_val'])
        
        # Criar pipeline EXATAMENTE igual ao treinamento.py
        pipeline = Pipeline([
            ("pca", PCA(n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=42)), 
            ("ocsvm", OneClassSVM(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=False, max_iter=-1))
        ]).set_params(**params)
        
        # Treinar modelo
        pipeline.fit(x_train_scaled)
        
        # Avaliar
        scores = -pipeline.decision_function(x_val_scaled)
        y_val_binary = (data['y_val'] == 1).astype(int)  # 1 para benign, 0 para attack
        
        # Usar AUC como métrica
        auc = roc_auc_score(y_val_binary, scores)
        
        return auc
        
    except Exception as e:
        print(f"Erro no trial {trial.number}: {e}")
        return 0.0

def tune_autoencoder(trial, data):
    """Otimizar hiperparâmetros do Autoencoder"""
    
    # Hiperparâmetros para testar
    params = {
        'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128, 256]),
        'encoding_dim': trial.suggest_categorical('encoding_dim', [8, 16, 32, 64]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'l2_reg': trial.suggest_float('l2_reg', 1e-6, 1e-2, log=True)
    }
    
    try:
        # Escalar dados
        scaler = MinMaxScaler()
        x_train_scaled = scaler.fit_transform(data['x_train'])
        x_val_scaled = scaler.transform(data['x_val'])
        
        # Construir modelo
        input_dim = x_train_scaled.shape[1]
        
        # Encoder
        input_layer = Input(shape=(input_dim,))
        hidden = Dense(params['hidden_dim'], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(params['l2_reg']))(input_layer)
        encoded = Dense(params['encoding_dim'], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(params['l2_reg']))(hidden)
        
        # Decoder  
        decoded = Dense(params['hidden_dim'], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(params['l2_reg']))(encoded)
        output_layer = Dense(input_dim, activation='sigmoid')(decoded)
        
        # Autoencoder
        autoencoder = Model(input_layer, output_layer)
        autoencoder.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mse')
        
        # Treinar com early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = autoencoder.fit(
            x_train_scaled, x_train_scaled,
            epochs=50,  # Máximo 50 épocas
            batch_size=params['batch_size'],
            validation_data=(x_val_scaled[data['y_val'] == 1], x_val_scaled[data['y_val'] == 1]),  # Apenas benign para validação
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Avaliar: calcular erro de reconstrução
        x_val_benign = x_val_scaled[data['y_val'] == 1]
        x_val_malicious = x_val_scaled[data['y_val'] == -1]
        
        # Erros de reconstrução
        benign_errors = np.mean(np.square(x_val_benign - autoencoder.predict(x_val_benign, verbose=0)), axis=1)
        malicious_errors = np.mean(np.square(x_val_malicious - autoencoder.predict(x_val_malicious, verbose=0)), axis=1)
        
        # Separabilidade: malicious deve ter erro maior que benign
        separability = np.mean(malicious_errors) - np.mean(benign_errors)
        
        # Limpar memória
        del autoencoder
        tf.keras.backend.clear_session()
        
        return separability
        
    except Exception as e:
        print(f"Erro no trial {trial.number}: {e}")
        tf.keras.backend.clear_session()
        return 0.0

def tune_random_forest(trial, data):
    """Otimizar hiperparâmetros do Random Forest - MESMAS VARIÁVEIS do treinamento.py"""
    
    # Tunar exatamente as mesmas variáveis do treinamento.py
    params = {
        "n_estimators": trial.suggest_int('n_estimators', 50, 200),  # atual: 97
        "max_samples": trial.suggest_float('max_samples', 0.7, 1.0),  # atual: 0.9034128710297624
        "max_features": trial.suggest_float('max_features', 0.1, 0.5),  # atual: 0.1751204590963604
        "min_samples_leaf": trial.suggest_int('min_samples_leaf', 1, 5)  # atual: 1
    }
    
    try:
        # Escalar dados
        scaler = QuantileTransformer(n_quantiles=500, random_state=42)
        x_train_scaled = scaler.fit_transform(data['x_train'])
        x_val_scaled = scaler.transform(data['x_val'])
        
        # Criar modelo EXATAMENTE igual ao treinamento.py
        model = RandomForestClassifier(random_state=42).set_params(**params)
        
        # Treinar modelo
        model.fit(x_train_scaled, data['y_train'])
        
        # Avaliar
        y_pred = model.predict(x_val_scaled)
        f1 = f1_score(data['y_val'], y_pred, average='weighted')
        
        return f1
        
    except Exception as e:
        print(f"Erro no trial {trial.number}: {e}")
        return 0.0

def run_optimization(model_name):
    """Executar otimização para um modelo específico"""
    
    print(f"\n🎯 OTIMIZANDO {model_name.upper()}")
    print("-" * 50)
    
    # Carregar dados
    data = load_tune_data(model_name)
    config = TUNE_CONFIG[model_name]
    
    # Configurar Optuna
    study_name = f"{model_name}_optimization"
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # Função objetivo
    if model_name == 'ocsvm':
        objective = lambda trial: tune_ocsvm(trial, data)
    elif model_name == 'ae':
        objective = lambda trial: tune_autoencoder(trial, data)
    elif model_name == 'rf':
        objective = lambda trial: tune_random_forest(trial, data)
    
    # Executar otimização
    print(f"🚀 Iniciando {config['n_trials']} trials (timeout: {config['timeout']/60:.0f}min)")
    start_time = time.time()
    
    study.optimize(
        objective,
        n_trials=config['n_trials'],
        timeout=config['timeout'],
        show_progress_bar=True
    )
    
    optimization_time = time.time() - start_time
    
    # Resultados
    print(f"\n✅ Otimização concluída em {optimization_time/60:.1f} minutos!")
    print(f"🏆 Melhor score: {study.best_value:.4f}")
    print(f"🎯 Melhores parâmetros:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    
    # Salvar resultados
    results = {
        'model': model_name,
        'best_score': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
        'optimization_time': optimization_time,
        'timestamp': datetime.now().isoformat()
    }
    
    # Salvar em arquivo
    results_file = os.path.join(results_dir, f"{model_name}_best_params.pkl")
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Salvar relatório texto
    report_file = os.path.join(results_dir, f"{model_name}_optimization_report.txt")
    with open(report_file, 'w') as f:
        f.write(f"RELATÓRIO DE OTIMIZAÇÃO - {model_name.upper()}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Trials executados: {len(study.trials)}\n")
        f.write(f"Tempo de otimização: {optimization_time/60:.1f} minutos\n")
        f.write(f"Melhor score: {study.best_value:.4f}\n\n")
        f.write("MELHORES PARÂMETROS:\n")
        for key, value in study.best_params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nTOP 5 TRIALS:\n")
        for i, trial in enumerate(sorted(study.trials, key=lambda x: x.value or 0, reverse=True)[:5]):
            f.write(f"{i+1}. Score: {trial.value:.4f} | Params: {trial.params}\n")
    
    print(f"💾 Resultados salvos em: {results_file}")
    print(f"📄 Relatório salvo em: {report_file}")
    
    return results

def main():
    """Função principal"""
    
    print(f"📅 Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    total_start = time.time()
    
    # Lista de modelos para otimizar
    models_to_tune = ['ocsvm', 'rf', 'ae']  # Ordem: mais rápido primeiro
    
    all_results = {}
    
    for model in models_to_tune:
        try:
            results = run_optimization(model)
            all_results[model] = results
        except Exception as e:
            print(f"❌ Erro na otimização de {model}: {e}")
            continue
    
    total_time = time.time() - total_start
    
    # Resumo final
    print(f"\n🎉 TUNAGEM COMPLETA!")
    print("=" * 80)
    print(f"⏱️ Tempo total: {total_time/60:.1f} minutos")
    
    if all_results:
        print(f"\n📊 RESUMO DOS MELHORES PARÂMETROS:")
        for model, results in all_results.items():
            print(f"\n{model.upper()}:")
            print(f"  Score: {results['best_score']:.4f}")
            print(f"  Tempo: {results['optimization_time']/60:.1f}min")
            for key, value in results['best_params'].items():
                print(f"  {key}: {value}")
    
    # Salvar resumo geral
    summary_file = os.path.join(results_dir, "tunagem_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("RESUMO GERAL DA TUNAGEM\n")
        f.write("=" * 50 + "\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tempo total: {total_time/60:.1f} minutos\n")
        f.write(f"Modelos otimizados: {len(all_results)}\n\n")
        
        for model, results in all_results.items():
            f.write(f"{model.upper()}:\n")
            f.write(f"  Melhor score: {results['best_score']:.4f}\n")
            f.write(f"  Trials: {results['n_trials']}\n")
            f.write(f"  Tempo: {results['optimization_time']/60:.1f}min\n")
            f.write("  Parâmetros:\n")
            for key, value in results['best_params'].items():
                f.write(f"    {key}: {value}\n")
            f.write("\n")
    
    print(f"\n💾 Resumo geral salvo em: {summary_file}")
    print(f"📁 Todos os resultados em: {results_dir}")

if __name__ == "__main__":
    main()
