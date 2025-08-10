#!/usr/bin/env python
# coding: utf-8

"""
ANÁLISE DO DATASET - Multi-Stage HIDS
Análise completa do dataset para entender estrutura, qualidade e viabilidade
"""

import os
import pandas as pd
import numpy as np
import time
from datetime import datetime

print("🔍 ANÁLISE COMPLETA DO DATASET")
print("=" * 80)

# Configurações
clean_dir = "/home/CIN/jbsn3/multi-stage-hierarchical-ids/ids-dataset-cleaning/cicids2018/clean"

def format_number(num):
    """Formatar números grandes"""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return str(num)

def analyze_dataframe(df, name):
    """Análise detalhada de um dataframe"""
    print(f"\n📊 ANÁLISE: {name.upper()}")
    print("-" * 50)
    
    # Informações básicas
    print(f"📈 Dimensões: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
    print(f"💾 Memória: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Qualidade dos dados
    null_count = df.isnull().sum().sum()
    duplicate_count = df.duplicated().sum()
    
    print(f"🔍 Qualidade:")
    print(f"  • Valores nulos: {null_count:,}")
    print(f"  • Duplicatas: {duplicate_count:,}")
    print(f"  • % Duplicatas: {(duplicate_count/len(df)*100):.2f}%")
    
    # Tipos de dados
    print(f"📋 Tipos de dados:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  • {dtype}: {count} colunas")
    
    # Estatísticas das colunas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"🔢 Colunas numéricas: {len(numeric_cols)}")
        
        # Verificar valores infinitos
        inf_counts = np.isinf(df[numeric_cols]).sum().sum()
        if inf_counts > 0:
            print(f"  ⚠️  Valores infinitos: {inf_counts:,}")
        
        # Range de valores
        print(f"📊 Ranges (primeiras 5 colunas numéricas):")
        for col in numeric_cols[:5]:
            min_val = df[col].min()
            max_val = df[col].max()
            print(f"  • {col}: [{min_val:.2f}, {max_val:.2f}]")
    
    return {
        'rows': df.shape[0],
        'cols': df.shape[1], 
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'nulls': null_count,
        'duplicates': duplicate_count,
        'numeric_cols': len(numeric_cols)
    }

try:
    # Carregar dados
    print("📊 Carregando dados...")
    start_time = time.time()
    
    df_benign = pd.read_parquet(f"{clean_dir}/all_benign.parquet")
    df_malicious = pd.read_parquet(f"{clean_dir}/all_malicious.parquet")
    
    load_time = time.time() - start_time
    print(f"⏱️ Tempo de carregamento: {load_time:.2f}s")
    
    # Análise individual
    benign_stats = analyze_dataframe(df_benign, "BENIGN")
    malicious_stats = analyze_dataframe(df_malicious, "MALICIOUS")
    
    # Análise da distribuição de ataques
    print(f"\n🎯 DISTRIBUIÇÃO DE ATAQUES:")
    print("-" * 50)
    attack_counts = df_malicious['Label'].value_counts()
    total_malicious = len(df_malicious)
    
    for attack, count in attack_counts.items():
        percentage = (count / total_malicious) * 100
        print(f"  • {attack:<15}: {count:>8,} ({percentage:>5.1f}%)")
    
    # Verificar problemas conhecidos
    print(f"\n🔍 VERIFICAÇÕES DE QUALIDADE:")
    print("-" * 50)
    
    # NaN em labels
    nan_labels = df_malicious['Label'].isnull().sum()
    if nan_labels > 0:
        print(f"  ⚠️  Labels NaN: {nan_labels:,}")
        print(f"      → Isso pode causar erros no treinamento!")
    else:
        print(f"  ✅ Labels: Sem valores NaN")
    
    # Consistência de colunas
    benign_cols = set(df_benign.columns)
    malicious_cols = set(df_malicious.columns)
    
    if benign_cols == malicious_cols:
        print(f"  ✅ Colunas: Consistentes entre benign e malicious")
    else:
        only_benign = benign_cols - malicious_cols
        only_malicious = malicious_cols - benign_cols
        print(f"  ⚠️  Colunas inconsistentes:")
        if only_benign:
            print(f"      → Apenas em benign: {list(only_benign)}")
        if only_malicious:
            print(f"      → Apenas em malicious: {list(only_malicious)}")
    
    # Resumo geral
    total_samples = benign_stats['rows'] + malicious_stats['rows']
    total_memory = benign_stats['memory_mb'] + malicious_stats['memory_mb']
    
    print(f"\n📈 RESUMO GERAL:")
    print("=" * 50)
    print(f"📊 Total de amostras: {total_samples:,}")
    print(f"  • Benign: {benign_stats['rows']:,} ({benign_stats['rows']/total_samples*100:.1f}%)")
    print(f"  • Malicious: {malicious_stats['rows']:,} ({malicious_stats['rows']/total_samples*100:.1f}%)")
    print(f"  • Proporção Benign/Malicious: {benign_stats['rows']/malicious_stats['rows']:.1f}:1")
    
    print(f"\n💾 Recursos necessários:")
    print(f"  • Memória total: {total_memory:.1f} MB")
    print(f"  • Features: {benign_stats['cols']} colunas")
    
    # Estimativas para treinamento
    print(f"\n🚀 ESTIMATIVAS DE TREINAMENTO:")
    print("-" * 50)
    
    # Com as proporções atuais do treinamento.py
    ocsvm_train = int(benign_stats['rows'] * 0.15)
    ocsvm_val = int(benign_stats['rows'] * 0.65) 
    ocsvm_test = int(benign_stats['rows'] * 0.20)
    
    ae_val = int(benign_stats['rows'] * 0.65)
    ae_test = int(benign_stats['rows'] * 0.20)
    
    rf_train = int(benign_stats['rows'] * 0.10)
    rf_val = int(benign_stats['rows'] * 0.70)
    rf_test = int(benign_stats['rows'] * 0.20)
    
    print(f"📊 OCSVM:")
    print(f"  • Treino: {format_number(ocsvm_train)} amostras benign")
    print(f"  • Validação: {format_number(ocsvm_val)} amostras (benign + malicious)")
    print(f"  • Teste: {format_number(ocsvm_test)} amostras")
    
    print(f"📊 Autoencoder:")
    print(f"  • Validação: {format_number(ae_val)} amostras")
    print(f"  • Teste: {format_number(ae_test)} amostras")
    
    print(f"📊 Random Forest:")
    print(f"  • Treino benign: {format_number(rf_train)} amostras")
    print(f"  • Malicious: {format_number(malicious_stats['rows'])} amostras (todas)")
    print(f"  • Validação: {format_number(rf_val)} amostras")
    
    # Recomendações
    print(f"\n💡 RECOMENDAÇÕES:")
    print("-" * 50)
    
    if total_memory > 1000:  # > 1GB
        print(f"  ⚠️  Dataset muito grande ({total_memory:.0f}MB)")
        print(f"      → Considere usar amostras menores para testes")
        print(f"      → Valores fixos podem ser melhores que proporções")
    
    if benign_stats['rows'] / malicious_stats['rows'] > 50:
        print(f"  ⚠️  Dataset muito desbalanceado")
        print(f"      → Considere técnicas de balanceamento")
    
    if nan_labels > 0:
        print(f"  🔧 AÇÃO NECESSÁRIA: Corrigir NaN em labels antes do treino")
    
    # Sugestão de configuração otimizada
    print(f"\n🎯 CONFIGURAÇÃO SUGERIDA PARA TESTES:")
    print("-" * 50)
    
    # Sugerir valores fixos baseados no tamanho do dataset
    if total_samples > 1_000_000:  # Dataset muito grande
        print("Dataset grande detectado - usando valores fixos menores:")
        print("OCSVM: train_size=20000, val_size=30000, test_size=25000")
        print("AE: train_size=50000, val_size=40000, test_size=25000") 
        print("RF: train_size=10000, val_size=15000, test_size=25000")
    else:  # Dataset menor
        print("Dataset médio - pode usar proporções:")
        print("Configuração atual está adequada")
    
except FileNotFoundError as e:
    print(f"❌ Erro: Arquivos não encontrados em {clean_dir}")
    print(f"   Verifique se o caminho está correto")
    print(f"   Erro: {e}")
except Exception as e:
    print(f"❌ Erro inesperado: {e}")
    import traceback
    traceback.print_exc()

print(f"\n✅ Análise concluída em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
