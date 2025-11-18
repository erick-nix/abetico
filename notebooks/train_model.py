"""
TREINAMENTO DO MODELO FINAL
============================

Este script treina o modelo Random Forest escolhido no model_selection.py
e salva o modelo treinado + scaler para uso em produção.

Baseado nos resultados de model_selection.py:
    - Modelo: Random Forest Classifier
    - Features: Glucose, BMI, DiabetesPedigreeFunction, Age
    - Melhores parâmetros do GridSearch

Output:
    - ../results/diabetes_model.pkl: Modelo treinado
    - ../results/scaler.pkl: Scaler para normalização
    - ../results/training_report.txt: Relatório de treinamento

Uso:
    python train_model.py
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import os
from datetime import datetime

print("="*60)
print("TREINAMENTO DO MODELO FINAL - DIABETES PREDICTOR")
print("="*60)

# ============================================
# 1. CARREGAR E PREPARAR DADOS
# ============================================
print("\n[1/5] Carregando dados...")

# Ajuste o caminho se necessário
data_path = '../data/diabetes.csv'
if not os.path.exists(data_path):
    data_path = 'data/diabetes.csv'

data = pd.read_csv(data_path)
print(f"✓ Dados carregados: {data.shape[0]} amostras, {data.shape[1]} features")

# Separar features e target
# Usar as mesmas features selecionadas no model_selection.py
selected_features = ['Glucose', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = data[selected_features]
y = data['Outcome']

print(f"✓ Features selecionadas: {selected_features}")
print(f"✓ Distribuição de classes: {dict(y.value_counts())}")

# ============================================
# 2. DIVIDIR DADOS E NORMALIZAR
# ============================================
print("\n[2/5] Dividindo dados e normalizando...")

# Dividir em treino e teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"✓ Treino: {X_train.shape[0]} amostras")
print(f"✓ Teste: {X_test.shape[0]} amostras")

# Normalizar os dados (importante para performance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Dados normalizados com StandardScaler")

# ============================================
# 3. TREINAR MODELO FINAL
# ============================================
print("\n[3/5] Treinando Random Forest com melhores parâmetros...")

# Usar os melhores parâmetros encontrados no GridSearch do model_selection.py
# Ajuste esses valores se o seu GridSearch encontrou outros valores
best_params = {
    'n_estimators': 50,
    'max_depth': 5,
    'min_samples_split': 10,
    'class_weight': 'balanced'  # Para lidar com desbalanceamento
}

model = RandomForestClassifier(**best_params)
model.fit(X_train_scaled, y_train)

print("✓ Modelo treinado com sucesso!")
print(f"  Parâmetros: {best_params}")

# ============================================
# 4. AVALIAR MODELO
# ============================================
print("\n[4/5] Avaliando performance no conjunto de teste...")

# Predições
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred)

print("\n📊 MÉTRICAS DE PERFORMANCE:")
print(f"  Acurácia:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precisão:  {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  AUC-ROC:   {auc_roc:.4f}")

print("\n📊 MATRIZ DE CONFUSÃO:")
print(f"  [[TN={cm[0,0]:3d}  FP={cm[0,1]:3d}]")
print(f"   [FN={cm[1,0]:3d}  TP={cm[1,1]:3d}]]")

# Feature importance
print("\n📊 IMPORTÂNCIA DAS FEATURES:")
importances = model.feature_importances_
for feature, importance in zip(selected_features, importances):
    print(f"  {feature:25s}: {importance:.4f}")

# ============================================
# 5. SALVAR MODELO E SCALER
# ============================================
print("\n[5/5] Salvando modelo e scaler...")

# Criar diretórios organizados
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(base_dir, 'results')
models_dir = os.path.join(results_dir, 'models')

# Criar diretórios se não existirem
os.makedirs(models_dir, exist_ok=True)

# Salvar modelo
model_path = os.path.join(models_dir, 'diabetes_model.pkl')
joblib.dump(model, model_path)
print(f"✓ Modelo salvo em: {model_path}")

# Salvar scaler
scaler_path = os.path.join(models_dir, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"✓ Scaler salvo em: {scaler_path}")

# Salvar metadata (features usadas, etc)
metadata = {
    'features': selected_features,
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'model_params': best_params,
    'metrics': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_roc': float(auc_roc)
    },
    'data_shape': {
        'train': X_train.shape[0],
        'test': X_test.shape[0]
    }
}

metadata_path = os.path.join(models_dir, 'model_metadata.pkl')
joblib.dump(metadata, metadata_path)
print(f"✓ Metadata salvo em: {metadata_path}")

# ============================================
# COPIAR PARA src/model/ (para uso em produção)
# ============================================
import shutil

src_model_dir = os.path.join(base_dir, 'src', 'model')
os.makedirs(src_model_dir, exist_ok=True)

# Copiar modelo e scaler
shutil.copy(model_path, os.path.join(src_model_dir, 'diabetes_model.pkl'))
shutil.copy(scaler_path, os.path.join(src_model_dir, 'scaler.pkl'))
print(f"\n✓ Arquivos copiados para produção: {src_model_dir}")
print("  - diabetes_model.pkl")
print("  - scaler.pkl")

print("\n✅ Treinamento concluído com sucesso!")
print(f"📁 Modelos salvos em: {models_dir}")
print(f"📁 Modelos de produção em: {src_model_dir}")