"""
MODEL SELECTION & COMPARISON
=============================

Este script foi usado para análise exploratória e seleção do melhor
modelo de Machine Learning para predição de diabetes.

Objetivo:
    - Comparar Random Forest vs AdaBoost
    - Determinar o modelo final para produção

Resultado:
    - Random Forest foi escolhido (AUC: 0.8124, F1: 0.6139)
    - Features selecionadas (utilizado o feature_selection.py): Glucose, BMI, DiabetesPedigreeFunction, Age

Uso:
    python model_selection.py

Output:
    - model_comparison.png: Gráficos comparativos
    - Prints no console com métricas detalhadas

Nota: Este arquivo é apenas para consulta e documentação do processo.
      O modelo de produção está em src/model.py

Autor: Erick-nix
Data: Novembro 2025
"""

# Suprimir warnings
import warnings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*MatplotlibDeprecationWarning.*')

# Carregar dados do CSV
data_path = '../data/diabetes.csv' if os.path.exists('../data/diabetes.csv') else 'data/diabetes.csv'
data = pd.read_csv(data_path)

# De acordo com o feature_selection.py os melhores features são: Glucose, BMI, DiabetesPedigreeFunction, Age
X = data.drop(['Outcome', 'BloodPressure', 'SkinThickness', 'Pregnancies', 'Insulin'], axis=1)  # ou o nome da coluna alvo no seu CSV
y = data['Outcome'] # Alvo

# ============================================
# TREINAMENTO E COMPARAÇÃO DOS MODELOS
# ============================================

print("\n" + "="*60)
print("TREINAMENTO E COMPARAÇÃO DE MODELOS")
print("="*60)

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nDados de treino: {X_train.shape[0]} amostras")
print(f"Dados de teste: {X_test.shape[0]} amostras")

# ============================================
# 1. RANDOM FOREST CLASSIFIER
# ============================================
print("\n" + "-"*60)
print("1. RANDOM FOREST CLASSIFIER")
print("-"*60)

# Configurar K-Fold Cross Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Hiperparâmetros para GridSearch
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    cv=kfold,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("Treinando Random Forest com GridSearchCV...")
rf_grid.fit(X_train, y_train)

print(f"\nMelhores parâmetros: {rf_grid.best_params_}")
print(f"Melhor score no treino (CV): {rf_grid.best_score_:.4f}")

# Predição no conjunto de teste
rf_pred = rf_grid.best_estimator_.predict(X_test)
rf_pred_proba = rf_grid.best_estimator_.predict_proba(X_test)[:, 1]

# Métricas
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print("\nMétricas no conjunto de teste:")
print(f"  Acurácia:  {rf_accuracy:.4f}")
print(f"  Precisão:  {rf_precision:.4f}")
print(f"  Recall:    {rf_recall:.4f}")
print(f"  F1-Score:  {rf_f1:.4f}")

# ============================================
# 2. ADABOOST CLASSIFIER
# ============================================
print("\n" + "-"*60)
print("2. ADABOOST CLASSIFIER")
print("-"*60)

# Hiperparâmetros para GridSearch
ada_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5, 1.0]
}

ada_grid = GridSearchCV(
    AdaBoostClassifier(random_state=42),
    ada_params,
    cv=kfold,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("Treinando AdaBoost com GridSearchCV...")
ada_grid.fit(X_train, y_train)

print(f"\nMelhores parâmetros: {ada_grid.best_params_}")
print(f"Melhor score no treino (CV): {ada_grid.best_score_:.4f}")

# Predição no conjunto de teste
ada_pred = ada_grid.best_estimator_.predict(X_test)
ada_pred_proba = ada_grid.best_estimator_.predict_proba(X_test)[:, 1]

# Métricas
ada_accuracy = accuracy_score(y_test, ada_pred)
ada_precision = precision_score(y_test, ada_pred)
ada_recall = recall_score(y_test, ada_pred)
ada_f1 = f1_score(y_test, ada_pred)

print("\nMétricas no conjunto de teste:")
print(f"  Acurácia:  {ada_accuracy:.4f}")
print(f"  Precisão:  {ada_precision:.4f}")
print(f"  Recall:    {ada_recall:.4f}")
print(f"  F1-Score:  {ada_f1:.4f}")

# ============================================
# 3. COMPARAÇÃO VISUAL DOS MODELOS
# ============================================
print("\n" + "="*60)
print("GERANDO VISUALIZAÇÕES COMPARATIVAS")
print("="*60)

# Criar figura com múltiplos subplots
fig = plt.figure(figsize=(16, 12))

# -----------------
# 3.1 Comparação de Métricas
# -----------------
ax1 = plt.subplot(2, 3, 1)
metrics = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
rf_values = [rf_accuracy, rf_precision, rf_recall, rf_f1]
ada_values = [ada_accuracy, ada_precision, ada_recall, ada_f1]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, rf_values, width, label='Random Forest', color='#2ecc71', alpha=0.8)
bars2 = ax1.bar(x + width/2, ada_values, width, label='AdaBoost', color='#e74c3c', alpha=0.8)

ax1.set_xlabel('Métricas', fontsize=10, fontweight='bold')
ax1.set_ylabel('Score', fontsize=10, fontweight='bold')
ax1.set_title('Comparação de Métricas', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0, 1])

# Adicionar valores nas barras
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)

# -----------------
# 3.2 Matriz de Confusão - Random Forest
# -----------------
ax2 = plt.subplot(2, 3, 2)
rf_cm = confusion_matrix(y_test, rf_pred)
im2 = ax2.imshow(rf_cm, interpolation='nearest', cmap='Greens')
ax2.set_title('Matriz de Confusão\nRandom Forest', fontsize=12, fontweight='bold')
plt.colorbar(im2, ax=ax2)

classes = ['Não Diabético', 'Diabético']
tick_marks = np.arange(len(classes))
ax2.set_xticks(tick_marks)
ax2.set_yticks(tick_marks)
ax2.set_xticklabels(classes, rotation=45, ha='right')
ax2.set_yticklabels(classes)

# Adicionar texto na matriz
for i in range(rf_cm.shape[0]):
    for j in range(rf_cm.shape[1]):
        ax2.text(j, i, format(rf_cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if rf_cm[i, j] > rf_cm.max() / 2 else "black",
                fontsize=14, fontweight='bold')

ax2.set_ylabel('Valor Real', fontsize=10, fontweight='bold')
ax2.set_xlabel('Predição', fontsize=10, fontweight='bold')

# -----------------
# 3.3 Matriz de Confusão - AdaBoost
# -----------------
ax3 = plt.subplot(2, 3, 3)
ada_cm = confusion_matrix(y_test, ada_pred)
im3 = ax3.imshow(ada_cm, interpolation='nearest', cmap='Reds')
ax3.set_title('Matriz de Confusão\nAdaBoost', fontsize=12, fontweight='bold')
plt.colorbar(im3, ax=ax3)

ax3.set_xticks(tick_marks)
ax3.set_yticks(tick_marks)
ax3.set_xticklabels(classes, rotation=45, ha='right')
ax3.set_yticklabels(classes)

# Adicionar texto na matriz
for i in range(ada_cm.shape[0]):
    for j in range(ada_cm.shape[1]):
        ax3.text(j, i, format(ada_cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if ada_cm[i, j] > ada_cm.max() / 2 else "black",
                fontsize=14, fontweight='bold')

ax3.set_ylabel('Valor Real', fontsize=10, fontweight='bold')
ax3.set_xlabel('Predição', fontsize=10, fontweight='bold')

# -----------------
# 3.4 Curva ROC
# -----------------
ax4 = plt.subplot(2, 3, 4)

# Random Forest ROC
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_pred_proba)
rf_auc = auc(rf_fpr, rf_tpr)

# AdaBoost ROC
ada_fpr, ada_tpr, _ = roc_curve(y_test, ada_pred_proba)
ada_auc = auc(ada_fpr, ada_tpr)

ax4.plot(rf_fpr, rf_tpr, color='#2ecc71', lw=2, label=f'Random Forest (AUC = {rf_auc:.3f})')
ax4.plot(ada_fpr, ada_tpr, color='#e74c3c', lw=2, label=f'AdaBoost (AUC = {ada_auc:.3f})')
ax4.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Chance (AUC = 0.5)')

ax4.set_xlim([0.0, 1.0])
ax4.set_ylim([0.0, 1.05])
ax4.set_xlabel('Taxa de Falso Positivo', fontsize=10, fontweight='bold')
ax4.set_ylabel('Taxa de Verdadeiro Positivo', fontsize=10, fontweight='bold')
ax4.set_title('Curva ROC', fontsize=12, fontweight='bold')
ax4.legend(loc="lower right")
ax4.grid(alpha=0.3)

# -----------------
# 3.5 Comparação de Scores CV
# -----------------
ax5 = plt.subplot(2, 3, 5)

rf_cv_scores = cross_val_score(rf_grid.best_estimator_, X, y, cv=kfold, scoring='accuracy')
ada_cv_scores = cross_val_score(ada_grid.best_estimator_, X, y, cv=kfold, scoring='accuracy')

bp = ax5.boxplot([rf_cv_scores, ada_cv_scores],
                  tick_labels=['Random Forest', 'AdaBoost'],
                  patch_artist=True,
                  notch=True)

colors = ['#2ecc71', '#e74c3c']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax5.set_ylabel('Acurácia (Cross-Validation)', fontsize=10, fontweight='bold')
ax5.set_title('Distribuição dos Scores (5-Fold CV)', fontsize=12, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)

# Adicionar médias
means = [rf_cv_scores.mean(), ada_cv_scores.mean()]
ax5.plot([1, 2], means, 'D', color='gold', markersize=10, label='Média', zorder=3)
ax5.legend()

# -----------------
# 3.6 Tabela Resumo
# -----------------
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Criar dados da tabela
table_data = [
    ['Métrica', 'Random Forest', 'AdaBoost', 'Melhor'],
    ['Acurácia', f'{rf_accuracy:.4f}', f'{ada_accuracy:.4f}',
     'RF' if rf_accuracy > ada_accuracy else 'Ada'],
    ['Precisão', f'{rf_precision:.4f}', f'{ada_precision:.4f}',
     'RF' if rf_precision > ada_precision else 'Ada'],
    ['Recall', f'{rf_recall:.4f}', f'{ada_recall:.4f}',
     'RF' if rf_recall > ada_recall else 'Ada'],
    ['F1-Score', f'{rf_f1:.4f}', f'{ada_f1:.4f}',
     'RF' if rf_f1 > ada_f1 else 'Ada'],
    ['AUC-ROC', f'{rf_auc:.4f}', f'{ada_auc:.4f}',
     'RF' if rf_auc > ada_auc else 'Ada'],
    ['CV Mean', f'{rf_cv_scores.mean():.4f}', f'{ada_cv_scores.mean():.4f}',
     'RF' if rf_cv_scores.mean() > ada_cv_scores.mean() else 'Ada']
]

table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.3, 0.25, 0.25, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Estilizar cabeçalho
for i in range(4):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Colorir linhas alternadas
for i in range(1, len(table_data)):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')

ax6.set_title('Resumo Comparativo', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()

# Salvar gráfico em results/visualizations/
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
viz_dir = os.path.join(base_dir, 'results', 'visualizations')
os.makedirs(viz_dir, exist_ok=True)
output_path = os.path.join(viz_dir, 'model_comparison.png')

plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nGráfico salvo em: {output_path}")
plt.show()

# ============================================
# 4. RESULTADO FINAL
# ============================================
print("\n" + "="*60)
print("RESULTADO FINAL")
print("="*60)

# Determinar melhor modelo baseado em múltiplas métricas
rf_total = rf_accuracy + rf_precision + rf_recall + rf_f1 + rf_auc
ada_total = ada_accuracy + ada_precision + ada_recall + ada_f1 + ada_auc

if rf_total > ada_total:
    print("\nMELHOR MODELO: Random Forest")
    print(f"   Score Total: {rf_total:.4f}")
    print("\n   Principais Métricas:")
    print(f"   - Acurácia: {rf_accuracy:.4f}")
    print(f"   - F1-Score: {rf_f1:.4f}")
    print(f"   - AUC-ROC:  {rf_auc:.4f}")
else:
    print("\nMELHOR MODELO: AdaBoost")
    print(f"   Score Total: {ada_total:.4f}")
    print("\n   Principais Métricas:")
    print(f"   - Acurácia: {ada_accuracy:.4f}")
    print(f"   - F1-Score: {ada_f1:.4f}")
    print(f"   - AUC-ROC:  {ada_auc:.4f}")

print("\n" + "="*60)
