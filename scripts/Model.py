# ================================
# Importación de bibliotecas necesarias
# ================================
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import joblib

# ================================
# Carga del dataset
# ================================
df = pd.read_csv("./data/data_corr.csv")  # Dataset con variables ya corregidas por correlación

# ================================
# Separación de variables predictoras y etiqueta
# ================================
X = df.drop(columns=["Bankrupt?"])   # Variables independientes
y = df["Bankrupt?"]                  # Variable dependiente (etiqueta)

# ================================
# División en conjuntos de entrenamiento y prueba
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ================================
# Configuración del modelo XGBoost y SMOTE
# ================================
smote = SMOTE(sampling_strategy=0.05, random_state=42)  # Aumenta la clase minoritaria al 5% del total

xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",            # AUC bajo la curva Precision-Recall (útil en clases desbalanceadas)
    learning_rate=0.02,             # Tasa de aprendizaje
    n_estimators=3500,              # Número total de árboles
    max_depth=10,                   # Profundidad máxima de los árboles
    min_child_weight=2,             # Mínimo número de observaciones por hoja
    subsample=0.8,                  # Porcentaje de muestra para cada árbol
    colsample_bytree=0.8,           # Porcentaje de características para cada árbol
    scale_pos_weight=13,           # Peso para clase positiva (usado por desbalanceo)
    random_state=42
)

# ================================
# Pipeline: Aplicar SMOTE + Entrenar modelo
# ================================
pipeline = Pipeline([
    ("smote", smote),
    ("xgb", xgb_model)
])

pipeline.fit(X_train, y_train)

# ================================
# Predicciones
# ================================
# Probabilidades predichas
y_proba = pipeline.predict_proba(X_test)[:, 1]

# Umbral específico para clasificación binaria
threshold = 0.5983815
y_pred = (y_proba >= threshold).astype(int)

# ================================
# Evaluación del modelo
# ================================
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("PR AUC:", average_precision_score(y_test, y_proba))

# ================================
# Guardar el modelo entrenado (EMOTE + XGBoost) en un archivo
# ================================
joblib.dump(pipeline, 'model.pkl')

# ================================
# Notas adicionales:
# ================================
# - Se utiliza SMOTE dentro del pipeline para que no ocurra data leakage.
# - El umbral se ha personalizado (no se usa 0.5 por defecto)
# - El modelo XGBoost está configurado para casos de alta desproporción entre clases.
