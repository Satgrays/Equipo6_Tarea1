# ================================
# Importación de bibliotecas necesarias
# ================================
import os
import pandas as pd
import kagglehub
import warnings

# Ignorar advertencias futuras
warnings.filterwarnings("ignore", category=FutureWarning)

# ================================
# Carga del dataset (descarga si no existe localmente)
# ================================

# Ruta destino final
local_file = "./data/data.csv"

# Si ya existe, cargar directamente
if os.path.exists(local_file):
    print("Archivo ya existe en ./data/")
else:
    print("Descargando y moviendo desde KaggleHub...")

    # Descargar y extraer el dataset (ya descomprimido)
    path = kagglehub.dataset_download("fedesoriano/company-bankruptcy-prediction")

    # Buscar .csv dentro del folder
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No está el archivo .csv.")

    # Asegurar carpeta ./data
    os.makedirs("./data", exist_ok=True)

    # Mover el primer CSV encontrado
    source_csv = os.path.join(path, csv_files[0])
    os.rename(source_csv, local_file)

# Leer el archivo CSV
data = pd.read_csv(local_file)

# ================================
# Preprocesamiento de datos
# ================================

# Renombrar columnas para facilitar su uso (eliminar espacios)
data.columns = [col.replace(' ', '_') for col in data.columns]

# Convertir la variable objetivo a tipo entero (0 o 1)
data['Bankrupt?'] = data['Bankrupt?'].astype(int)

# Mostrar información del DataFrame
print("Información del DataFrame:")
print(data.info())

# Mostrar estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(data.describe())

# Mostrar los primeros registros del DataFrame
print("\nPrimeros registros del DataFrame:")
print(data.head())

# ================================
# Selección de variables relevantes basadas en correlación
# ================================
matriz = data.corr()  # Matriz de correlaciones
corr = matriz["Bankrupt?"].drop("Bankrupt?").sort_values(ascending=False)  # Correlaciones con la variable objetivo

# Filtrar solo variables con alta correlación positiva o negativa (umbral: ±0.2)
corr = corr[(corr >= 0.2) | (corr <= -0.2)]

# Obtener los nombres de columnas relevantes y añadir la variable objetivo
columnas = corr.index.tolist()
columnas.append("Bankrupt?")
data_1 = data[columnas]  # Subconjunto del DataFrame con solo estas columnas

# Reemplazar espacios en nombres de columnas por guiones bajos (por compatibilidad)
data_1.columns = [c.replace(' ', '_') for c in data_1.columns]

# ================================
# Filtrado de outliers en casos de empresas **no quebradas** (Bankrupt? == 0)
# ================================
class_0 = data_1[data_1['Bankrupt?'] == 0]  # Empresas no quebradas

# Columnas específicas sobre las que se desea filtrar outliers
columns_to_check = [
    '_Net_profit_before_tax/Paid-in_capital',
    '_Per_Share_Net_profit_before_tax_(Yuan_¥)',
    '_Retained_Earnings_to_Total_Assets',
    '_Persistent_EPS_in_the_Last_Four_Seasons'
]

# Inicializar un índice booleano que marca las filas válidas
valid_indices = pd.Series([True] * len(class_0), index=class_0.index)

# Filtrar outliers por IQR (rango intercuartílico)
for col in columns_to_check:
    Q1 = class_0[col].quantile(0.25)
    Q3 = class_0[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    # Mantener solo las filas dentro del rango permitido
    valid_indices &= (class_0[col] >= lower_limit) & (class_0[col] <= upper_limit)

# Aplicar el filtro al DataFrame original de clase 0
data_filtered = class_0[valid_indices]

# Unir empresas filtradas de clase 0 con todas las empresas quebradas (clase 1)
data_2 = pd.concat([data_filtered, data_1[data_1['Bankrupt?'] == 1]])

# ================================
# 💾 Guardado de resultados
# ================================

# Guardar versión con variables correlacionadas
data_1.to_csv("./data/data_corr.csv", index=False)
