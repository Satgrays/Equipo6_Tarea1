# Predicción de Bancarrota Empresarial

Este repositorio contiene un servicio de predicción de bancarrota utilizando datos históricos de empresas. El servicio está desplegado en Azure y puede realizar inferencias mediante una API diseñada por el equipo.

## Pasos para usar este repositorio

### Paso 1: Forkea este repositorio
Haz clic en el botón "Fork" en la parte superior derecha para copiar este repositorio a tu cuenta de GitHub.

### Paso 2: Clona el repositorio
Usa el siguiente comando para clonar el repositorio en tu máquina local:

```bash
git clone <url_del_fork>
cd <nombre_del_repositorio>
```

### Paso 3: Instala las dependencias, limpia los datos y entrena el modelo
```bash
Ejecuta el archivo run.py
```

### Paso 6: Prueba la API
Prueba el servicio de predicción con los datos proporcionados en el siguiente enlace  
[Descarga los datos de prueba aquí](https://drive.google.com/file/d/1nlao4hDgZ0nkw6Lp_m7bPqGS0YT-Od6u/view?usp=drive_link)

### Paso 8: Realiza inferencias con la API
Para realizar inferencias con este modelo, utiliza la API desplegada siguiendo las instrucciones. Sustituye el archivo prueba.csv por los datos que desees consultar.

```bash
curl -X POST "https://prediccionbancarrota.azurewebsites.net/api/v1/predict" \