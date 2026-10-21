# Instrucciones para Análisis de Datasets

## 1. Instalar Dependencia Faltante

```bash
# Asegúrate de estar en el environment correcto
conda activate tesis

# Instalar pm4py (necesario para leer archivos XES)
pip install pm4py>=2.7.0
```

## 2. Ejecutar Análisis Completo

```bash
# Ejecutar el script de análisis
python analyze_datasets.py
```

Esto analizará los 4 datasets:
- ✓ HIRING (benchmark/hiring/hiring_log_high.xes)
- ✓ HOSPITAL (benchmark/hospital/hospital_log_high.xes)
- ✓ LENDING (benchmark/lending/lending_log_high.xes)
- ✓ RENTING (benchmark/renting/renting_log_high.xes)

## 3. Salida Esperada

El script generará:

### En Consola:
Para cada dataset mostrará:
- **Información General**: casos, eventos, actividades
- **Estadísticas de Longitud de Trazas**: min/max/media/mediana/std
- **Distribución de Variantes**: total, top 10, únicas
- **Recursos**: frecuencia de uso por recurso
- **Atributos de Casos**: todos los atributos disponibles
- **Distribución de Actividades**: top 15 actividades
- **Información Temporal**: duración de casos (si hay timestamps)

### Archivo de Salida:
`dataset_statistics.json` - Estadísticas completas en formato JSON

### Resumen Comparativo:
Tabla comparativa de los 4 datasets con métricas clave

## 4. Verificación Rápida

Antes de ejecutar el análisis completo, verifica que los archivos existen:

```bash
ls -lh benchmark/hiring/hiring_log_high.xes
ls -lh benchmark/hospital/hospital_log_high.xes
ls -lh benchmark/lending/lending_log_high.xes
ls -lh benchmark/renting/renting_log_high.xes
```

## 5. Ejemplo de Salida

```
================================================================================
ANALIZANDO: HIRING
================================================================================
Archivo: benchmark/hiring/hiring_log_high.xes
Cargando log...

--- Información General ---
Total de casos: 10,000
Total de eventos: 150,000
Actividades únicas: 25
Promedio eventos/caso: 15.00

--- Estadísticas de Longitud de Trazas ---
Mínima: 3
Máxima: 45
Media: 15.23
Mediana: 14
Desviación estándar: 5.67

Distribución de longitudes (top 10):
  Longitud 12: 1,234 casos ( 12.34%)
  Longitud 13: 1,189 casos ( 11.89%)
  ...

--- Variantes de Proceso ---
Total de variantes únicas: 2,345
Variante más común: 789 casos (7.89%)

Top 10 variantes más frecuentes:
   1.   789 casos ( 7.89%): A → B → C → D → E ... (12 actividades total)
   2.   654 casos ( 6.54%): A → B → C → F → G ... (15 actividades total)
  ...

--- Recursos ---
Columna de recurso: org:resource

  Total de recursos únicos: 45

  Top 10 recursos más activos:
     1. Recruiter_1: 12,345 eventos (8.23%)
     2. Recruiter_2: 11,234 eventos (7.49%)
    ...

--- Atributos de Casos ---
Atributos de caso encontrados: 5

  case:concept:name:
    Valores únicos: 10,000

  case:application_type:
    Valores únicos: 3
    Valores:
      Standard: 7,500
      Fast_Track: 2,000
      Special: 500
  ...

--- Distribución de Actividades ---
Total de actividades únicas: 25

Top 15 actividades más frecuentes:
   1. Application Received: 10,000 (6.67%)
   2. Background Check: 9,850 (6.57%)
   3. Interview Scheduled: 8,500 (5.67%)
  ...
```

## 6. Uso de Resultados

Una vez generado `dataset_statistics.json`, puedes:

```bash
# Ver estadísticas de un dataset específico
cat dataset_statistics.json | jq '.[0]'  # HIRING
cat dataset_statistics.json | jq '.[1]'  # HOSPITAL
cat dataset_statistics.json | jq '.[2]'  # LENDING
cat dataset_statistics.json | jq '.[3]'  # RENTING

# Ver solo las longitudes de trazas
cat dataset_statistics.json | jq '.[].trace_lengths'

# Ver variantes
cat dataset_statistics.json | jq '.[].variants'
```

## 7. Troubleshooting

### Error: "No module named 'pm4py'"
```bash
conda activate tesis
pip install pm4py>=2.7.0
```

### Error: "FileNotFoundError"
Verifica que estás en el directorio correcto:
```bash
pwd  # Debe mostrar: .../pm-samples
ls benchmark/  # Debe mostrar: hiring/ hospital/ lending/ renting/
```

### Script muy lento
Los archivos XES son grandes (25-35 MB cada uno). El análisis completo puede tomar 2-5 minutos.

Para analizar solo un dataset, edita `analyze_datasets.py` y comenta las líneas de los datasets que no necesitas.

## 8. Siguiente Paso

Una vez que tengas las estadísticas, puedes usarlas para:

1. **Seleccionar casos de prueba** para los benchmarks
2. **Entender la complejidad** de cada proceso
3. **Identificar variantes representativas** para adaptar
4. **Documentar las características** de cada dataset en tu tesis

## 9. Comando Completo (Todo en uno)

```bash
# Desde el directorio pm-samples
conda activate tesis
pip install pm4py>=2.7.0
python analyze_datasets.py > dataset_analysis_output.txt 2>&1
cat dataset_analysis_output.txt
```

Esto:
1. Activa el environment
2. Instala pm4py si falta
3. Ejecuta el análisis
4. Guarda la salida en un archivo
5. Muestra el resultado
