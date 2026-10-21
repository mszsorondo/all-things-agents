#!/usr/bin/env python3
"""
Análisis completo de datasets XES
Genera estadísticas detalladas para Hiring, Hospital, Lending y Renting
"""

import sys
from pathlib import Path
from collections import Counter
import json

try:
    import pm4py
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"ERROR: Falta instalar dependencias")
    print(f"  {e}")
    print("\nPara instalar:")
    print("  conda activate tesis")
    print("  pip install pm4py pandas numpy")
    sys.exit(1)


def analyze_dataset(xes_path: str, dataset_name: str):
    """Analiza un dataset XES y retorna estadísticas completas."""
    print(f"\n{'='*80}")
    print(f"ANALIZANDO: {dataset_name}")
    print(f"{'='*80}")
    print(f"Archivo: {xes_path}")

    # Leer log
    print("Cargando log...")
    log = pm4py.read_xes(xes_path)

    # Información básica
    n_cases = log['case:concept:name'].nunique()
    n_events = len(log)
    activities = log['concept:name'].unique()
    n_activities = len(activities)

    print(f"\n--- Información General ---")
    print(f"Total de casos: {n_cases:,}")
    print(f"Total de eventos: {n_events:,}")
    print(f"Actividades únicas: {n_activities}")
    print(f"Promedio eventos/caso: {n_events/n_cases:.2f}")

    # Agrupar por caso
    grouped = log.groupby('case:concept:name')

    # 1. ESTADÍSTICAS DE LONGITUD DE TRAZAS
    trace_lengths = grouped.size()

    print(f"\n--- Estadísticas de Longitud de Trazas ---")
    print(f"Mínima: {trace_lengths.min()}")
    print(f"Máxima: {trace_lengths.max()}")
    print(f"Media: {trace_lengths.mean():.2f}")
    print(f"Mediana: {trace_lengths.median():.0f}")
    print(f"Desviación estándar: {trace_lengths.std():.2f}")

    # Distribución de longitudes
    length_dist = trace_lengths.value_counts().sort_index()
    print(f"\nDistribución de longitudes (top 10):")
    for length, count in length_dist.head(10).items():
        pct = (count / n_cases) * 100
        print(f"  Longitud {length:2d}: {count:5d} casos ({pct:5.2f}%)")

    # 2. DISTRIBUCIÓN DE VARIANTES
    print(f"\n--- Variantes de Proceso ---")

    # Obtener variantes (secuencia de actividades por caso)
    variants = {}
    for case_id, case_df in grouped:
        variant = tuple(case_df['concept:name'].tolist())
        variants[variant] = variants.get(variant, 0) + 1

    n_variants = len(variants)
    variant_counts = sorted(variants.items(), key=lambda x: x[1], reverse=True)

    print(f"Total de variantes únicas: {n_variants:,}")
    print(f"Variante más común: {variant_counts[0][1]} casos ({(variant_counts[0][1]/n_cases)*100:.2f}%)")

    # Top 10 variantes
    print(f"\nTop 10 variantes más frecuentes:")
    for i, (variant, count) in enumerate(variant_counts[:10], 1):
        pct = (count / n_cases) * 100
        variant_str = ' → '.join(variant[:5])  # Primeras 5 actividades
        if len(variant) > 5:
            variant_str += f" ... ({len(variant)} actividades total)"
        print(f"  {i:2d}. {count:5d} casos ({pct:5.2f}%): {variant_str}")

    # Distribución de frecuencia de variantes
    variant_freq_dist = Counter([count for _, count in variant_counts])
    print(f"\nDistribución de frecuencia de variantes:")
    print(f"  Variantes con 1 caso (únicas): {variant_freq_dist.get(1, 0):,} ({(variant_freq_dist.get(1, 0)/n_variants)*100:.2f}%)")
    print(f"  Variantes con 2-5 casos: {sum(variant_freq_dist.get(i, 0) for i in range(2, 6)):,}")
    print(f"  Variantes con 6-10 casos: {sum(variant_freq_dist.get(i, 0) for i in range(6, 11)):,}")
    print(f"  Variantes con >10 casos: {sum(count for freq, count in variant_freq_dist.items() if freq > 10):,}")

    # 3. RECURSOS INVOLUCRADOS
    print(f"\n--- Recursos ---")

    resource_cols = [col for col in log.columns if 'resource' in col.lower() or 'org:resource' in col.lower()]

    if resource_cols:
        for res_col in resource_cols:
            print(f"\nColumna de recurso: {res_col}")
            resources = log[res_col].dropna()
            if len(resources) > 0:
                n_resources = resources.nunique()
                print(f"  Total de recursos únicos: {n_resources}")

                # Frecuencia de recursos
                resource_freq = resources.value_counts()
                print(f"\n  Top 10 recursos más activos:")
                for i, (resource, count) in enumerate(resource_freq.head(10).items(), 1):
                    pct = (count / len(resources)) * 100
                    print(f"    {i:2d}. {resource}: {count:,} eventos ({pct:.2f}%)")
            else:
                print(f"  (Sin datos de recursos)")
    else:
        print("No se encontraron columnas de recursos en el log")
        print(f"Columnas disponibles: {', '.join(log.columns)}")

    # 4. ATRIBUTOS DE CASOS
    print(f"\n--- Atributos de Casos ---")

    # Atributos a nivel de caso (case:*)
    case_attrs = [col for col in log.columns if col.startswith('case:')]

    if case_attrs:
        print(f"Atributos de caso encontrados: {len(case_attrs)}")
        for attr in case_attrs:
            unique_vals = log[attr].nunique()
            print(f"\n  {attr}:")
            print(f"    Valores únicos: {unique_vals}")

            # Mostrar algunos ejemplos si son pocos
            if unique_vals <= 20:
                val_counts = log[attr].value_counts()
                print(f"    Valores:")
                for val, count in val_counts.head(10).items():
                    print(f"      {val}: {count:,}")
            else:
                # Solo mostrar tipo y rango si son muchos
                dtype = log[attr].dtype
                print(f"    Tipo: {dtype}")
                if pd.api.types.is_numeric_dtype(log[attr]):
                    print(f"    Rango: {log[attr].min()} - {log[attr].max()}")
    else:
        print("No se encontraron atributos de caso (case:*)")

    # Todos los atributos disponibles
    print(f"\n--- Todos los Atributos del Log ---")
    print(f"Total de columnas: {len(log.columns)}")
    for col in log.columns:
        dtype = log[col].dtype
        n_unique = log[col].nunique()
        n_null = log[col].isnull().sum()
        print(f"  {col}:")
        print(f"    Tipo: {dtype}, Únicos: {n_unique:,}, Nulos: {n_null:,}")

    # 5. ACTIVIDADES
    print(f"\n--- Distribución de Actividades ---")
    activity_freq = log['concept:name'].value_counts()
    print(f"Total de actividades únicas: {n_activities}")
    print(f"\nTop 15 actividades más frecuentes:")
    for i, (activity, count) in enumerate(activity_freq.head(15).items(), 1):
        pct = (count / n_events) * 100
        print(f"  {i:2d}. {activity}: {count:,} ({pct:.2f}%)")

    # 6. TEMPORAL (si hay timestamps)
    if 'time:timestamp' in log.columns:
        print(f"\n--- Información Temporal ---")
        log['time:timestamp'] = pd.to_datetime(log['time:timestamp'])

        print(f"Fecha más antigua: {log['time:timestamp'].min()}")
        print(f"Fecha más reciente: {log['time:timestamp'].max()}")
        print(f"Duración total: {log['time:timestamp'].max() - log['time:timestamp'].min()}")

        # Duración de casos
        case_durations = grouped['time:timestamp'].agg(['min', 'max'])
        case_durations['duration'] = case_durations['max'] - case_durations['min']

        print(f"\nDuración de casos:")
        print(f"  Mínima: {case_durations['duration'].min()}")
        print(f"  Máxima: {case_durations['duration'].max()}")
        print(f"  Media: {case_durations['duration'].mean()}")
        print(f"  Mediana: {case_durations['duration'].median()}")

    # Retornar datos estructurados
    stats = {
        'dataset_name': dataset_name,
        'file_path': xes_path,
        'basic': {
            'n_cases': int(n_cases),
            'n_events': int(n_events),
            'n_activities': int(n_activities),
            'avg_events_per_case': float(n_events / n_cases)
        },
        'trace_lengths': {
            'min': int(trace_lengths.min()),
            'max': int(trace_lengths.max()),
            'mean': float(trace_lengths.mean()),
            'median': float(trace_lengths.median()),
            'std': float(trace_lengths.std())
        },
        'variants': {
            'n_unique': int(n_variants),
            'top_variant_count': int(variant_counts[0][1]),
            'top_variant_pct': float((variant_counts[0][1]/n_cases)*100),
            'unique_variants': int(variant_freq_dist.get(1, 0))
        },
        'activities': list(activities),
        'case_attributes': case_attrs,
        'all_columns': list(log.columns)
    }

    return stats


def generate_histograms(all_stats):
    """Genera histogramas ASCII de distribución de variantes."""
    print(f"\n\n{'='*80}")
    print("HISTOGRAMAS DE DISTRIBUCIÓN DE VARIANTES")
    print(f"{'='*80}\n")

    for stats in all_stats:
        print(f"\n--- {stats['dataset_name']} ---")
        print(f"Total variantes: {stats['variants']['n_unique']:,}")
        print(f"Variantes únicas (1 caso): {stats['variants']['unique_variants']:,} ({(stats['variants']['unique_variants']/stats['variants']['n_unique'])*100:.1f}%)")
        print(f"Variante más común: {stats['variants']['top_variant_count']} casos ({stats['variants']['top_variant_pct']:.2f}%)")


def main():
    """Ejecuta análisis completo de todos los datasets."""
    datasets = [
        ('benchmark/hiring/hiring_log_high.xes', 'HIRING'),
        ('benchmark/hospital/hospital_log_high.xes', 'HOSPITAL'),
        ('benchmark/lending/lending_log_high.xes', 'LENDING'),
        ('benchmark/renting/renting_log_high.xes', 'RENTING')
    ]

    all_stats = []

    for xes_path, dataset_name in datasets:
        if not Path(xes_path).exists():
            print(f"\nWARNING: No se encontró {xes_path}")
            continue

        try:
            stats = analyze_dataset(xes_path, dataset_name)
            all_stats.append(stats)
        except Exception as e:
            print(f"\nERROR analizando {dataset_name}:")
            print(f"  {e}")
            import traceback
            traceback.print_exc()

    # Generar histogramas
    if all_stats:
        generate_histograms(all_stats)

    # Guardar estadísticas completas
    output_file = Path('dataset_statistics.json')
    with open(output_file, 'w') as f:
        json.dump(all_stats, f, indent=2, default=str)

    print(f"\n\n{'='*80}")
    print(f"Estadísticas guardadas en: {output_file}")
    print(f"{'='*80}\n")

    # Resumen comparativo
    print(f"\n{'='*80}")
    print("RESUMEN COMPARATIVO")
    print(f"{'='*80}\n")

    print(f"{'Dataset':<12} {'Casos':>10} {'Eventos':>12} {'Actividades':>12} {'Variantes':>12} {'Prom.Long':>10}")
    print("-" * 80)
    for stats in all_stats:
        print(f"{stats['dataset_name']:<12} "
              f"{stats['basic']['n_cases']:>10,} "
              f"{stats['basic']['n_events']:>12,} "
              f"{stats['basic']['n_activities']:>12} "
              f"{stats['variants']['n_unique']:>12,} "
              f"{stats['trace_lengths']['mean']:>10.2f}")


if __name__ == '__main__':
    main()
