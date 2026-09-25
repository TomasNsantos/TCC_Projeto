r"""M2 (XGBoost+SHAP), Dia 6 — treino real no grid completo (378.000
linhas de treino), métricas estratificadas em test, e SHAP nas duas
visões (agregado + por_pi). Etapa 8a do plano aprovado — modelo COMPLETO,
todas as features (a Etapa 8b, condicional, fica para depois, disparada
por leitura humana do ranking produzido aqui, não decidida neste script).

Gate de tempo já cumprido no Prompt 2 (`scripts/medir_tempo_m2_amostra.py`,
commit `3bbf2e0`): projeção de 25,5s para o treino completo, bem abaixo
do teto de 1h — este script roda o treino de verdade, não uma amostra.

Uso (a partir da raiz do projeto, para `import src....` resolver):
    python -m scripts.rodar_m2_grid_completo
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.pipeline.m2_xgboost import (
    amostrar_para_shap,
    calcular_metricas_m2,
    calcular_shap_values,
    carregar_dataset_m2,
    preparar_matriz_treino,
    ranking_importancia_shap,
    treinar_xgboost,
)

DIRETORIO_FEATURES_V3 = Path("output/features_agregadas_v3")

N_AMOSTRA_SHAP = 18_000
SEED_AMOSTRA_SHAP = 42

_COLUNAS_CONTAGEM_VOLUME_BRUTO = (
    "fonte_a_n_eventos_total",
    "fonte_a_volume_total",
    "fonte_a_volume_medio",
    "fonte_b_n_eventos_total",
)
_COLUNAS_FLAG_VAZIO = ("fonte_a_vazia", "fonte_b_vazia")


def _marcar_categoria(feature: str) -> str:
    """Rótulo de apresentação (não interpretação) — só sinaliza qual
    categoria a feature pertence, para leitura humana do ranking."""
    if feature in _COLUNAS_CONTAGEM_VOLUME_BRUTO:
        return "[CONTAGEM/VOLUME BRUTO]"
    if feature in _COLUNAS_FLAG_VAZIO:
        return "[FLAG DE VAZIO]"
    return ""


def main() -> None:
    print("=== M2 (XGBoost+SHAP), Dia 6 — treino real no grid completo ===")
    print()

    print("Carregando features/metadados (output/features_agregadas_v3/)...")
    t0 = time.perf_counter()
    features, metadados = carregar_dataset_m2(
        DIRETORIO_FEATURES_V3 / "features.parquet", DIRETORIO_FEATURES_V3 / "metadados.parquet"
    )
    print(f"  carregado em {time.perf_counter() - t0:.2f}s -- {len(features)} linhas totais")
    print()

    x_completo, y_completo = preparar_matriz_treino(features, metadados)

    mascara_train = (metadados["split"] == "train").to_numpy()
    mascara_val = (metadados["split"] == "val").to_numpy()
    mascara_test = (metadados["split"] == "test").to_numpy()

    x_train = x_completo[mascara_train]
    y_train = y_completo[mascara_train]
    x_val = x_completo[mascara_val]
    y_val = y_completo[mascara_val]
    x_test = x_completo[mascara_test].reset_index(drop=True)
    y_test = y_completo[mascara_test].reset_index(drop=True)
    metadados_test = metadados[mascara_test].reset_index(drop=True)

    print(f"train: {len(x_train)} linhas | val: {len(x_val)} linhas | test: {len(x_test)} linhas")
    print()

    print(f"Treinando XGBoost no grid completo ({len(x_train)} linhas de treino, {len(x_val)} de val)...")
    t0 = time.perf_counter()
    modelo = treinar_xgboost(x_train, y_train, x_val, y_val, early_stopping_rounds=20)
    tempo_treino = time.perf_counter() - t0
    n_arvores_default = modelo.get_params().get("n_estimators")
    print(f"  treino completo em {tempo_treino:.2f}s ({tempo_treino / 60:.2f}min)")
    print(f"  melhor iteração (early stopping): {modelo.best_iteration} (n_estimators default: {n_arvores_default})")
    print()

    print("Gerando y_pred/y_proba sobre X_test...")
    y_pred = modelo.predict(x_test)
    y_proba = modelo.predict_proba(x_test)[:, 1]

    previsoes = pd.DataFrame(
        {
            "y_true": y_test.to_numpy(),
            "y_pred": y_pred,
            "y_proba": y_proba,
            "grupo": metadados_test["grupo"].to_numpy(),
            "delta_t": metadados_test["delta_t"].to_numpy(),
            "recompensa": metadados_test["recompensa"].to_numpy(),
            "pi": metadados_test["pi"].to_numpy(),
        }
    )

    print("Calculando métricas estratificadas (delta_t x recompensa x pi x subgrupo_positivo)...")
    metricas = calcular_metricas_m2(previsoes)
    print(f"  {len(metricas)} linhas de estrato")
    print()

    print(f"Amostrando para SHAP (~{N_AMOSTRA_SHAP} linhas do teste, estratificada grupo x pi x delta_t, seed={SEED_AMOSTRA_SHAP})...")
    t0 = time.perf_counter()
    x_amostra_shap = amostrar_para_shap(x_test, metadados_test, n_amostra=N_AMOSTRA_SHAP, seed=SEED_AMOSTRA_SHAP)
    tempo_amostragem_shap = time.perf_counter() - t0
    metadados_amostra_shap = metadados_test.loc[x_amostra_shap.index].reset_index(drop=True)
    x_amostra_shap = x_amostra_shap.reset_index(drop=True)
    print(f"  amostrar_para_shap em {tempo_amostragem_shap:.2f}s -- {len(x_amostra_shap)} linhas")
    print()

    print(f"Calculando SHAP values (TreeExplainer) sobre a amostra de {len(x_amostra_shap)} linhas...")
    t0 = time.perf_counter()
    shap_values = calcular_shap_values(modelo, x_amostra_shap)
    tempo_shap = time.perf_counter() - t0
    print(f"  calcular_shap_values em {tempo_shap:.2f}s ({tempo_shap / 60:.2f}min) -- shape {shap_values.values.shape}")
    print()

    ranking = ranking_importancia_shap(shap_values, metadados_amostra_shap)

    # -------------------------------------------------------------------
    # Relatório final
    # -------------------------------------------------------------------
    tempo_total_script = time.perf_counter() - t0

    print("=" * 100)
    print("1. TREINO REAL")
    print("=" * 100)
    print(f"Tempo real do treino completo: {tempo_treino:.2f}s ({tempo_treino / 60:.2f}min)")
    print(f"Melhor iteração (early stopping): {modelo.best_iteration} de n_estimators default={n_arvores_default}")
    print()

    print("=" * 100)
    print("2. MÉTRICAS ESTRATIFICADAS (delta_t x recompensa x pi x subgrupo_positivo) -- TABELA COMPLETA")
    print("=" * 100)
    colunas_exibicao = [
        "subgrupo_positivo", "delta_t", "recompensa", "pi",
        "n_positivas", "n_negativas", "vp", "fp", "vn", "fn",
        "precisao", "recall", "f1", "auroc", "fpr",
    ]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(metricas[colunas_exibicao].sort_values(["subgrupo_positivo", "delta_t", "recompensa", "pi"]).to_string(index=False))
    print()

    print("=" * 100)
    print("3. SHAP -- TEMPO E RANKING")
    print("=" * 100)
    print(f"Tempo real de calcular_shap_values ({len(x_amostra_shap)} linhas): {tempo_shap:.2f}s")
    print()

    print("-" * 100)
    print("3a. Ranking AGREGADO (todas as 26 features, sem corte)")
    print("-" * 100)
    agregado = ranking["agregado"].copy()
    agregado["categoria"] = agregado["feature"].map(_marcar_categoria)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(agregado.to_string(index=False))
    print()

    print("-" * 100)
    print("3b. Ranking POR PI (todas as combinações feature x pi, sem corte)")
    print("-" * 100)
    por_pi = ranking["por_pi"].copy()
    por_pi["categoria"] = por_pi["feature"].map(_marcar_categoria)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(por_pi.to_string(index=False))
    print()

    print("=" * 100)
    print("4. LEGENDA DE CATEGORIAS (só organização, sem julgamento de dominância)")
    print("=" * 100)
    print(f"[CONTAGEM/VOLUME BRUTO]: {', '.join(_COLUNAS_CONTAGEM_VOLUME_BRUTO)}")
    print(f"[FLAG DE VAZIO]: {', '.join(_COLUNAS_FLAG_VAZIO)}")
    print("(demais features sem marcação: Bloco A/B restante -- IET/offset/razão --, coordenação, Bloco C)")


if __name__ == "__main__":
    main()
