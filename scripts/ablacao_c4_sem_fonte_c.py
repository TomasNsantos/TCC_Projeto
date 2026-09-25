r"""M2 (XGBoost+SHAP), Dia 6 — Prompt 3.5: sanity check pontual de C4
(A+B, sem Fonte C), comparado diretamente com C7 (A+B+C, Prompt 3,
`scripts/rodar_m2_grid_completo.py`).

**Não é o estudo de ablação C1-C7 completo** (§5.4.1, Semanas 14-17) —
motivado por um achado específico do Prompt 3: `fonte_c_media` domina o
ranking SHAP do modelo completo (SHAP médio ~6,90, ~7x a segunda
feature), 13/26 features em SHAP=0.000000 (incluindo TODO o Bloco de
coordenação), e F1/AUROC=1.000 para `positiva_ativa` em TODOS os níveis
de π sem degradação — risco direto ao critério de sucesso falsificável
do TCC (§5.4.5, C7 precisa superar significativamente max(C1,C2,C3)).

Reaproveita toda a infraestrutura de `src/pipeline/m2_xgboost.py`
(Prompts 1-3) — `preparar_matriz_treino` já aceita `excluir_colunas`,
usado aqui para remover as 5 colunas `fonte_c_*` (confirmadas contra o
schema real do parquet, não assumidas: `fonte_c_media`, `fonte_c_min`,
`fonte_c_max`, `fonte_c_std`, `fonte_c_n_unidades` — nenhuma outra com
esse prefixo existe no dataset).

**Não decide nada sobre o resultado** — só produz os números (métricas
estratificadas + ranking SHAP nas duas visões) para leitura humana,
mesmo princípio já seguido em `rodar_m2_grid_completo.py`.

Uso (a partir da raiz do projeto, para `import src....` resolver):
    python -m scripts.ablacao_c4_sem_fonte_c
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

_COLUNAS_FONTE_C = (
    "fonte_c_media",
    "fonte_c_min",
    "fonte_c_max",
    "fonte_c_std",
    "fonte_c_n_unidades",
)
"""Confirmadas contra o schema real de output/features_agregadas_v3/features.parquet
(`[c for c in features.columns if c.startswith("fonte_c_")]`) -- exatamente
estas 5, nenhuma outra com esse prefixo."""


def main() -> None:
    print("=== M2 (XGBoost+SHAP), Dia 6 — Prompt 3.5: sanity check C4 (A+B, sem Fonte C) ===")
    print()

    print("Carregando features/metadados (output/features_agregadas_v3/)...")
    t0 = time.perf_counter()
    features, metadados = carregar_dataset_m2(
        DIRETORIO_FEATURES_V3 / "features.parquet", DIRETORIO_FEATURES_V3 / "metadados.parquet"
    )
    print(f"  carregado em {time.perf_counter() - t0:.2f}s -- {len(features)} linhas totais")

    colunas_fonte_c_presentes = [c for c in features.columns if c.startswith("fonte_c_")]
    assert set(colunas_fonte_c_presentes) == set(_COLUNAS_FONTE_C), (
        f"Schema mudou desde a confirmação: colunas fonte_c_* no parquet são "
        f"{colunas_fonte_c_presentes}, esperado {_COLUNAS_FONTE_C}."
    )
    print(f"  colunas fonte_c_* confirmadas e excluídas: {colunas_fonte_c_presentes}")
    print()

    x_completo, y_completo = preparar_matriz_treino(features, metadados, excluir_colunas=_COLUNAS_FONTE_C)

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
    print(f"colunas de X (sem fonte_c_*): {list(x_train.columns)}")
    print()

    print(f"Treinando XGBoost SEM Fonte C ({len(x_train)} linhas de treino, {len(x_val)} de val)...")
    t0 = time.perf_counter()
    modelo = treinar_xgboost(x_train, y_train, x_val, y_val, early_stopping_rounds=20)
    tempo_treino = time.perf_counter() - t0
    print(f"  treino completo em {tempo_treino:.2f}s ({tempo_treino / 60:.2f}min)")
    print(f"  melhor iteração (early stopping): {modelo.best_iteration}")
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
    print("=" * 100)
    print("1. TREINO REAL -- C4 (A+B, sem Fonte C)")
    print("=" * 100)
    print(f"Tempo real do treino completo: {tempo_treino:.2f}s ({tempo_treino / 60:.2f}min)")
    print(f"Melhor iteração (early stopping): {modelo.best_iteration}")
    print()

    print("=" * 100)
    print("2. MÉTRICAS ESTRATIFICADAS -- C4 (A+B, sem Fonte C) -- TABELA COMPLETA")
    print("   (comparar linha a linha com C7 do Prompt 3 -- mesma chave delta_t/recompensa/pi/subgrupo_positivo)")
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
    print("3. SHAP -- TEMPO E RANKING -- C4 (A+B, sem Fonte C)")
    print("=" * 100)
    print(f"Tempo real de calcular_shap_values ({len(x_amostra_shap)} linhas): {tempo_shap:.2f}s")
    print()

    print("-" * 100)
    print("3a. Ranking AGREGADO -- C4 (A+B, sem Fonte C) -- todas as 21 features restantes, sem corte")
    print("-" * 100)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(ranking["agregado"].to_string(index=False))
    print()

    print("-" * 100)
    print("3b. Ranking POR PI -- C4 (A+B, sem Fonte C) -- todas as combinações feature x pi, sem corte")
    print("-" * 100)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(ranking["por_pi"].to_string(index=False))
    print()


if __name__ == "__main__":
    main()
