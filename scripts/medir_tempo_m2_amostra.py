r"""M2 (XGBoost+SHAP), Dia 6 — medição de tempo numa amostra, ANTES de
treinar no grid completo (378.000 linhas de treino). Gate obrigatório do
plano aprovado (decisão 5): projeta linearmente o tempo de treino no
grid completo a partir de uma amostra de ~25k linhas; se a projeção
ultrapassar ~1h, PARA e reporta — não decide sozinho reduzir dataset ou
trocar hiperparâmetros.

**Não é o treino real** — só mede tempo. `treinar_xgboost` roda sobre uma
amostra aleatória simples de 25k linhas de treino (não estratificada,
`random_state=42`), com `X_val`/`y_val` COMPLETOS (81.000 linhas, mesmo
tamanho que a rodada real vai usar — early stopping é parte do custo real
do treino, não deve ser escondido/reduzido só para a medição).

Mede também, sobre o MESMO modelo treinado na amostra (não treina de novo
só para medir SHAP): tempo de `amostrar_para_shap` (~15-20k linhas do
teste, estratificada `grupo × pi × delta_t`) e tempo de
`calcular_shap_values` (TreeExplainer) nessa amostra.

Uso (a partir da raiz do projeto, para `import src....` resolver):
    python -m scripts.medir_tempo_m2_amostra
"""

from __future__ import annotations

import time
from pathlib import Path

from src.pipeline.m2_xgboost import (
    amostrar_para_shap,
    calcular_shap_values,
    carregar_dataset_m2,
    preparar_matriz_treino,
    treinar_xgboost,
)

DIRETORIO_FEATURES_V3 = Path("output/features_agregadas_v3")

N_AMOSTRA_TREINO = 25_000
"""Tamanho da amostra aleatória simples de treino usada só para medir
tempo — NÃO é o treino real (378.000 linhas)."""

SEED_AMOSTRA_TREINO = 42
"""Seed da amostra aleatória simples de N_AMOSTRA_TREINO linhas de
train — documentada explicitamente para reprodutibilidade da medição,
mesma convenção de seed explícita já usada em todo o projeto."""

N_AMOSTRA_SHAP = 18_000
"""Tamanho alvo da amostra estratificada para SHAP (decisão 7 do plano:
~15-20k linhas) — meio da faixa pedida."""

SEED_AMOSTRA_SHAP = 42

LIMITE_PROJECAO_SEGUNDOS = 3600.0
"""Gate obrigatório do plano (decisão 5) — se a projeção linear para o
grid completo ultrapassar isso, o script para e reporta, sem prosseguir."""


def main() -> None:
    print("=== M2 (XGBoost+SHAP), Dia 6 — medição de tempo numa amostra ===")
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

    x_train_completo = x_completo[mascara_train]
    y_train_completo = y_completo[mascara_train]
    x_val = x_completo[mascara_val]
    y_val = y_completo[mascara_val]
    x_test = x_completo[mascara_test]
    metadados_test = metadados[mascara_test].reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)

    n_train_total = len(x_train_completo)
    print(f"train: {n_train_total} linhas totais | val: {len(x_val)} linhas | test: {len(x_test)} linhas")
    print()

    print(f"Amostrando {N_AMOSTRA_TREINO} linhas de train (aleatória simples, seed={SEED_AMOSTRA_TREINO})...")
    x_amostra_treino = x_train_completo.sample(n=N_AMOSTRA_TREINO, random_state=SEED_AMOSTRA_TREINO)
    y_amostra_treino = y_train_completo.loc[x_amostra_treino.index]
    print(f"  amostra: {len(x_amostra_treino)} linhas")
    print()

    print(f"Treinando XGBoost na amostra de {N_AMOSTRA_TREINO} linhas, com val COMPLETO ({len(x_val)} linhas) para early stopping...")
    t0 = time.perf_counter()
    modelo = treinar_xgboost(x_amostra_treino, y_amostra_treino, x_val, y_val, early_stopping_rounds=20)
    tempo_treino_amostra = time.perf_counter() - t0
    print(f"  treino (amostra {N_AMOSTRA_TREINO} + val completo) em {tempo_treino_amostra:.2f}s ({tempo_treino_amostra / 60:.2f}min)")
    print(f"  melhor iteração (early stopping): {modelo.best_iteration}")
    print()

    fator_escala = n_train_total / N_AMOSTRA_TREINO
    projecao_segundos = tempo_treino_amostra * fator_escala
    print("Projeção linear para o grid completo de treino:")
    print(f"  {tempo_treino_amostra:.2f}s (amostra {N_AMOSTRA_TREINO}) x ({n_train_total} / {N_AMOSTRA_TREINO} = {fator_escala:.4f}x escala)")
    print(f"  = {projecao_segundos:.1f}s ({projecao_segundos / 60:.2f}min, {projecao_segundos / 3600:.3f}h)")
    print("  (X_val/y_val mantidos no tamanho COMPLETO na projeção -- não escalam com o treino, já estão no tamanho real)")
    print()

    print(f"Amostrando para SHAP (~{N_AMOSTRA_SHAP} linhas do teste, estratificada grupo x pi x delta_t, seed={SEED_AMOSTRA_SHAP})...")
    t0 = time.perf_counter()
    x_amostra_shap = amostrar_para_shap(x_test, metadados_test, n_amostra=N_AMOSTRA_SHAP, seed=SEED_AMOSTRA_SHAP)
    tempo_amostragem_shap = time.perf_counter() - t0
    print(f"  amostrar_para_shap em {tempo_amostragem_shap:.2f}s -- {len(x_amostra_shap)} linhas")
    print()

    print(f"Calculando SHAP values (TreeExplainer) sobre a amostra de {len(x_amostra_shap)} linhas (modelo treinado na amostra de {N_AMOSTRA_TREINO})...")
    t0 = time.perf_counter()
    shap_values = calcular_shap_values(modelo, x_amostra_shap)
    tempo_shap = time.perf_counter() - t0
    print(f"  calcular_shap_values em {tempo_shap:.2f}s ({tempo_shap / 60:.2f}min) -- shape {shap_values.values.shape}")
    print()

    print("=" * 100)
    print("RESUMO DA MEDIÇÃO")
    print("=" * 100)
    print(f"Treino amostra ({N_AMOSTRA_TREINO} linhas + val completo {len(x_val)}): {tempo_treino_amostra:.2f}s ({tempo_treino_amostra / 60:.2f}min)")
    print(f"Projeção linear para grid completo ({n_train_total} linhas de treino, val completo mantido): "
          f"{projecao_segundos:.1f}s ({projecao_segundos / 60:.2f}min, {projecao_segundos / 3600:.3f}h)")
    print(f"amostrar_para_shap (~{N_AMOSTRA_SHAP} linhas): {tempo_amostragem_shap:.2f}s")
    print(f"calcular_shap_values ({len(x_amostra_shap)} linhas): {tempo_shap:.2f}s ({tempo_shap / 60:.2f}min)")
    print()

    if projecao_segundos > LIMITE_PROJECAO_SEGUNDOS:
        print("!" * 100)
        print(f"GATE: projeção ({projecao_segundos / 3600:.3f}h) ULTRAPASSA o limite de {LIMITE_PROJECAO_SEGUNDOS / 3600:.1f}h.")
        print("PARANDO -- não treinar no grid completo sem decisão do usuário (decisão 5 do plano aprovado).")
        print("!" * 100)
    else:
        print("=" * 100)
        print(f"GATE: projeção ({projecao_segundos / 3600:.3f}h) DENTRO do limite de {LIMITE_PROJECAO_SEGUNDOS / 3600:.1f}h.")
        print("Pronto para o Prompt 3 (treino real no grid completo).")
        print("=" * 100)


if __name__ == "__main__":
    main()
