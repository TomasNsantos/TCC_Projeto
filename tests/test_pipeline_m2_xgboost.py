"""Testes de src/pipeline/m2_xgboost.py (carregar_dataset_m2,
preparar_matriz_treino, treinar_xgboost, calcular_metricas_m2,
amostrar_para_shap, calcular_shap_values, ranking_importancia_shap)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pipeline.m2_xgboost import (
    _COLUNAS_CAMADA_1_CRUA,
    _COLUNAS_EXCLUIDAS_FEATURES,
    amostrar_para_shap,
    calcular_metricas_m2,
    calcular_shap_values,
    carregar_dataset_m2,
    preparar_matriz_treino,
    ranking_importancia_shap,
    treinar_xgboost,
)

_DIR_FEATURES_V3 = Path("output/features_agregadas_v3")
_V3_DISPONIVEL = (_DIR_FEATURES_V3 / "features.parquet").exists() and (_DIR_FEATURES_V3 / "metadados.parquet").exists()

_COLUNAS_FEATURE_REAIS = (
    "fonte_a_n_eventos_total", "fonte_a_volume_total", "fonte_a_volume_medio",
    "fonte_a_volume_por_timestep_std", "fonte_a_volume_por_timestep_min",
    "fonte_a_volume_por_timestep_max", "fonte_a_razao_eventos_volume",
    "fonte_a_iet_media", "fonte_a_iet_std", "fonte_a_iet_cv",
    "fonte_a_offset_primeiro_evento", "fonte_a_vazia",
    "fonte_b_n_eventos_total", "fonte_b_iet_media", "fonte_b_iet_std",
    "fonte_b_iet_cv", "fonte_b_offset_primeiro_evento", "fonte_b_vazia",
    "tau_kendall_ab", "lag_correlacao_cruzada_otimo", "correlacao_cruzada_maxima",
    "fonte_c_media", "fonte_c_min", "fonte_c_max", "fonte_c_std", "fonte_c_n_unidades",
)


def _features_sinteticas(n: int, seed: int = 0) -> pd.DataFrame:
    """Fixture sintética pequena com todas as 26 colunas reais de feature
    + classe/window_id, valores aleatórios simples (sem NaN, para não
    interferir nos testes de forma/estratificação -- testes de NaN já
    são cobertos em test_pipeline_features_agregadas.py)."""
    rng = np.random.default_rng(seed)
    dados = {"classe": ["negativa"] * n, "window_id": np.arange(n)}
    for coluna in _COLUNAS_FEATURE_REAIS:
        if coluna in ("fonte_a_vazia", "fonte_b_vazia"):
            dados[coluna] = rng.random(n) < 0.3
        elif coluna in ("fonte_a_n_eventos_total", "fonte_b_n_eventos_total", "fonte_c_n_unidades"):
            dados[coluna] = rng.integers(0, 20, n)
        else:
            dados[coluna] = rng.random(n) * 10
    return pd.DataFrame(dados)


def _metadados_sinteticos(
    n: int,
    grupos: list[str] | None = None,
    pi_valores: list[float] | None = None,
    delta_t_valores: list[float] | None = None,
    split: str = "train",
    seed: int = 0,
) -> pd.DataFrame:
    """Fixture sintética de metadados -- ciclando grupos/pi/delta_t
    fornecidos para garantir presença de todos os estratos pedidos nos
    testes de estratificação."""
    rng = np.random.default_rng(seed)
    grupos = grupos or ["negativa"]
    pi_valores = pi_valores or [0.0]
    delta_t_valores = delta_t_valores or [2.0]
    return pd.DataFrame(
        {
            "classe": ["negativa" if [g for g in grupos][i % len(grupos)] == "negativa" else "positiva" for i in range(n)],
            "window_id": np.arange(n),
            "split": split,
            "grupo": [grupos[i % len(grupos)] for i in range(n)],
            "contrato_ativado": [np.nan if grupos[i % len(grupos)] == "negativa" else float(grupos[i % len(grupos)] == "positiva_ativa") for i in range(n)],
            "beta": 1,
            "delta_t": [delta_t_valores[i % len(delta_t_valores)] for i in range(n)],
            "g": "pool",
            "pi": [pi_valores[i % len(pi_valores)] for i in range(n)],
            "recompensa": 1.0,
            "rho": 0.0,
            "seed": 1,
            "ambas_vazias": False,
        }
    )


# ---------------------------------------------------------------------------
# preparar_matriz_treino
# ---------------------------------------------------------------------------


def test_preparar_matriz_treino_y_binario_negativa_zero_resto_um() -> None:
    features = _features_sinteticas(6)
    metadados = _metadados_sinteticos(
        6, grupos=["negativa", "positiva_ativa", "positiva_inativa"]
    )

    x, y = preparar_matriz_treino(features, metadados)

    assert list(y) == [0, 1, 1, 0, 1, 1]
    assert y.name == "y"


def test_preparar_matriz_treino_exclui_colunas_de_chave() -> None:
    features = _features_sinteticas(4)
    metadados = _metadados_sinteticos(4)

    x, _ = preparar_matriz_treino(features, metadados)

    assert "classe" not in x.columns
    assert "window_id" not in x.columns
    for coluna in _COLUNAS_FEATURE_REAIS:
        assert coluna in x.columns


def test_preparar_matriz_treino_exclui_colunas_extras_etapa_8b() -> None:
    features = _features_sinteticas(4)
    metadados = _metadados_sinteticos(4)

    x, _ = preparar_matriz_treino(features, metadados, excluir_colunas=_COLUNAS_CAMADA_1_CRUA)

    for coluna in _COLUNAS_CAMADA_1_CRUA:
        assert coluna not in x.columns
    # colunas de coordenação/Bloco C continuam presentes
    assert "tau_kendall_ab" in x.columns
    assert "fonte_c_media" in x.columns


def test_preparar_matriz_treino_tamanhos_diferentes_levanta_erro() -> None:
    features = _features_sinteticas(4)
    metadados = _metadados_sinteticos(5)

    with pytest.raises(ValueError):
        preparar_matriz_treino(features, metadados)


# ---------------------------------------------------------------------------
# calcular_metricas_m2 -- duas linhas por estrato, não uma degenerada por grupo
# ---------------------------------------------------------------------------


def _previsoes_sinteticas() -> pd.DataFrame:
    """Um único estrato (delta_t=2.0, recompensa=1.0, pi=0.0), com
    negativa/positiva_ativa/positiva_inativa presentes -- confusão
    conhecida à mão para cada subgrupo positivo."""
    linhas = []
    # negativas: 3 vn, 1 fp
    for y_pred, y_proba in [(0, 0.1), (0, 0.2), (0, 0.3), (1, 0.9)]:
        linhas.append({"grupo": "negativa", "y_true": 0, "y_pred": y_pred, "y_proba": y_proba})
    # positiva_ativa: 3 vp, 1 fn
    for y_pred, y_proba in [(1, 0.8), (1, 0.9), (1, 0.95), (0, 0.4)]:
        linhas.append({"grupo": "positiva_ativa", "y_true": 1, "y_pred": y_pred, "y_proba": y_proba})
    # positiva_inativa: 2 vp, 2 fn
    for y_pred, y_proba in [(1, 0.7), (1, 0.6), (0, 0.3), (0, 0.2)]:
        linhas.append({"grupo": "positiva_inativa", "y_true": 1, "y_pred": y_pred, "y_proba": y_proba})

    df = pd.DataFrame(linhas)
    df["delta_t"] = 2.0
    df["recompensa"] = 1.0
    df["pi"] = 0.0
    return df


def test_calcular_metricas_m2_produz_duas_linhas_por_estrato() -> None:
    previsoes = _previsoes_sinteticas()

    metricas = calcular_metricas_m2(previsoes)

    assert len(metricas) == 2
    assert set(metricas["subgrupo_positivo"]) == {"positiva_ativa", "positiva_inativa"}


def test_calcular_metricas_m2_matriz_confusao_correta_por_subgrupo() -> None:
    previsoes = _previsoes_sinteticas()

    metricas = calcular_metricas_m2(previsoes).set_index("subgrupo_positivo")

    ativa = metricas.loc["positiva_ativa"]
    assert ativa["vp"] == 3
    assert ativa["fn"] == 1
    assert ativa["fp"] == 1
    assert ativa["vn"] == 3

    inativa = metricas.loc["positiva_inativa"]
    assert inativa["vp"] == 2
    assert inativa["fn"] == 2
    assert inativa["fp"] == 1
    assert inativa["vn"] == 3

    # fpr é o mesmo nas duas linhas -- mesmo conjunto de negativas reaproveitado
    assert ativa["fpr"] == inativa["fpr"] == pytest.approx(1 / 4)


def test_calcular_metricas_m2_auroc_calculavel_com_duas_classes() -> None:
    previsoes = _previsoes_sinteticas()

    metricas = calcular_metricas_m2(previsoes).set_index("subgrupo_positivo")

    assert not np.isnan(metricas.loc["positiva_ativa", "auroc"])
    assert not np.isnan(metricas.loc["positiva_inativa", "auroc"])


def test_calcular_metricas_m2_estrato_sem_negativas_e_omitido() -> None:
    previsoes = _previsoes_sinteticas()
    previsoes = previsoes[previsoes["grupo"] != "negativa"]

    metricas = calcular_metricas_m2(previsoes)

    assert metricas.empty


def test_calcular_metricas_m2_colunas_obrigatorias_faltando_levanta_erro() -> None:
    previsoes = _previsoes_sinteticas().drop(columns=["pi"])

    with pytest.raises(ValueError):
        calcular_metricas_m2(previsoes)


# ---------------------------------------------------------------------------
# amostrar_para_shap -- estratificação grupo x pi x delta_t
# ---------------------------------------------------------------------------


def test_amostrar_para_shap_respeita_tamanho_aproximado() -> None:
    n_total = 1000
    features = _features_sinteticas(n_total)
    x = features.drop(columns=list(_COLUNAS_EXCLUIDAS_FEATURES))
    metadados = _metadados_sinteticos(
        n_total,
        grupos=["negativa", "positiva_ativa", "positiva_inativa"],
        pi_valores=[0.0, 0.5, 0.9],
        delta_t_valores=[0.0, 2.0, 24.0],
        split="test",
    )

    amostra = amostrar_para_shap(x, metadados, n_amostra=200, seed=42)

    # tolerância generosa -- amostragem por grupo é aproximada, não exata
    assert 100 <= len(amostra) <= 300


def test_amostrar_para_shap_cobre_todos_os_estratos_presentes() -> None:
    n_total = 900
    features = _features_sinteticas(n_total)
    x = features.drop(columns=list(_COLUNAS_EXCLUIDAS_FEATURES))
    metadados = _metadados_sinteticos(
        n_total,
        grupos=["negativa", "positiva_ativa", "positiva_inativa"],
        pi_valores=[0.0, 0.5],
        delta_t_valores=[2.0, 24.0],
        split="test",
    )

    amostra = amostrar_para_shap(x, metadados, n_amostra=400, seed=1)
    metadados_amostra = metadados.loc[amostra.index]

    estratos_originais = set(map(tuple, metadados[["grupo", "pi", "delta_t"]].drop_duplicates().to_numpy(dtype=object)))
    estratos_amostra = set(map(tuple, metadados_amostra[["grupo", "pi", "delta_t"]].drop_duplicates().to_numpy(dtype=object)))

    assert estratos_amostra == estratos_originais


def test_amostrar_para_shap_tamanhos_diferentes_levanta_erro() -> None:
    features = _features_sinteticas(5)
    x = features.drop(columns=list(_COLUNAS_EXCLUIDAS_FEATURES))
    metadados = _metadados_sinteticos(6, split="test")

    with pytest.raises(ValueError):
        amostrar_para_shap(x, metadados, n_amostra=3, seed=0)


# ---------------------------------------------------------------------------
# ranking_importancia_shap -- duas chaves do dict, formatos esperados
# ---------------------------------------------------------------------------


class _ExplanationFalso:
    """Stub mínimo de shap.Explanation -- só os atributos que
    ranking_importancia_shap lê (values, feature_names)."""

    def __init__(self, values: np.ndarray, feature_names: list[str]) -> None:
        self.values = values
        self.feature_names = feature_names


def test_ranking_importancia_shap_retorna_duas_chaves() -> None:
    n, k = 20, 3
    rng = np.random.default_rng(0)
    valores = rng.normal(size=(n, k))
    nomes = ["feat_a", "feat_b", "feat_c"]
    shap_values = _ExplanationFalso(valores, nomes)
    metadados_amostra = pd.DataFrame({"pi": [0.0] * 10 + [0.5] * 10})

    ranking = ranking_importancia_shap(shap_values, metadados_amostra)

    assert set(ranking.keys()) == {"agregado", "por_pi"}


def test_ranking_importancia_shap_agregado_formato_e_ordenacao() -> None:
    valores = np.array([[1.0, -5.0, 0.5], [2.0, 3.0, -0.5]])
    nomes = ["feat_a", "feat_b", "feat_c"]
    shap_values = _ExplanationFalso(valores, nomes)
    metadados_amostra = pd.DataFrame({"pi": [0.0, 0.0]})

    ranking = ranking_importancia_shap(shap_values, metadados_amostra)
    agregado = ranking["agregado"]

    assert list(agregado.columns) == ["feature", "shap_medio_abs"]
    # feat_b: media(|-5|,|3|)=4.0 -- maior; feat_a: media(1,2)=1.5; feat_c: media(0.5,0.5)=0.5
    assert list(agregado["feature"]) == ["feat_b", "feat_a", "feat_c"]
    assert agregado["shap_medio_abs"].iloc[0] == pytest.approx(4.0)


def test_ranking_importancia_shap_por_pi_formato_e_quebra_por_pi() -> None:
    # pi=0.0: feat_a domina; pi=0.5: feat_b domina
    valores = np.array([[10.0, 0.1], [0.1, 10.0]])
    nomes = ["feat_a", "feat_b"]
    shap_values = _ExplanationFalso(valores, nomes)
    metadados_amostra = pd.DataFrame({"pi": [0.0, 0.5]})

    ranking = ranking_importancia_shap(shap_values, metadados_amostra)
    por_pi = ranking["por_pi"]

    assert list(por_pi.columns) == ["pi", "feature", "shap_medio_abs"]
    assert set(por_pi["pi"]) == {0.0, 0.5}

    top_pi_0 = por_pi[por_pi["pi"] == 0.0].iloc[0]
    assert top_pi_0["feature"] == "feat_a"

    top_pi_05 = por_pi[por_pi["pi"] == 0.5].iloc[0]
    assert top_pi_05["feature"] == "feat_b"


def test_ranking_importancia_shap_tamanhos_diferentes_levanta_erro() -> None:
    valores = np.zeros((5, 2))
    shap_values = _ExplanationFalso(valores, ["a", "b"])
    metadados_amostra = pd.DataFrame({"pi": [0.0, 0.5]})  # só 2 linhas, valores tem 5

    with pytest.raises(ValueError):
        ranking_importancia_shap(shap_values, metadados_amostra)


# ---------------------------------------------------------------------------
# Integração leve: treino real minúsculo (não é o grid completo) +
# calcular_shap_values sobre o modelo treinado
# ---------------------------------------------------------------------------


def test_treinar_xgboost_e_calcular_shap_values_fluxo_minimo() -> None:
    """Fluxo mínimo de ponta a ponta com dados sintéticos pequenos --
    NÃO é o grid completo (isso é escopo de um prompt futuro), só
    confirma que as funções se encaixam sem erro."""
    n = 60
    features = _features_sinteticas(n, seed=7)
    metadados = _metadados_sinteticos(
        n, grupos=["negativa", "positiva_ativa"], split="train", seed=7
    )

    x, y = preparar_matriz_treino(features, metadados)
    x_train, y_train = x.iloc[:40], y.iloc[:40]
    x_val, y_val = x.iloc[40:], y.iloc[40:]

    modelo = treinar_xgboost(x_train, y_train, x_val, y_val, early_stopping_rounds=5)

    shap_values = calcular_shap_values(modelo, x_val)

    assert shap_values.values.shape[0] == len(x_val)


# ---------------------------------------------------------------------------
# Integração leve com o parquet real (Dia 4-5)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _V3_DISPONIVEL, reason="output/features_agregadas_v3 não disponível localmente")
def test_carregar_dataset_m2_le_parquet_real_e_valida_alinhamento() -> None:
    features, metadados = carregar_dataset_m2(
        _DIR_FEATURES_V3 / "features.parquet", _DIR_FEATURES_V3 / "metadados.parquet"
    )

    assert len(features) == len(metadados)
    assert len(features) == 540_000
    for coluna in _COLUNAS_FEATURE_REAIS:
        assert coluna in features.columns
    assert "grupo" in metadados.columns
    assert "split" in metadados.columns


@pytest.mark.skipif(not _V3_DISPONIVEL, reason="output/features_agregadas_v3 não disponível localmente")
def test_preparar_matriz_treino_sobre_dataset_real_produz_y_binario_coerente() -> None:
    features, metadados = carregar_dataset_m2(
        _DIR_FEATURES_V3 / "features.parquet", _DIR_FEATURES_V3 / "metadados.parquet"
    )

    x, y = preparar_matriz_treino(features, metadados)

    assert set(y.unique()) <= {0, 1}
    assert "classe" not in x.columns
    assert "window_id" not in x.columns
    # positiva_inativa (grupo != negativa) deve estar em y==1
    idx_inativa = metadados.index[metadados["grupo"] == "positiva_inativa"][0]
    assert y.iloc[idx_inativa] == 1
