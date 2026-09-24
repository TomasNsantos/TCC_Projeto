"""Testes de src/pipeline/features_agregadas.py (extrair_features_bloco_a/b,
extrair_features_coordenacao, extrair_features_bloco_c,
montar_vetor_features_agregadas)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pipeline.binning import bin_janelas
from src.pipeline.features_agregadas import (
    _LAG_MAXIMO_CORRELACAO_CRUZADA,
    extrair_features_bloco_a,
    extrair_features_bloco_b,
    extrair_features_bloco_c,
    extrair_features_coordenacao,
    montar_vetor_features_agregadas,
)

_DIR_V3 = Path("output/dataset_producao_v3")
_V3_DISPONIVEL = _DIR_V3.exists() and any(_DIR_V3.glob("*.h5"))


def _metadados_sinteticos(linhas: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(linhas)


def _fonte_a_sintetica(linhas: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# Bloco A -- casos básicos
# ---------------------------------------------------------------------------


def test_bloco_a_calcula_contagem_iet_offset_a_mao() -> None:
    """2 eventos em timesteps 1 e 4 -- IET = 3 (um único intervalo),
    std=0.0 (um único intervalo, ddof=0), offset = 1."""
    fonte_a = _fonte_a_sintetica(
        [
            {"classe": "negativa", "window_id": 0, "timestep": 1, "n_eventos": 2, "volume": 10.0},
            {"classe": "negativa", "window_id": 0, "timestep": 4, "n_eventos": 1, "volume": 5.0},
        ]
    )
    fonte_b_vazia = pd.DataFrame({"classe": pd.Series(dtype=object), "window_id": pd.Series(dtype=int), "timestamp": pd.Series(dtype=float)})
    metadados = _metadados_sinteticos([{"classe": "negativa", "window_id": 0, "delta_t": 5.0, "split": "train"}])
    binado = bin_janelas(fonte_a, fonte_b_vazia, metadados)

    resultado = extrair_features_bloco_a(binado, fonte_a).set_index(["classe", "window_id"]).loc[("negativa", 0)]

    assert resultado["fonte_a_n_eventos_total"] == 3
    assert resultado["fonte_a_offset_primeiro_evento"] == 1
    assert resultado["fonte_a_iet_media"] == 3.0
    assert resultado["fonte_a_iet_std"] == 0.0
    assert resultado["fonte_a_vazia"] is np.False_ or resultado["fonte_a_vazia"] == False  # noqa: E712


def test_bloco_a_volume_total_e_medio() -> None:
    fonte_a = _fonte_a_sintetica(
        [
            {"classe": "negativa", "window_id": 0, "timestep": 0, "n_eventos": 2, "volume": 10.0},
            {"classe": "negativa", "window_id": 0, "timestep": 1, "n_eventos": 1, "volume": 5.0},
        ]
    )
    fonte_b_vazia = pd.DataFrame({"classe": pd.Series(dtype=object), "window_id": pd.Series(dtype=int), "timestamp": pd.Series(dtype=float)})
    metadados = _metadados_sinteticos([{"classe": "negativa", "window_id": 0, "delta_t": 2.0, "split": "train"}])
    binado = bin_janelas(fonte_a, fonte_b_vazia, metadados)

    resultado = extrair_features_bloco_a(binado, fonte_a).set_index(["classe", "window_id"]).loc[("negativa", 0)]

    assert resultado["fonte_a_volume_total"] == 15.0
    assert resultado["fonte_a_volume_medio"] == 5.0  # 15.0 / 3 eventos


# ---------------------------------------------------------------------------
# fonte_a_razao_eventos_volume
# ---------------------------------------------------------------------------


def test_fonte_a_razao_eventos_volume_caso_simples() -> None:
    fonte_a = _fonte_a_sintetica(
        [{"classe": "negativa", "window_id": 0, "timestep": 0, "n_eventos": 4, "volume": 8.0}]
    )
    fonte_b_vazia = pd.DataFrame({"classe": pd.Series(dtype=object), "window_id": pd.Series(dtype=int), "timestamp": pd.Series(dtype=float)})
    metadados = _metadados_sinteticos([{"classe": "negativa", "window_id": 0, "delta_t": 1.0, "split": "train"}])
    binado = bin_janelas(fonte_a, fonte_b_vazia, metadados)

    resultado = extrair_features_bloco_a(binado, fonte_a).set_index(["classe", "window_id"]).loc[("negativa", 0)]

    assert resultado["fonte_a_razao_eventos_volume"] == pytest.approx(4.0 / 8.0)


def test_fonte_a_razao_eventos_volume_vazia_e_nan() -> None:
    fonte_a_vazia = pd.DataFrame({"classe": pd.Series(dtype=object), "window_id": pd.Series(dtype=int), "timestep": pd.Series(dtype=int), "n_eventos": pd.Series(dtype=int), "volume": pd.Series(dtype=float)})
    fonte_b_vazia = pd.DataFrame({"classe": pd.Series(dtype=object), "window_id": pd.Series(dtype=int), "timestamp": pd.Series(dtype=float)})
    metadados = _metadados_sinteticos([{"classe": "negativa", "window_id": 0, "delta_t": 2.0, "split": "train"}])
    binado = bin_janelas(fonte_a_vazia, fonte_b_vazia, metadados)

    resultado = extrair_features_bloco_a(binado, fonte_a_vazia).set_index(["classe", "window_id"]).loc[("negativa", 0)]

    assert resultado["fonte_a_vazia"] == True  # noqa: E712
    assert np.isnan(resultado["fonte_a_razao_eventos_volume"])
    assert np.isnan(resultado["fonte_a_volume_total"]) or resultado["fonte_a_volume_total"] == 0.0


# ---------------------------------------------------------------------------
# Casos de borda de vazio -- fonte_a_vazia/fonte_b_vazia
# ---------------------------------------------------------------------------


def test_janela_totalmente_vazia_em_a_produz_estatisticas_nan() -> None:
    fonte_a_vazia = pd.DataFrame({"classe": pd.Series(dtype=object), "window_id": pd.Series(dtype=int), "timestep": pd.Series(dtype=int), "n_eventos": pd.Series(dtype=int), "volume": pd.Series(dtype=float)})
    fonte_b = pd.DataFrame({"classe": ["negativa"], "window_id": [0], "timestamp": [0.5]})
    metadados = _metadados_sinteticos([{"classe": "negativa", "window_id": 0, "delta_t": 2.0, "split": "train"}])
    binado = bin_janelas(fonte_a_vazia, fonte_b, metadados)

    resultado = extrair_features_bloco_a(binado, fonte_a_vazia).set_index(["classe", "window_id"]).loc[("negativa", 0)]

    assert resultado["fonte_a_n_eventos_total"] == 0
    assert resultado["fonte_a_vazia"] == True  # noqa: E712
    assert np.isnan(resultado["fonte_a_iet_media"])
    assert np.isnan(resultado["fonte_a_offset_primeiro_evento"])


def test_caso_positiva_inativa_tudo_nan_em_a_e_b_mas_c_tem_valor_real() -> None:
    """Fixture sintética simulando positiva_inativa: Fonte A/B vazias,
    Fonte C com valor real (resíduo de adesão sincera não paga)."""
    fonte_a_vazia = pd.DataFrame({"classe": pd.Series(dtype=object), "window_id": pd.Series(dtype=int), "timestep": pd.Series(dtype=int), "n_eventos": pd.Series(dtype=int), "volume": pd.Series(dtype=float)})
    fonte_b_vazia = pd.DataFrame({"classe": pd.Series(dtype=object), "window_id": pd.Series(dtype=int), "timestamp": pd.Series(dtype=float)})
    metadados = _metadados_sinteticos(
        [{"classe": "positiva", "window_id": 7, "delta_t": 2.0, "split": "test", "grupo": "positiva_inativa"}]
    )
    binado = bin_janelas(fonte_a_vazia, fonte_b_vazia, metadados)

    bloco_a = extrair_features_bloco_a(binado, fonte_a_vazia).set_index(["classe", "window_id"]).loc[("positiva", 7)]
    bloco_b = extrair_features_bloco_b(binado).set_index(["classe", "window_id"]).loc[("positiva", 7)]
    coordenacao = extrair_features_coordenacao(binado).set_index(["classe", "window_id"]).loc[("positiva", 7)]

    assert bloco_a["fonte_a_vazia"] == True  # noqa: E712
    assert bloco_b["fonte_b_vazia"] == True  # noqa: E712
    for col in ("fonte_a_iet_media", "fonte_a_iet_std", "fonte_a_offset_primeiro_evento", "fonte_a_volume_total"):
        assert np.isnan(bloco_a[col])
    for col in ("fonte_b_iet_media", "fonte_b_iet_std", "fonte_b_offset_primeiro_evento"):
        assert np.isnan(bloco_b[col])
    assert np.isnan(coordenacao["tau_kendall_ab"])
    assert np.isnan(coordenacao["lag_correlacao_cruzada_otimo"])
    assert np.isnan(coordenacao["correlacao_cruzada_maxima"])

    fonte_c = pd.DataFrame(
        {
            "classe": ["positiva"] * 5,
            "window_id": [7] * 5,
            "unidade": [0, 1, 2, 3, 4],
            "fracao_candidato_alvo": [0.35, 0.4, 0.37, 0.33, 0.38],
        }
    )
    bloco_c = extrair_features_bloco_c(fonte_c).set_index(["classe", "window_id"]).loc[("positiva", 7)]
    assert not np.isnan(bloco_c["fonte_c_media"])
    assert bloco_c["fonte_c_n_unidades"] == 5


def test_delta_t_zero_positiva_iet_nan_mesmo_com_muitos_eventos_nao_vazia() -> None:
    """Achado real do grid completo do v3: com delta_t=0.0, TODOS os
    eventos de desembolso da classe positiva caem em timestep=0 (ver
    binning.py/CLAUDE.md, 'timestep estruturalmente impossível' para
    timestep>=1) -- há só 1 TIMESTEP distinto com evento, mesmo com
    dezenas de EVENTOS. IET (intervalo entre timesteps consecutivos com
    evento) fica indefinido (NaN) mesmo com fonte_a_vazia=False -- não é
    bug, é a mesma leitura de 'IET indefinido com <2 pontos' já aplicada
    ao caso de contagem de eventos, agora à contagem de timesteps
    distintos. n_eventos_total/volume/offset continuam calculáveis
    normalmente (não dependem de ter >=2 timesteps distintos)."""
    fonte_a = _fonte_a_sintetica(
        [{"classe": "positiva", "window_id": 0, "timestep": 0, "n_eventos": 32, "volume": 64.0}]
    )
    fonte_b_vazia = pd.DataFrame({"classe": pd.Series(dtype=object), "window_id": pd.Series(dtype=int), "timestamp": pd.Series(dtype=float)})
    metadados = _metadados_sinteticos([{"classe": "positiva", "window_id": 0, "delta_t": 0.0, "split": "train"}])
    binado = bin_janelas(fonte_a, fonte_b_vazia, metadados)

    resultado = extrair_features_bloco_a(binado, fonte_a).set_index(["classe", "window_id"]).loc[("positiva", 0)]

    assert resultado["fonte_a_vazia"] == False  # noqa: E712
    assert resultado["fonte_a_n_eventos_total"] == 32
    assert resultado["fonte_a_offset_primeiro_evento"] == 0.0
    assert not np.isnan(resultado["fonte_a_volume_total"])
    assert np.isnan(resultado["fonte_a_iet_media"])
    assert np.isnan(resultado["fonte_a_iet_std"])
    assert np.isnan(resultado["fonte_a_iet_cv"])


def test_negativa_com_janelas_esparsas_produz_nan_so_nas_vazias() -> None:
    """Múltiplas janelas negativas, algumas com evento em A e outras sem
    -- confirma comportamento correto por janela, não agregado incorreto
    por causa de esparsidade do estrato."""
    fonte_a = _fonte_a_sintetica(
        [
            {"classe": "negativa", "window_id": 0, "timestep": 0, "n_eventos": 1, "volume": 3.0},
            # window_id=1 não tem nenhuma linha em fonte_a -- vazia
        ]
    )
    fonte_b_vazia = pd.DataFrame({"classe": pd.Series(dtype=object), "window_id": pd.Series(dtype=int), "timestamp": pd.Series(dtype=float)})
    metadados = _metadados_sinteticos(
        [
            {"classe": "negativa", "window_id": 0, "delta_t": 2.0, "split": "train"},
            {"classe": "negativa", "window_id": 1, "delta_t": 2.0, "split": "train"},
        ]
    )
    binado = bin_janelas(fonte_a, fonte_b_vazia, metadados)

    resultado = extrair_features_bloco_a(binado, fonte_a).set_index(["classe", "window_id"])

    assert resultado.loc[("negativa", 0), "fonte_a_vazia"] == False  # noqa: E712
    assert not np.isnan(resultado.loc[("negativa", 0), "fonte_a_volume_total"])
    assert resultado.loc[("negativa", 1), "fonte_a_vazia"] == True  # noqa: E712
    assert np.isnan(resultado.loc[("negativa", 1), "fonte_a_volume_total"])


# ---------------------------------------------------------------------------
# tau_kendall_ab via binning
# ---------------------------------------------------------------------------


def test_tau_kendall_calculado_a_mao_via_scipy_direto() -> None:
    from scipy.stats import kendalltau

    fonte_a = _fonte_a_sintetica(
        [
            {"classe": "negativa", "window_id": 0, "timestep": t, "n_eventos": n, "volume": float(n)}
            for t, n in enumerate([1, 3, 2, 5, 4])
        ]
    )
    fonte_b = pd.DataFrame(
        {
            "classe": ["negativa"] * 6,
            "window_id": [0] * 6,
            "timestamp": [0.1, 1.1, 1.2, 2.1, 3.1, 4.1],  # produz contagem_b = [1,2,1,1,1]
        }
    )
    metadados = _metadados_sinteticos([{"classe": "negativa", "window_id": 0, "delta_t": 5.0, "split": "train"}])
    binado = bin_janelas(fonte_a, fonte_b, metadados)

    resultado = extrair_features_coordenacao(binado).set_index(["classe", "window_id"]).loc[("negativa", 0)]

    contagem_a = binado.sort_values("timestep")["contagem_a"].to_numpy()
    contagem_b = binado.sort_values("timestep")["contagem_b"].to_numpy()
    tau_esperado = kendalltau(contagem_a, contagem_b).statistic

    assert resultado["tau_kendall_ab"] == pytest.approx(tau_esperado)


def test_tau_kendall_nan_quando_uma_fonte_vazia() -> None:
    fonte_a_vazia = pd.DataFrame({"classe": pd.Series(dtype=object), "window_id": pd.Series(dtype=int), "timestep": pd.Series(dtype=int), "n_eventos": pd.Series(dtype=int), "volume": pd.Series(dtype=float)})
    fonte_b = pd.DataFrame({"classe": ["negativa"], "window_id": [0], "timestamp": [0.5]})
    metadados = _metadados_sinteticos([{"classe": "negativa", "window_id": 0, "delta_t": 3.0, "split": "train"}])
    binado = bin_janelas(fonte_a_vazia, fonte_b, metadados)

    resultado = extrair_features_coordenacao(binado).set_index(["classe", "window_id"]).loc[("negativa", 0)]

    assert np.isnan(resultado["tau_kendall_ab"])


# ---------------------------------------------------------------------------
# Correlação cruzada -- lag conhecido embutido
# ---------------------------------------------------------------------------


def test_correlacao_cruzada_detecta_lag_positivo_conhecido() -> None:
    """contagem_b = contagem_a deslocado 2 timesteps PARA FRENTE (atrasado)
    -- convenção: lag>0 significa B atrasado em relação a A."""
    n_timesteps = 12
    rng = np.random.default_rng(42)
    base = rng.integers(0, 5, size=n_timesteps).astype(float)
    lag_verdadeiro = 2

    contagem_a_bruta = base
    contagem_b_bruta = np.zeros(n_timesteps)
    contagem_b_bruta[lag_verdadeiro:] = base[: n_timesteps - lag_verdadeiro]

    fonte_a = _fonte_a_sintetica(
        [
            {"classe": "negativa", "window_id": 0, "timestep": t, "n_eventos": int(c), "volume": float(c)}
            for t, c in enumerate(contagem_a_bruta)
            if c > 0
        ]
    )
    linhas_b = []
    for t, c in enumerate(contagem_b_bruta):
        for _ in range(int(c)):
            linhas_b.append({"classe": "negativa", "window_id": 0, "timestamp": t + 0.5})
    fonte_b = pd.DataFrame(linhas_b) if linhas_b else pd.DataFrame({"classe": pd.Series(dtype=object), "window_id": pd.Series(dtype=int), "timestamp": pd.Series(dtype=float)})

    metadados = _metadados_sinteticos([{"classe": "negativa", "window_id": 0, "delta_t": float(n_timesteps), "split": "train"}])
    binado = bin_janelas(fonte_a, fonte_b, metadados)

    resultado = extrair_features_coordenacao(binado).set_index(["classe", "window_id"]).loc[("negativa", 0)]

    assert resultado["lag_correlacao_cruzada_otimo"] == lag_verdadeiro
    assert resultado["correlacao_cruzada_maxima"] == pytest.approx(1.0, abs=1e-6)


def test_lag_maximo_correlacao_cruzada_e_constante_documentada() -> None:
    assert _LAG_MAXIMO_CORRELACAO_CRUZADA == 6


# ---------------------------------------------------------------------------
# Bloco C -- wrapper genérico N=5 e N=1
# ---------------------------------------------------------------------------


def test_bloco_c_com_n_5_unidades_secao() -> None:
    fonte_c_secao = pd.DataFrame(
        {
            "classe": ["negativa"] * 5,
            "window_id": [0] * 5,
            "unidade": [0, 1, 2, 3, 4],
            "fracao_candidato_alvo": [0.2, 0.3, 0.25, 0.28, 0.22],
        }
    )
    resultado = extrair_features_bloco_c(fonte_c_secao).set_index(["classe", "window_id"]).loc[("negativa", 0)]

    assert resultado["fonte_c_n_unidades"] == 5
    assert resultado["fonte_c_media"] == pytest.approx(np.mean([0.2, 0.3, 0.25, 0.28, 0.22]))
    assert not np.isnan(resultado["fonte_c_std"])


def test_bloco_c_com_n_1_unidade_municipio_std_e_nan() -> None:
    fonte_c_municipio = pd.DataFrame(
        {
            "classe": ["negativa"],
            "window_id": [0],
            "unidade": [0],
            "fracao_candidato_alvo": [0.27],
        }
    )
    resultado = extrair_features_bloco_c(fonte_c_municipio).set_index(["classe", "window_id"]).loc[("negativa", 0)]

    assert resultado["fonte_c_n_unidades"] == 1
    assert resultado["fonte_c_media"] == 0.27
    assert np.isnan(resultado["fonte_c_std"])


# ---------------------------------------------------------------------------
# Separação metadado / features / auditoria -- regressão contra vazamento
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _V3_DISPONIVEL, reason="dataset v3 não disponível localmente")
def test_ambas_vazias_nunca_entra_em_features_so_em_metadados() -> None:
    caminho = sorted(_DIR_V3.glob("*.h5"))[0]
    resultado = montar_vetor_features_agregadas(caminho)

    assert "ambas_vazias" not in resultado["features"].columns
    assert "ambas_vazias" in resultado["metadados"].columns
    assert "fonte_a_vazia" in resultado["features"].columns
    assert "fonte_b_vazia" in resultado["features"].columns


# ---------------------------------------------------------------------------
# Integração leve com arquivos reais do v3
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _V3_DISPONIVEL, reason="dataset v3 não disponível localmente")
def test_integracao_leve_com_arquivo_real_recompensa_0_5() -> None:
    candidatos = sorted(_DIR_V3.glob("*recompensa-0.5*.h5"))
    assert candidatos, "esperava pelo menos um arquivo com recompensa=0.5 no v3"
    caminho = candidatos[0]

    resultado = montar_vetor_features_agregadas(caminho)

    assert set(resultado.keys()) == {"features", "metadados"}
    assert len(resultado["features"]) == len(resultado["metadados"])
    assert len(resultado["features"]) > 0
    assert resultado["metadados"]["window_id"].notna().all()
    # positiva_inativa deve estar presente e ter fonte_a_vazia=True em toda linha
    metadados = resultado["metadados"]
    inativas_idx = metadados.index[metadados["grupo"] == "positiva_inativa"]
    if len(inativas_idx) > 0:
        assert resultado["features"].loc[inativas_idx, "fonte_a_vazia"].all()


@pytest.mark.skipif(not _V3_DISPONIVEL, reason="dataset v3 não disponível localmente")
def test_integracao_leve_processa_qualquer_arquivo_sem_erro() -> None:
    caminho = sorted(_DIR_V3.glob("*.h5"))[0]

    resultado = montar_vetor_features_agregadas(caminho)

    colunas_esperadas = {
        "fonte_a_n_eventos_total",
        "fonte_a_vazia",
        "fonte_b_n_eventos_total",
        "fonte_b_vazia",
        "tau_kendall_ab",
        "lag_correlacao_cruzada_otimo",
        "correlacao_cruzada_maxima",
        "fonte_c_media",
        "fonte_c_n_unidades",
    }
    assert colunas_esperadas.issubset(set(resultado["features"].columns))
