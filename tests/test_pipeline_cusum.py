"""Testes de src/pipeline/cusum.py (cusum_unilateral, aplicar_cusum_por_janela,
calcular_metricas_m1, fracao_disparos_no_piso_da_referencia)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pipeline.binning import bin_janelas, calcular_referencia_negativa_treino, calcular_zscore
from src.pipeline.cusum import (
    aplicar_cusum_por_janela,
    calcular_metricas_m1,
    cusum_unilateral,
    fracao_disparos_no_piso_da_referencia,
)
from src.pipeline.loader import carregar_arquivo

_DIR_V3 = Path("output/dataset_producao_v3")
_V3_DISPONIVEL = _DIR_V3.exists() and any(_DIR_V3.glob("*.h5"))


# ---------------------------------------------------------------------------
# cusum_unilateral -- série sintética, degrau controlado
# ---------------------------------------------------------------------------


def test_cusum_dispara_no_timestep_exato_do_degrau() -> None:
    """Ruído baixo (z-score ~0) antes do degrau, salto acima do limiar
    depois -- confirma que S+ cruza h exatamente no timestep esperado,
    nem antes nem depois."""
    # k=0.5, h=5.0: antes do degrau (zscore=0), S+ fica em 0 (max(0, 0-0.5)=0).
    # No degrau (zscore=3.0 a partir do timestep 5), S+ acumula 2.5/timestep:
    # t=5: 2.5, t=6: 5.0 -> dispara em t=6.
    zscore = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0])
    disparou, timestep_disparo = cusum_unilateral(zscore, k=0.5, h=5.0)

    assert disparou is True
    assert timestep_disparo == 6


def test_cusum_nao_dispara_sem_degrau() -> None:
    zscore = np.array([0.1, -0.2, 0.3, 0.0, 0.1])
    disparou, timestep_disparo = cusum_unilateral(zscore, k=0.5, h=5.0)

    assert disparou is False
    assert timestep_disparo is None


def test_cusum_timestep_disparo_e_o_primeiro_cruzamento_nao_o_maximo() -> None:
    """S+ continua subindo depois do primeiro cruzamento -- timestep_disparo
    deve ser o PRIMEIRO ponto em que S+ >= h, não o timestep de valor
    máximo de S+."""
    zscore = np.array([5.0, 5.0, 5.0, 5.0])  # k=0: S+ = [5, 10, 15, 20], cruza h=5 já em t=0
    disparou, timestep_disparo = cusum_unilateral(zscore, k=0.0, h=5.0)

    assert disparou is True
    assert timestep_disparo == 0  # não 3 (onde S+ é máximo)


def test_cusum_h_nao_positivo_levanta_erro() -> None:
    with pytest.raises(ValueError, match="h deve ser > 0"):
        cusum_unilateral(np.array([1.0, 2.0]), k=0.5, h=0.0)
    with pytest.raises(ValueError, match="h deve ser > 0"):
        cusum_unilateral(np.array([1.0, 2.0]), k=0.5, h=-1.0)


def test_cusum_impossivel_com_tamanho_errado_levanta_erro() -> None:
    with pytest.raises(ValueError, match="mesmo tamanho"):
        cusum_unilateral(np.array([1.0, 2.0, 3.0]), k=0.5, h=5.0, impossivel=np.array([False, True]))


# ---------------------------------------------------------------------------
# cusum_unilateral -- timesteps impossíveis pulados (não zerados)
# ---------------------------------------------------------------------------


def test_timesteps_impossiveis_nao_disparam_por_si_sos() -> None:
    """Série onde TODA a evidência é ruído (nunca cruzaria h), com alguns
    timesteps marcados impossíveis interpostos -- confirma que marcar como
    impossível não introduz disparo por si só (não é lido como um desvio
    positivo artificial)."""
    zscore = np.array([0.1, 10.0, 0.1, 10.0, 0.1])  # os "10.0" são impossíveis
    impossivel = np.array([False, True, False, True, False])

    disparou, timestep_disparo = cusum_unilateral(zscore, k=0.5, h=5.0, impossivel=impossivel)

    assert disparou is False
    assert timestep_disparo is None


def test_timesteps_impossiveis_nao_suprimem_disparo_real() -> None:
    """Evidência real o bastante para disparar, com timesteps impossíveis
    interpostos -- confirma que pular (não zerar) não impede o acumulador
    de continuar de onde parou."""
    # zscore real nos timesteps validos: 3.0, 3.0, 3.0 (t=0,2,4) -- k=0.5:
    # t=0: 2.5, [t=1 impossivel, pula], t=2: 2.5+2.5=5.0 -> dispara em t=2.
    zscore = np.array([3.0, 999.0, 3.0, 999.0, 3.0])
    impossivel = np.array([False, True, False, True, False])

    disparou, timestep_disparo = cusum_unilateral(zscore, k=0.5, h=5.0, impossivel=impossivel)

    assert disparou is True
    assert timestep_disparo == 2  # não o timestep 1 (impossível, ignorado)


def test_disparo_sticky_atraves_de_timesteps_impossiveis_ate_o_final_da_janela() -> None:
    """Teste pedido explicitamente pelo usuário: cruzamento de h em
    timestep=0 (caso positiva/delta_t=0.0), seguido de timesteps 1-23
    TODOS marcados impossíveis -- o disparo deve permanecer True (flag
    sticky ao longo da varredura), não reavaliado/perdido pelo estado
    final de S+ após os timesteps pulados. timestep_disparo deve ser 0,
    não None nem o último timestep pulado."""
    n_timesteps = 24
    zscore = np.zeros(n_timesteps)
    zscore[0] = 100.0  # disparo garantido já em t=0
    impossivel = np.zeros(n_timesteps, dtype=bool)
    impossivel[1:] = True  # timesteps 1..23 impossíveis, como em positiva/delta_t=0.0

    disparou, timestep_disparo = cusum_unilateral(zscore, k=0.5, h=5.0, impossivel=impossivel)

    assert disparou is True
    assert timestep_disparo == 0


# ---------------------------------------------------------------------------
# cusum_unilateral -- k/h expostos e testáveis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "h, deveria_disparar",
    [(5.0, True), (100.0, False)],
)
def test_h_maior_dispara_menos(h: float, deveria_disparar: bool) -> None:
    zscore = np.array([3.0, 3.0, 3.0, 3.0])
    disparou, _ = cusum_unilateral(zscore, k=0.5, h=h)
    assert disparou is deveria_disparar


@pytest.mark.parametrize(
    "k, deveria_disparar",
    [(0.1, True), (2.9, False)],
)
def test_k_maior_dispara_menos(k: float, deveria_disparar: bool) -> None:
    zscore = np.array([3.0, 3.0, 3.0, 3.0])
    disparou, _ = cusum_unilateral(zscore, k=k, h=5.0)
    assert disparou is deveria_disparar


# ---------------------------------------------------------------------------
# aplicar_cusum_por_janela -- fixtures sintéticas
# ---------------------------------------------------------------------------


def _com_zscore_sintetico(linhas: list[dict]) -> pd.DataFrame:
    base = {"recompensa": 1.0, "pi": 0.0, "delta_t": 2.0, "rho": 0.0, "beta": 1, "g": "pool", "seed": 1, "split": "test", "grupo": "positiva_ativa"}
    return pd.DataFrame([{**base, **linha} for linha in linhas])


def test_aplicar_cusum_or_entre_fonte_a_e_b() -> None:
    """disparo_m1 deve ser True se A OU B disparar (decisão de design:
    OR entre fontes)."""
    linhas = [
        {"classe": "positiva", "window_id": 0, "timestep": t, "zscore_a": 3.0, "zscore_b": 0.0, "timestep_estruturalmente_impossivel": False}
        for t in range(4)
    ]  # só A dispara (k=0.5,h=5 -> S+_a cruza; zscore_b=0 nunca cruza)
    com_zscore = _com_zscore_sintetico(linhas)

    resultado = aplicar_cusum_por_janela(com_zscore, k=0.5, h=5.0)

    assert len(resultado) == 1
    linha = resultado.iloc[0]
    assert linha["disparo_a"] == True
    assert linha["disparo_b"] == False
    assert linha["disparo_m1"] == True


def test_aplicar_cusum_janelas_nao_se_misturam() -> None:
    linhas = [
        {"classe": "positiva", "window_id": 0, "timestep": t, "zscore_a": 3.0, "zscore_b": 0.0, "timestep_estruturalmente_impossivel": False}
        for t in range(4)
    ] + [
        {"classe": "negativa", "window_id": 0, "timestep": t, "zscore_a": 0.0, "zscore_b": 0.0, "timestep_estruturalmente_impossivel": False}
        for t in range(4)
    ]
    com_zscore = _com_zscore_sintetico(linhas)

    resultado = aplicar_cusum_por_janela(com_zscore, k=0.5, h=5.0)

    assert len(resultado) == 2
    positiva = resultado[resultado["classe"] == "positiva"].iloc[0]
    negativa = resultado[resultado["classe"] == "negativa"].iloc[0]
    assert positiva["disparo_m1"] == True
    assert negativa["disparo_m1"] == False


def test_aplicar_cusum_nao_mistura_janelas_de_combinacoes_diferentes_com_mesmo_window_id() -> None:
    """Regressão de um bug real encontrado ao rodar sobre o v3 completo:
    duas janelas com MESMO (classe, window_id) mas COMBINAÇÕES diferentes
    (delta_t distinto, simulando dois arquivos/seeds concatenados) devem
    ser tratadas como janelas SEPARADAS -- window_id não é global, é
    reiniciado em 0 por classe em CADA arquivo (mesmo princípio já
    validado em binning.bin_janelas). Sem a chave completa
    (_CHAVE_JANELA_CUSUM), o CUSUM misturaria os z-scores das duas na
    mesma sequência de acumulador, produzindo um resultado incorreto."""
    linhas = (
        [
            {"classe": "positiva", "window_id": 0, "timestep": t, "zscore_a": 0.0, "zscore_b": 0.0, "timestep_estruturalmente_impossivel": False, "delta_t": 2.0}
            for t in range(4)
        ]
        + [
            {"classe": "positiva", "window_id": 0, "timestep": t, "zscore_a": 3.0, "zscore_b": 0.0, "timestep_estruturalmente_impossivel": False, "delta_t": 24.0}
            for t in range(4)
        ]
    )
    com_zscore = _com_zscore_sintetico(linhas)

    resultado = aplicar_cusum_por_janela(com_zscore, k=0.5, h=5.0)

    assert len(resultado) == 2, "duas combinações diferentes com mesmo window_id devem virar DUAS linhas, não uma"
    linha_dt2 = resultado[resultado["delta_t"] == 2.0].iloc[0]
    linha_dt24 = resultado[resultado["delta_t"] == 24.0].iloc[0]
    assert linha_dt2["disparo_m1"] == False  # zscore_a=0.0 -- nunca dispara
    assert linha_dt24["disparo_m1"] == True   # zscore_a=3.0 -- dispara (mesma lógica do teste OR acima)


def test_aplicar_cusum_propaga_timestep_disparo() -> None:
    linhas = [
        {"classe": "positiva", "window_id": 0, "timestep": t, "zscore_a": 3.0, "zscore_b": 0.0, "timestep_estruturalmente_impossivel": False}
        for t in range(4)
    ]
    com_zscore = _com_zscore_sintetico(linhas)

    resultado = aplicar_cusum_por_janela(com_zscore, k=0.5, h=5.0)

    linha = resultado.iloc[0]
    assert linha["timestep_disparo_a"] == 1  # t=0: 2.5, t=1: 5.0 -> dispara
    assert pd.isna(linha["timestep_disparo_b"])


def test_aplicar_cusum_sem_colunas_obrigatorias_levanta_erro() -> None:
    with pytest.raises(ValueError, match="colunas obrigatórias"):
        aplicar_cusum_por_janela(pd.DataFrame({"classe": ["positiva"]}), k=0.5, h=5.0)


def test_aplicar_cusum_vazio_levanta_erro() -> None:
    """DataFrame com TODAS as colunas obrigatórias (incluindo a chave de
    combinação, ver _CHAVE_JANELA_CUSUM) mas 0 linhas -- caso genuíno de
    "vazio" (distinto de "faltando colunas", já coberto pelo teste
    anterior)."""
    colunas = [
        "classe", "window_id", "timestep", "zscore_a", "zscore_b", "timestep_estruturalmente_impossivel",
        "delta_t", "recompensa", "pi", "rho", "beta", "g", "seed",
    ]
    vazio = pd.DataFrame(columns=colunas)
    with pytest.raises(ValueError, match="vazio"):
        aplicar_cusum_por_janela(vazio, k=0.5, h=5.0)


def test_aplicar_cusum_timestep_duplicado_na_chave_completa_levanta_erro() -> None:
    """Guarda análoga à já existente em bin_janelas, mas para um caso
    diferente: aqui a chave COMPLETA (classe, window_id, delta_t,
    recompensa, pi, rho, beta, g, seed, timestep) está duplicada -- duas
    linhas para o MESMO timestep da MESMA janela, não ambiguidade entre
    combinações diferentes. Isso indica dado corrompido/mal formado na
    entrada e corromperia silenciosamente a sequência do acumulador CUSUM
    (dois valores de zscore_a para o mesmo t) se não fosse rejeitado."""
    linhas = [
        {"classe": "positiva", "window_id": 0, "timestep": 0, "zscore_a": 3.0, "zscore_b": 0.0, "timestep_estruturalmente_impossivel": False},
        {"classe": "positiva", "window_id": 0, "timestep": 0, "zscore_a": 999.0, "zscore_b": 0.0, "timestep_estruturalmente_impossivel": False},  # timestep=0 duplicado
    ]
    com_zscore = _com_zscore_sintetico(linhas)

    with pytest.raises(ValueError, match="duplicado"):
        aplicar_cusum_por_janela(com_zscore, k=0.5, h=5.0)


# ---------------------------------------------------------------------------
# calcular_metricas_m1 -- confusão 2x2 calculada à mão, estratificação
# ---------------------------------------------------------------------------


def _previsoes_sinteticas(linhas: list[dict]) -> pd.DataFrame:
    base = {"recompensa": 1.0, "pi": 0.0, "delta_t": 2.0, "rho": 0.0, "beta": 1, "g": "pool"}
    return pd.DataFrame([{**base, **linha} for linha in linhas])


def test_metricas_m1_confusao_2x2_calculada_a_mao() -> None:
    """1 estrato (delta_t/recompensa/pi fixos): 2 positiva_ativa (1 disparo,
    1 não) + 2 negativa (1 disparo=FP, 1 não) -> vp=1, fn=1, fp=1, vn=1 ->
    precisao=0.5, recall=0.5, f1=0.5, fpr=0.5, na linha subgrupo_positivo=
    "positiva_ativa". positiva_inativa não está presente nesta fixture --
    deve aparecer mesmo assim (vp=0, fn=0, n_positivas=0)."""
    linhas = [
        {"grupo": "positiva_ativa", "disparo_m1": True},   # vp
        {"grupo": "positiva_ativa", "disparo_m1": False},  # fn
        {"grupo": "negativa", "disparo_m1": True},          # fp
        {"grupo": "negativa", "disparo_m1": False},         # vn
    ]
    previsoes = _previsoes_sinteticas(linhas)

    metricas = calcular_metricas_m1(previsoes)

    assert len(metricas) == 2  # 1 estrato de combinação x 2 subgrupos positivos
    assert set(metricas["subgrupo_positivo"]) == {"positiva_ativa", "positiva_inativa"}

    linha_ativa = metricas[metricas["subgrupo_positivo"] == "positiva_ativa"].iloc[0]
    assert linha_ativa["vp"] == 1
    assert linha_ativa["fp"] == 1
    assert linha_ativa["vn"] == 1
    assert linha_ativa["fn"] == 1
    assert linha_ativa["precisao"] == pytest.approx(0.5)
    assert linha_ativa["recall"] == pytest.approx(0.5)
    assert linha_ativa["f1"] == pytest.approx(0.5)
    assert linha_ativa["fpr"] == pytest.approx(0.5)

    linha_inativa = metricas[metricas["subgrupo_positivo"] == "positiva_inativa"].iloc[0]
    assert linha_inativa["n_positivas"] == 0
    assert linha_inativa["vp"] == 0
    assert linha_inativa["fn"] == 0
    assert linha_inativa["fpr"] == pytest.approx(0.5)  # mesmo fp/vn das negativas, reaproveitado


def test_metricas_m1_positiva_ativa_e_inativa_comparam_com_as_mesmas_negativas() -> None:
    """fpr deve ser IGUAL nas duas linhas do mesmo estrato de combinação --
    as negativas usadas são as mesmas (mesmo delta_t/recompensa/pi), só o
    subgrupo positivo pareado muda."""
    linhas = [
        {"grupo": "positiva_ativa", "disparo_m1": True},
        {"grupo": "positiva_inativa", "disparo_m1": False},
        {"grupo": "negativa", "disparo_m1": True},
        {"grupo": "negativa", "disparo_m1": False},
        {"grupo": "negativa", "disparo_m1": False},
    ]
    previsoes = _previsoes_sinteticas(linhas)

    metricas = calcular_metricas_m1(previsoes)

    fprs = metricas["fpr"].unique()
    assert len(fprs) == 1  # mesmo fpr nas duas linhas
    assert fprs[0] == pytest.approx(1 / 3)
    assert (metricas["n_negativas"] == 3).all()


def test_metricas_m1_estrato_sem_negativas_e_omitido() -> None:
    """(delta_t, recompensa, pi) sem NENHUMA janela negativa não pode ter
    fp/vn/fpr calculados -- omitido do resultado, sem levantar erro."""
    linhas = [{"grupo": "positiva_ativa", "disparo_m1": True}]
    previsoes = _previsoes_sinteticas(linhas)

    metricas = calcular_metricas_m1(previsoes)

    assert len(metricas) == 0


def test_metricas_m1_estratifica_por_delta_t_sem_agregar() -> None:
    """Duas combinações diferentes de delta_t -> estratos SEPARADOS na
    saída, não agregados juntos."""
    linhas = [
        {"grupo": "negativa", "disparo_m1": False, "delta_t": 2.0},
        {"grupo": "positiva_ativa", "disparo_m1": True, "delta_t": 2.0},
        {"grupo": "negativa", "disparo_m1": True, "delta_t": 24.0},
        {"grupo": "positiva_ativa", "disparo_m1": True, "delta_t": 24.0},
    ]
    previsoes = _previsoes_sinteticas(linhas)

    metricas = calcular_metricas_m1(previsoes)

    assert set(metricas["delta_t"]) == {2.0, 24.0}
    assert len(metricas) == 4  # 2 delta_t x 2 subgrupos positivos


def test_metricas_m1_pi_diferente_produz_estratos_separados() -> None:
    """Requisito explícito: duas combinações idênticas em delta_t/recompensa
    mas com pi diferente NÃO devem ser agregadas juntas -- pi participa do
    agrupamento de calcular_metricas_m1, não só delta_t x recompensa."""
    linhas = [
        {"grupo": "negativa", "disparo_m1": False, "pi": 0.0},
        {"grupo": "positiva_ativa", "disparo_m1": True, "pi": 0.0},
        {"grupo": "negativa", "disparo_m1": True, "pi": 0.9},
        {"grupo": "positiva_ativa", "disparo_m1": False, "pi": 0.9},
    ]
    previsoes = _previsoes_sinteticas(linhas)

    metricas = calcular_metricas_m1(previsoes)

    assert set(metricas["pi"]) == {0.0, 0.9}
    linha_pi_0 = metricas[(metricas["pi"] == 0.0) & (metricas["subgrupo_positivo"] == "positiva_ativa")].iloc[0]
    linha_pi_09 = metricas[(metricas["pi"] == 0.9) & (metricas["subgrupo_positivo"] == "positiva_ativa")].iloc[0]
    assert linha_pi_0["vn"] == 1 and linha_pi_0["fp"] == 0
    assert linha_pi_09["fp"] == 1 and linha_pi_09["vn"] == 0


def test_metricas_m1_denominador_zero_produz_nan_nao_zero() -> None:
    """Estrato sem nenhum disparo -> precisao NaN (não 0.0), pois
    vp+fp=0 -- "sem dado" != "desempenho zero"."""
    linhas = [
        {"grupo": "positiva_ativa", "disparo_m1": False},
        {"grupo": "negativa", "disparo_m1": False},
    ]
    previsoes = _previsoes_sinteticas(linhas)

    metricas = calcular_metricas_m1(previsoes)

    linha_ativa = metricas[metricas["subgrupo_positivo"] == "positiva_ativa"].iloc[0]
    assert pd.isna(linha_ativa["precisao"])
    assert linha_ativa["recall"] == pytest.approx(0.0)  # vp=0, fn=1 -> recall=0/1=0.0, definido


def test_metricas_m1_sem_colunas_obrigatorias_levanta_erro() -> None:
    with pytest.raises(ValueError, match="colunas obrigatórias"):
        calcular_metricas_m1(pd.DataFrame({"grupo": ["negativa"]}))


# ---------------------------------------------------------------------------
# fracao_disparos_no_piso_da_referencia
# ---------------------------------------------------------------------------


def _referencia_sintetica(linhas: list[dict]) -> pd.DataFrame:
    base = {"recompensa": 1.0, "pi": 0.0, "delta_t": 2.0, "rho": 0.0, "beta": 1, "g": "pool"}
    return pd.DataFrame([{**base, **linha} for linha in linhas])


def test_fracao_no_piso_identifica_disparo_com_desvio_no_piso() -> None:
    from src.pipeline.binning import _EPSILON_DESVIO_PADRAO_ZERO

    previsoes = _previsoes_sinteticas(
        [{"grupo": "positiva_ativa", "disparo_m1": True, "disparo_a": True, "disparo_b": False, "timestep_disparo_a": 2, "timestep_disparo_b": None}]
    )
    referencia = _referencia_sintetica(
        [{"timestep": 2, "desvio_a": _EPSILON_DESVIO_PADRAO_ZERO, "desvio_b": 1.0}]
    )

    resultado = fracao_disparos_no_piso_da_referencia(previsoes, referencia)

    linha_a = resultado[resultado["fonte"] == "a"].iloc[0]
    assert linha_a["n_disparos"] == 1
    assert linha_a["n_disparos_no_piso"] == 1
    assert linha_a["fracao_no_piso"] == pytest.approx(1.0)


def test_fracao_no_piso_identifica_disparo_com_desvio_real() -> None:
    previsoes = _previsoes_sinteticas(
        [{"grupo": "positiva_ativa", "disparo_m1": True, "disparo_a": True, "disparo_b": False, "timestep_disparo_a": 2, "timestep_disparo_b": None}]
    )
    referencia = _referencia_sintetica([{"timestep": 2, "desvio_a": 2.5, "desvio_b": 1.0}])

    resultado = fracao_disparos_no_piso_da_referencia(previsoes, referencia)

    linha_a = resultado[resultado["fonte"] == "a"].iloc[0]
    assert linha_a["n_disparos_no_piso"] == 0
    assert linha_a["fracao_no_piso"] == pytest.approx(0.0)


def test_fracao_no_piso_sem_nenhum_disparo_retorna_nan_nao_zero() -> None:
    previsoes = _previsoes_sinteticas(
        [{"grupo": "negativa", "disparo_m1": False, "disparo_a": False, "disparo_b": False, "timestep_disparo_a": None, "timestep_disparo_b": None}]
    )
    referencia = _referencia_sintetica([{"timestep": 0, "desvio_a": 1.0, "desvio_b": 1.0}])

    resultado = fracao_disparos_no_piso_da_referencia(previsoes, referencia)

    assert (resultado["n_disparos"] == 0).all()
    assert resultado["fracao_no_piso"].isna().all()


# ---------------------------------------------------------------------------
# Integração leve com um arquivo real do v3, ponta a ponta
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _V3_DISPONIVEL, reason="output/dataset_producao_v3/ não encontrado neste ambiente")
def test_integracao_ponta_a_ponta_arquivo_real_do_v3() -> None:
    """carregar_arquivo -> bin_janelas -> referência -> calcular_zscore ->
    aplicar_cusum_por_janela -> calcular_metricas_m1, sem erro, schema correto."""
    caminho = _DIR_V3 / "beta-1_delta_t-2.0000_g-pool_pi-0.0000_recompensa-0.5000_rho-0.0000_seed-1.h5"
    assert caminho.exists()

    carregado = carregar_arquivo(caminho)
    binado = bin_janelas(carregado["fonte_a"], carregado["fonte_b"], carregado["metadados_janela"])
    referencia = calcular_referencia_negativa_treino(binado)
    com_zscore = calcular_zscore(binado, referencia)

    previsoes = aplicar_cusum_por_janela(com_zscore, k=0.5, h=5.0)
    assert {"disparo_a", "disparo_b", "disparo_m1", "timestep_disparo_a", "timestep_disparo_b", "grupo", "pi", "delta_t", "recompensa"} <= set(
        previsoes.columns
    )
    assert previsoes["window_id"].notna().all()

    metricas = calcular_metricas_m1(previsoes)
    assert {"subgrupo_positivo", "delta_t", "recompensa", "pi", "f1", "precisao", "recall", "fpr"} <= set(metricas.columns)
    assert set(metricas["subgrupo_positivo"]) <= {"positiva_ativa", "positiva_inativa"}

    piso = fracao_disparos_no_piso_da_referencia(previsoes, referencia)
    assert set(piso["fonte"]) == {"a", "b"}
