"""Testes de src/pipeline/binning.py (bin_janelas,
calcular_referencia_negativa_treino, calcular_zscore)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.generator.adversarial_mode import CenarioAdversarial
from src.generator.normal_mode import CenarioNormal
from src.pipeline.binning import (
    _EPSILON_DESVIO_PADRAO_ZERO,
    _janela_teto,
    bin_janelas,
    calcular_referencia_negativa_treino,
    calcular_zscore,
)
from src.pipeline.config import JANELA_TRAFEGO_FUNDO_QUANDO_DELTA_T_ZERO
from src.pipeline.loader import carregar_arquivo
from src.pipeline.storage import escrever_run_hdf5

_DIR_V3 = Path("output/dataset_producao_v3")
_V3_DISPONIVEL = _DIR_V3.exists() and any(_DIR_V3.glob("*.h5"))


# ---------------------------------------------------------------------------
# _janela_teto
# ---------------------------------------------------------------------------


def test_janela_teto_delta_t_zero_usa_constante() -> None:
    assert _janela_teto(0.0) == JANELA_TRAFEGO_FUNDO_QUANDO_DELTA_T_ZERO == 24.0


@pytest.mark.parametrize("delta_t", [2.0, 24.0, 10.0])
def test_janela_teto_delta_t_nao_zero_e_identidade(delta_t: float) -> None:
    assert _janela_teto(delta_t) == delta_t


# ---------------------------------------------------------------------------
# bin_janelas -- fixture sintética simples, construída à mão
# ---------------------------------------------------------------------------


def _metadados_sinteticos(linhas: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(linhas)


def test_bin_janelas_reindexa_fonte_a_preenchendo_zeros() -> None:
    """Fonte A já vem binada (timestep/n_eventos) -- reindexar para
    preencher timesteps sem linha com n_eventos=0, sem buraco."""
    fonte_a = pd.DataFrame(
        {"classe": ["negativa", "negativa"], "window_id": [0, 0], "timestep": [0, 3], "n_eventos": [5, 2]}
    )
    fonte_b = pd.DataFrame({"classe": pd.Series(dtype=object), "window_id": pd.Series(dtype=int), "timestamp": pd.Series(dtype=float)})
    metadados = _metadados_sinteticos([{"classe": "negativa", "window_id": 0, "delta_t": 4.0, "split": "train"}])

    binado = bin_janelas(fonte_a, fonte_b, metadados)

    assert len(binado) == 4  # ceil(4.0) = 4 timesteps: 0,1,2,3
    contagem = binado.set_index("timestep")["contagem_a"]
    assert contagem[0] == 5
    assert contagem[1] == 0
    assert contagem[2] == 0
    assert contagem[3] == 2


def test_bin_janelas_bina_fonte_b_corretamente() -> None:
    """Fonte B (timestamps brutos) deve ser binada via contagem_por_timestep
    -- 3 eventos em [0,1), 1 evento em [2,3)."""
    fonte_a = pd.DataFrame({"classe": pd.Series(dtype=object), "window_id": pd.Series(dtype=int), "timestep": pd.Series(dtype=int), "n_eventos": pd.Series(dtype=int)})
    fonte_b = pd.DataFrame(
        {
            "classe": ["negativa"] * 4,
            "window_id": [0] * 4,
            "timestamp": [0.1, 0.5, 0.9, 2.3],
        }
    )
    metadados = _metadados_sinteticos([{"classe": "negativa", "window_id": 0, "delta_t": 4.0, "split": "train"}])

    binado = bin_janelas(fonte_a, fonte_b, metadados)

    contagem = binado.set_index("timestep")["contagem_b"]
    assert contagem[0] == 3
    assert contagem[1] == 0
    assert contagem[2] == 1
    assert contagem[3] == 0


def test_bin_janelas_alinhamento_de_indice_entre_a_e_b() -> None:
    """contagem_a e contagem_b devem compartilhar o MESMO índice de
    timestep (mesmo n_timesteps), mesmo vindo de fontes com formatos
    de entrada diferentes (A já binada, B bruta)."""
    fonte_a = pd.DataFrame({"classe": ["negativa"], "window_id": [0], "timestep": [1], "n_eventos": [2]})
    fonte_b = pd.DataFrame({"classe": ["negativa"], "window_id": [0], "timestamp": [2.5]})
    metadados = _metadados_sinteticos([{"classe": "negativa", "window_id": 0, "delta_t": 4.0, "split": "train"}])

    binado = bin_janelas(fonte_a, fonte_b, metadados)

    assert list(binado["timestep"]) == [0, 1, 2, 3]
    assert list(binado["contagem_a"]) == [0, 2, 0, 0]
    assert list(binado["contagem_b"]) == [0, 0, 1, 0]


def test_bin_janelas_janela_totalmente_vazia_produz_serie_zero_nao_ausencia() -> None:
    """Requisito explícito: janela sem NENHUMA linha em fonte_a/fonte_b
    (ex. positiva_inativa) deve produzir série toda zero, não ficar
    ausente do resultado."""
    fonte_a_vazia = pd.DataFrame({"classe": pd.Series(dtype=object), "window_id": pd.Series(dtype=int), "timestep": pd.Series(dtype=int), "n_eventos": pd.Series(dtype=int)})
    fonte_b_vazia = pd.DataFrame({"classe": pd.Series(dtype=object), "window_id": pd.Series(dtype=int), "timestamp": pd.Series(dtype=float)})
    metadados = _metadados_sinteticos([{"classe": "positiva", "window_id": 5, "delta_t": 4.0, "split": "test", "grupo": "positiva_inativa"}])

    binado = bin_janelas(fonte_a_vazia, fonte_b_vazia, metadados)

    assert len(binado) == 4  # janela presente, não ausente
    assert (binado["contagem_a"] == 0).all()
    assert (binado["contagem_b"] == 0).all()
    assert (binado["window_id"] == 5).all()
    assert (binado["grupo"] == "positiva_inativa").all()


def test_bin_janelas_marca_timesteps_estruturalmente_impossiveis() -> None:
    """Positiva com delta_t=0.0: timestep 0 é possível (único onde eventos
    podem cair), timesteps 1..23 são estruturalmente impossíveis (teto
    comum = 24, mas o mecanismo do gerador nunca coloca evento ali para a
    classe positiva quando delta_t=0.0)."""
    fonte_a_vazia = pd.DataFrame({"classe": pd.Series(dtype=object), "window_id": pd.Series(dtype=int), "timestep": pd.Series(dtype=int), "n_eventos": pd.Series(dtype=int)})
    fonte_b_vazia = pd.DataFrame({"classe": pd.Series(dtype=object), "window_id": pd.Series(dtype=int), "timestamp": pd.Series(dtype=float)})
    metadados = _metadados_sinteticos([{"classe": "positiva", "window_id": 0, "delta_t": 0.0, "split": "train"}])

    binado = bin_janelas(fonte_a_vazia, fonte_b_vazia, metadados)

    assert len(binado) == 24  # teto comum = JANELA_TRAFEGO_FUNDO_QUANDO_DELTA_T_ZERO = 24
    impossiveis = binado.set_index("timestep")["timestep_estruturalmente_impossivel"]
    assert impossiveis[0] == False
    assert impossiveis[1:].all()


def test_bin_janelas_negativa_delta_t_zero_nao_e_marcada_impossivel() -> None:
    """A classe NEGATIVA em delta_t=0.0 tem janela real de 24.0 (pós-fix)
    -- nenhum timestep dela é estruturalmente impossível."""
    fonte_a_vazia = pd.DataFrame({"classe": pd.Series(dtype=object), "window_id": pd.Series(dtype=int), "timestep": pd.Series(dtype=int), "n_eventos": pd.Series(dtype=int)})
    fonte_b_vazia = pd.DataFrame({"classe": pd.Series(dtype=object), "window_id": pd.Series(dtype=int), "timestamp": pd.Series(dtype=float)})
    metadados = _metadados_sinteticos([{"classe": "negativa", "window_id": 0, "delta_t": 0.0, "split": "train"}])

    binado = bin_janelas(fonte_a_vazia, fonte_b_vazia, metadados)

    assert len(binado) == 24
    assert not binado["timestep_estruturalmente_impossivel"].any()


def test_bin_janelas_multiplas_janelas_nao_se_misturam() -> None:
    fonte_a = pd.DataFrame(
        {"classe": ["negativa", "positiva"], "window_id": [0, 0], "timestep": [0, 0], "n_eventos": [3, 9]}
    )
    fonte_b = pd.DataFrame({"classe": pd.Series(dtype=object), "window_id": pd.Series(dtype=int), "timestamp": pd.Series(dtype=float)})
    metadados = _metadados_sinteticos(
        [
            {"classe": "negativa", "window_id": 0, "delta_t": 2.0, "split": "train"},
            {"classe": "positiva", "window_id": 0, "delta_t": 2.0, "split": "train"},
        ]
    )

    binado = bin_janelas(fonte_a, fonte_b, metadados)

    contagem_neg = binado[(binado["classe"] == "negativa") & (binado["timestep"] == 0)]["contagem_a"].iloc[0]
    contagem_pos = binado[(binado["classe"] == "positiva") & (binado["timestep"] == 0)]["contagem_a"].iloc[0]
    assert contagem_neg == 3
    assert contagem_pos == 9


# ---------------------------------------------------------------------------
# bin_janelas com dados reais do v3 (loader real, não sintético)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _V3_DISPONIVEL, reason="output/dataset_producao_v3/ não encontrado neste ambiente")
def test_bin_janelas_com_dataset_real_delta_t_zero() -> None:
    """Confirma contra um arquivo REAL do v3 (delta_t=0.0) que a negativa
    passa a ter contagem não-trivial (pós-fix) e a positiva concentra tudo
    em timestep=0, com o resto marcado impossível."""
    caminho = _DIR_V3 / "beta-1_delta_t-0.0000_g-pool_pi-0.0000_recompensa-1.0000_rho-0.0000_seed-1.h5"
    assert caminho.exists()

    carregado = carregar_arquivo(caminho)
    binado = bin_janelas(carregado["fonte_a"], carregado["fonte_b"], carregado["metadados_janela"])

    assert set(binado["timestep"].unique()) == set(range(24))

    negativa = binado[binado["classe"] == "negativa"]
    assert negativa["contagem_a"].sum() > 0, "negativa em delta_t=0.0 deveria ter eventos pós-fix"
    assert not negativa["timestep_estruturalmente_impossivel"].any()

    positiva = binado[binado["classe"] == "positiva"]
    positiva_timestep_0 = positiva[positiva["timestep"] == 0]
    positiva_resto = positiva[positiva["timestep"] >= 1]
    assert positiva_resto["contagem_a"].sum() == 0
    assert positiva_resto["timestep_estruturalmente_impossivel"].all()
    assert not positiva_timestep_0["timestep_estruturalmente_impossivel"].any()


# ---------------------------------------------------------------------------
# calcular_referencia_negativa_treino
# ---------------------------------------------------------------------------


def _binado_sintetico(linhas: list[dict]) -> pd.DataFrame:
    base = {"recompensa": 1.0, "pi": 0.0, "delta_t": 2.0, "rho": 0.0, "beta": 1, "g": "pool"}
    return pd.DataFrame([{**base, **linha} for linha in linhas])


def test_referencia_calculada_a_mao_combinacao_pequena() -> None:
    """3 janelas negativas de treino, mesma combinação, timestep=0:
    contagem_a = [2, 4, 6] -> media=4.0, std amostral (ddof=1) = 2.0."""
    binado = _binado_sintetico(
        [
            {"classe": "negativa", "split": "train", "window_id": 0, "timestep": 0, "contagem_a": 2, "contagem_b": 1},
            {"classe": "negativa", "split": "train", "window_id": 1, "timestep": 0, "contagem_a": 4, "contagem_b": 1},
            {"classe": "negativa", "split": "train", "window_id": 2, "timestep": 0, "contagem_a": 6, "contagem_b": 1},
        ]
    )

    referencia = calcular_referencia_negativa_treino(binado)

    assert len(referencia) == 1
    linha = referencia.iloc[0]
    assert linha["media_a"] == pytest.approx(4.0)
    assert linha["desvio_a"] == pytest.approx(2.0)
    assert linha["media_b"] == pytest.approx(1.0)
    assert linha["n_janelas"] == 3


def test_referencia_ignora_positiva_e_split_nao_train() -> None:
    binado = _binado_sintetico(
        [
            {"classe": "negativa", "split": "train", "window_id": 0, "timestep": 0, "contagem_a": 10, "contagem_b": 0},
            {"classe": "negativa", "split": "test", "window_id": 1, "timestep": 0, "contagem_a": 999, "contagem_b": 0},
            {"classe": "positiva", "split": "train", "window_id": 0, "timestep": 0, "contagem_a": 999, "contagem_b": 0},
        ]
    )

    referencia = calcular_referencia_negativa_treino(binado)

    assert len(referencia) == 1
    assert referencia.iloc[0]["media_a"] == pytest.approx(10.0)
    assert referencia.iloc[0]["n_janelas"] == 1


def test_referencia_levanta_erro_sem_nenhuma_janela_negativa_treino() -> None:
    binado = _binado_sintetico(
        [{"classe": "positiva", "split": "train", "window_id": 0, "timestep": 0, "contagem_a": 1, "contagem_b": 0}]
    )

    with pytest.raises(ValueError, match="Nenhuma janela negativa"):
        calcular_referencia_negativa_treino(binado)


def test_referencia_desvio_zero_nao_produz_nan_nem_inf() -> None:
    """Timestep onde contagem_a é constante (0) em todas as janelas de
    treino -- std amostral = 0.0 -- deve ser pisado em EPSILON, não deixado
    como 0.0 (que causaria divisão por zero depois em calcular_zscore)."""
    binado = _binado_sintetico(
        [
            {"classe": "negativa", "split": "train", "window_id": 0, "timestep": 0, "contagem_a": 0, "contagem_b": 0},
            {"classe": "negativa", "split": "train", "window_id": 1, "timestep": 0, "contagem_a": 0, "contagem_b": 0},
            {"classe": "negativa", "split": "train", "window_id": 2, "timestep": 0, "contagem_a": 0, "contagem_b": 0},
        ]
    )

    referencia = calcular_referencia_negativa_treino(binado)

    assert not referencia[["media_a", "desvio_a", "media_b", "desvio_b"]].isna().any().any()
    assert np.isfinite(referencia[["media_a", "desvio_a", "media_b", "desvio_b"]].to_numpy()).all()
    assert referencia.iloc[0]["desvio_a"] == pytest.approx(_EPSILON_DESVIO_PADRAO_ZERO)


def test_referencia_uma_unica_janela_std_indefinido_tratado() -> None:
    """n=1 janela contribuindo -> pandas std (ddof=1) retorna NaN por
    definição (denominador n-1=0) -- deve ser tratado como o caso
    'sem variância estimável', mesmo destino do std==0.0 explícito."""
    binado = _binado_sintetico(
        [{"classe": "negativa", "split": "train", "window_id": 0, "timestep": 0, "contagem_a": 5, "contagem_b": 2}]
    )

    referencia = calcular_referencia_negativa_treino(binado)

    assert referencia.iloc[0]["desvio_a"] == pytest.approx(_EPSILON_DESVIO_PADRAO_ZERO)
    assert referencia.iloc[0]["desvio_b"] == pytest.approx(_EPSILON_DESVIO_PADRAO_ZERO)
    assert referencia.iloc[0]["n_janelas"] == 1


def test_referencia_agrupamento_nao_mistura_combinacoes_diferentes() -> None:
    """recompensa/delta_t diferentes -> linhas de referência SEPARADAS,
    não agregadas juntas."""
    base_a = {"classe": "negativa", "split": "train", "timestep": 0, "contagem_b": 0}
    binado = pd.concat(
        [
            _binado_sintetico([{**base_a, "window_id": 0, "contagem_a": 10, "recompensa": 0.5}]),
            _binado_sintetico([{**base_a, "window_id": 0, "contagem_a": 90, "recompensa": 1.5}]),
            _binado_sintetico([{**base_a, "window_id": 0, "contagem_a": 50, "delta_t": 24.0}]),
        ],
        ignore_index=True,
    )

    referencia = calcular_referencia_negativa_treino(binado)

    assert len(referencia) == 3  # 3 combinações distintas, nenhuma agregada com outra
    medias = sorted(referencia["media_a"].tolist())
    assert medias == [10.0, 50.0, 90.0]


def test_referencia_poola_seeds_dentro_da_mesma_combinacao() -> None:
    """seed NÃO entra no agrupamento -- duas seeds da MESMA combinação
    devem ser agregadas juntas, aumentando n_janelas."""
    binado = _binado_sintetico(
        [
            {"classe": "negativa", "split": "train", "window_id": 0, "timestep": 0, "contagem_a": 2, "contagem_b": 0, "seed": 1},
            {"classe": "negativa", "split": "train", "window_id": 0, "timestep": 0, "contagem_a": 8, "contagem_b": 0, "seed": 2},
        ]
    )

    referencia = calcular_referencia_negativa_treino(binado)

    assert len(referencia) == 1  # uma única linha, seeds pooladas
    assert referencia.iloc[0]["n_janelas"] == 2
    assert referencia.iloc[0]["media_a"] == pytest.approx(5.0)


@pytest.mark.skipif(not _V3_DISPONIVEL, reason="output/dataset_producao_v3/ não encontrado neste ambiente")
def test_referencia_calculavel_para_combinacao_delta_t_zero_do_v3() -> None:
    """Requisito explícito: a referência deve ser calculável para
    delta_t=0.0 pós-fix (onde antes do fix a negativa não tinha nenhum
    evento -- agora tem)."""
    caminho = _DIR_V3 / "beta-1_delta_t-0.0000_g-pool_pi-0.0000_recompensa-0.5000_rho-0.0000_seed-1.h5"
    assert caminho.exists()

    carregado = carregar_arquivo(caminho)
    binado = bin_janelas(carregado["fonte_a"], carregado["fonte_b"], carregado["metadados_janela"])
    referencia = calcular_referencia_negativa_treino(binado)

    assert len(referencia) == 24  # 24 timesteps, 1 combinação (1 arquivo = 1 combinação)
    assert not referencia[["media_a", "desvio_a", "media_b", "desvio_b"]].isna().any().any()
    assert np.isfinite(referencia[["media_a", "desvio_a", "media_b", "desvio_b"]].to_numpy()).all()
    assert (referencia["desvio_a"] > 0).all()
    assert (referencia["desvio_b"] > 0).all()


# ---------------------------------------------------------------------------
# calcular_zscore
# ---------------------------------------------------------------------------


def test_zscore_calculo_basico() -> None:
    referencia = pd.DataFrame(
        [
            {
                "recompensa": 1.0, "pi": 0.0, "delta_t": 2.0, "rho": 0.0, "beta": 1, "g": "pool",
                "timestep": 0, "media_a": 4.0, "desvio_a": 2.0, "media_b": 1.0, "desvio_b": 0.5,
            }
        ]
    )
    binado = _binado_sintetico(
        [{"classe": "positiva", "split": "test", "window_id": 7, "timestep": 0, "contagem_a": 10, "contagem_b": 2}]
    )

    resultado = calcular_zscore(binado, referencia)

    assert resultado.iloc[0]["zscore_a"] == pytest.approx((10 - 4.0) / 2.0)
    assert resultado.iloc[0]["zscore_b"] == pytest.approx((2 - 1.0) / 0.5)


def test_zscore_com_desvio_padrao_zero_da_referencia_nao_quebra() -> None:
    """Referência com desvio_a no piso EPSILON (produzido por
    calcular_referencia_negativa_treino quando std==0.0) -- aplicar
    z-score não deve levantar exceção nem produzir NaN/inf."""
    referencia = pd.DataFrame(
        [
            {
                "recompensa": 1.0, "pi": 0.0, "delta_t": 2.0, "rho": 0.0, "beta": 1, "g": "pool",
                "timestep": 0, "media_a": 0.0, "desvio_a": _EPSILON_DESVIO_PADRAO_ZERO,
                "media_b": 0.0, "desvio_b": _EPSILON_DESVIO_PADRAO_ZERO,
            }
        ]
    )
    binado = _binado_sintetico(
        [{"classe": "positiva", "split": "test", "window_id": 0, "timestep": 0, "contagem_a": 3, "contagem_b": 0}]
    )

    resultado = calcular_zscore(binado, referencia)

    assert np.isfinite(resultado["zscore_a"].to_numpy()).all()
    assert np.isfinite(resultado["zscore_b"].to_numpy()).all()
    assert resultado.iloc[0]["zscore_a"] > 0  # contagem 3 >> media 0, desvio quase-zero -> z grande e positivo
    assert resultado.iloc[0]["zscore_b"] == pytest.approx(0.0)  # contagem==media==0 -> z=0, não NaN


def test_zscore_preserva_numero_de_linhas() -> None:
    referencia = pd.DataFrame(
        [
            {
                "recompensa": 1.0, "pi": 0.0, "delta_t": 2.0, "rho": 0.0, "beta": 1, "g": "pool",
                "timestep": t, "media_a": 1.0, "desvio_a": 1.0, "media_b": 1.0, "desvio_b": 1.0,
            }
            for t in range(2)
        ]
    )
    binado = _binado_sintetico(
        [
            {"classe": "negativa", "split": "test", "window_id": 0, "timestep": 0, "contagem_a": 1, "contagem_b": 1},
            {"classe": "negativa", "split": "test", "window_id": 0, "timestep": 1, "contagem_a": 2, "contagem_b": 2},
        ]
    )

    resultado = calcular_zscore(binado, referencia)

    assert len(resultado) == len(binado)


def test_zscore_aplica_em_qualquer_grupo_split() -> None:
    """z-score deve ser aplicável a qualquer grupo/split (não só
    negativa/train, que é usado só para CALCULAR a referência)."""
    referencia = pd.DataFrame(
        [
            {
                "recompensa": 1.0, "pi": 0.0, "delta_t": 2.0, "rho": 0.0, "beta": 1, "g": "pool",
                "timestep": 0, "media_a": 5.0, "desvio_a": 1.0, "media_b": 0.0, "desvio_b": 1.0,
            }
        ]
    )
    binado = _binado_sintetico(
        [
            {"classe": "positiva", "split": "val", "window_id": 0, "timestep": 0, "contagem_a": 20, "contagem_b": 0},
            {"classe": "negativa", "split": "test", "window_id": 3, "timestep": 0, "contagem_a": 5, "contagem_b": 0},
        ]
    )

    resultado = calcular_zscore(binado, referencia)

    assert resultado.iloc[0]["zscore_a"] == pytest.approx(15.0)  # positiva val, longe da referência
    assert resultado.iloc[1]["zscore_a"] == pytest.approx(0.0)  # negativa test, na média


# ---------------------------------------------------------------------------
# Teste de equivalência: implementação ANTIGA (iterrows, mantida só aqui,
# não no módulo final) vs. implementação NOVA (vetorizada) de bin_janelas,
# sobre arquivos reais do v3 -- confirma que a vetorização não mudou
# comportamento.
# ---------------------------------------------------------------------------


def _bin_uma_janela_fonte_b_antigo(timestamps: np.ndarray, n_timesteps: int) -> np.ndarray:
    """Cópia exata da função auxiliar da implementação pré-vetorização."""
    from src.generator.normal_mode.trafego import contagem_por_timestep

    return contagem_por_timestep(timestamps, janela=float(n_timesteps))


def _bin_janelas_implementacao_antiga(
    fonte_a: pd.DataFrame, fonte_b: pd.DataFrame, metadados_janela: pd.DataFrame
) -> pd.DataFrame:
    """Cópia EXATA de `bin_janelas` como existia antes da vetorização
    (iterrows + filtro repetido por linha) -- mantida só neste arquivo de
    teste, para o teste de equivalência. NÃO reflete o módulo atual;
    qualquer bug já presente na versão antiga é preservado aqui de
    propósito, para comparação byte-a-byte com a versão nova."""
    if "delta_t" not in metadados_janela.columns:
        raise ValueError("metadados_janela precisa da coluna 'delta_t' (metadado de cenário do loader).")

    partes = []
    for _, meta_janela in metadados_janela.iterrows():
        classe = meta_janela["classe"]
        window_id = meta_janela["window_id"]
        delta_t = float(meta_janela["delta_t"])

        n_timesteps = int(np.ceil(_janela_teto(delta_t)))
        timesteps = np.arange(n_timesteps)

        fa_janela = fonte_a[(fonte_a["classe"] == classe) & (fonte_a["window_id"] == window_id)]
        contagem_a = np.zeros(n_timesteps, dtype=int)
        if len(fa_janela) > 0:
            indices_validos = fa_janela["timestep"].to_numpy()
            valores = fa_janela["n_eventos"].to_numpy()
            dentro_do_range = indices_validos < n_timesteps
            contagem_a[indices_validos[dentro_do_range]] = valores[dentro_do_range]

        fb_janela = fonte_b[(fonte_b["classe"] == classe) & (fonte_b["window_id"] == window_id)]
        contagem_b = _bin_uma_janela_fonte_b_antigo(fb_janela["timestamp"].to_numpy(), n_timesteps)

        impossivel = (classe == "positiva") & (delta_t == 0.0) & (timesteps >= 1)

        bloco = pd.DataFrame(
            {
                "classe": classe,
                "window_id": window_id,
                "timestep": timesteps,
                "contagem_a": contagem_a,
                "contagem_b": contagem_b,
                "timestep_estruturalmente_impossivel": impossivel,
            }
        )
        for coluna in meta_janela.index:
            if coluna not in ("classe", "window_id"):
                bloco[coluna] = meta_janela[coluna]

        partes.append(bloco)

    if not partes:
        return pd.DataFrame(
            columns=["classe", "window_id", "timestep", "contagem_a", "contagem_b", "timestep_estruturalmente_impossivel"]
        )

    return pd.concat(partes, ignore_index=True)


def _normalizar_para_comparacao(df: pd.DataFrame) -> pd.DataFrame:
    """Ordena linhas e colunas de forma determinística, para que a
    comparação de equivalência não dependa de ordem (não é uma garantia
    de API de nenhuma das duas implementações)."""
    colunas_ordenadas = sorted(df.columns)
    df = df[colunas_ordenadas].sort_values(colunas_ordenadas).reset_index(drop=True)
    return df


@pytest.mark.skipif(not _V3_DISPONIVEL, reason="output/dataset_producao_v3/ não encontrado neste ambiente")
@pytest.mark.parametrize(
    "nome_arquivo",
    [
        "beta-1_delta_t-0.0000_g-pool_pi-0.0000_recompensa-0.5000_rho-0.0000_seed-1.h5",
        "beta-1_delta_t-24.0000_g-pool_pi-0.9000_recompensa-1.0000_rho-0.5000_seed-2.h5",
        "beta-1_delta_t-2.0000_g-pool_pi-0.0000_recompensa-0.5000_rho-0.0000_seed-1.h5",
    ],
)
def test_equivalencia_implementacao_antiga_vs_nova(nome_arquivo: str) -> None:
    """Compara a implementação antiga (iterrows) com a nova (vetorizada)
    sobre 3 arquivos reais completos do v3 -- um com delta_t=0.0 (caso do
    fix, onde a classe negativa tem eventos reais pós-fix e a positiva
    tem timesteps estruturalmente impossíveis), um com delta_t=24.0
    (caso "normal", sem a assimetria de delta_t=0), e um com delta_t=2.0
    (13% das janelas negativas sem nenhuma linha em fonte_a nesta
    combinação específica -- exercita bem o caminho "grupo sem evento é
    pulado" da vetorização de Fonte B, sem recorrer a um extremo
    artificial como pi=0.95, que mascararia quase tudo por um motivo não
    relacionado à lógica sendo testada). DataFrames devem ser idênticos
    após normalização de ordem."""
    caminho = _DIR_V3 / nome_arquivo
    assert caminho.exists(), f"arquivo esperado não encontrado no v3: {caminho}"

    carregado = carregar_arquivo(caminho)

    resultado_antigo = _bin_janelas_implementacao_antiga(
        carregado["fonte_a"], carregado["fonte_b"], carregado["metadados_janela"]
    )
    resultado_novo = bin_janelas(carregado["fonte_a"], carregado["fonte_b"], carregado["metadados_janela"])

    antigo_normalizado = _normalizar_para_comparacao(resultado_antigo)
    novo_normalizado = _normalizar_para_comparacao(resultado_novo)

    assert list(antigo_normalizado.columns) == list(novo_normalizado.columns)
    assert len(antigo_normalizado) == len(novo_normalizado)
    pd.testing.assert_frame_equal(antigo_normalizado, novo_normalizado, check_dtype=False)
