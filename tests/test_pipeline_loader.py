"""Testes de src/pipeline/loader.py (carregar_arquivo, carregar_diretorio,
extrair_parametros_cenario, _derivar_grupo)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.generator.adversarial_mode import CenarioAdversarial
from src.generator.normal_mode import CenarioNormal
from src.pipeline.loader import (
    COLUNAS_METADADO_CENARIO,
    COLUNAS_METADADO_JANELA,
    ParametrosCenario,
    _derivar_grupo,
    carregar_arquivo,
    carregar_diretorio,
    extrair_parametros_cenario,
)
from src.pipeline.storage import escrever_run_hdf5

_DIR_V3 = Path("output/dataset_producao_v3")
_V3_DISPONIVEL = _DIR_V3.exists() and any(_DIR_V3.glob("*.h5"))


def _fonte_a_df(rows: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["timestep", "n_eventos", "volume", "timestamp_medio", "dispersao_timestamp"])


def _nome_com_params(delta_t: float = 2.0, recompensa: float = 1.0, pi: float = 0.0, rho: float = 0.0, seed: int = 1) -> Path:
    """Nome de arquivo sintético seguindo o padrão real de produção --
    usado para fixtures locais (não arquivos reais do v3)."""
    return Path(
        f"beta-1_delta_t-{delta_t:.4f}_g-pool_pi-{pi:.4f}_"
        f"recompensa-{recompensa:.4f}_rho-{rho:.4f}_seed-{seed}.h5"
    )


@pytest.fixture
def dataset_fixture(tmp_path) -> Path:
    """Um HDF5 pequeno com 3 janelas: uma positiva ativa (com Fonte A/B),
    uma positiva inativa (Fonte A/B genuinamente vazias -- resolver_desembolso
    nunca roda, ver model.py) e uma negativa (com Fonte A/B). Nomeado
    seguindo o padrão de parâmetros de cenário no nome do arquivo."""
    cenario_positivo_ativo = CenarioAdversarial(
        fonte_a=_fonte_a_df([[1, 2, 20.0, 1.5, 0.3], [2, 1, 10.0, 2.1, 0.0]]),
        fonte_b=np.array([1.5, 2.5, 2.05]),
        resultado_por_secao=pd.Series([0.6, 0.7, 0.65, 0.62, 0.58]),
        resultado_por_municipio=pd.Series([0.63]),
        resultado_por_estado=pd.Series([0.63]),
        contrato_ativado=True,
    )
    cenario_positivo_inativo = CenarioAdversarial(
        fonte_a=_fonte_a_df([]),
        fonte_b=np.array([]),
        resultado_por_secao=pd.Series([0.1, 0.2, 0.15, 0.12, 0.18]),
        resultado_por_municipio=pd.Series([0.15]),
        resultado_por_estado=pd.Series([0.15]),
        contrato_ativado=False,
    )
    cenario_negativo = CenarioNormal(
        fonte_a=_fonte_a_df([[1, 1, 5.0, 1.0, 0.0]]),
        fonte_b=np.array([3.0]),
        resultado_por_secao=pd.Series([0.3, 0.31, 0.29, 0.28, 0.32]),
        resultado_por_municipio=pd.Series([0.3]),
        resultado_por_estado=pd.Series([0.3]),
    )

    seeds = [np.random.SeedSequence(i) for i in range(6)]
    janelas_positivas = [
        (0, cenario_positivo_ativo, seeds[0], seeds[1]),
        (1, cenario_positivo_inativo, seeds[2], seeds[3]),
    ]
    janelas_negativas = [(0, cenario_negativo, seeds[4], seeds[5])]

    caminho = tmp_path / _nome_com_params()
    escrever_run_hdf5(caminho, janelas_positivas, janelas_negativas)
    return caminho


# ---------------------------------------------------------------------------
# extrair_parametros_cenario -- Requisito 4
# ---------------------------------------------------------------------------


def test_extrair_parametros_cenario_nome_sintetico() -> None:
    caminho = _nome_com_params(delta_t=24.0, recompensa=1.5, pi=0.9, rho=0.5, seed=7)
    params = extrair_parametros_cenario(caminho)

    assert params == ParametrosCenario(beta=1, delta_t=24.0, g="pool", pi=0.9, recompensa=1.5, rho=0.5, seed=7)


def test_extrair_parametros_cenario_levanta_erro_para_nome_invalido() -> None:
    with pytest.raises(ValueError, match="fora do padrão esperado"):
        extrair_parametros_cenario(Path("arquivo_qualquer.h5"))


@pytest.mark.skipif(not _V3_DISPONIVEL, reason="output/dataset_producao_v3/ não encontrado neste ambiente")
@pytest.mark.parametrize(
    "nome_arquivo, esperado",
    [
        (
            "beta-1_delta_t-0.0000_g-pool_pi-0.0000_recompensa-0.5000_rho-0.0000_seed-1.h5",
            ParametrosCenario(beta=1, delta_t=0.0, g="pool", pi=0.0, recompensa=0.5, rho=0.0, seed=1),
        ),
        (
            "beta-1_delta_t-2.0000_g-pool_pi-0.9000_recompensa-1.0000_rho-0.5000_seed-2.h5",
            ParametrosCenario(beta=1, delta_t=2.0, g="pool", pi=0.9, recompensa=1.0, rho=0.5, seed=2),
        ),
        (
            "beta-1_delta_t-24.0000_g-pool_pi-0.9500_recompensa-1.5000_rho-1.0000_seed-1.h5",
            ParametrosCenario(beta=1, delta_t=24.0, g="pool", pi=0.95, recompensa=1.5, rho=1.0, seed=1),
        ),
    ],
)
def test_extrair_parametros_cenario_arquivos_reais_do_v3(nome_arquivo: str, esperado: ParametrosCenario) -> None:
    """Confirma o parse contra 3 nomes de arquivo REAIS do v3 (não
    inventados) -- os 3 níveis de delta_t do design fatorial, valores
    de pi/recompensa/rho/seed distintos entre si."""
    caminho = _DIR_V3 / nome_arquivo
    assert caminho.exists(), f"arquivo esperado não encontrado no v3: {caminho}"

    assert extrair_parametros_cenario(caminho) == esperado


# ---------------------------------------------------------------------------
# _derivar_grupo -- os três casos
# ---------------------------------------------------------------------------


def test_derivar_grupo_tres_casos() -> None:
    classe = pd.Series(["negativa", "positiva", "positiva", "positiva"])
    contrato_ativado = pd.Series([np.nan, 1.0, 0.0, 1.0])

    grupo = _derivar_grupo(classe, contrato_ativado)

    assert list(grupo) == ["negativa", "positiva_ativa", "positiva_inativa", "positiva_ativa"]


def test_derivar_grupo_preserva_indice() -> None:
    classe = pd.Series(["negativa", "positiva"], index=[10, 20])
    contrato_ativado = pd.Series([np.nan, 1.0], index=[10, 20])

    grupo = _derivar_grupo(classe, contrato_ativado)

    assert list(grupo.index) == [10, 20]


# ---------------------------------------------------------------------------
# carregar_arquivo -- join por (classe, window_id), Requisitos 1/2/3/4/5/7
# ---------------------------------------------------------------------------


def test_metadados_janela_e_a_lista_completa_de_janelas(dataset_fixture: Path) -> None:
    """Requisito 1: metadados_janela tem TODAS as janelas, incluindo a
    positiva inativa (sem nenhuma linha em fonte_a/fonte_b)."""
    resultado = carregar_arquivo(dataset_fixture)
    md = resultado["metadados_janela"]

    assert len(md) == 3  # 2 positivas + 1 negativa
    chaves = set(zip(md["classe"], md["window_id"]))
    assert chaves == {("positiva", 0), ("positiva", 1), ("negativa", 0)}


def test_janela_positiva_inativa_nao_e_descartada_do_metadado(dataset_fixture: Path) -> None:
    """Comportamento pedido explicitamente: uma janela cujo fonte_a/fonte_b
    estão vazios (positiva_inativa) não deve quebrar nem desaparecer de
    metadados_janela -- só não aparece em fonte_a/fonte_b (que não têm
    linha nenhuma para ela, por não haver evento)."""
    resultado = carregar_arquivo(dataset_fixture)
    md = resultado["metadados_janela"]

    linha_inativa = md[(md["classe"] == "positiva") & (md["window_id"] == 1)]
    assert len(linha_inativa) == 1
    assert linha_inativa["grupo"].iloc[0] == "positiva_inativa"
    assert linha_inativa["contrato_ativado"].iloc[0] == 0.0

    fa = resultado["fonte_a"]
    fb = resultado["fonte_b"]
    assert not ((fa["classe"] == "positiva") & (fa["window_id"] == 1)).any()
    assert not ((fb["classe"] == "positiva") & (fb["window_id"] == 1)).any()


def test_join_fonte_a_nao_duplica_nem_perde_linhas(dataset_fixture: Path) -> None:
    """Requisito 2: join por (classe, window_id) -- confirma contagem exata
    de linhas antes/depois do merge, para a tabela com mais linhas
    (fonte_a: 2 linhas na positiva ativa + 1 na negativa = 3)."""
    bruto = pd.read_hdf(dataset_fixture, "fonte_a")
    resultado = carregar_arquivo(dataset_fixture)
    fa = resultado["fonte_a"]

    assert len(fa) == len(bruto) == 3
    # cada linha bruta deve aparecer exatamente uma vez após o join
    assert fa.groupby(["classe", "window_id", "timestep"]).size().max() == 1


def test_join_fonte_b_nao_duplica_nem_perde_linhas(dataset_fixture: Path) -> None:
    bruto = pd.read_hdf(dataset_fixture, "fonte_b")
    resultado = carregar_arquivo(dataset_fixture)
    fb = resultado["fonte_b"]

    assert len(fb) == len(bruto) == 4  # 3 eventos positiva ativa + 1 negativa


def test_join_fonte_c_secao_todas_as_5_secoes_por_janela_ativa(dataset_fixture: Path) -> None:
    """fonte_c_secao tem 5 linhas por janela (uma por seção) -- confirma
    que o join preserva as 5 linhas de cada janela sem duplicar."""
    resultado = carregar_arquivo(dataset_fixture)
    fc = resultado["fonte_c_secao"]

    linhas_janela_ativa = fc[(fc["classe"] == "positiva") & (fc["window_id"] == 0)]
    assert len(linhas_janela_ativa) == 5
    assert set(linhas_janela_ativa["unidade"]) == {0, 1, 2, 3, 4}


def test_grupo_correto_nas_tres_tabelas_de_fonte(dataset_fixture: Path) -> None:
    """Grupo anexado corretamente após o join, não só em metadados_janela."""
    resultado = carregar_arquivo(dataset_fixture)

    fa = resultado["fonte_a"]
    grupo_positiva_ativa = fa.loc[(fa["classe"] == "positiva") & (fa["window_id"] == 0), "grupo"].unique()
    assert list(grupo_positiva_ativa) == ["positiva_ativa"]

    grupo_negativa = fa.loc[fa["classe"] == "negativa", "grupo"].unique()
    assert list(grupo_negativa) == ["negativa"]

    fc = resultado["fonte_c_secao"]
    grupo_inativa = fc.loc[(fc["classe"] == "positiva") & (fc["window_id"] == 1), "grupo"].unique()
    assert list(grupo_inativa) == ["positiva_inativa"]


def test_parametros_cenario_anexados_como_metadado_em_todas_as_tabelas(dataset_fixture: Path) -> None:
    """Requisito 4/5: parâmetros do nome do arquivo aparecem como colunas
    de metadado em toda tabela retornada, com o valor correto."""
    resultado = carregar_arquivo(dataset_fixture)

    for nome_tabela, df in resultado.items():
        for coluna in COLUNAS_METADADO_CENARIO:
            assert coluna in df.columns, f"{coluna!r} ausente em {nome_tabela!r}"
        if len(df) > 0:
            assert (df["delta_t"] == 2.0).all()
            assert (df["recompensa"] == 1.0).all()
            assert (df["pi"] == 0.0).all()
            assert (df["rho"] == 0.0).all()
            assert (df["seed"] == 1).all()
            assert (df["g"] == "pool").all()
            assert (df["beta"] == 1).all()


def test_colunas_metadado_janela_presentes_em_metadados(dataset_fixture: Path) -> None:
    """split/grupo (Requisito 5) disponíveis para filtro em metadados_janela."""
    resultado = carregar_arquivo(dataset_fixture)
    md = resultado["metadados_janela"]

    for coluna in COLUNAS_METADADO_JANELA:
        assert coluna in md.columns, f"{coluna!r} ausente em metadados_janela"


def test_colunas_metadado_e_dados_sao_distinguiveis_por_convencao(dataset_fixture: Path) -> None:
    """Requisito 5 -- separação clara entre metadado e feature: nenhuma
    coluna de dado real (n_eventos, volume, timestamp, fracao_candidato_alvo)
    coincide em nome com as colunas de metadado, então filtrar por
    `df.columns.difference(COLUNAS_METADADO_CENARIO + COLUNAS_METADADO_JANELA)`
    isola exatamente as colunas de dado."""
    resultado = carregar_arquivo(dataset_fixture)
    metadados_conhecidos = set(COLUNAS_METADADO_CENARIO) | set(COLUNAS_METADADO_JANELA)

    colunas_dado_fonte_a = set(resultado["fonte_a"].columns) - metadados_conhecidos
    assert colunas_dado_fonte_a == {"timestep", "n_eventos", "volume", "timestamp_medio", "dispersao_timestamp"}

    colunas_dado_fonte_c = set(resultado["fonte_c_secao"].columns) - metadados_conhecidos
    assert colunas_dado_fonte_c == {"unidade", "fracao_candidato_alvo"}


def test_nao_ha_agregacao_por_janela(dataset_fixture: Path) -> None:
    """Requisito 7 -- dados permanecem "longos": fonte_a da janela ativa
    tem 2 linhas (2 timesteps), não 1 linha agregada."""
    resultado = carregar_arquivo(dataset_fixture)
    fa = resultado["fonte_a"]

    linhas_janela_ativa = fa[(fa["classe"] == "positiva") & (fa["window_id"] == 0)]
    assert len(linhas_janela_ativa) == 2  # não agregado a 1 linha


# ---------------------------------------------------------------------------
# carregar_diretorio -- Requisito 6
# ---------------------------------------------------------------------------


def test_carregar_diretorio_concatena_multiplos_arquivos(tmp_path) -> None:
    cenario_a = CenarioAdversarial(
        fonte_a=_fonte_a_df([[0, 1, 1.0, 0.5, 0.0]]),
        fonte_b=np.array([0.5]),
        resultado_por_secao=pd.Series([0.5]),
        resultado_por_municipio=pd.Series([0.5]),
        resultado_por_estado=pd.Series([0.5]),
        contrato_ativado=True,
    )
    cenario_neg = CenarioNormal(
        fonte_a=_fonte_a_df([[0, 1, 1.0, 0.5, 0.0]]),
        fonte_b=np.array([0.5]),
        resultado_por_secao=pd.Series([0.3]),
        resultado_por_municipio=pd.Series([0.3]),
        resultado_por_estado=pd.Series([0.3]),
    )
    seeds = [np.random.SeedSequence(i) for i in range(4)]

    caminho_1 = tmp_path / _nome_com_params(recompensa=0.5, seed=1)
    escrever_run_hdf5(caminho_1, [(0, cenario_a, seeds[0], seeds[1])], [(0, cenario_neg, seeds[2], seeds[3])])

    caminho_2 = tmp_path / _nome_com_params(recompensa=1.5, seed=2)
    escrever_run_hdf5(caminho_2, [(0, cenario_a, seeds[0], seeds[1])], [(0, cenario_neg, seeds[2], seeds[3])])

    resultado = carregar_diretorio(tmp_path)

    assert len(resultado["metadados_janela"]) == 4  # 2 arquivos x 2 janelas cada
    assert set(resultado["metadados_janela"]["recompensa"]) == {0.5, 1.5}
    assert set(resultado["metadados_janela"]["seed"]) == {1, 2}


def test_carregar_diretorio_levanta_erro_para_diretorio_vazio(tmp_path) -> None:
    diretorio_vazio = tmp_path / "vazio"
    diretorio_vazio.mkdir()

    with pytest.raises(FileNotFoundError, match="Nenhum arquivo .h5"):
        carregar_diretorio(diretorio_vazio)


def test_carregar_diretorio_aceita_subconjunto_de_tabelas(tmp_path) -> None:
    cenario_a = CenarioAdversarial(
        fonte_a=_fonte_a_df([[0, 1, 1.0, 0.5, 0.0]]),
        fonte_b=np.array([0.5]),
        resultado_por_secao=pd.Series([0.5]),
        resultado_por_municipio=pd.Series([0.5]),
        resultado_por_estado=pd.Series([0.5]),
        contrato_ativado=True,
    )
    seeds = [np.random.SeedSequence(i) for i in range(2)]
    caminho = tmp_path / _nome_com_params()
    escrever_run_hdf5(caminho, [(0, cenario_a, seeds[0], seeds[1])], [])

    resultado = carregar_diretorio(tmp_path, tabelas=("fonte_a",))

    assert set(resultado.keys()) == {"metadados_janela", "fonte_a"}


@pytest.mark.skipif(not _V3_DISPONIVEL, reason="output/dataset_producao_v3/ não encontrado neste ambiente")
def test_carregar_arquivo_real_do_v3_positiva_inativa_presente(tmp_path) -> None:
    """Mesmo comportamento do teste com fixture local (janela positiva
    inativa não descartada), agora contra um arquivo REAL do v3 com
    recompensa=0.5 (nível onde ~90% das janelas positivas ficam inativas,
    ver CLAUDE.md) -- garante que o comportamento não é um artefato só da
    fixture sintética."""
    caminho = _DIR_V3 / "beta-1_delta_t-0.0000_g-pool_pi-0.0000_recompensa-0.5000_rho-0.0000_seed-1.h5"
    assert caminho.exists()

    resultado = carregar_arquivo(caminho)
    md = resultado["metadados_janela"]

    positivas_inativas = md[(md["classe"] == "positiva") & (md["grupo"] == "positiva_inativa")]
    assert len(positivas_inativas) > 0, "esperava-se ao menos uma janela positiva_inativa em recompensa=0.5"

    window_ids_inativos = set(positivas_inativas["window_id"])
    fa = resultado["fonte_a"]
    fa_positiva = fa[fa["classe"] == "positiva"]
    window_ids_com_fonte_a = set(fa_positiva["window_id"])

    # nenhuma janela inativa deveria ter linha em fonte_a (Fase 2 nunca
    # roda para contrato_ativado=False, ver model.py::resolver_desembolso)
    assert window_ids_inativos.isdisjoint(window_ids_com_fonte_a)
