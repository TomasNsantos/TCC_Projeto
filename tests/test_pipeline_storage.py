"""Testes de src/pipeline/storage.py (calcular_split, escrever_run_hdf5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.generator.adversarial_mode import CenarioAdversarial
from src.generator.normal_mode import CenarioNormal
from src.pipeline.storage import calcular_split, escrever_run_hdf5


@pytest.mark.parametrize(
    "window_id, esperado",
    [(699, "train"), (700, "val"), (849, "val"), (850, "test")],
)
def test_calcular_split_fronteiras(window_id: int, esperado: str) -> None:
    assert calcular_split(window_id, n_janelas=1000) == esperado


def _fonte_a_df(rows: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["timestep", "n_eventos", "volume"])


@pytest.fixture
def cenario_positivo_ativo() -> CenarioAdversarial:
    return CenarioAdversarial(
        fonte_a=_fonte_a_df([[1, 2, 20.0]]),
        fonte_b=np.array([1.5, 2.5]),
        resultado_por_secao=pd.Series([0.6, 0.7]),
        resultado_por_municipio=pd.Series([0.65]),
        resultado_por_estado=pd.Series([0.65]),
        contrato_ativado=True,
    )


@pytest.fixture
def cenario_positivo_inativo() -> CenarioAdversarial:
    return CenarioAdversarial(
        fonte_a=_fonte_a_df([]),
        fonte_b=np.array([]),
        resultado_por_secao=pd.Series([0.1, 0.2]),
        resultado_por_municipio=pd.Series([0.15]),
        resultado_por_estado=pd.Series([0.15]),
        contrato_ativado=False,
    )


@pytest.fixture
def cenario_negativo() -> CenarioNormal:
    return CenarioNormal(
        fonte_a=_fonte_a_df([[1, 1, 5.0]]),
        fonte_b=np.array([3.0]),
        resultado_por_secao=pd.Series([0.3, 0.3]),
        resultado_por_municipio=pd.Series([0.3]),
        resultado_por_estado=pd.Series([0.3]),
    )


def test_escrever_run_hdf5_cria_seis_tabelas_com_linhas_esperadas(
    tmp_path, cenario_positivo_ativo, cenario_positivo_inativo, cenario_negativo
) -> None:
    seeds = [np.random.SeedSequence(i) for i in range(6)]
    janelas_positivas = [(0, cenario_positivo_ativo, seeds[0], seeds[1]), (1, cenario_positivo_inativo, seeds[2], seeds[3])]
    janelas_negativas = [(0, cenario_negativo, seeds[4], seeds[5])]

    caminho = tmp_path / "run.h5"
    escrever_run_hdf5(caminho, janelas_positivas, janelas_negativas)

    assert caminho.exists()

    fonte_a = pd.read_hdf(caminho, "fonte_a")
    assert len(fonte_a) == 2  # 1 linha da janela positiva ativa + 1 da negativa; a positiva inativa nao contribui
    assert set(fonte_a.columns) == {"window_id", "classe", "split", "timestep", "n_eventos", "volume"}

    fonte_b = pd.read_hdf(caminho, "fonte_b")
    assert len(fonte_b) == 3  # 2 eventos da positiva ativa + 1 da negativa

    for tabela in ("fonte_c_secao", "fonte_c_municipio", "fonte_c_estado"):
        df = pd.read_hdf(caminho, tabela)
        assert set(df["window_id"]) <= {0, 1}
        assert set(df["classe"]) == {"positiva", "negativa"}

    metadados = pd.read_hdf(caminho, "metadados_janela")
    assert len(metadados) == 3


def test_contrato_ativado_e_nulo_para_classe_negativa(
    tmp_path, cenario_positivo_ativo, cenario_negativo
) -> None:
    seeds = [np.random.SeedSequence(i) for i in range(4)]
    caminho = tmp_path / "run.h5"
    escrever_run_hdf5(caminho, [(0, cenario_positivo_ativo, seeds[0], seeds[1])], [(0, cenario_negativo, seeds[2], seeds[3])])

    metadados = pd.read_hdf(caminho, "metadados_janela")
    positiva = metadados[metadados["classe"] == "positiva"].iloc[0]
    negativa = metadados[metadados["classe"] == "negativa"].iloc[0]

    assert positiva["contrato_ativado"] == 1.0
    assert pd.isna(negativa["contrato_ativado"])


def test_escrever_run_hdf5_com_todas_janelas_vazias_nao_quebra(tmp_path) -> None:
    """Caso de fronteira: se nenhuma janela de uma classe teve eventos em
    Fonte A/B, a tabela correspondente fica vazia mas com dtypes corretos
    (nao 'object' -- ver bug encontrado na implementacao)."""
    cenario_sem_eventos = CenarioAdversarial(
        fonte_a=_fonte_a_df([]),
        fonte_b=np.array([]),
        resultado_por_secao=pd.Series([0.0]),
        resultado_por_municipio=pd.Series([0.0]),
        resultado_por_estado=pd.Series([0.0]),
        contrato_ativado=False,
    )
    seeds = [np.random.SeedSequence(i) for i in range(2)]
    caminho = tmp_path / "run.h5"

    escrever_run_hdf5(caminho, [(0, cenario_sem_eventos, seeds[0], seeds[1])], [])

    fonte_a = pd.read_hdf(caminho, "fonte_a")
    assert fonte_a.empty
    assert fonte_a["window_id"].dtype.kind == "i"

    fonte_b = pd.read_hdf(caminho, "fonte_b")
    assert fonte_b.empty
    assert fonte_b["window_id"].dtype.kind == "i"
