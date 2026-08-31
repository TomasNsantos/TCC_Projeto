"""Testes de src/pipeline/geracao.py (gerar_par_de_classes_real)."""

from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import pytest

from src.pipeline.config import ParametrosPopulacionaisStub, ParametrosStubGeracao
from src.pipeline.geracao import gerar_par_de_classes_real

# recompensa alta + threshold_range default => adesao quase universal, contrato
# ativa com folga acima do resultado_alvo default (0.5) na maioria das janelas.
_PARAMS_ATIVA_CONTRATO = {"g": "pool", "delta_t": 20.0, "recompensa": 10.0, "rho": 0.3, "beta": 1}


@pytest.fixture
def populacionais() -> ParametrosPopulacionaisStub:
    return ParametrosPopulacionaisStub(n_agentes=100, n_secoes=4, n_candidatos=3)


@pytest.fixture
def stub_geracao() -> ParametrosStubGeracao:
    return ParametrosStubGeracao(tau_kendall=0.5, taxa_fonte_a=1.0, volume_medio_fonte_a=1000.0, taxa_fonte_b=1.0)


def test_gerar_par_de_classes_real_produz_hdf5_com_contagens_corretas(
    populacionais: ParametrosPopulacionaisStub, stub_geracao: ParametrosStubGeracao, tmp_path
) -> None:
    resultado = gerar_par_de_classes_real(
        _PARAMS_ATIVA_CONTRATO, seed=1, n_janelas=5, populacionais=populacionais, stub_geracao=stub_geracao, diretorio_output=tmp_path
    )

    assert resultado["n_janelas_ok"] == 10  # 5 positivas + 5 negativas
    assert resultado["n_janelas_falha"] == 0
    assert resultado["n_contrato_nao_ativado"] >= 0

    caminho = resultado["caminho_output"]
    metadados = pd.read_hdf(caminho, "metadados_janela")
    assert len(metadados) == 10
    assert set(metadados["classe"]) == {"positiva", "negativa"}

    n_positivas_ativas = (metadados.loc[metadados["classe"] == "positiva", "contrato_ativado"] == 1.0).sum()
    n_positivas_inativas = (metadados.loc[metadados["classe"] == "positiva", "contrato_ativado"] == 0.0).sum()
    assert n_positivas_ativas + n_positivas_inativas == 5
    assert n_positivas_inativas == resultado["n_contrato_nao_ativado"]

    assert metadados.loc[metadados["classe"] == "negativa", "contrato_ativado"].isna().all()

    # com recompensa=10.0 e defaults de ElectionModel, contrato deveria ativar
    # em pelo menos algumas janelas -- nao so checar que a contagem bate, mas
    # que o cenario de teste de fato exercita o caminho "ativado".
    assert n_positivas_ativas > 0


def test_reprodutibilidade_mesma_seed_produz_mesmo_hdf5(
    populacionais: ParametrosPopulacionaisStub, stub_geracao: ParametrosStubGeracao, tmp_path
) -> None:
    resultado_1 = gerar_par_de_classes_real(
        _PARAMS_ATIVA_CONTRATO, seed=7, n_janelas=4, populacionais=populacionais, stub_geracao=stub_geracao, diretorio_output=tmp_path / "run1"
    )
    resultado_2 = gerar_par_de_classes_real(
        _PARAMS_ATIVA_CONTRATO, seed=7, n_janelas=4, populacionais=populacionais, stub_geracao=stub_geracao, diretorio_output=tmp_path / "run2"
    )

    for tabela in ("fonte_a", "fonte_b", "fonte_c_secao", "fonte_c_municipio", "fonte_c_estado", "metadados_janela"):
        df1 = pd.read_hdf(resultado_1["caminho_output"], tabela)
        df2 = pd.read_hdf(resultado_2["caminho_output"], tabela)
        assert df1.equals(df2), f"tabela {tabela} difere entre as duas execucoes"


def test_n_candidatos_um_levanta_erro_claro_antes_de_gerar_qualquer_janela(
    stub_geracao: ParametrosStubGeracao, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    populacionais_invalido = ParametrosPopulacionaisStub(n_candidatos=1)

    mock_election_model = Mock(side_effect=AssertionError("ElectionModel nao deveria ser construido"))
    monkeypatch.setattr("src.pipeline.geracao.ElectionModel", mock_election_model)

    with pytest.raises(ValueError, match="n_candidatos"):
        gerar_par_de_classes_real(
            _PARAMS_ATIVA_CONTRATO, seed=1, n_janelas=5, populacionais=populacionais_invalido, stub_geracao=stub_geracao, diretorio_output=tmp_path
        )

    mock_election_model.assert_not_called()


def test_granularidade_diferente_de_pool_usa_unidade_alvo_zero(
    populacionais: ParametrosPopulacionaisStub, stub_geracao: ParametrosStubGeracao, tmp_path
) -> None:
    params = {**_PARAMS_ATIVA_CONTRATO, "g": "secao"}

    resultado = gerar_par_de_classes_real(
        params, seed=3, n_janelas=3, populacionais=populacionais, stub_geracao=stub_geracao, diretorio_output=tmp_path
    )

    assert resultado["n_janelas_falha"] == 0
    assert resultado["n_janelas_ok"] == 6
