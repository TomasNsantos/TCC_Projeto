"""Testes de src/pipeline/geracao.py (gerar_par_de_classes_real)."""

from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import pytest

from src.pipeline.config import ParametrosPopulacionaisStub, ParametrosStubGeracao
from src.pipeline.geracao import gerar_par_de_classes_real

# recompensa alta + threshold_range default => adesao quase universal, contrato
# ativa com folga acima do resultado_alvo default (0.5) na maioria das janelas.
# pi=0.0 preserva o comportamento de antes da integracao de pi ao gerador real
# (retrocompatibilidade estrita) -- testes especificos sobre pi sobrescrevem.
_PARAMS_ATIVA_CONTRATO = {"g": "pool", "delta_t": 20.0, "recompensa": 10.0, "rho": 0.3, "beta": 1, "pi": 0.0}


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


def test_fragmentacao_beta_chega_ate_o_hdf5(
    populacionais: ParametrosPopulacionaisStub, stub_geracao: ParametrosStubGeracao, tmp_path
) -> None:
    """Rede de segurança para o eixo de robustez de β, que ainda não está
    conectado a orquestrar/orquestrar_paralelo (ver CLAUDE.md/plano da
    tarefa de teste e2e) -- confirma que a fragmentação já testada
    isoladamente em layer2_copula.aplicar_batching (len(timestamps)*beta)
    e em gerar_cenario_adversarial (Fonte B pareada 1:1 com
    eventos_desembolso já fragmentado) de fato se propaga até o HDF5
    gravado por gerar_par_de_classes_real, não só até as camadas internas.

    Mesma seed/params nas duas chamadas, variando só beta: com o mesmo
    conjunto de agentes pagos (Fase 2 não depende de beta, só a
    fragmentação pós-decisão), len(fonte_b) deve escalar exatamente por
    beta -- eventos_desembolso vira (n_pagos * beta) via aplicar_batching
    (np.repeat), e gerar_fonte_b devolve "mesmo tamanho de
    fonte_a_timestamps" (docstring de layer2_copula.gerar_fonte_b).
    """
    params_beta_1 = {**_PARAMS_ATIVA_CONTRATO, "beta": 1}
    params_beta_5 = {**_PARAMS_ATIVA_CONTRATO, "beta": 5}

    resultado_beta_1 = gerar_par_de_classes_real(
        params_beta_1, seed=42, n_janelas=5, populacionais=populacionais, stub_geracao=stub_geracao, diretorio_output=tmp_path / "beta1"
    )
    resultado_beta_5 = gerar_par_de_classes_real(
        params_beta_5, seed=42, n_janelas=5, populacionais=populacionais, stub_geracao=stub_geracao, diretorio_output=tmp_path / "beta5"
    )

    # filtra só classe positiva: fonte_a/fonte_b da classe negativa vêm de
    # trafego de fundo independente (gerar_cenario_normal), que não passa
    # por resolver_desembolso/aplicar_batching -- não deveria escalar com
    # beta, e misturar as duas classes na soma esconderia isso.
    fonte_a_1 = pd.read_hdf(resultado_beta_1["caminho_output"], "fonte_a")
    fonte_a_5 = pd.read_hdf(resultado_beta_5["caminho_output"], "fonte_a")
    fonte_a_1_positiva = fonte_a_1[fonte_a_1["classe"] == "positiva"]
    fonte_a_5_positiva = fonte_a_5[fonte_a_5["classe"] == "positiva"]

    fonte_b_1 = pd.read_hdf(resultado_beta_1["caminho_output"], "fonte_b")
    fonte_b_5 = pd.read_hdf(resultado_beta_5["caminho_output"], "fonte_b")
    fonte_b_1_positiva = fonte_b_1[fonte_b_1["classe"] == "positiva"]
    fonte_b_5_positiva = fonte_b_5[fonte_b_5["classe"] == "positiva"]

    # mesma seed => mesmo conjunto de agentes pagos por janela (Fase 2 nao
    # depende de beta) => n_eventos de fonte_a (bucketizado) soma igual por
    # janela antes de fragmentar; beta so multiplica a contagem de eventos.
    assert len(fonte_a_1_positiva) > 0, "cenario de teste nao gerou nenhum evento de Fonte A -- nao exercita a fragmentacao"
    assert fonte_a_5_positiva["n_eventos"].sum() == fonte_a_1_positiva["n_eventos"].sum() * 5

    assert len(fonte_b_1_positiva) > 0, "cenario de teste nao ativou o contrato em nenhuma janela -- nao exercita Fonte B"
    assert len(fonte_b_5_positiva) == len(fonte_b_1_positiva) * 5

    # classe negativa nao deveria escalar com beta -- confirma que o efeito
    # observado acima e especifico da classe positiva/Fase 2, nao um
    # artefato de outra fonte de variacao entre as duas execucoes.
    fonte_a_1_negativa = fonte_a_1[fonte_a_1["classe"] == "negativa"]
    fonte_a_5_negativa = fonte_a_5[fonte_a_5["classe"] == "negativa"]
    assert fonte_a_5_negativa["n_eventos"].sum() == fonte_a_1_negativa["n_eventos"].sum()


def test_pi_alto_reduz_eventos_de_fonte_a_positiva_e_negativa(
    populacionais: ParametrosPopulacionaisStub, stub_geracao: ParametrosStubGeracao, tmp_path
) -> None:
    """Rede de seguranca para a integracao de pi ao gerador real: confirma
    que params["pi"] chega ate ElectionModel(pi=...)/gerar_cenario_normal
    e reduz o numero de eventos observados em Fonte A das duas classes,
    nao so dentro das camadas isoladas (model.py/trafego.py, ja testadas
    em tarefas anteriores)."""
    params_pi_zero = {**_PARAMS_ATIVA_CONTRATO, "pi": 0.0}
    params_pi_alto = {**_PARAMS_ATIVA_CONTRATO, "pi": 0.9}

    resultado_pi_zero = gerar_par_de_classes_real(
        params_pi_zero, seed=42, n_janelas=5, populacionais=populacionais, stub_geracao=stub_geracao, diretorio_output=tmp_path / "pi0"
    )
    resultado_pi_alto = gerar_par_de_classes_real(
        params_pi_alto, seed=42, n_janelas=5, populacionais=populacionais, stub_geracao=stub_geracao, diretorio_output=tmp_path / "pi9"
    )

    fonte_a_pi_zero = pd.read_hdf(resultado_pi_zero["caminho_output"], "fonte_a")
    fonte_a_pi_alto = pd.read_hdf(resultado_pi_alto["caminho_output"], "fonte_a")

    for classe in ("positiva", "negativa"):
        n_pi_zero = fonte_a_pi_zero.loc[fonte_a_pi_zero["classe"] == classe, "n_eventos"].sum()
        n_pi_alto = fonte_a_pi_alto.loc[fonte_a_pi_alto["classe"] == classe, "n_eventos"].sum()
        assert n_pi_zero > 0, f"cenario de teste nao gerou eventos de Fonte A para classe {classe} com pi=0.0"
        assert n_pi_alto < n_pi_zero, f"pi=0.9 nao reduziu n_eventos de Fonte A para classe {classe}"


def test_pi_reprodutibilidade_mesma_seed_produz_mesmo_hdf5(
    populacionais: ParametrosPopulacionaisStub, stub_geracao: ParametrosStubGeracao, tmp_path
) -> None:
    """Confirma que a nova derivacao local de seed_pi_positiva/seed_pi_negativa
    (via seed_modelo.spawn(1)[0]) e deterministica -- mesma seed de topo
    produz o mesmo HDF5 em duas execucoes, com pi>0 exercitando o novo
    caminho de codigo (nao so pi=0.0, que nao consome random_state_pi)."""
    params_pi_alto = {**_PARAMS_ATIVA_CONTRATO, "pi": 0.9}

    resultado_1 = gerar_par_de_classes_real(
        params_pi_alto, seed=7, n_janelas=4, populacionais=populacionais, stub_geracao=stub_geracao, diretorio_output=tmp_path / "run1"
    )
    resultado_2 = gerar_par_de_classes_real(
        params_pi_alto, seed=7, n_janelas=4, populacionais=populacionais, stub_geracao=stub_geracao, diretorio_output=tmp_path / "run2"
    )

    for tabela in ("fonte_a", "fonte_b", "fonte_c_secao", "fonte_c_municipio", "fonte_c_estado", "metadados_janela"):
        df1 = pd.read_hdf(resultado_1["caminho_output"], tabela)
        df2 = pd.read_hdf(resultado_2["caminho_output"], tabela)
        assert df1.equals(df2), f"tabela {tabela} difere entre as duas execucoes com pi=0.9"
