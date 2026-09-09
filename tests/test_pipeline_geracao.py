"""Testes de src/pipeline/geracao.py (gerar_par_de_classes_real)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from src.generator.layer1_abm import ElectionModel as _ElectionModelReal
from src.pipeline.config import ParametrosPopulacionaisStub, ParametrosStubGeracao
from src.pipeline.geracao import gerar_par_de_classes_real

_FIXTURES_CANDIDATO_ALVO = Path(__file__).parent / "fixtures" / "candidato_alvo_retrocompat"

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


def _monkeypatch_election_model_espiao(monkeypatch: pytest.MonkeyPatch, chamadas: list) -> None:
    """Espiona o argumento candidato_alvo de cada chamada a ElectionModel,
    sem impedir que ela rode de verdade -- alveja src.pipeline.geracao.ElectionModel
    (o NOME COMO IMPORTADO dentro de geracao.py via
    `from src.generator.layer1_abm import ElectionModel`), nao
    src.generator.layer1_abm.ElectionModel na origem: patchar na origem nao
    afeta a referencia que geracao.py ja importou para seu proprio namespace
    no momento do import do modulo (mesmo padrao ja usado em
    test_n_candidatos_um_levanta_erro_claro_antes_de_gerar_qualquer_janela)."""

    def _construtor_espiao(*args, **kwargs):
        chamadas.append(kwargs["candidato_alvo"])
        return _ElectionModelReal(*args, **kwargs)

    monkeypatch.setattr("src.pipeline.geracao.ElectionModel", _construtor_espiao)


def test_candidato_alvo_none_varia_entre_janelas_em_pelo_menos_uma_classe(
    stub_geracao: ParametrosStubGeracao, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """candidato_alvo=None deve sortear um valor por janela, nao usar um
    unico valor fixo para o run inteiro.

    Calculo de falso positivo: evento de falha = "as duas classes sao
    internamente constantes ao mesmo tempo" (todas as 20 janelas de uma
    classe sortearam o mesmo candidato). Para uma classe com 20 sorteios
    uniformes independentes em {0,...,4} (n_candidatos=5),
    P(constante) = 5 * (1/5)**20 = 5**-19 ~= 2e-14. Como as duas classes
    usam seed_modelo's independentes (raizes distintas de derivar_seeds),
    P(as duas constantes ao mesmo tempo) = (5**-19)**2 ~= 4e-27 --
    extremamente improvavel, nao e flaky na pratica.
    """
    populacionais_none = ParametrosPopulacionaisStub(n_agentes=50, n_secoes=4, n_candidatos=5, candidato_alvo=None)
    chamadas: list = []
    _monkeypatch_election_model_espiao(monkeypatch, chamadas)

    gerar_par_de_classes_real(
        _PARAMS_ATIVA_CONTRATO, seed=1, n_janelas=20, populacionais=populacionais_none, stub_geracao=stub_geracao, diretorio_output=tmp_path
    )

    assert len(chamadas) == 40  # 20 positiva + 20 negativa, dois loops sequenciais
    positiva, negativa = chamadas[:20], chamadas[20:]
    assert all(isinstance(c, int) for c in chamadas), "candidato_alvo deve ser int puro, nao np.int64"
    assert len(set(positiva)) > 1 or len(set(negativa)) > 1, (
        "candidato_alvo=None nao variou entre janelas em nenhuma das duas classes -- "
        "sorteio pode nao estar acontecendo por janela"
    )


def test_candidato_alvo_none_e_reprodutivel_com_mesma_seed(
    stub_geracao: ParametrosStubGeracao, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    populacionais_none = ParametrosPopulacionaisStub(n_agentes=50, n_secoes=4, n_candidatos=5, candidato_alvo=None)

    chamadas_1: list = []
    _monkeypatch_election_model_espiao(monkeypatch, chamadas_1)
    gerar_par_de_classes_real(
        _PARAMS_ATIVA_CONTRATO, seed=3, n_janelas=10, populacionais=populacionais_none, stub_geracao=stub_geracao, diretorio_output=tmp_path / "run1"
    )

    chamadas_2: list = []
    _monkeypatch_election_model_espiao(monkeypatch, chamadas_2)
    gerar_par_de_classes_real(
        _PARAMS_ATIVA_CONTRATO, seed=3, n_janelas=10, populacionais=populacionais_none, stub_geracao=stub_geracao, diretorio_output=tmp_path / "run2"
    )

    assert chamadas_1 == chamadas_2


def test_candidato_alvo_zero_explicito_reproduz_hdf5_do_codigo_anterior_a_esta_tarefa(
    tmp_path,
) -> None:
    """Retrocompatibilidade verificada contra o código PRÉ-tarefa, não
    contra outra chamada pós-tarefa (que só provaria determinismo interno
    do código novo, não retrocompatibilidade com o antigo).

    Fixtures em tests/fixtures/candidato_alvo_retrocompat/*.pkl foram
    geradas ANTES de qualquer edição desta tarefa em geracao.py (código do
    commit 9df5774 -- só o tipo de candidato_alvo tinha mudado para
    int | None, sem lógica de sorteio), com:
        populacionais = ParametrosPopulacionaisStub(n_agentes=100, n_secoes=4,
            n_candidatos=3, candidato_alvo=0)
        stub_geracao = ParametrosStubGeracao(tau_kendall=0.5, taxa_fonte_a=1.0,
            volume_medio_fonte_a=1000.0, taxa_fonte_b=1.0)
        params = {"g": "pool", "delta_t": 20.0, "recompensa": 10.0, "rho": 0.3,
            "beta": 1, "pi": 0.0}
        gerar_par_de_classes_real(params, seed=99, n_janelas=4, ...)
    Comparação via pd.testing.assert_frame_equal(check_like=False) contra
    o conteúdo lido de volta via pd.read_hdf de cada uma das 6 tabelas --
    não hash bruto do arquivo .h5 (verificado empiricamente que
    pd.HDFStore/PyTables embute timestamp/metadado interno a cada escrita,
    então duas escritas do MESMO DataFrame produzem arquivos com hashes
    diferentes mesmo sem nenhuma mudança de conteúdo).
    """
    populacionais = ParametrosPopulacionaisStub(n_agentes=100, n_secoes=4, n_candidatos=3, candidato_alvo=0)
    stub_geracao = ParametrosStubGeracao(tau_kendall=0.5, taxa_fonte_a=1.0, volume_medio_fonte_a=1000.0, taxa_fonte_b=1.0)
    params = {"g": "pool", "delta_t": 20.0, "recompensa": 10.0, "rho": 0.3, "beta": 1, "pi": 0.0}

    resultado = gerar_par_de_classes_real(
        params, seed=99, n_janelas=4, populacionais=populacionais, stub_geracao=stub_geracao, diretorio_output=tmp_path
    )

    for tabela in ("fonte_a", "fonte_b", "fonte_c_secao", "fonte_c_municipio", "fonte_c_estado", "metadados_janela"):
        df_novo = pd.read_hdf(resultado["caminho_output"], tabela)
        df_referencia = pd.read_pickle(_FIXTURES_CANDIDATO_ALVO / f"{tabela}.pkl")
        pd.testing.assert_frame_equal(df_novo, df_referencia, check_like=False)


def test_candidato_alvo_none_positiva_e_negativa_sorteiam_independentemente(
    stub_geracao: ParametrosStubGeracao, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prova de independencia entre classes do MESMO window_id -- decisao
    deliberada, nao um descuido: positiva e negativa usam seed_modelo's
    distintas (raizes diferentes de derivar_seeds), entao seus sorteios de
    candidato_alvo nunca compartilham entropia.

    Calculo de falso positivo, DIFERENTE do teste de variancia acima:
    evento de falha aqui e "para TODO indice i, os dois valores pareados
    (positiva[i], negativa[i]) coincidem por acaso" -- nao "uma classe e
    constante". Por indice, com as duas classes sorteando uniformemente e
    independentemente em {0,...,4} (n_candidatos=5),
    P(positiva[i] == negativa[i]) = 1/5. Os 20 pareamentos sao
    independentes entre si (seeds independentes por janela via
    derivar_seeds), entao P(todos os 20 pareamentos coincidem) =
    (1/5)**20 ~= 1e-14 -- extremamente improvavel.
    """
    populacionais_none = ParametrosPopulacionaisStub(n_agentes=50, n_secoes=4, n_candidatos=5, candidato_alvo=None)
    chamadas: list = []
    _monkeypatch_election_model_espiao(monkeypatch, chamadas)

    gerar_par_de_classes_real(
        _PARAMS_ATIVA_CONTRATO, seed=5, n_janelas=20, populacionais=populacionais_none, stub_geracao=stub_geracao, diretorio_output=tmp_path
    )

    positiva, negativa = chamadas[:20], chamadas[20:]
    divergencias = [i for i in range(20) if positiva[i] != negativa[i]]
    assert len(divergencias) > 0, (
        "candidato_alvo sorteado foi identico entre positiva e negativa em TODAS as janelas -- "
        "sinal de acoplamento indevido entre as duas classes (deveriam ser independentes)"
    )
