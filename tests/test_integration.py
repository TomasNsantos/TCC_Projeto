"""Testes de integração Camada 1 -> Camada 2, antecipando os Sanity Checks (PLANO §5.2.3)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kendalltau

from src.generator.layer1_abm import ElectionModel
from src.generator.layer2_copula import gerar_fonte_b


def _fonte_a_timestamps(model: ElectionModel) -> np.ndarray:
    fonte_a = model.fonte_a_eventos_fronteira()
    return np.repeat(fonte_a["timestep"].to_numpy(dtype=float), fonte_a["n_eventos"].to_numpy())


def test_sanity_check_1_lambda_zero_nao_gera_sinal_espurio() -> None:
    """Sem incentivo (proxy de lambda=0), o contrato nao ativa e Fonte A/B ficam vazias."""
    model = ElectionModel(n_agentes=200, prop_racional=1.0, recompensa=0.0, resultado_alvo=0.1, seed=1)
    model.run()
    model.resolver_desembolso()

    assert model.contrato_ativado is False

    fonte_a = _fonte_a_timestamps(model)
    assert fonte_a.size == 0

    fonte_b = gerar_fonte_b(fonte_a, janela=model.delta_t, tau_kendall=0.6, random_state=1)
    assert fonte_b.size == 0


def test_sanity_check_4_modo_normal_tau_kendall_baixo() -> None:
    """tau_Kendall(A, B) < 0.1 quando a Camada 2 opera em modo normal (tau alvo = 0)."""
    # resultado_alvo baixo garante ativacao do contrato; delta_t largo e rho=0
    # (default) espalham o desembolso em varios timesteps inteiros distintos,
    # necessario para estimar tau_Kendall com uma amostra nao-degenerada.
    model = ElectionModel(
        n_agentes=500,
        prop_racional=1.0,
        recompensa=5.0,
        threshold_range=(0.0, 0.9),
        resultado_alvo=0.0,
        delta_t=200.0,
        seed=1,
    )
    model.run()
    model.resolver_desembolso()

    assert model.contrato_ativado is True

    fonte_a = _fonte_a_timestamps(model)
    assert fonte_a.size > 30  # amostra suficiente para estimar tau_Kendall
    assert np.unique(fonte_a).size > 5  # eventos espalhados em varios timesteps

    fonte_b = gerar_fonte_b(fonte_a, janela=model.delta_t, tau_kendall=0.0, random_state=2)

    tau_empirico, _ = kendalltau(fonte_a, fonte_b)
    assert abs(tau_empirico) < 0.1


def test_sanity_check_2_rho_alto_caracteriza_limite_conhecido() -> None:
    """Caracterizacao de um limite conhecido do gerador -- NAO e uma regressao.

    O Sanity Check 2 do PLANO (rho=1.0, lambda maximo, esperado tau_Kendall(A,B)
    > 0.4) FALHA nesta configuracao: tau empirico fica bem abaixo de 0.4.
    Investigado e documentado no notebook `validacao_visual_desistencia.ipynb`
    (celulas 9-13) e em CLAUDE.md: rho=1.0 concentra os timestamps de Fonte A
    num desvio-padrao muito pequeno (por construcao de
    `_amostrar_timestamps_desembolso`, proporcional a
    `_SIGMA_FRACAO_DELTA_T * delta_t`), e tau_Kendall -- uma estatistica de
    postos -- perde poder discriminativo quando uma das variaveis tem
    variancia propria muito pequena, mesmo com o parametro interno da copula
    (theta) pedindo dependencia forte.

    Este teste fixa esse comportamento como conhecido: se ele comecar a
    falhar porque tau passou a superar 0.4 (ou porque o std deixou de
    colapsar), o comportamento do gerador mudou e a nota correspondente em
    CLAUDE.md/no notebook precisa ser revisitada -- nao trate isso como uma
    regressao no sentido usual de "conserte para o teste voltar a passar".

    Nota sobre o metodo de extracao dos timestamps de Fonte A: ao contrario
    dos Sanity Checks 1 e 4 neste mesmo arquivo (que usam `_fonte_a_timestamps()`,
    bucketizando via `fonte_a_eventos_fronteira()`), este teste usa os
    timestamps BRUTOS de `model.eventos_desembolso` diretamente. Escolha
    deliberada, nao um descuido: verificado que os dois metodos dao a mesma
    conclusao nesta configuracao (bruto: tau=0.1417; bucketizado: tau=0.1344
    -- diferenca de ~5%, ambos bem abaixo do limiar de 0.4), mas bucketizar
    arredondaria os timestamps para o timestep inteiro mais proximo, o que
    atenuaria ainda mais uma variancia que ja esta colapsada por construcao
    (ver std abaixo) -- tornaria o teste mais conservador (mais dificil do
    tau subir por acaso), nao mais correto. Usar o bruto mede a perda de
    poder do tau_Kendall na fonte mais direta possivel, sem uma segunda
    fonte de atenuacao (o binning) se somando ao efeito que o teste quer
    isolar.
    """
    # Configuracao identica a celula 9 do notebook de validacao (nao inventar
    # novos parametros). beta nao e especificado (default=1): aplicar_batching
    # e no-op exato em beta=1, entao a Tarefa 1 (integracao de batching) nao
    # interfere nos valores caracterizados aqui.
    model = ElectionModel(
        n_agentes=500,
        prop_racional=1.0,
        recompensa=10.0,
        threshold_range=(0.0, 0.3),
        resultado_alvo=0.0,
        delta_t=200.0,
        rho=1.0,
        prob_conformidade=0.9,
        seed=1,
    )
    model.run()
    model.resolver_desembolso()

    timestamps_a = np.array([t for t, _ in model.eventos_desembolso])

    # Causa raiz: rho=1.0 colapsa a variancia de Fonte A para perto do
    # sigma do ramo concentrado de _amostrar_timestamps_desembolso.
    sigma_esperado = ElectionModel._SIGMA_FRACAO_DELTA_T * model.delta_t
    assert timestamps_a.std() == pytest.approx(sigma_esperado, rel=0.5)

    # Consequencia: com Fonte A quase sem variancia propria, tau_Kendall(A,B)
    # nao atinge o limiar de 0.4 do Sanity Check 2, mesmo pedindo tau_kendall=0.8
    # (dependencia forte) na Camada 2.
    fonte_b = gerar_fonte_b(timestamps_a, janela=model.delta_t, tau_kendall=0.8, random_state=2)
    tau_empirico, _ = kendalltau(timestamps_a, fonte_b)
    assert tau_empirico < 0.4


def test_tau_kendall_cai_monotonicamente_conforme_rho_aumenta() -> None:
    """Documenta, via guarda de regressao, o achado contraintuitivo do notebook:
    mais coordenacao (rho maior) produz MENOS correlacao de postos entre A e B,
    porque rho concentra Fonte A e reduz a variancia de que tau_Kendall precisa
    para detectar dependencia -- nao mais correlacao, como a intuicao ingenua
    sugeriria. Mesma configuracao base da celula 12 do notebook de validacao."""
    kwargs = dict(
        n_agentes=500,
        prop_racional=1.0,
        recompensa=10.0,
        threshold_range=(0.0, 0.3),
        resultado_alvo=0.0,
        delta_t=200.0,
        prob_conformidade=0.9,
        seed=1,
    )

    taus = []
    for rho in (0.0, 0.3, 0.6, 1.0):
        model = ElectionModel(rho=rho, **kwargs)
        model.run()
        model.resolver_desembolso()

        timestamps_a = np.array([t for t, _ in model.eventos_desembolso])
        fonte_b = gerar_fonte_b(timestamps_a, janela=model.delta_t, tau_kendall=0.8, random_state=2)
        tau_empirico, _ = kendalltau(timestamps_a, fonte_b)
        taus.append(tau_empirico)

    for tau_atual, tau_seguinte in zip(taus, taus[1:]):
        assert tau_atual > tau_seguinte
