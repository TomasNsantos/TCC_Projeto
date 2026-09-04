"""Orquestração de um cenário completo da classe negativa (modo normal).

Combina Fonte C real (eleição sem CSC, ``ElectionModel`` com ``recompensa=0``)
com Fonte A/B de tráfego de fundo independente (``trafego.py``), produzindo a
mesma estrutura que a classe positiva expõe, para tratamento uniforme rio
abaixo no pipeline de features (PLANO §5.2.4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.generator.layer1_abm.model import ElectionModel
from src.generator.normal_mode.trafego import (
    RandomState,
    gerar_fonte_a_normal,
    gerar_fonte_b_normal,
)


@dataclass(frozen=True)
class CenarioNormal:
    """Uma amostra completa da classe negativa: Fonte A, Fonte B e Fonte C.

    Attributes
    ----------
    fonte_a : pd.DataFrame
        Tráfego de fundo (``gerar_fonte_a_normal``), colunas
        ``timestep``/``n_eventos``/``volume``.
    fonte_b : np.ndarray
        Tráfego de fundo (``gerar_fonte_b_normal``), timestamps.
    resultado_por_secao : pd.Series
        Resultado eleitoral real (sem CSC) por seção.
    resultado_por_municipio : pd.Series
        Resultado eleitoral real por município.
    resultado_por_estado : pd.Series
        Resultado eleitoral real por estado.
    """

    fonte_a: pd.DataFrame
    fonte_b: np.ndarray
    resultado_por_secao: pd.Series
    resultado_por_municipio: pd.Series
    resultado_por_estado: pd.Series


def gerar_cenario_normal(
    modelo_eleicao: ElectionModel,
    janela: float,
    taxa_fonte_a: float,
    volume_medio_fonte_a: float,
    taxa_fonte_b: float,
    random_state_fonte_a: RandomState = None,
    random_state_fonte_b: RandomState = None,
    random_state_pi: RandomState = None,
) -> CenarioNormal:
    r"""Gera uma amostra da classe negativa: eleição real sem CSC + tráfego de fundo.

    Roda ``modelo_eleicao`` só até o fim da Fase 1 (``run()``) — nunca chama
    ``resolver_desembolso()``, que é específico do mecanismo de pagamento
    condicional a um CSC e não se aplica aqui, já que não há contrato de
    compra de voto nesta amostra. Fonte C vem diretamente do resultado
    eleitoral real do modelo (``resultado_eleitoral_por_secao/municipio/estado``);
    Fonte A/B vêm de processos de Poisson homogêneos independentes
    (``trafego.py``), com fluxos de aleatoriedade completamente separados de
    ``modelo_eleicao`` e um do outro — ``random_state_fonte_a`` e
    ``random_state_fonte_b`` são parâmetros distintos propositalmente, para
    que não exista nenhum caminho no código onde a semente de uma fonte
    influencia a outra.

    **π (privacidade) — sem parâmetro próprio nesta função, lido de
    ``modelo_eleicao.pi``.** ``fonte_a`` é obtida via
    ``gerar_fonte_a_normal(..., pi=modelo_eleicao.pi, random_state_pi=random_state_pi)``
    — mesma razão de `gerar_cenario_adversarial` (`adversarial_mode/cenario.py`):
    não duplicar o valor de π como um segundo parâmetro solto, que
    poderia divergir do valor já configurado no `ElectionModel` recebido.
    `gerar_fonte_b_normal` continua sem nenhum parâmetro de π — tráfego de
    fundo de Fonte B, estruturalmente irredutível pelo mesmo motivo já
    documentado para a Fonte B da classe positiva (ver
    `trafego.py`/CLAUDE.md).

    Parameters
    ----------
    modelo_eleicao : ElectionModel
        Instância ainda não executada (``steps == 0``), com ``recompensa=0.0``
        e ``n_candidatos > 1`` já configurados pelo caller. Todos os demais
        parâmetros populacionais/de granularidade (incluindo ``pi``) ficam a
        critério de quem constrói o modelo — esta função não os duplica.
    janela : float
        Duração da janela de observação de Fonte A/B (independente de
        ``modelo_eleicao.n_steps``, que rege só a campanha de adesão).
    taxa_fonte_a, volume_medio_fonte_a, taxa_fonte_b : float
        Parâmetros de ``gerar_fonte_a_normal``/``gerar_fonte_b_normal`` —
        suposições v0 sem valor calibrado, ver docstring de ``trafego.py``.
    random_state_fonte_a, random_state_fonte_b : int | np.random.Generator | None
        Sementes independentes para Fonte A e Fonte B.
    random_state_pi : int | np.random.Generator | None
        Semente para a máscara de privacidade π, repassada a
        ``gerar_fonte_a_normal`` — independente de ``random_state_fonte_a``/
        ``random_state_fonte_b``. Não consultada quando
        ``modelo_eleicao.pi == 0.0``.

    Returns
    -------
    CenarioNormal

    Raises
    ------
    ValueError
        Se ``modelo_eleicao.recompensa != 0`` (não é um cenário sem suborno);
        se ``modelo_eleicao.steps != 0`` (o modelo já foi executado — esta
        função é quem chama ``run()``); ou se ``modelo_eleicao.n_candidatos
        == 1`` — nesse caso o voto de base nunca é consultado
        (``ElectionModel._voto_e_candidato_alvo``) e, com adesão zero
        (``recompensa=0``), o resultado por seção fica exatamente zero em
        todo lugar — degenerando no mesmo cenário do Sanity Check 1, que
        esta função existe para diferenciar.
    """
    if modelo_eleicao.recompensa != 0.0:
        raise ValueError(
            "gerar_cenario_normal exige modelo_eleicao.recompensa == 0.0 — não há CSC no cenário normal."
        )
    if modelo_eleicao.steps != 0:
        raise ValueError(
            "modelo_eleicao já foi executado — gerar_cenario_normal() deve receber um ElectionModel ainda não rodado."
        )
    if modelo_eleicao.n_candidatos == 1:
        raise ValueError(
            "gerar_cenario_normal exige modelo_eleicao.n_candidatos > 1 — com n_candidatos=1 e "
            "recompensa=0.0 o voto de base nunca é consultado e o resultado por seção fica "
            "degenerado (exatamente zero em todo lugar), reproduzindo o cenário do Sanity Check 1."
        )

    modelo_eleicao.run()

    fonte_a = gerar_fonte_a_normal(
        janela,
        taxa_fonte_a,
        volume_medio_fonte_a,
        random_state_fonte_a,
        pi=modelo_eleicao.pi,
        random_state_pi=random_state_pi,
    )
    fonte_b = gerar_fonte_b_normal(janela, taxa_fonte_b, random_state_fonte_b)

    return CenarioNormal(
        fonte_a=fonte_a,
        fonte_b=fonte_b,
        resultado_por_secao=modelo_eleicao.resultado_eleitoral_por_secao(),
        resultado_por_municipio=modelo_eleicao.resultado_eleitoral_por_municipio(),
        resultado_por_estado=modelo_eleicao.resultado_eleitoral_por_estado(),
    )
