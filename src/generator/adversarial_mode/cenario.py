"""Orquestração de um cenário completo da classe positiva (modo adversarial).

Combina a simulação real do CSC (``ElectionModel``: Fase 1 → Fase 2 →
Fonte C) com o acoplamento Fonte A↔B via cópula Clayton (Camada 2),
produzindo a mesma estrutura que ``normal_mode.gerar_cenario_normal``
expõe para a classe negativa, para tratamento uniforme rio abaixo no
pipeline de features (PLANO §5.2.4). Antes desta função, essa composição só
existia manualmente repetida em ``tests/test_integration.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.generator.layer1_abm.model import ElectionModel
from src.generator.layer2_copula import gerar_fonte_b

RandomState = int | np.random.Generator | None


@dataclass(frozen=True)
class CenarioAdversarial:
    """Uma amostra completa da classe positiva: Fonte A, Fonte B, Fonte C e ativação.

    ``contrato_ativado`` não tem equivalente em ``CenarioNormal``: no modo
    normal a Fase 2 nunca roda, então não existe contrato a ativar. Aqui a
    ativação é o evento condicional central — decide se Fonte A/B têm
    conteúdo (desembolso + acoplamento via cópula) ou ficam vazias.

    Attributes
    ----------
    fonte_a : pd.DataFrame
        Desembolso (``ElectionModel.fonte_a_eventos_fronteira``), colunas
        ``timestep``/``n_eventos``/``volume``. Vazia se ``contrato_ativado``
        for ``False``.
    fonte_b : np.ndarray
        Timestamps de Fonte B, acoplados a Fonte A via cópula Clayton
        (``layer2_copula.gerar_fonte_b``). Vazia se ``contrato_ativado`` for
        ``False``.
    resultado_por_secao : pd.Series
        Resultado eleitoral real (com CSC) por seção.
    resultado_por_municipio : pd.Series
        Resultado eleitoral real por município.
    resultado_por_estado : pd.Series
        Resultado eleitoral real por estado.
    contrato_ativado : bool
        Se o CSC ativou (``ElectionModel.contrato_ativado``).
    """

    fonte_a: pd.DataFrame
    fonte_b: np.ndarray
    resultado_por_secao: pd.Series
    resultado_por_municipio: pd.Series
    resultado_por_estado: pd.Series
    contrato_ativado: bool


def gerar_cenario_adversarial(
    modelo_eleicao: ElectionModel,
    tau_kendall: float,
    random_state_fonte_b: RandomState = None,
    random_state_pi: RandomState = None,
) -> CenarioAdversarial:
    r"""Gera uma amostra da classe positiva: CSC real + Fonte B acoplada via cópula.

    Roda ``modelo_eleicao`` por completo (Fase 1 via ``run()``, Fase 2 via
    ``resolver_desembolso()``) e, se o contrato ativar, acopla Fonte B a
    Fonte A via ``layer2_copula.gerar_fonte_b``. Se não ativar, ``fonte_b``
    fica vazia SEM chamar ``gerar_fonte_b`` — branch explícito, não uma
    consequência do próprio guard de array-vazio de ``gerar_fonte_b``, para
    que a garantia "a cópula não roda sem dado de entrada" seja verificável
    por chamada, não só pelo resultado.

    Os timestamps passados à cópula são os BRUTOS de
    ``modelo_eleicao.eventos_desembolso`` (floats, um por evento de
    desembolso já fragmentado por β), não a versão bucketizada de
    ``fonte_a_eventos_fronteira()`` — mesmo raciocínio já registrado em
    ``test_sanity_check_2_rho_alto_caracteriza_limite_conhecido``
    (`tests/test_integration.py`): bucketizar arredondaria para o timestep
    inteiro mais próximo, atenuando a variância antes de medir/gerar a
    dependência com Fonte B. O campo ``fonte_a`` do retorno, por outro lado,
    usa a versão bucketizada — é o formato que o resto do pipeline espera;
    só os timestamps internos passados à cópula são brutos.

    **π (privacidade) — sem parâmetro próprio nesta função, lido de
    ``modelo_eleicao.pi``.** ``fonte_a`` é obtida via
    ``modelo_eleicao.fonte_a_eventos_fronteira(random_state_pi=random_state_pi)``,
    que já consulta ``self.pi`` internamente — não duplicamos o valor de π
    como um segundo parâmetro aqui, para não abrir a possibilidade de
    divergir do valor configurado no modelo. **A garantia de que Fonte B
    nunca é afetada por π não vem da ORDEM em que esta função chama as
    coisas** (``gerar_fonte_b`` roda antes de
    ``fonte_a_eventos_fronteira`` no código acima, mas isso é incidental)
    — vem de como ``fonte_a_eventos_fronteira`` é implementada: ela
    mascara uma CÓPIA local dos eventos, nunca reatribui nem filtra
    ``self.eventos_desembolso`` in-place (ver docstring do método/CLAUDE.md).
    ``gerar_fonte_b`` já usa ``modelo_eleicao.eventos_desembolso`` bruto
    diretamente (linha acima), então mesmo que a ordem das duas chamadas
    fosse invertida, o resultado seria idêntico — a proteção é estrutural,
    não posicional.

    Parameters
    ----------
    modelo_eleicao : ElectionModel
        Instância configurada, ainda não executada (``steps == 0``). Todos
        os parâmetros populacionais, de granularidade e do design fatorial
        (incluindo ``recompensa``, ``rho``, ``beta``, ``pi``) ficam a
        critério de quem constrói o modelo — esta função não os duplica.
    tau_kendall : float
        Força-alvo de dependência entre Fonte A e Fonte B na cópula Clayton
        (``[0, 1)``, ver ``gerar_fonte_b``). Obrigatório e sem default:
        stub v0 sem valor calibrado, e independente de ``rho`` — ``rho``
        controla a concentração temporal do timing de desembolso (Fase 2,
        dentro de ``ElectionModel``); ``tau_kendall`` controla a dependência
        estatística A↔B (Camada 2). Dois eixos diferentes, não confundir
        (mesma convenção de "suposição v0 sem calibração" que
        ``taxa``/``volume_medio`` já usam em ``normal_mode/trafego.py``).
    random_state_fonte_b : int | np.random.Generator | None
        Semente para a cópula (``gerar_fonte_b``) — independente do RNG
        interno de ``modelo_eleicao`` (``modelo_eleicao.rng``, que já rege
        Fase 1/Fase 2, incluindo o timing de desembolso e a fragmentação
        por β).
    random_state_pi : int | np.random.Generator | None
        Semente para a máscara de privacidade π, repassada a
        ``fonte_a_eventos_fronteira`` — independente de
        ``random_state_fonte_b`` e de ``modelo_eleicao.rng``. Não
        consultada quando ``modelo_eleicao.pi == 0.0``.

    Returns
    -------
    CenarioAdversarial

    Raises
    ------
    ValueError
        Se ``modelo_eleicao.steps != 0`` — o modelo já foi executado; esta
        função é quem chama ``run()``.
    """
    if modelo_eleicao.steps != 0:
        raise ValueError(
            "modelo_eleicao já foi executado — gerar_cenario_adversarial() deve receber um ElectionModel ainda não rodado."
        )

    modelo_eleicao.run()
    modelo_eleicao.resolver_desembolso()

    if modelo_eleicao.contrato_ativado:
        timestamps_a = np.array([t for t, _ in modelo_eleicao.eventos_desembolso])
        fonte_b = gerar_fonte_b(
            timestamps_a,
            janela=modelo_eleicao.delta_t,
            tau_kendall=tau_kendall,
            random_state=random_state_fonte_b,
        )
    else:
        fonte_b = np.array([])

    return CenarioAdversarial(
        fonte_a=modelo_eleicao.fonte_a_eventos_fronteira(random_state_pi=random_state_pi),
        fonte_b=fonte_b,
        resultado_por_secao=modelo_eleicao.resultado_eleitoral_por_secao(),
        resultado_por_municipio=modelo_eleicao.resultado_eleitoral_por_municipio(),
        resultado_por_estado=modelo_eleicao.resultado_eleitoral_por_estado(),
        contrato_ativado=modelo_eleicao.contrato_ativado,
    )
