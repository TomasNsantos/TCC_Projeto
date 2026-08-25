"""Camada 2 do gerador sintético: estrutura de dependência via cópula Clayton.

Acopla a Fonte B (interações com oráculo) à Fonte A (eventos de fronteira),
gerada pela Camada 1 (`layer1_abm`). A Fonte C (resultado eleitoral) não passa
por este acoplamento — é gerada diretamente pela simulação de agentes.

Clayton (e não Gumbel) porque a Fonte B não tem dimensão de magnitude: o único
eixo de dependência coerente é proximidade temporal (cauda inferior — valores
pequenos de tempo-até-evento coocorrendo), não magnitude conjunta grande
(cauda superior). O batching reforça esse argumento: β reduz a amplitude dos
picos mas comprime o agrupamento temporal em [0, Δt/β], preservando o sinal de
proximidade temporal mesmo sob evasão adversarial.
"""

from __future__ import annotations

import numpy as np
from copulas.bivariate.clayton import Clayton

RandomState = int | np.random.Generator | None


def _pseudo_observacoes(timestamps: np.ndarray, janela: float) -> np.ndarray:
    """Normaliza timestamps para (0, 1) dentro da janela de observação."""
    u = np.asarray(timestamps, dtype=float) / janela
    return np.clip(u, 1e-9, 1 - 1e-9)


def gerar_fonte_b(
    fonte_a_timestamps: np.ndarray,
    janela: float,
    tau_kendall: float,
    random_state: RandomState = None,
) -> np.ndarray:
    r"""Gera timestamps de Fonte B condicionados à Fonte A via cópula Clayton.

    Fonte B é um evento discreto/episódico (apenas timestamp de ativação do
    oráculo, sem dimensão de magnitude). A dependência com a Fonte A é
    controlada por ``tau_kendall``: τ ≈ 0 reproduz o modo normal (processos de
    Poisson independentes — Sanity Check 4); τ > 0.4 reproduz coordenação
    adversarial.

    A conversão τ → θ usa a forma fechada da própria cópula Clayton
    (``compute_theta``, θ = 2τ/(1-τ)), sem necessidade de fitar a partir de
    dados brutos.

    Parameters
    ----------
    fonte_a_timestamps : np.ndarray
        Timestamps dos eventos de Fonte A (Camada 1), em ``[0, janela]``.
    janela : float
        Duração da janela de observação (ex.: Δt de divulgação eleitoral).
    tau_kendall : float
        Força de dependência alvo entre Fonte A e Fonte B, em ``[0, 1)``.
    random_state : int | np.random.Generator | None
        Semente para reprodutibilidade.

    Returns
    -------
    np.ndarray
        Timestamps de Fonte B, mesmo tamanho de ``fonte_a_timestamps``.
    """
    fonte_a_timestamps = np.asarray(fonte_a_timestamps, dtype=float)
    rng = np.random.default_rng(random_state)

    if fonte_a_timestamps.size == 0:
        return np.array([])

    if not 0 <= tau_kendall < 1:
        raise ValueError("tau_kendall deve estar em [0, 1) — Clayton só modela dependência positiva.")

    if tau_kendall == 0:
        # Modo normal: Fonte B é um processo de Poisson independente de A.
        return rng.uniform(0, janela, size=fonte_a_timestamps.size)

    copula = Clayton(random_state=random_state)
    copula.tau = tau_kendall
    copula.theta = copula.compute_theta()

    u_a = _pseudo_observacoes(fonte_a_timestamps, janela)
    y = rng.uniform(0, 1, size=u_a.size)
    u_b = copula.percent_point(y, u_a)

    return u_b * janela


def aplicar_batching(
    timestamps: np.ndarray,
    delta_t: float,
    beta: int,
    random_state: RandomState = None,
) -> np.ndarray:
    r"""Fragmenta eventos de Fonte A em β saques distribuídos em [0, Δt/β].

    Estratégia de evasão do adversário: em vez de um único evento, fragmenta
    em β sub-eventos, reduzindo a amplitude do pico observável por fator 1/β
    e comprimindo o agrupamento temporal na janela [0, Δt/β] a partir do
    timestamp original.

    Parameters
    ----------
    timestamps : np.ndarray
        Timestamps originais dos eventos de Fonte A (não fragmentados, β=1).
    delta_t : float
        Atraso de divulgação eleitoral (Δt) — define a escala da janela de
        fragmentação.
    beta : int
        Número de saques fragmentados por evento original.
    random_state : int | np.random.Generator | None
        Semente para reprodutibilidade.

    Returns
    -------
    np.ndarray
        Timestamps fragmentados, tamanho ``len(timestamps) * beta``.
    """
    timestamps = np.asarray(timestamps, dtype=float)
    rng = np.random.default_rng(random_state)

    if timestamps.size == 0 or beta <= 1:
        return timestamps

    janela_fragmento = delta_t / beta
    offsets = rng.uniform(0, janela_fragmento, size=(timestamps.size, beta))
    fragmentados = timestamps[:, None] + offsets
    return fragmentados.ravel()
