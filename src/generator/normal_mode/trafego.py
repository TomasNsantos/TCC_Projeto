"""Tráfego de fundo do modo normal (classe negativa): Poisson homogêneo independente.

Modo normal (PLANO §5.2.2): "processos de Poisson independentes para A e B",
τ_Kendall ≈ 0. Cada fonte é um único processo de Poisson homogêneo — sem
superposição de múltiplos tipos de contrato (decisão v0 deliberada, mantém
simplicidade proporcional a esta fase) — e cada função cria seu próprio
``np.random.default_rng`` local, sem nunca compartilhar ou derivar estado de
RNG entre Fonte A e Fonte B. Isso é o que garante independência
mecanicamente, não só empiricamente: não existe caminho no código onde uma
semente de uma fonte influencia a sequência da outra.

``taxa`` e ``volume_medio`` são suposições v0 sem valor calibrado — não há
default nas duas funções de geração, propositalmente: a ordem de grandeza
correta só pode ser decidida depois que os níveis de λ (baixa/média/alta) do
design fatorial tiverem valores numéricos no gerador, o que ainda não
aconteceu (ver CLAUDE.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RandomState = int | np.random.Generator | None

_SIGMA_LOG_VOLUME: float = 0.5
"""Desvio-padrão (em escala log) do volume por evento de Fonte A normal —
placeholder v0 só para evitar volume idêntico em todo evento, sem
calibração real."""


def gerar_fonte_a_normal(
    janela: float,
    taxa: float,
    volume_medio: float,
    random_state: RandomState = None,
) -> pd.DataFrame:
    r"""Gera Fonte A do modo normal: Poisson homogêneo, volume log-normal por evento.

    Tráfego de fundo plausível (ex.: outros contratos legítimos — DeFi,
    mercados de predição, votação corporativa — usando a mesma infraestrutura
    de oráculo/L2), sem qualquer relação com a eleição sendo simulada. Mesmo
    formato de saída de ``ElectionModel.fonte_a_eventos_fronteira`` para que
    o pipeline de features trate classe positiva e negativa uniformemente.

    Parameters
    ----------
    janela : float
        Duração da janela de observação.
    taxa : float
        Taxa do processo de Poisson (eventos por unidade de tempo). Suposição
        v0 sem valor calibrado — ver docstring do módulo.
    volume_medio : float
        Média alvo da distribuição log-normal de volume por evento
        (``LogNormal(μ, σ)`` com ``σ`` fixo em ``_SIGMA_LOG_VOLUME`` e
        ``μ = log(volume_medio) - σ²/2``, forma fechada para que a média da
        log-normal seja exatamente ``volume_medio``). Suposição v0 sem valor
        calibrado.
    random_state : int | np.random.Generator | None
        Semente para reprodutibilidade — gera um ``Generator`` local,
        independente de qualquer outro usado para Fonte B.

    Returns
    -------
    pd.DataFrame
        Colunas ``timestep`` (int), ``n_eventos`` (int), ``volume`` (float) —
        mesmo formato de ``fonte_a_eventos_fronteira``.
    """
    rng = np.random.default_rng(random_state)
    n_total = rng.poisson(taxa * janela)

    if n_total == 0:
        return pd.DataFrame(
            {
                "timestep": pd.Series(dtype=int),
                "n_eventos": pd.Series(dtype=int),
                "volume": pd.Series(dtype=float),
            }
        )

    timestamps = rng.uniform(0, janela, size=n_total)
    sigma = _SIGMA_LOG_VOLUME
    mu = np.log(volume_medio) - sigma**2 / 2
    volumes = rng.lognormal(mean=mu, sigma=sigma, size=n_total)

    timesteps = pd.Series(np.floor(timestamps).astype(int))
    contagem = (
        timesteps.value_counts()
        .sort_index()
        .rename_axis("timestep")
        .reset_index(name="n_eventos")
    )
    volume_por_timestep = pd.Series(volumes).groupby(timesteps.values).sum()
    contagem["volume"] = contagem["timestep"].map(volume_por_timestep).to_numpy()
    return contagem


def gerar_fonte_b_normal(
    janela: float,
    taxa: float,
    random_state: RandomState = None,
) -> np.ndarray:
    r"""Gera Fonte B do modo normal: timestamps de Poisson homogêneo independente.

    Fonte B continua sem dimensão de magnitude (evento discreto/episódico),
    mesmo formato de saída de ``gerar_fonte_b`` (Camada 2 acoplada). Usa seu
    próprio ``np.random.default_rng`` local — nunca compartilhado com Fonte
    A — para que a independência estatística seja garantida pela ausência de
    qualquer caminho de acoplamento no código, não apenas observada
    empiricamente via τ_Kendall baixo.

    Parameters
    ----------
    janela : float
        Duração da janela de observação.
    taxa : float
        Taxa do processo de Poisson (eventos por unidade de tempo). Suposição
        v0 sem valor calibrado — ver docstring do módulo.
    random_state : int | np.random.Generator | None
        Semente para reprodutibilidade — gera um ``Generator`` local,
        independente de qualquer outro usado para Fonte A.

    Returns
    -------
    np.ndarray
        Timestamps de Fonte B em ``[0, janela]``.
    """
    rng = np.random.default_rng(random_state)
    n = rng.poisson(taxa * janela)

    if n == 0:
        return np.array([])

    return rng.uniform(0, janela, size=n)


def contagem_por_timestep(timestamps: np.ndarray, janela: float) -> np.ndarray:
    r"""Bina timestamps em contagem por timestep inteiro em ``[0, janela)``.

    Utilitário genérico (não específico do modo normal) para estimar
    correlação cruzada entre duas fontes sem pareamento elemento-a-elemento
    natural — caso de Fonte A/B independentes, que têm contagens de eventos
    diferentes em geral. Corresponde a "τ_Kendall(A,B) estimado por janela
    deslizante" (PLANO §5.3.1): bina ambas as fontes na mesma grade de
    timesteps e compara os vetores de contagem alinhados por tempo, em vez
    dos timestamps brutos.

    Parameters
    ----------
    timestamps : np.ndarray
        Timestamps de uma fonte, em ``[0, janela]``.
    janela : float
        Duração da janela de observação — define o tamanho do vetor de saída.

    Returns
    -------
    np.ndarray
        Vetor de inteiros de tamanho ``ceil(janela)``, contagem de eventos
        por timestep (índice = timestep, preenchido com zero onde não há
        evento).
    """
    n_timesteps = int(np.ceil(janela))
    timestamps = np.asarray(timestamps, dtype=float)

    if timestamps.size == 0:
        return np.zeros(n_timesteps, dtype=int)

    timesteps_int = np.clip(np.floor(timestamps).astype(int), 0, n_timesteps - 1)
    return np.bincount(timesteps_int, minlength=n_timesteps)
