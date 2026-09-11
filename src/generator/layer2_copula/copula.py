"""Camada 2 do gerador sintético: estrutura de dependência via cópula Clayton.

Acopla a Fonte B (interações com oráculo) à Fonte A (eventos de fronteira),
gerada pela Camada 1 (`layer1_abm`). A Fonte C (resultado eleitoral) não passa
por este acoplamento — é gerada diretamente pela simulação de agentes.

Clayton (e não Gumbel) porque a Fonte B não tem dimensão de magnitude: o único
eixo de dependência coerente é proximidade temporal (cauda inferior — valores
pequenos de tempo-até-evento coocorrendo), não magnitude conjunta grande
(cauda superior). O batching reforça esse argumento: β reduz a amplitude dos
picos mas comprime o agrupamento temporal numa janela fixa a partir do
timestamp original, preservando o sinal de proximidade temporal mesmo sob
evasão adversarial (ver ``_JANELA_FRAGMENTACAO_FIXA``/``aplicar_batching``).
"""

from __future__ import annotations

import numpy as np
from copulas.bivariate.clayton import Clayton

RandomState = int | np.random.Generator | None

_JANELA_FRAGMENTACAO_FIXA: float = 12 / 3600
"""Largura fixa da janela de fragmentação de ``aplicar_batching``, em
timesteps (1 timestep = 1 hora, ver docstring de ``delta_t`` em
``ElectionModel``) — ``12/3600 ≈ 0.00333`` timesteps.

12 segundos = tempo de bloco/slot do Ethereum pós-Merge sob Proof-of-Stake
(determinístico por slot, especificação oficial do protocolo) — referência
de tempo real de rede escolhida pelos orientadores (reunião 2026-09) para
calibrar a granularidade de evasão do adversário, substituindo a fórmula
anterior ``delta_t/beta``. A fórmula anterior fazia a janela ENCOLHER
conforme β crescia, produzindo amplitude de pico não-monotônica em β (achado
documentado em CLAUDE.md) — a janela fixa, por não depender de β nem de
delta_t, remove essa causa raiz sem alterar o critério de validação do
Sanity Check 3 (degradação monotônica do F1 conforme β aumenta continua
sendo a expectativa, a reverificar quando o detector existir).

Não confundir com a granularidade de bucketização de Fonte A (1 timestep =
1 hora, ``int(np.floor(t))`` em ``fonte_a_eventos_fronteira``/
``gerar_fonte_a_normal``) — esta constante opera sobre o timestamp
CONTÍNUO, antes da bucketização; o schema do HDF5 (coluna ``timestep``,
inteiro) não é afetado por esta mudança."""


def _pseudo_observacoes(timestamps: np.ndarray, janela: float) -> np.ndarray:
    """Normaliza timestamps para (0, 1) dentro da janela de observação.

    Levanta ``ValueError`` se ``janela <= 0`` — ``timestamps / janela``
    com ``janela == 0`` produz ``NaN`` silencioso (``0/0``), que o
    ``np.clip`` seguinte NÃO filtra (``NaN`` falha as duas comparações do
    clip e atravessa sem alteração). Defesa em profundidade: o chamador
    (``gerar_fonte_b``) já trata ``janela <= 0`` antes de chegar aqui —
    este guard protege qualquer caminho futuro que esqueça de fazer o
    mesmo, não deveria disparar no fluxo normal.
    """
    if janela <= 0:
        raise ValueError(
            "janela de observação não pode ser <= 0 — chamador deveria ter tratado esse caso antes de chegar aqui."
        )
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

    **Caso degenerado — ``janela <= 0`` (ex. Δt=0h, PLANO §5.2.2,
    ``Δt ∈ {0h, 2h, 24h}``): Fonte B = cópia de Fonte A, cópula não é
    chamada.** Com janela de largura zero, todos os timestamps de Fonte A
    coincidem no mesmo instante — massa pontual sem variância. τ_Kendall
    (medida de dependência baseada em postos) não é estatisticamente
    definível nesse caso — não há variação para medir correlação entre A
    e B. Retornar Fonte B idêntica a Fonte A é a única leitura
    matematicamente consistente de "sem janela de observação": não é uma
    escolha arbitrária de negócio, é a ausência de qualquer alternativa
    bem definida (achado: sem esse tratamento, ``_pseudo_observacoes``
    calcula ``0/0 = NaN``, corrompendo toda a coluna — visto empiricamente
    em 22% dos arquivos de uma rodada de validação de pipeline com
    ``delta_t=0.0`` e contrato ativado; ver CLAUDE.md).

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

    if janela <= 0:
        # Massa pontual (todos os timestamps coincidem) -- tau_Kendall
        # indefinido, sem variação para medir dependência. Fonte B = Fonte A.
        return fonte_a_timestamps.copy()

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
    r"""Fragmenta eventos de Fonte A em β saques distribuídos numa janela fixa.

    Estratégia de evasão do adversário: em vez de um único evento, fragmenta
    em β sub-eventos, reduzindo a amplitude do pico observável por fator 1/β
    e comprimindo o agrupamento temporal na janela
    ``[0, _JANELA_FRAGMENTACAO_FIXA]`` a partir do timestamp original — largura
    FIXA, independente de β e de ``delta_t`` (ver docstring de
    ``_JANELA_FRAGMENTACAO_FIXA`` para a calibração em 12s/tempo de bloco
    Ethereum pós-Merge e o motivo da mudança em relação à fórmula anterior
    ``delta_t/beta``, que fazia a janela encolher conforme β crescia).

    Parameters
    ----------
    timestamps : np.ndarray
        Timestamps originais dos eventos de Fonte A (não fragmentados, β=1).
    delta_t : float
        Atraso de divulgação eleitoral (Δt) — mantido como parâmetro
        obrigatório por compatibilidade de assinatura com o resto do
        pipeline (`ElectionModel.resolver_desembolso`), mas NÃO determina
        mais a largura da janela de fragmentação (ver
        ``_JANELA_FRAGMENTACAO_FIXA``) — os timestamps originais (que já
        vivem em ``[0, delta_t]``, amostrados por
        ``_amostrar_timestamps_desembolso``) são usados como offset base
        dos sub-eventos, não `delta_t` em si.
    beta : int
        Número de saques fragmentados por evento original — determina
        quantos sub-eventos, não mais a largura da janela em que caem.
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

    offsets = rng.uniform(0, _JANELA_FRAGMENTACAO_FIXA, size=(timestamps.size, beta))
    fragmentados = timestamps[:, None] + offsets
    return fragmentados.ravel()
