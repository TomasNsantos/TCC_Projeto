"""Configuração da grade fatorial: dataclasses, expansão em combinações e seeds.

Camada de configuração pura — não chama nenhum dos geradores
(`ElectionModel`, `gerar_cenario_adversarial`, `gerar_cenario_normal`). O
runner que efetivamente executa cada combinação fica para uma tarefa futura.

Design fatorial (PLANO §5.2.2, §5.4.3): β ∈ {1, 5, 20} é explicitamente
listado como "experimento de robustez", separado dos outros cinco eixos do
fatorial principal (π × g × Δt × λ × ρ) — §5.4.3 confirma que o experimento
de β é "separado do fatorial principal, fixando a configuração de melhor
desempenho" dele. Por isso ``expandir_grade`` (grid principal) e
``expandir_grade_robustez`` (experimento de β) são duas funções distintas,
não uma só com uma flag — a separação reflete a estrutura do experimento no
PLANO, não é uma escolha de implementação.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

N_JANELAS_POR_CLASSE_PADRAO: int = 1000
"""Janelas por classe (PLANO §5.2.4: "1.000 janelas... por classe") — único
valor nesta tarefa com lastro direto no PLANO; todo o resto em
``ParametrosPopulacionaisStub``/``ParametrosStubGeracao`` é stub v0 sem
calibração (ver CLAUDE.md)."""

_CASAS_DECIMAIS_RUN_ID: int = 4
"""Casas decimais fixas usadas para formatar floats em ``run_id`` — garante
que a string não varie por causa de representação de ponto flutuante."""

_CLASSES_VALIDAS = ("positiva", "negativa")


@dataclass
class GradeFatorial:
    """Eixos do design fatorial principal (PLANO §5.2.2), exceto β.

    Attributes
    ----------
    g : list[str]
        Rótulos de granularidade eleitoral. O mapeamento exato para
        ``ElectionModel.granularidade``/``unidade_alvo`` — que tem um
        quarto valor, ``"pool"``, fora do ``g ∈ {seção, município, estado}``
        do PLANO — fica para quando o runner existir; não decidido aqui.
    delta_t : list[float]
        Atraso de divulgação (Δt do design fatorial).
    recompensa : list[float]
        Proxy stub de λ (intensidade adversarial) já usado em
        ``ElectionModel`` — não posso nomear o campo ``lambda``, é palavra
        reservada em Python. Sem valor calibrado, mesma ressalva do resto
        do projeto (ver CLAUDE.md).
    rho : list[float]
        Grau de coordenação (ρ).
    beta : list[int]
        Fragmentação (β). Existe como campo aqui para completude da grade,
        mas **não é cruzado** por ``expandir_grade`` — β só varia no
        experimento de robustez (``expandir_grade_robustez``, ver docstring
        do módulo). Não fica sem uso por descuido; é a estrutura do PLANO.
    seeds : list[int]
        Sementes de topo, uma árvore de derivação por seed (ver
        ``derivar_seeds``).
    """

    g: list[str]
    delta_t: list[float]
    recompensa: list[float]
    rho: list[float]
    beta: list[int]
    seeds: list[int]


@dataclass
class ParametrosPopulacionaisStub:
    """Parâmetros populacionais — stub v0, não calibrado, pendente de
    consenso com orientadores (mesma linguagem de CLAUDE.md).

    Todos os defaults são copiados literalmente dos defaults de
    ``ElectionModel.__init__`` (`src/generator/layer1_abm/model.py`) — não
    são uma segunda fonte de verdade sobre o valor provisório, são a mesma.
    """

    n_agentes: int = 100
    alpha_beta: tuple[float, float] = (2.0, 2.0)
    prop_racional: float = 0.9
    n_secoes: int = 5
    n_candidatos: int = 1
    candidato_alvo: int = 0
    prob_conformidade: float = 1.0


@dataclass
class ParametrosStubGeracao:
    """Parâmetros de geração de Fonte A/B — stub v0, sem default (nenhum dos
    quatro), forçando quem constrói a reconhecer explicitamente que são
    valores não calibrados, em vez de deixar passar despercebido.

    Attributes
    ----------
    tau_kendall : float
        Força-alvo de dependência A↔B na cópula Clayton
        (``layer2_copula.gerar_fonte_b``). Independente de ρ — ρ controla a
        concentração temporal do timing de desembolso (Fase 2 de
        ``ElectionModel``), não a dependência estatística entre fontes.
        Pendência de validação com o Prof. Alexandre (mesma nota já
        registrada em ``adversarial_mode/cenario.py`` e em CLAUDE.md).
    taxa_fonte_a, volume_medio_fonte_a, taxa_fonte_b : float
        Parâmetros de ``normal_mode.gerar_fonte_a_normal``/
        ``gerar_fonte_b_normal`` — mesma convenção de "sem default,
        obrigatório" já usada nessas funções.
    """

    tau_kendall: float
    taxa_fonte_a: float
    volume_medio_fonte_a: float
    taxa_fonte_b: float


@dataclass
class RobustezBeta:
    """Experimento de robustez adversarial (PLANO §5.4.3): fixa uma única
    configuração-base do grid principal e varia só β (e a seed).

    Attributes
    ----------
    base_config : dict
        Uma combinação já expandida do grid principal (uma linha de
        ``expandir_grade``) — "a configuração de melhor desempenho", per
        §5.4.3. Obrigatório: não há valor do PLANO para qual configuração
        usar.
    seeds : list[int]
        Sementes para o experimento de robustez. Obrigatório, mesma razão.
    beta : list[int]
        Default ``[1, 5, 20]`` — não é um valor inventado, é o literal do
        PLANO (§5.2.2, §5.4.3).
    """

    base_config: dict
    seeds: list[int]
    beta: list[int] = field(default_factory=lambda: [1, 5, 20])


def expandir_grade(grade: GradeFatorial) -> list[dict]:
    """Produto cartesiano de ``g × delta_t × recompensa × rho × seeds``.

    β fica de fora do produto (ver docstring do módulo e de
    ``GradeFatorial.beta``) — toda combinação retornada recebe ``beta=1``,
    o baseline do fatorial principal.

    Parameters
    ----------
    grade : GradeFatorial

    Returns
    -------
    list[dict]
        Uma combinação por dict, chaves ``g``, ``delta_t``, ``recompensa``,
        ``rho``, ``beta`` (sempre ``1``), ``seed``.
    """
    combinacoes = itertools.product(grade.g, grade.delta_t, grade.recompensa, grade.rho, grade.seeds)
    return [
        {"g": g, "delta_t": delta_t, "recompensa": recompensa, "rho": rho, "beta": 1, "seed": seed}
        for g, delta_t, recompensa, rho, seed in combinacoes
    ]


def expandir_grade_robustez(robustez: RobustezBeta) -> list[dict]:
    """Produto cartesiano de ``beta × seeds`` em torno de ``robustez.base_config``.

    Parameters
    ----------
    robustez : RobustezBeta

    Returns
    -------
    list[dict]
        Uma combinação por dict — todos os campos de ``base_config`` mais
        ``beta``/``seed`` daquela linha (sobrescrevendo qualquer ``beta``/
        ``seed`` já presente em ``base_config``, já que são os eixos
        variando neste experimento).
    """
    combinacoes = itertools.product(robustez.beta, robustez.seeds)
    return [{**robustez.base_config, "beta": beta, "seed": seed} for beta, seed in combinacoes]


def _formatar_valor(valor: object) -> str:
    if isinstance(valor, float):
        return f"{valor:.{_CASAS_DECIMAIS_RUN_ID}f}"
    return str(valor)


def run_id(params: dict, seed: int) -> str:
    """Identificador determinístico e legível de uma combinação de parâmetros.

    Formato ``"chave1-valor1_chave2-valor2..._seed-N"``, chaves ordenadas
    (independente da ordem de inserção do dict) e floats formatados com
    ``_CASAS_DECIMAIS_RUN_ID`` casas decimais fixas — dois floats que só
    diferem por erro de representação de ponto flutuante (ex. ``0.1 + 0.2``
    vs. ``0.3``) produzem o mesmo ``run_id``.

    Parameters
    ----------
    params : dict
        Parâmetros escalares (str/int/float/bool) de uma combinação — ex.
        uma linha de ``expandir_grade``/``expandir_grade_robustez``. Não
        lida com valores aninhados (não é o caso de uso aqui).
    seed : int

    Returns
    -------
    str
    """
    partes = [f"{chave}-{_formatar_valor(valor)}" for chave, valor in sorted(params.items())]
    return "_".join(partes) + f"_seed-{seed}"


def derivar_seeds(seed: int, n_janelas: int, classe: str) -> list[tuple[np.random.SeedSequence, np.random.SeedSequence]]:
    """Hierarquia de três níveis de seeds: raiz por classe → uma por janela → (modelo, Fonte B).

    ``SeedSequence([seed, flag])`` como raiz, usando a lista ``[seed, flag]``
    como entropia (não um offset aritmético somado a ``seed``) — é isso que
    garante independência entre as árvores de classe positiva e negativa,
    nativamente via ``SeedSequence``. A raiz gera ``n_janelas`` sub-seeds
    (``.spawn``); cada uma gera, por sua vez, exatamente duas sub-seeds
    (``seed_modelo``, ``seed_fonte_b``) para aquela janela.

    Parameters
    ----------
    seed : int
        Semente de topo (mesma para as duas classes — a independência entre
        classes vem do ``flag``, não de usar seeds de topo diferentes).
    n_janelas : int
    classe : str
        ``"positiva"`` (flag ``0``) ou ``"negativa"`` (flag ``1``).

    Returns
    -------
    list[tuple[np.random.SeedSequence, np.random.SeedSequence]]
        Um par ``(seed_modelo, seed_fonte_b)`` por janela, na ordem
        ``0..n_janelas-1`` — essa ordem define o ``window_id`` no runner,
        precisa ser estável.

    Raises
    ------
    ValueError
        Se ``classe`` não for ``"positiva"`` nem ``"negativa"`` — evita que
        um erro de digitação caia silenciosamente no ramo "negativa".
    """
    if classe not in _CLASSES_VALIDAS:
        raise ValueError(f"classe deve ser um de {_CLASSES_VALIDAS}, recebido {classe!r}.")

    flag = 0 if classe == "positiva" else 1
    raiz = np.random.SeedSequence([seed, flag])

    return [tuple(sub_seed.spawn(2)) for sub_seed in raiz.spawn(n_janelas)]
