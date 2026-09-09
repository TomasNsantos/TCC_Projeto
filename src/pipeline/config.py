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
import math
from dataclasses import dataclass, field

import numpy as np

N_JANELAS_POR_CLASSE_PADRAO: int = 1000
"""Janelas por classe (PLANO §5.2.4: "1.000 janelas... por classe") — único
valor nesta tarefa com lastro direto no PLANO; todo o resto em
``ParametrosPopulacionaisStub``/``ParametrosStubGeracao`` é stub v0 sem
calibração (ver CLAUDE.md)."""

_N_SECOES_DEFAULT: int = 5
"""Default de ``ParametrosPopulacionaisStub.n_secoes`` — extraído como
constante de módulo (não só um literal no default do campo) porque também
serve de referência única para a checagem de ambiguidade em
``eleitores_por_secao``: é o valor contra o qual ``__post_init__`` compara
para decidir se ``n_secoes`` "foi deixado no default" ou "foi setado
explicitamente". Uma dataclass não distingue "passado explicitamente com o
mesmo valor do default" de "não passado" — comparar contra este literal é
uma limitação conhecida e aceita (ver docstring de
``ParametrosPopulacionaisStub``), não contornada com um sentinel/Optional
mais complexo."""

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
    pi : list[float]
        Nível de privacidade (π), primeiro eixo do fatorial principal do
        PLANO ("π × g × Δt × λ × ρ", §5.2.2) — valores tipicamente em
        ``{0.50, 0.75, 0.90, 0.95}`` conforme o PLANO, mas sem default
        aqui: é o caller que decide os valores, mesma convenção de
        ``g``/``rho``/etc. Não confundir com o π já consumido por
        ``ElectionModel``/``gerar_fonte_a_normal`` (tarefas anteriores) —
        este campo só expõe π como eixo do GRID; a conexão entre uma
        combinação expandida e a construção real de ``ElectionModel``
        continua acontecendo em ``src/pipeline/geracao.py``, que ainda não
        lê a chave ``"pi"`` do dict retornado por ``expandir_grade``
        (pendência registrada em CLAUDE.md, não resolvida nesta tarefa).
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
    pi: list[float]
    delta_t: list[float]
    recompensa: list[float]
    rho: list[float]
    beta: list[int]
    seeds: list[int]


@dataclass
class ParametrosPopulacionaisStub:
    """Parâmetros populacionais.

    ``n_agentes``, ``alpha_beta`` e ``prop_racional`` foram travados com
    os orientadores (Item 1, reunião 2026-09): ``alpha_beta=(2.0, 2.0)`` e
    ``prop_racional=0.9`` já coincidem com os defaults abaixo;
    ``n_agentes=500`` (principal) / ``5000`` (sensibilidade) — o default
    desta dataclass agora É ``500`` (cenário principal); o cenário de
    sensibilidade (``5000``) continua sendo um override explícito no
    PONTO DE CHAMADA, não um segundo default aqui. ``eleitores_por_secao``
    também foi travado nessa reunião, ver docstring do campo abaixo.
    ``n_candidatos`` foi travado numa reunião seguinte (Item 2, reunião
    2026-09): ``5`` como valor principal de desenvolvimento (primeira
    rodada) — a faixa 6–10 fica para exploração posterior, ao simular
    sistemas proporcionais, se houver tempo; não implementada nesta
    tarefa, só o default. Um ``n_secoes`` explícito fora do mecanismo de
    ``eleitores_por_secao`` segue stub v0, não calibrado, pendente de
    consenso com orientadores (mesma linguagem geral de CLAUDE.md).

    A maioria dos defaults ainda espelha ``ElectionModel.__init__``
    (`src/generator/layer1_abm/model.py`) — não são uma segunda fonte de
    verdade sobre o valor provisório, são a mesma — EXCETO ``n_agentes``
    e ``n_candidatos``: aqui são ``500``/``5`` (travados pelos Itens 1/2),
    enquanto os defaults de ``ElectionModel`` permanecem ``100``/``1``
    (não alterados, fora do escopo dessas decisões) — divergência
    deliberada, não um descuido.

    Attributes
    ----------
    candidato_alvo : int | None
        Índice do candidato-alvo, em ``[0, n_candidatos)`` quando ``int``.
        ``None`` significa "sortear o candidato-alvo por janela/cenário"
        — funcionalidade ainda não implementada aqui (fica para uma tarefa
        futura em ``src/pipeline/geracao.py``); esta tarefa só amplia o
        TIPO aceito, sem lógica de sorteio. Um ``int`` explícito preserva
        exatamente o comportamento atual — mesmo princípio de
        retrocompatibilidade estrita v0 já usado por ``pi=0.0``
        (`src/generator/privacidade.py`/`ElectionModel`) e ``beta=1``
        (`layer2_copula.aplicar_batching`): o valor por si só não muda
        nada até que o código consumidor (`geracao.py`) seja atualizado
        para reconhecer ``None`` e agir sobre ele. Default continua ``0``,
        idêntico ao de antes desta tarefa.
    eleitores_por_secao : int | None
        Forma PREFERENCIAL de configurar ``n_secoes`` de forma realista —
        aproximação de seções eleitorais brasileiras (~300–400
        eleitores/seção), decisão confirmada com os orientadores (Item 1,
        reunião de 2026-09) para uso EXCLUSIVO no cenário de sensibilidade
        (``n_agentes=5000``); no cenário principal (``n_agentes=500``),
        este campo deve ser deixado em ``None`` e ``n_secoes`` permanece
        fixo no valor default. Não hardcodado aqui como default diferente
        de ``None`` — o valor exato dentro da faixa 300–400 é escolhido no
        PONTO DE CHAMADA, não nesta dataclass. Quando setado (não
        ``None``), recalcula ``n_secoes = ceil(n_agentes / eleitores_por_secao)``
        em ``__post_init__`` (mesmo padrão de ``math.ceil`` já usado em
        ``ElectionModel`` para ``n_municipios``/``n_estados`` — nunca
        ``round()``/``floor()``). Mutuamente exclusivo com um ``n_secoes``
        explícito diferente do default (``_N_SECOES_DEFAULT``): setar os
        dois de forma ambígua (`n_secoes` explícito != default E
        `eleitores_por_secao` setado) levanta ``ValueError`` em
        ``__post_init__`` — ver `Raises` abaixo, e a limitação conhecida
        documentada em `_N_SECOES_DEFAULT`. Um ``n_secoes`` explícito
        continua válido para uso direto/testes onde o acoplamento com
        ``n_agentes`` não é relevante.

    Raises
    ------
    ValueError
        Se ``eleitores_por_secao is not None`` e ``n_secoes !=
        _N_SECOES_DEFAULT`` (os dois campos setados de forma ambígua —
        ver `eleitores_por_secao` acima).
    """

    n_agentes: int = 500
    alpha_beta: tuple[float, float] = (2.0, 2.0)
    prop_racional: float = 0.9
    n_secoes: int = _N_SECOES_DEFAULT
    n_candidatos: int = 5
    candidato_alvo: int | None = 0
    prob_conformidade: float = 1.0
    eleitores_por_secao: int | None = None

    def __post_init__(self) -> None:
        if self.eleitores_por_secao is not None:
            if self.n_secoes != _N_SECOES_DEFAULT:
                raise ValueError(
                    "n_secoes e eleitores_por_secao não podem ser setados simultaneamente de forma "
                    f"ambígua: n_secoes={self.n_secoes!r} (diferente do default {_N_SECOES_DEFAULT!r}) "
                    f"e eleitores_por_secao={self.eleitores_por_secao!r} foram passados juntos. "
                    "Escolha um dos dois: deixe n_secoes no default para que eleitores_por_secao "
                    "determine n_secoes, ou deixe eleitores_por_secao em None e configure n_secoes "
                    "diretamente."
                )
            self.n_secoes = math.ceil(self.n_agentes / self.eleitores_por_secao)


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
    """Produto cartesiano de ``g × pi × delta_t × recompensa × rho × seeds``.

    β fica de fora do produto (ver docstring do módulo e de
    ``GradeFatorial.beta``) — toda combinação retornada recebe ``beta=1``,
    o baseline do fatorial principal.

    Parameters
    ----------
    grade : GradeFatorial

    Returns
    -------
    list[dict]
        Uma combinação por dict, chaves ``g``, ``pi``, ``delta_t``,
        ``recompensa``, ``rho``, ``beta`` (sempre ``1``), ``seed``.
    """
    combinacoes = itertools.product(grade.g, grade.pi, grade.delta_t, grade.recompensa, grade.rho, grade.seeds)
    return [
        {"g": g, "pi": pi, "delta_t": delta_t, "recompensa": recompensa, "rho": rho, "beta": 1, "seed": seed}
        for g, pi, delta_t, recompensa, rho, seed in combinacoes
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
