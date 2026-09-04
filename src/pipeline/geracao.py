"""Implementação real de `gerar_par_de_classes`: chama o gerador de verdade
(Camada 1 + Camada 2 + `adversarial_mode`/`normal_mode`) para cada janela e
grava o resultado em HDF5. `runner.py` não importa este módulo — só depende
da interface `GerarParDeClasses`; a ligação entre os dois é feita por
`criar_gerador_real`.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from src.generator.adversarial_mode import gerar_cenario_adversarial
from src.generator.layer1_abm import ElectionModel
from src.generator.normal_mode import gerar_cenario_normal
from src.pipeline.config import ParametrosPopulacionaisStub, ParametrosStubGeracao, derivar_seeds, run_id
from src.pipeline.runner import GerarParDeClasses
from src.pipeline.storage import JanelaNegativa, JanelaPositiva, escrever_run_hdf5

_EIXOS_GRADE = ("g", "pi", "delta_t", "recompensa", "rho", "beta")


def _seed_sequence_para_int(seed_sequence: np.random.SeedSequence) -> int:
    """Deriva um int simples e determinístico de um `SeedSequence`.

    Achado durante a implementação: `copulas.bivariate.clayton.Clayton`
    (usada por `gerar_fonte_b` quando `tau_kendall != 0`) só aceita
    `int`/`np.random.RandomState` para `random_state` — rejeita
    `SeedSequence` e até `np.random.Generator` diretamente (validação
    própria da lib `copulas`, mais restrita que a de `layer2_copula.gerar_fonte_b`,
    que aceita qualquer coisa que `np.random.default_rng` aceite). Nenhum
    teste anterior pegou isso porque nenhum usava `SeedSequence` como seed
    até este pipeline existir. Contorno local (não altero `copula.py`,
    módulo já testado de uma tarefa anterior): derivo um `int` determinístico
    via `SeedSequence.generate_state`, que ambas as APIs aceitam.
    """
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def gerar_par_de_classes_real(
    params: dict,
    seed: int,
    n_janelas: int,
    populacionais: ParametrosPopulacionaisStub,
    stub_geracao: ParametrosStubGeracao,
    diretorio_output: Path,
) -> dict:
    r"""Gera `n_janelas` janelas de cada classe (positiva/negativa) e grava em HDF5.

    Implementação real da interface `GerarParDeClasses` (`runner.py`).
    `params` é lido só pelas chaves de eixo de grade (``g``, ``pi``,
    ``delta_t``, ``recompensa``, ``rho``, ``beta``) — mesmo que venha com
    chaves extras
    de `populacionais`/`stub_geracao` misturadas (como acontece quando
    chamada via `orquestrar()`, que funde tudo num dict só antes de
    invocar a função injetada), essas chaves extras são ignoradas: os
    objetos `populacionais`/`stub_geracao` recebidos separadamente são a
    única fonte de verdade para tudo que não é eixo de grade, evitando
    ambiguidade entre as duas representações do mesmo valor.

    Mapeamento ``g`` → ``granularidade``/``unidade_alvo`` de `ElectionModel`
    (decisão v0, pendente de revisão — ver CLAUDE.md): ``g`` vira
    `granularidade` diretamente; quando `!= "pool"`, `unidade_alvo=0` —
    stub, sempre válido (toda config populacional tem pelo menos 1
    seção/município/estado), mas não é uma escolha definitiva de qual
    unidade mirar.

    ``resultado_alvo``/``threshold_range`` de `ElectionModel` não são
    parâmetros de `GradeFatorial`/`ParametrosPopulacionaisStub` (tarefa
    anterior) — ficam no default da própria classe (`0.5`/`(0.2, 0.8)`).
    Limite de escopo desta camada de config, não esquecimento; documentado
    em CLAUDE.md.

    Fase de derivação de seeds: `derivar_seeds(seed, n_janelas, "positiva"/
    "negativa")` dá exatamente 2 sub-seeds por janela
    (`seed_modelo`, `seed_fonte_b`) — suficiente para a classe positiva
    (`ElectionModel(seed=seed_modelo)` + `gerar_cenario_adversarial(...,
    random_state_fonte_b=seed_fonte_b)`), mas a classe negativa precisa de
    uma TERCEIRA fonte de aleatoriedade independente
    (`gerar_cenario_normal`'s `random_state_fonte_a`). Reusar `seed_modelo`
    diretamente para isso acoplaria deterministicamente o RNG interno do
    `ElectionModel` (Fonte C) ao gerador de tráfego de Fonte A — dois
    `np.random.default_rng` semeados com o MESMO valor compartilham o
    stream de bits subjacente (PCG64), mesmo chamando métodos diferentes;
    isso introduziria um canal de correlação espúria C↔A que nenhum sanity
    check existente cobre (SC1 mede desempenho sob λ=0; SC4 mede A↔B).
    Por isso a seed de Fonte A da classe negativa é derivada LOCALMENTE,
    aqui, via ``seed_modelo.spawn(1)[0]`` — mesmo mecanismo de
    `SeedSequence.spawn` já usado no resto do pipeline, sem alterar o
    contrato de `derivar_seeds` (que continua devolvendo só o par).

    **Quarta fonte de aleatoriedade: a máscara de privacidade π.** Mesmo
    raciocínio acima, agora aplicado a `random_state_pi` (consumido por
    `mascara_sobrevivencia_pi` dentro de `gerar_cenario_adversarial`/
    `gerar_cenario_normal`): reusar `seed_fonte_b` ou `seed_modelo`
    diretamente para isso acoplaria deterministicamente dois RNGs que
    deveriam ser independentes — na classe positiva, `seed_modelo` já
    alimenta `ElectionModel(seed=seed_modelo)`; na negativa, `seed_modelo`
    já alimenta `ElectionModel` E `seed_fonte_a_negativa` também deriva
    dela. Por isso `seed_pi_positiva`/`seed_pi_negativa` são derivadas
    LOCALMENTE, aqui, via `seed_modelo.spawn(1)[0]` — chamada adicional de
    `.spawn()` sobre a mesma `seed_modelo` já usada para
    `seed_fonte_a_negativa`; cada chamada de `.spawn()` devolve um filho
    diferente e independente via `spawn_key` interno incrementado, então
    chamar de novo sobre a mesma `seed_modelo` é seguro e não colide com
    `seed_fonte_a_negativa`.

    **Nota sobre `_seed_sequence_para_int`:** só `seed_fonte_b` passa por
    essa conversão antes de chegar a `gerar_cenario_adversarial`/
    `gerar_cenario_normal`, pela restrição já documentada da lib `copulas`
    (só aceita `int`/`RandomState`, usada quando `tau_kendall != 0`).
    `seed_fonte_a_negativa`, `seed_pi_positiva` e `seed_pi_negativa` usam
    `SeedSequence` diretamente, SEM essa conversão — `mascara_sobrevivencia_pi`
    (assim como `gerar_fonte_a_normal`) flui para `np.random.default_rng`,
    que aceita `SeedSequence` nativamente. Aplicar `_seed_sequence_para_int`
    nessas três sementes seria inofensivo mas criaria uma inconsistência
    de padrão sem motivo real, sugerindo a um leitor futuro que
    `mascara_sobrevivencia_pi` compartilha a restrição da lib `copulas` —
    não compartilha.

    Falhas são capturadas por janela individual, não abortam o lote — só
    propaga (`RuntimeError`) se TODAS as janelas de uma classe falharem
    (sinal de erro sistemático, não pontual).

    Parameters
    ----------
    params : dict
        Combinação da grade — só ``g``/``pi``/``delta_t``/``recompensa``/
        ``rho``/``beta`` são lidos.
    seed : int
    n_janelas : int
        Por classe (positiva e negativa geram `n_janelas` cada).
    populacionais : ParametrosPopulacionaisStub
    stub_geracao : ParametrosStubGeracao
    diretorio_output : Path
        Diretório onde o HDF5 é gravado (criado se não existir). O nome do
        arquivo é ``f"{run_id}.h5"``, recomputado aqui só a partir dos
        eixos de grade — consistente com o `run_id` que `orquestrar()` já
        usa para esta mesma combinação no manifesto.

    Returns
    -------
    dict
        ``{"n_janelas_ok", "n_janelas_falha", "n_contrato_nao_ativado",
        "caminho_output"}`` — formato que `orquestrar()` espera.

    Raises
    ------
    ValueError
        Se ``populacionais.n_candidatos <= 1`` — checado logo no topo,
        antes de qualquer geração de janela (`gerar_cenario_normal` exige
        `n_candidatos > 1`; sem essa checagem antecipada, o erro só
        apareceria depois de já ter gerado as janelas positivas inteiras).
    RuntimeError
        Se todas as janelas de uma classe falharem.
    """
    if populacionais.n_candidatos <= 1:
        raise ValueError(
            "populacionais.n_candidatos deve ser > 1 para gerar a classe negativa "
            f"(recebido {populacionais.n_candidatos}) — gerar_cenario_normal exige voto de "
            "base ativo (n_candidatos=1 degenera o resultado para zero em todo lugar). "
            "Ajuste ParametrosPopulacionaisStub antes de rodar o pipeline."
        )

    granularidade = params["g"]
    unidade_alvo = 0 if granularidade != "pool" else None

    seeds_positivas = derivar_seeds(seed, n_janelas, "positiva")
    seeds_negativas = derivar_seeds(seed, n_janelas, "negativa")

    janelas_positivas: list[JanelaPositiva] = []
    falhas_positivas = 0
    ultimo_erro_positiva: Exception | None = None
    for window_id, (seed_modelo, seed_fonte_b) in enumerate(seeds_positivas):
        try:
            modelo = ElectionModel(
                n_agentes=populacionais.n_agentes,
                alpha_beta=populacionais.alpha_beta,
                prop_racional=populacionais.prop_racional,
                n_secoes=populacionais.n_secoes,
                n_candidatos=populacionais.n_candidatos,
                candidato_alvo=populacionais.candidato_alvo,
                prob_conformidade=populacionais.prob_conformidade,
                recompensa=params["recompensa"],
                delta_t=params["delta_t"],
                rho=params["rho"],
                beta=params["beta"],
                pi=params["pi"],
                granularidade=granularidade,
                unidade_alvo=unidade_alvo,
                seed=seed_modelo,
            )
            seed_pi_positiva = seed_modelo.spawn(1)[0]
            cenario = gerar_cenario_adversarial(
                modelo,
                stub_geracao.tau_kendall,
                _seed_sequence_para_int(seed_fonte_b),
                random_state_pi=seed_pi_positiva,
            )
            janelas_positivas.append((window_id, cenario, seed_modelo, seed_fonte_b))
        except Exception as erro:  # noqa: BLE001 - falha por janela nao aborta o lote
            falhas_positivas += 1
            ultimo_erro_positiva = erro

    if falhas_positivas == n_janelas:
        raise RuntimeError(
            f"Todas as {n_janelas} janelas da classe positiva falharam — erro sistemático, "
            f"não pontual. Último erro: {ultimo_erro_positiva}"
        )

    janelas_negativas: list[JanelaNegativa] = []
    falhas_negativas = 0
    ultimo_erro_negativa: Exception | None = None
    for window_id, (seed_modelo, seed_fonte_b) in enumerate(seeds_negativas):
        try:
            modelo = ElectionModel(
                n_agentes=populacionais.n_agentes,
                alpha_beta=populacionais.alpha_beta,
                prop_racional=populacionais.prop_racional,
                n_secoes=populacionais.n_secoes,
                n_candidatos=populacionais.n_candidatos,
                candidato_alvo=populacionais.candidato_alvo,
                prob_conformidade=populacionais.prob_conformidade,
                recompensa=0.0,
                delta_t=params["delta_t"],
                rho=params["rho"],
                beta=params["beta"],
                pi=params["pi"],
                granularidade=granularidade,
                unidade_alvo=unidade_alvo,
                seed=seed_modelo,
            )
            seed_fonte_a_negativa = seed_modelo.spawn(1)[0]
            seed_pi_negativa = seed_modelo.spawn(1)[0]
            cenario = gerar_cenario_normal(
                modelo,
                janela=params["delta_t"],
                taxa_fonte_a=stub_geracao.taxa_fonte_a,
                volume_medio_fonte_a=stub_geracao.volume_medio_fonte_a,
                taxa_fonte_b=stub_geracao.taxa_fonte_b,
                random_state_fonte_a=seed_fonte_a_negativa,
                random_state_fonte_b=seed_fonte_b,
                random_state_pi=seed_pi_negativa,
            )
            janelas_negativas.append((window_id, cenario, seed_modelo, seed_fonte_b))
        except Exception as erro:  # noqa: BLE001 - falha por janela nao aborta o lote
            falhas_negativas += 1
            ultimo_erro_negativa = erro

    if falhas_negativas == n_janelas:
        raise RuntimeError(
            f"Todas as {n_janelas} janelas da classe negativa falharam — erro sistemático, "
            f"não pontual. Último erro: {ultimo_erro_negativa}"
        )

    eixos_grade = {eixo: params[eixo] for eixo in _EIXOS_GRADE}
    diretorio_output.mkdir(parents=True, exist_ok=True)
    caminho_output = diretorio_output / f"{run_id(eixos_grade, seed)}.h5"

    escrever_run_hdf5(caminho_output, janelas_positivas, janelas_negativas)

    n_contrato_nao_ativado = sum(1 for _, cenario, _, _ in janelas_positivas if not cenario.contrato_ativado)

    return {
        "n_janelas_ok": len(janelas_positivas) + len(janelas_negativas),
        "n_janelas_falha": falhas_positivas + falhas_negativas,
        "n_contrato_nao_ativado": n_contrato_nao_ativado,
        "caminho_output": str(caminho_output),
    }


def criar_gerador_real(
    populacionais: ParametrosPopulacionaisStub,
    stub_geracao: ParametrosStubGeracao,
    diretorio_output: Path,
) -> GerarParDeClasses:
    """Fecha `populacionais`/`stub_geracao`/`diretorio_output` via
    `functools.partial`, devolvendo um callable de 3 argumentos
    (`params, seed, n_janelas`) compatível com `runner.orquestrar()`.

    Não pedido explicitamente, mas necessário para ligar
    `gerar_par_de_classes_real` (6 parâmetros, bom para teste direto) ao
    contrato `GerarParDeClasses` (3 parâmetros) que `orquestrar()` já
    espera.
    """
    return functools.partial(
        gerar_par_de_classes_real,
        populacionais=populacionais,
        stub_geracao=stub_geracao,
        diretorio_output=diretorio_output,
    )
