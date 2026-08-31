"""Laço de orquestração: expande a grade, consulta o manifesto e decide o
que rodar/pular/repetir. Não chama o gerador real — `gerar_par_de_classes`
é injetada, para que o teste desta tarefa passe um fake sem importar
`ElectionModel`. A implementação real (chamando
`gerar_cenario_adversarial`/`gerar_cenario_normal` e gravando HDF5) fica
para uma tarefa futura.
"""

from __future__ import annotations

import os
from typing import Callable

from joblib import Parallel, delayed

from src.pipeline.config import (
    GradeFatorial,
    ParametrosPopulacionaisStub,
    ParametrosStubGeracao,
    expandir_grade,
    run_id,
)
from src.pipeline.manifest import Manifesto

GerarParDeClasses = Callable[[dict, int, int], dict]
"""Assinatura da função injetada: ``(params, seed, n_janelas) -> dict``.

``params`` é a combinação da grade SEM a chave ``"seed"`` (o valor da seed
já vai separado, mesma convenção de ``config.run_id``), mesclada com
``parametros_stub``/``populacionais``. ``n_janelas`` é o número de janelas
a gerar por classe (positiva e negativa).

Espera-se que a função gere as janelas, grave o resultado (HDF5, em tarefa
futura) e retorne um dict com as chaves ``n_janelas_ok``, ``n_janelas_falha``,
``n_contrato_nao_ativado`` e ``caminho_output`` — ou levante qualquer
exceção, que ``orquestrar`` captura e registra como ``"failed"`` sem
abortar as combinações restantes.
"""


def orquestrar(
    grade: GradeFatorial,
    parametros_stub: ParametrosStubGeracao,
    populacionais: ParametrosPopulacionaisStub,
    n_janelas_por_classe: int,
    manifesto: Manifesto,
    gerar_par_de_classes: GerarParDeClasses,
) -> None:
    r"""Expande a grade e processa cada combinação pendente, de forma resumível.

    Duas passadas: (1) registra TODAS as combinações da grade no manifesto
    via ``registrar_pending`` (idempotente — não afeta linhas já existentes,
    de nenhum status); (2) processa só as combinações cujo ``run_id`` ainda
    está em ``manifesto.pendentes()`` (não é ``"success"``) — isso inclui
    combinações que nunca rodaram, que ficaram ``"running"`` órfãs de uma
    execução anterior interrompida, e que falharam antes.

    Para cada combinação pendente: marca ``"running"``, chama
    ``gerar_par_de_classes``; se retornar normalmente, marca ``"success"``
    com as contagens do retorno; se levantar qualquer exceção, marca
    ``"failed"`` com a mensagem e continua para a próxima combinação — uma
    falha isolada nunca aborta o lote inteiro.

    **Aviso operacional — trocar `parametros_stub`/`populacionais` entre
    execuções não é detectado automaticamente.** O ``run_id`` de cada
    combinação é calculado só a partir dos eixos da grade (``g``,
    ``delta_t``, ``recompensa``, ``rho``, ``beta``, ``seed``) — não inclui
    valores de ``parametros_stub``/``populacionais``, porque eles não
    variam dentro de uma mesma grade e incluí-los tornaria o ``run_id``
    ilegível. Consequência: como esses parâmetros são stub v0 (pendentes de
    calibração, ver CLAUDE.md), se você trocar `tau_kendall`, `n_agentes`
    etc. entre uma execução e outra usando o MESMO arquivo de manifesto,
    todas as combinações já marcadas ``"success"`` com o stub antigo serão
    puladas — os HDF5s gerados com os parâmetros desatualizados continuam
    sendo tratados como atuais, sem aviso. Apague o arquivo do manifesto
    antes de rodar com valores de stub diferentes.

    Parameters
    ----------
    grade : GradeFatorial
    parametros_stub : ParametrosStubGeracao
        Repassado a ``gerar_par_de_classes`` (mesclado em ``params``) — não
        usado para nada além disso nesta tarefa.
    populacionais : ParametrosPopulacionaisStub
        Idem.
    n_janelas_por_classe : int
    manifesto : Manifesto
    gerar_par_de_classes : GerarParDeClasses
    """
    combinacoes_por_run_id = _registrar_grade(grade, manifesto)
    pendentes = set(manifesto.pendentes())

    for rid, combinacao in combinacoes_por_run_id.items():
        if rid not in pendentes:
            continue

        seed = combinacao["seed"]
        params_completos = _params_completos(combinacao, parametros_stub, populacionais)

        manifesto.marcar_running(rid)
        try:
            resultado = gerar_par_de_classes(params_completos, seed, n_janelas_por_classe)
        except Exception as erro:  # noqa: BLE001 - falha isolada não deve abortar o lote
            manifesto.marcar_failed(rid, str(erro))
            continue

        manifesto.marcar_success(
            rid,
            n_janelas_ok=resultado["n_janelas_ok"],
            n_janelas_falha=resultado["n_janelas_falha"],
            n_contrato_nao_ativado=resultado["n_contrato_nao_ativado"],
            caminho_output=resultado["caminho_output"],
        )


def _registrar_grade(grade: GradeFatorial, manifesto: Manifesto) -> dict[str, dict]:
    """Expande a grade e registra cada combinação como ``"pending"``
    (idempotente — ver ``Manifesto.registrar_pending``).

    Compartilhado por ``orquestrar``/``orquestrar_paralelo`` para que as
    duas formas de execução expandam e registrem a grade exatamente da
    mesma maneira.
    """
    combinacoes_por_run_id: dict[str, dict] = {}
    for combinacao in expandir_grade(grade):
        seed = combinacao["seed"]
        params = {k: v for k, v in combinacao.items() if k != "seed"}
        rid = run_id(params, seed)
        combinacoes_por_run_id[rid] = combinacao
        manifesto.registrar_pending(rid, combinacao, seed)
    return combinacoes_por_run_id


def _params_completos(
    combinacao: dict, parametros_stub: ParametrosStubGeracao, populacionais: ParametrosPopulacionaisStub
) -> dict:
    params = {k: v for k, v in combinacao.items() if k != "seed"}
    return {**params, **vars(parametros_stub), **vars(populacionais)}


def _rodar_uma_combinacao(
    rid: str,
    combinacao: dict,
    parametros_stub: ParametrosStubGeracao,
    populacionais: ParametrosPopulacionaisStub,
    n_janelas_por_classe: int,
    gerar_par_de_classes: GerarParDeClasses,
) -> tuple[str, dict]:
    """Roda uma combinação e devolve ``(run_id, resultado)`` — nunca escreve
    no manifesto.

    Chamada dentro de cada processo ``loky`` por ``orquestrar_paralelo``.
    ``resultado`` é ``{"status": "success", **retorno de gerar_par_de_classes}``
    ou ``{"status": "failed", "erro": str(excecao)}`` — nunca propaga a
    exceção, para que ``Parallel`` sempre devolva a lista completa de
    resultados (uma combinação falha não pode derrubar as outras rodando em
    paralelo, mesma regra da versão serial).
    """
    seed = combinacao["seed"]
    params_completos = _params_completos(combinacao, parametros_stub, populacionais)
    try:
        resultado = gerar_par_de_classes(params_completos, seed, n_janelas_por_classe)
    except Exception as erro:  # noqa: BLE001 - falha isolada não deve abortar o lote
        return rid, {"status": "failed", "erro": str(erro)}
    return rid, {"status": "success", **resultado}


def orquestrar_paralelo(
    grade: GradeFatorial,
    parametros_stub: ParametrosStubGeracao,
    populacionais: ParametrosPopulacionaisStub,
    n_janelas_por_classe: int,
    manifesto: Manifesto,
    gerar_par_de_classes: GerarParDeClasses,
    n_jobs: int | None = None,
) -> None:
    r"""Mesmo laço de ``orquestrar``, mas roda cada combinação pendente num
    processo separado via ``joblib.Parallel``.

    A unidade de paralelismo é a COMBINAÇÃO×SEED inteira (uma chamada a
    ``gerar_par_de_classes``, que já processa as 2.000 janelas — positiva +
    negativa — serialmente por dentro), não a janela individual: medido no
    benchmark de uma tarefa anterior (`scripts/benchmark_geracao.py`),
    ~7ms por janela, tempo dominado pelo VOLUME de chamadas (1.635 runs ×
    2.000 janelas no grid completo), não por latência por janela alta o
    suficiente para justificar paralelizar dentro de uma combinação.

    **Por que a escrita no manifesto continua serial, no processo
    principal:** ``Manifesto`` documenta explicitamente que não é
    thread-safe nem process-safe para escrita concorrente — é uma única
    conexão SQLite aberta uma vez no processo principal (ver docstring de
    ``Manifesto`` em ``manifest.py``: "a paralelização (tarefa futura)
    agrega os resultados de cada worker no processo principal antes de
    escrever aqui — os workers não escrevem no manifesto diretamente").
    Se cada worker chamasse ``marcar_running``/``marcar_success``/
    ``marcar_failed`` diretamente: (a) workers ``loky`` são processos
    separados, não enxergam a conexão SQLite já aberta no processo
    principal — precisariam abrir uma conexão própria sobre o mesmo
    arquivo, sob risco real de ``database is locked`` sob escrita
    concorrente do SQLite; (b) mesmo que funcionasse, violaria a garantia
    documentada de "escrita sempre serial". Por isso ``_rodar_uma_combinacao``
    (o worker) só RETORNA ``(run_id, resultado)`` — nunca toca o manifesto
    — e ``orquestrar_paralelo`` escreve todos os resultados serialmente,
    só depois que ``Parallel`` já devolveu a lista completa.

    **Por que ``backend="loky"`` (processos), não threads:**
    ``ElectionModel``/os geradores usam RNGs (``np.random.default_rng``)
    com estado interno mutável por instância — não há estado global
    compartilhado hoje, mas threads no mesmo processo Python correm sob
    risco estrutural de introduzir esse acoplamento no futuro (ex. um RNG
    guardado sem querer numa variável de módulo) sem nenhum aviso de tipo
    ou teste que pegasse isso. Processos separados (``loky``, o backend
    default do joblib) isolam essa possibilidade estruturalmente — cada
    worker tem sua própria memória, não há como um RNG de uma combinação
    vazar para outra rodando "ao mesmo tempo".

    Marca cada combinação pendente como ``"running"`` ANTES de disparar o
    lote paralelo (serial, no processo principal — barato, um ``UPDATE``
    por linha) para que uma interrupção no meio da execução deixe o
    manifesto num estado que ``pendentes()`` já reconhece como retentável
    (mesmo padrão de "running órfã" que ``orquestrar`` já trata).

    Parameters
    ----------
    grade, parametros_stub, populacionais, n_janelas_por_classe, manifesto, gerar_par_de_classes
        Mesmo significado de ``orquestrar``.
    n_jobs : int | None
        Processos paralelos. Default (``None``) usa
        ``max(1, os.cpu_count() - 1)`` — nunca menos que 1, mesmo em
        máquinas de 1 núcleo (``os.cpu_count()`` também pode devolver
        ``None`` em ambientes onde a contagem não é detectável).
    """
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 1) - 1)

    combinacoes_por_run_id = _registrar_grade(grade, manifesto)
    pendentes = set(manifesto.pendentes())

    a_rodar = [(rid, combinacao) for rid, combinacao in combinacoes_por_run_id.items() if rid in pendentes]
    for rid, _ in a_rodar:
        manifesto.marcar_running(rid)

    resultados = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_rodar_uma_combinacao)(rid, combinacao, parametros_stub, populacionais, n_janelas_por_classe, gerar_par_de_classes)
        for rid, combinacao in a_rodar
    )

    for rid, resultado in resultados:
        if resultado["status"] == "failed":
            manifesto.marcar_failed(rid, resultado["erro"])
            continue
        manifesto.marcar_success(
            rid,
            n_janelas_ok=resultado["n_janelas_ok"],
            n_janelas_falha=resultado["n_janelas_falha"],
            n_contrato_nao_ativado=resultado["n_contrato_nao_ativado"],
            caminho_output=resultado["caminho_output"],
        )

