"""Laço de orquestração: expande a grade, consulta o manifesto e decide o
que rodar/pular/repetir. Não chama o gerador real — `gerar_par_de_classes`
é injetada, para que o teste desta tarefa passe um fake sem importar
`ElectionModel`. A implementação real (chamando
`gerar_cenario_adversarial`/`gerar_cenario_normal` e gravando HDF5) fica
para uma tarefa futura.
"""

from __future__ import annotations

from typing import Callable

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
    combinacoes = expandir_grade(grade)

    combinacoes_por_run_id: dict[str, dict] = {}
    for combinacao in combinacoes:
        seed = combinacao["seed"]
        params = {k: v for k, v in combinacao.items() if k != "seed"}
        rid = run_id(params, seed)
        combinacoes_por_run_id[rid] = combinacao
        manifesto.registrar_pending(rid, combinacao, seed)

    pendentes = set(manifesto.pendentes())

    for rid, combinacao in combinacoes_por_run_id.items():
        if rid not in pendentes:
            continue

        seed = combinacao["seed"]
        params = {k: v for k, v in combinacao.items() if k != "seed"}
        params_completos = {**params, **vars(parametros_stub), **vars(populacionais)}

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

