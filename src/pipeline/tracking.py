"""Tracking MLflow do runner: um run do MLflow por `run_id` do lote
(combinação×seed), backend SQLite local — painel consultável via
``mlflow.search_runs()`` sem precisar abrir o `Manifesto` manualmente.

Sistema de tracking paralelo e independente do `Manifesto`
(`src/pipeline/manifest.py`) — não o substitui nem se funde com ele.
`Manifesto` continua sendo a fonte de verdade para resumabilidade
(``pendentes()``); MLflow é só observabilidade.
"""

from __future__ import annotations

from pathlib import Path

import mlflow

_MAX_PARAM_VAL_LENGTH = mlflow.utils.validation.MAX_PARAM_VAL_LENGTH
"""Limite real de `mlflow.log_param` (6000 chars na versão instalada) —
trunco `erro` explicitamente com este valor em vez de deixar o
comportamento de truncamento da lib agir implicitamente (haveria um erro
de validação, não truncamento silencioso, se um valor ultrapassasse o
limite sem tratamento)."""


def configurar_mlflow(caminho_db: Path) -> None:
    """Aponta o tracking URI do MLflow para um backend SQLite local.

    Parameters
    ----------
    caminho_db : Path
        Caminho do arquivo SQLite (criado pelo MLflow se não existir).
    """
    mlflow.set_tracking_uri(f"sqlite:///{caminho_db}")


def registrar_run_mlflow(run_id: str, params: dict, seed: int, resultado: dict) -> None:
    r"""Registra um run do MLflow para uma combinação×seed do lote.

    **Decisão: sucesso E falha geram run do MLflow, não só sucesso.** O
    `Manifesto` já é a fonte de verdade para resumabilidade e já inclui
    falhas em `pendentes()`; se o MLflow só mostrasse sucessos, daria uma
    vista parcial do lote (combinações que falharam simplesmente não
    apareceriam), forçando quem olha o painel a cruzar com o SQLite mesmo
    assim — contrariando o propósito de painel único. Falhas recebem tag
    ``status="failed"`` e a mensagem de erro como parâmetro, em vez de
    serem omitidas.

    **Paridade de parâmetros entre sucesso e falha:** ``params`` deve ser
    o MESMO dict mesclado (eixos da grade + parâmetros populacionais +
    stub de geração) nos dois casos — não uma versão parcial no branch de
    falha. Sem essa paridade, uma linha ``status="failed"`` em
    ``mlflow.search_runs()`` teria colunas de parâmetro faltando/`NaN` que
    uma linha ``status="success"`` tem, impedindo filtrar/comparar as duas
    de forma confiável. É responsabilidade de quem chama esta função
    (``orquestrar``/``orquestrar_paralelo``) garantir essa paridade.

    Parameters
    ----------
    run_id : str
        Usado como ``run_name`` do MLflow — mesmo `run_id` do `Manifesto`
        para essa combinação×seed.
    params : dict
        Parâmetros da combinação (grade + populacionais + stub de geração),
        SEM a chave ``seed`` (vai separada). Valores não-string são
        convertidos via ``str()`` antes de logar — `mlflow.log_params`
        exige valores str-coercíveis, e MLflow armazena parâmetros sempre
        como string internamente de qualquer forma.
    seed : int
    resultado : dict
        ``{"status": "success", "n_janelas_ok", "n_janelas_falha",
        "n_contrato_nao_ativado", "caminho_output"}`` ou
        ``{"status": "failed", "erro": str}``.
    """
    params_str = {chave: str(valor) for chave, valor in params.items()}
    params_str["seed"] = str(seed)

    with mlflow.start_run(run_name=run_id):
        mlflow.log_params(params_str)
        mlflow.set_tag("status", resultado["status"])

        if resultado["status"] == "success":
            mlflow.log_metrics(
                {
                    "n_janelas_ok": resultado["n_janelas_ok"],
                    "n_janelas_falha": resultado["n_janelas_falha"],
                    "n_contrato_nao_ativado": resultado["n_contrato_nao_ativado"],
                }
            )
            # tag, não artifact: HDF5s grandes demais para copiar para dentro
            # do storage do MLflow, já têm seu próprio local (diretorio_output).
            mlflow.set_tag("caminho_output", resultado["caminho_output"])
        else:
            mlflow.log_param("erro", resultado["erro"][:_MAX_PARAM_VAL_LENGTH])
