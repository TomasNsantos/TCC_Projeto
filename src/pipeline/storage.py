"""Gravação das janelas geradas em HDF5, formato longo, com split temporal
por convenção de ``window_id``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.generator.adversarial_mode import CenarioAdversarial
    from src.generator.normal_mode import CenarioNormal

JanelaPositiva = tuple[int, "CenarioAdversarial", np.random.SeedSequence, np.random.SeedSequence]
JanelaNegativa = tuple[int, "CenarioNormal", np.random.SeedSequence, np.random.SeedSequence]
"""``(window_id, cenario, seed_modelo, seed_fonte_b)`` — autocontido por
janela, não depende da posição na lista (janelas que falharam na geração
simplesmente não aparecem aqui, então a posição por si só não seria um
``window_id`` confiável)."""


def calcular_split(window_id: int, n_janelas: int) -> str:
    """Split temporal 70/15/15 por convenção de ``window_id``.

    ``window_id`` funciona aqui como PROXY de ordem cronológica, por
    convenção — não é um split baseado em timestamp real dentro da janela.
    Janelas são episódios i.i.d. (cada uma vem de uma sub-seed independente
    derivada em ``config.derivar_seeds``), não pontos ao longo de uma linha
    do tempo real compartilhada. A convenção "window_id maior = mais
    recente" existe só para dar ao protocolo de split (PLANO §5.2.4:
    "nenhuma janela de treino sobrepõe temporalmente as de teste") um
    critério determinístico e reprodutível de aplicar, não porque as
    janelas tenham uma relação cronológica genuína entre si.

    Parameters
    ----------
    window_id : int
        Em ``[0, n_janelas)``.
    n_janelas : int

    Returns
    -------
    str
        ``"train"`` (``window_id < 0.7 * n_janelas``), ``"val"``
        (``< 0.85 * n_janelas``) ou ``"test"`` (caso contrário).
    """
    if window_id < 0.7 * n_janelas:
        return "train"
    if window_id < 0.85 * n_janelas:
        return "val"
    return "test"


def _dataframe_vazio(dtypes: dict) -> pd.DataFrame:
    """DataFrame vazio com dtypes explícitos por coluna.

    Necessário porque ``pd.DataFrame(columns=[...])``/``pd.DataFrame([])``
    sem dados cria colunas ``object`` — e concatenar uma tabela ``object``
    (mesmo vazia) com uma tabela real de ``window_id`` inteiro "envenena" o
    resultado para ``object`` (confirmado empiricamente: PyTables recusa
    gravar ``window_id`` assim, "not [string] but [integer] object dtype").
    Mesmo padrão já usado em ``ElectionModel.fonte_a_eventos_fronteira``.
    """
    return pd.DataFrame({coluna: pd.Series(dtype=dtype) for coluna, dtype in dtypes.items()})


_DTYPES_FONTE_A = {"window_id": int, "classe": object, "split": object, "timestep": int, "n_eventos": int, "volume": float}
_DTYPES_FONTE_B = {"window_id": int, "classe": object, "split": object, "timestamp": float}
_DTYPES_FONTE_C = {"window_id": int, "classe": object, "split": object, "unidade": int, "fracao_candidato_alvo": float}
_DTYPES_METADADOS = {
    "window_id": int,
    "classe": object,
    "split": object,
    "contrato_ativado": float,
    "seed_modelo": object,
    "seed_fonte_b": object,
}


def _linhas_fonte_a(janelas: list, classe: str, n_janelas: int) -> pd.DataFrame:
    partes = []
    for window_id, cenario, _, _ in janelas:
        if cenario.fonte_a.empty:
            continue
        parte = cenario.fonte_a.copy()
        parte.insert(0, "window_id", window_id)
        parte.insert(1, "classe", classe)
        parte.insert(2, "split", calcular_split(window_id, n_janelas))
        partes.append(parte)
    if not partes:
        return _dataframe_vazio(_DTYPES_FONTE_A)
    return pd.concat(partes, ignore_index=True)


def _linhas_fonte_b(janelas: list, classe: str, n_janelas: int) -> pd.DataFrame:
    linhas = []
    for window_id, cenario, _, _ in janelas:
        split = calcular_split(window_id, n_janelas)
        for timestamp in cenario.fonte_b:
            linhas.append({"window_id": window_id, "classe": classe, "split": split, "timestamp": float(timestamp)})
    if not linhas:
        return _dataframe_vazio(_DTYPES_FONTE_B)
    return pd.DataFrame(linhas, columns=list(_DTYPES_FONTE_B)).astype(_DTYPES_FONTE_B)


def _linhas_fonte_c(janelas: list, classe: str, n_janelas: int, atributo: str) -> pd.DataFrame:
    linhas = []
    for window_id, cenario, _, _ in janelas:
        split = calcular_split(window_id, n_janelas)
        serie = getattr(cenario, atributo)
        for unidade, fracao in serie.items():
            linhas.append(
                {
                    "window_id": window_id,
                    "classe": classe,
                    "split": split,
                    "unidade": int(unidade),
                    "fracao_candidato_alvo": float(fracao),
                }
            )
    if not linhas:
        return _dataframe_vazio(_DTYPES_FONTE_C)
    return pd.DataFrame(linhas, columns=list(_DTYPES_FONTE_C)).astype(_DTYPES_FONTE_C)


def _linhas_metadados(janelas: list, classe: str, n_janelas: int) -> pd.DataFrame:
    linhas = []
    for window_id, cenario, seed_modelo, seed_fonte_b in janelas:
        contrato_ativado = getattr(cenario, "contrato_ativado", None)
        linhas.append(
            {
                "window_id": window_id,
                "classe": classe,
                "split": calcular_split(window_id, n_janelas),
                # float (1.0/0.0/NaN), não bool nullable: pandas "boolean" (nullable)
                # não é gravável em format="table" do PyTables (tables.BooleanCol
                # não existe — confirmado empiricamente). NaN para classe negativa,
                # que não tem conceito de contrato (Fase 2 nunca roda).
                "contrato_ativado": float(contrato_ativado) if contrato_ativado is not None else float("nan"),
                "seed_modelo": repr(seed_modelo),
                "seed_fonte_b": repr(seed_fonte_b),
            }
        )
    if not linhas:
        return _dataframe_vazio(_DTYPES_METADADOS)
    return pd.DataFrame(linhas, columns=list(_DTYPES_METADADOS)).astype(_DTYPES_METADADOS)


def escrever_run_hdf5(
    caminho: Path,
    janelas_positivas: list[JanelaPositiva],
    janelas_negativas: list[JanelaNegativa],
) -> None:
    r"""Grava as janelas geradas em HDF5, formato longo, 6 tabelas.

    Uma linha por (janela × unidade observável) em cada tabela — não uma
    linha por janela (exceto ``metadados_janela``, que é uma linha por
    janela). ``window_id`` é por classe (``0..n_janelas-1`` em cada uma,
    não um índice global compartilhado) — a coluna ``classe`` junto com
    ``window_id`` identifica uma janela unicamente. Sobrescreve ``caminho``
    se já existir (``mode="w"``) — cada chamada produz um arquivo
    autocontido e determinístico, não acumula sobre um arquivo anterior.

    Janelas que falharam na geração (capturadas por
    ``gerar_par_de_classes_real``) simplesmente não aparecem em
    ``janelas_positivas``/``janelas_negativas`` — não há linha "vazia" para
    elas em nenhuma tabela.

    Parameters
    ----------
    caminho : Path
    janelas_positivas : list[JanelaPositiva]
        ``(window_id, CenarioAdversarial, seed_modelo, seed_fonte_b)`` por
        janela bem-sucedida da classe positiva.
    janelas_negativas : list[JanelaNegativa]
        Idem, classe negativa (``CenarioNormal`` — sem ``contrato_ativado``,
        ver ``metadados_janela`` abaixo).

    Tabelas gravadas
    -----------------
    fonte_a : window_id, classe, split, timestep, n_eventos, volume
    fonte_b : window_id, classe, split, timestamp
    fonte_c_secao / fonte_c_municipio / fonte_c_estado :
        window_id, classe, split, unidade, fracao_candidato_alvo
    metadados_janela :
        window_id, classe, split, contrato_ativado, seed_modelo, seed_fonte_b
    """
    n_positivas = len(janelas_positivas)
    n_negativas = len(janelas_negativas)

    tabelas = {
        "fonte_a": pd.concat(
            [_linhas_fonte_a(janelas_positivas, "positiva", n_positivas), _linhas_fonte_a(janelas_negativas, "negativa", n_negativas)],
            ignore_index=True,
        ),
        "fonte_b": pd.concat(
            [_linhas_fonte_b(janelas_positivas, "positiva", n_positivas), _linhas_fonte_b(janelas_negativas, "negativa", n_negativas)],
            ignore_index=True,
        ),
        "fonte_c_secao": pd.concat(
            [
                _linhas_fonte_c(janelas_positivas, "positiva", n_positivas, "resultado_por_secao"),
                _linhas_fonte_c(janelas_negativas, "negativa", n_negativas, "resultado_por_secao"),
            ],
            ignore_index=True,
        ),
        "fonte_c_municipio": pd.concat(
            [
                _linhas_fonte_c(janelas_positivas, "positiva", n_positivas, "resultado_por_municipio"),
                _linhas_fonte_c(janelas_negativas, "negativa", n_negativas, "resultado_por_municipio"),
            ],
            ignore_index=True,
        ),
        "fonte_c_estado": pd.concat(
            [
                _linhas_fonte_c(janelas_positivas, "positiva", n_positivas, "resultado_por_estado"),
                _linhas_fonte_c(janelas_negativas, "negativa", n_negativas, "resultado_por_estado"),
            ],
            ignore_index=True,
        ),
        "metadados_janela": pd.concat(
            [_linhas_metadados(janelas_positivas, "positiva", n_positivas), _linhas_metadados(janelas_negativas, "negativa", n_negativas)],
            ignore_index=True,
        ),
    }

    with pd.HDFStore(str(caminho), mode="w") as store:
        for nome, tabela in tabelas.items():
            # format="table" com 0 linhas faz store.put() virar no-op silencioso
            # (PyTables não escreve o grupo — confirmado empiricamente: store.keys()
            # fica vazio, leitura posterior levanta KeyError). "fixed" grava
            # corretamente mesmo vazio; só perde a capacidade de query por coluna
            # via where=, irrelevante numa tabela sem linha nenhuma.
            formato = "table" if len(tabela) > 0 else "fixed"
            store.put(nome, tabela, format=formato)
