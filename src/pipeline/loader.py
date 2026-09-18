r"""Loader reutilizável do dataset de produção (v3 em diante) — lê os HDF5
gerados por `storage.escrever_run_hdf5`, junta cada fonte com
`metadados_janela` pela chave composta ``(classe, window_id)`` e devolve
DataFrames "longos" (uma linha por evento/seção), com metadados de cenário
anexados como colunas — sem agregar por janela.

Não é um script de EDA (não imprime nada, não escolhe amostra) — é a
camada de carregamento que qualquer notebook/script de feature engineering
(Semana 9-10) deve reusar em vez de reimplementar o join/parse de nome de
arquivo. A lógica de derivação de `grupo` reaproveita exatamente a mesma
regra já usada em `scripts/eda/eda_dataset_producao_v2.py::_carregar_grupos`
e no relatório visual do v3 (`scripts/eda/relatorio_visual_dataset_v3.py`)
— não reimplementada do zero, só extraída para um lugar único e testado.

``metadados_janela`` é a ÚNICA fonte de verdade sobre quais janelas
existem (Requisito 1) — nenhuma outra tabela é usada para decidir a lista
de janelas; uma janela sem nenhuma linha em Fonte A/B (ex. positiva
inativa, ou negativa com Fonte A vazia por acaso de Poisson) continua
presente no resultado via `metadados_janela`, mesmo que aquela fonte
específica não tenha nenhuma linha correspondente após o merge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_PADRAO_NOME_ARQUIVO = re.compile(
    r"beta-(?P<beta>[\d.]+)_delta_t-(?P<delta_t>[\d.]+)_g-(?P<g>\w+)_pi-(?P<pi>[\d.]+)_"
    r"recompensa-(?P<recompensa>[\d.]+)_rho-(?P<rho>[\d.]+)_seed-(?P<seed>\d+)\.h5"
)
"""Mesmo regex já usado em `scripts/eda/eda_dataset_producao_v2.py` — os
nomes de arquivo são gerados por `config.run_id` + `.h5`, formato estável
entre v2 e v3 (mesma convenção de nomenclatura, só o conteúdo mudou)."""

_TABELAS_FONTE = ("fonte_a", "fonte_b", "fonte_c_secao", "fonte_c_municipio", "fonte_c_estado")
"""Tabelas que recebem join com metadados_janela. Não inclui a própria
`metadados_janela` (é a base do join, não algo que se junta a si mesma)."""

_CHAVE_JANELA = ["classe", "window_id"]
"""Chave composta do join — `window_id` sozinho NÃO identifica uma janela
unicamente (é um índice por classe, `0..n_janelas-1` em cada uma
separadamente, ver `storage.py::escrever_run_hdf5`); só `(classe,
window_id)` juntos identificam uma janela."""

COLUNAS_METADADO_CENARIO = ("beta", "delta_t", "g", "pi", "recompensa", "rho", "seed")
"""Parâmetros do design fatorial extraídos do nome do arquivo (Requisito
4) — sempre colunas de METADADO, nunca features de treino. Qualquer
consumidor de M1/M2/M3 deve excluir estas colunas (junto com
`COLUNAS_METADADO_JANELA`) antes de treinar um modelo."""

COLUNAS_METADADO_JANELA = ("classe", "window_id", "split", "grupo", "contrato_ativado")
"""Colunas de identificação/filtro da janela (Requisito 5) — `grupo` e
`split` devem estar disponíveis para filtro/agrupamento, mas não são
features. `contrato_ativado` é mantida (não é um metadado de cenário, mas
também não é feature — é o que definiu `grupo`, incluída por
transparência/auditoria, não para ser usada como input de detecção)."""


@dataclass(frozen=True)
class ParametrosCenario:
    """Parâmetros do design fatorial extraídos do nome de um arquivo `.h5`.

    Attributes
    ----------
    beta, delta_t, g, pi, recompensa, rho, seed
        Mesmos nomes/semântica de `pipeline.config.GradeFatorial` — ver lá
        para a definição de cada eixo. `g` e `seed` não são float (g é
        string, seed é int); os demais são float.
    """

    beta: int
    delta_t: float
    g: str
    pi: float
    recompensa: float
    rho: float
    seed: int

    def como_dict(self) -> dict[str, float | int | str]:
        """Dict pronto para virar colunas — mesmas chaves de
        `COLUNAS_METADADO_CENARIO`, nessa ordem."""
        return {
            "beta": self.beta,
            "delta_t": self.delta_t,
            "g": self.g,
            "pi": self.pi,
            "recompensa": self.recompensa,
            "rho": self.rho,
            "seed": self.seed,
        }


def extrair_parametros_cenario(caminho: Path) -> ParametrosCenario:
    """Extrai os parâmetros do design fatorial do NOME do arquivo `.h5`.

    Parameters
    ----------
    caminho : Path
        Caminho de um arquivo gerado por `gerar_par_de_classes_real`/
        `config.run_id` — só o `.name` é usado, o resto do caminho é
        ignorado (funciona com caminho absoluto ou relativo).

    Returns
    -------
    ParametrosCenario

    Raises
    ------
    ValueError
        Se o nome do arquivo não bater com o padrão esperado
        (`beta-..._delta_t-..._g-..._pi-..._recompensa-..._rho-..._seed-....h5`).
    """
    m = _PADRAO_NOME_ARQUIVO.search(Path(caminho).name)
    if not m:
        raise ValueError(
            f"Nome de arquivo fora do padrão esperado (beta-*_delta_t-*_g-*_pi-*_"
            f"recompensa-*_rho-*_seed-*.h5): {Path(caminho).name!r}"
        )
    d = m.groupdict()
    return ParametrosCenario(
        beta=int(float(d["beta"])),
        delta_t=float(d["delta_t"]),
        g=d["g"],
        pi=float(d["pi"]),
        recompensa=float(d["recompensa"]),
        rho=float(d["rho"]),
        seed=int(d["seed"]),
    )


def _derivar_grupo(classe: pd.Series, contrato_ativado: pd.Series) -> pd.Series:
    """Deriva a coluna `grupo` (negativa / positiva_ativa / positiva_inativa)
    a partir de `classe` + `contrato_ativado` — mesma regra usada em
    `scripts/eda/eda_dataset_producao_v2.py::_carregar_grupos` (não
    reimplementada, só extraída para uso compartilhado).

    ``contrato_ativado`` é `NaN` para a classe negativa (Fase 2 nunca roda
    — ver `storage.py::_linhas_metadados`); comparações com `NaN` são
    sempre `False` em pandas, então a máscara `classe == "positiva"` já
    exclui a negativa dos dois ramos positivos sem precisar de um `fillna`
    explícito.

    Parameters
    ----------
    classe : pd.Series
        Valores ``"positiva"``/``"negativa"``.
    contrato_ativado : pd.Series
        `1.0`/`0.0`/`NaN`, mesmo índice de `classe`.

    Returns
    -------
    pd.Series
        Valores ``"negativa"``/``"positiva_ativa"``/``"positiva_inativa"``,
        mesmo índice de entrada.
    """
    grupo = pd.Series(pd.array([None] * len(classe), dtype=object), index=classe.index)
    grupo[classe == "negativa"] = "negativa"
    grupo[(classe == "positiva") & (contrato_ativado == 1.0)] = "positiva_ativa"
    grupo[(classe == "positiva") & (contrato_ativado == 0.0)] = "positiva_inativa"
    return grupo


def _metadados_com_grupo(caminho: Path) -> pd.DataFrame:
    """Lê `metadados_janela` de um arquivo e anexa a coluna `grupo`.

    É a base de todo o resto do loader — nenhuma outra tabela decide quais
    janelas existem (Requisito 1): mesmo que `fonte_a`/`fonte_b` estejam
    vazias para uma janela (ex. positiva inativa), a janela continua
    presente aqui.
    """
    md = pd.read_hdf(caminho, "metadados_janela")
    md = md.copy()
    md["grupo"] = _derivar_grupo(md["classe"], md["contrato_ativado"])
    return md


def _anexar_metadados_cenario(df: pd.DataFrame, params: ParametrosCenario) -> pd.DataFrame:
    """Anexa as colunas de `COLUNAS_METADADO_CENARIO` a `df` (mesmo valor
    em toda linha — um arquivo `.h5` é uma única combinação do design
    fatorial)."""
    df = df.copy()
    for chave, valor in params.como_dict().items():
        df[chave] = valor
    return df


def carregar_arquivo(caminho: Path) -> dict[str, pd.DataFrame]:
    r"""Carrega um único arquivo `.h5`: junta cada fonte com
    `metadados_janela` por `(classe, window_id)` e anexa `grupo` +
    parâmetros de cenário.

    Cada tabela retornada é "longa" (uma linha por evento, ou por
    seção/município/estado para Fonte C) — nenhuma agregação por janela é
    feita aqui (Requisito 7).

    Parameters
    ----------
    caminho : Path
        Caminho de um arquivo `.h5` gerado por `gerar_par_de_classes_real`.

    Returns
    -------
    dict[str, pd.DataFrame]
        Chaves ``"metadados_janela"``, ``"fonte_a"``, ``"fonte_b"``,
        ``"fonte_c_secao"``, ``"fonte_c_municipio"``, ``"fonte_c_estado"``.
        ``metadados_janela`` tem uma linha por janela (a lista completa,
        via Requisito 1) com `grupo` + parâmetros de cenário anexados. As
        demais têm uma linha por evento/seção, resultado de um LEFT JOIN
        de metadados_janela com a fonte correspondente — uma janela sem
        nenhuma linha naquela fonte simplesmente não aparece na tabela
        daquela fonte (não há linha "vazia" artificial), mas continua
        presente em `metadados_janela`.
    """
    metadados = _metadados_com_grupo(caminho)
    params = extrair_parametros_cenario(caminho)
    metadados = _anexar_metadados_cenario(metadados, params)

    # "split" não entra em colunas_join: fonte_a/b/c já persistem sua
    # própria coluna "split" (mesma fórmula calcular_split de
    # metadados_janela, ver storage.py) — trazê-la de novo via merge
    # criaria uma coluna "split_meta" duplicada e redundante.
    colunas_join = _CHAVE_JANELA + ["grupo"] + list(COLUNAS_METADADO_CENARIO) + ["contrato_ativado"]
    chave_para_join = metadados[colunas_join]

    resultado: dict[str, pd.DataFrame] = {"metadados_janela": metadados}
    for tabela in _TABELAS_FONTE:
        bruta = pd.read_hdf(caminho, tabela)
        # inner join: uma janela sem linha nesta fonte simplesmente não
        # aparece aqui — a lista completa de janelas continua em
        # metadados_janela (Requisito 1), não nesta tabela.
        resultado[tabela] = bruta.merge(chave_para_join, on=_CHAVE_JANELA, how="inner")

    return resultado


def carregar_diretorio(diretorio: Path, tabelas: tuple[str, ...] | None = None) -> dict[str, pd.DataFrame]:
    r"""Itera sobre todos os `.h5` de `diretorio`, carrega cada um via
    `carregar_arquivo` e concatena — mesma estrutura de retorno de
    `carregar_arquivo`, mas com todas as combinações do dataset juntas.

    Parameters
    ----------
    diretorio : Path
        Diretório com arquivos `.h5` (ex. ``output/dataset_producao_v3``).
    tabelas : tuple[str, ...] | None
        Subconjunto de tabelas a carregar (``"metadados_janela"`` sempre
        incluída implicitamente, mesmo se omitida). ``None`` (default)
        carrega todas — `metadados_janela` + as 5 tabelas de
        `_TABELAS_FONTE`. Útil para não carregar `fonte_b` (potencialmente
        a maior tabela) quando não for necessária.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mesmas chaves de `carregar_arquivo`, cada uma concatenada
        (`pd.concat(..., ignore_index=True)`) sobre todos os arquivos.

    Raises
    ------
    FileNotFoundError
        Se `diretorio` não existir ou não tiver nenhum `.h5`.
    """
    diretorio = Path(diretorio)
    caminhos = sorted(diretorio.glob("*.h5"))
    if not caminhos:
        raise FileNotFoundError(f"Nenhum arquivo .h5 encontrado em {diretorio}")

    tabelas_desejadas = set(tabelas) if tabelas is not None else {"metadados_janela", *_TABELAS_FONTE}
    tabelas_desejadas.add("metadados_janela")

    acumulado: dict[str, list[pd.DataFrame]] = {nome: [] for nome in tabelas_desejadas}
    for caminho in caminhos:
        carregado = carregar_arquivo(caminho)
        for nome in tabelas_desejadas:
            acumulado[nome].append(carregado[nome])

    return {nome: pd.concat(partes, ignore_index=True) for nome, partes in acumulado.items()}
