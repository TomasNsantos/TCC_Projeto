r"""Camada raw/binned (Fonte A/B) e referência negativa-treino para
z-score/CUSUM (Semana 9-10, PLANO §5.3.2 — M1).

Três peças, nesta ordem de dependência:

1. `bin_janelas` — reindexa Fonte A (já binada por `timestep`) e bina Fonte
   B (timestamps brutos, via `normal_mode.trafego.contagem_por_timestep`,
   reusada — não reimplementada) num índice comum de timestep por janela,
   preenchendo com zero explícito onde não há evento. Formato LONGO,
   indexado por ``(classe, window_id, timestep)``.
2. `calcular_referencia_negativa_treino` — média/desvio-padrão de
   ``contagem_a``/``contagem_b`` por ``(combinação completa × timestep)``,
   agregando sobre as seeds e as janelas de treino da classe negativa
   daquela combinação.
3. `calcular_zscore` — aplica uma referência a qualquer janela (qualquer
   grupo/split), retornando o desvio em unidades de desvio-padrão por
   timestep.

**Índice de timestep — teto único por combinação, igual para as duas
classes.** O índice de cada janela cobre ``[0, ceil(janela_teto))``, com
``janela_teto = max(delta_t, JANELA_TRAFEGO_FUNDO_QUANDO_DELTA_T_ZERO)`` —
MESMO teto para positiva e negativa, mesmo quando `delta_t == 0.0` (onde a
janela de observação REAL da classe negativa, pós-fix, é 24.0, mas a da
classe positiva continua sendo 0.0 — ver `geracao._janela_trafego_fundo`
e a entrada "Bug corrigido (2026-09-17)" em CLAUDE.md). Escolhido em vez
de um índice por classe (que produziria séries de tamanho diferente entre
positiva e negativa só em `delta_t=0.0`, complicando qualquer comparação
cruzada de z-score) — ao custo de introduzir timesteps ESTRUTURALMENTE
IMPOSSÍVEIS de ter evento na classe positiva quando `delta_t=0.0`
(`_amostrar_timestamps_desembolso` com `delta_t=0.0` produz
`uniform(0, 0)`, sempre exatamente `0.0` — não há mecanismo no gerador que
possa colocar um evento de Fonte A/B da classe positiva em
`timestep >= 1` nesse caso). Esses timesteps são marcados explicitamente
pela coluna `timestep_estruturalmente_impossivel` (ver `bin_janelas`), não
deixados para inferência de quem consome a tabela.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.generator.normal_mode.trafego import contagem_por_timestep
from src.pipeline.geracao import _janela_trafego_fundo

_EPSILON_DESVIO_PADRAO_ZERO = 1e-9
"""Piso aplicado ao desvio-padrão da referência antes de dividir, só nos
timesteps onde o desvio-padrão amostral é exatamente 0.0 (contagem
constante entre todas as janelas de referência daquele timestep — comum
em timesteps onde a contagem é quase sempre zero). Não é uma suposição de
variância "verdadeira" — é só o que evita 0/0=NaN ou x/0=inf sem aviso
(ver `calcular_zscore`); qualquer desvio de zero nesse timestep já produz
um z-score de magnitude grande (`diferença / 1e-9`), sinalizando
corretamente "isto nunca acontece na referência", sem propagar NaN/inf
para o resto do pipeline."""


def _janela_teto(delta_t: float) -> float:
    """Teto único do índice de timestep para uma combinação, IGUAL para
    positiva e negativa — ``max(delta_t, JANELA_TRAFEGO_FUNDO_QUANDO_DELTA_T_ZERO)``.

    Reusa `geracao._janela_trafego_fundo` (não duplica a regra
    ``24.0 se delta_t==0.0 senão delta_t``) — o teto da combinação é
    exatamente a janela real de observação da classe negativa, aplicada
    também à positiva por decisão de design (ver docstring do módulo).
    """
    return _janela_trafego_fundo(delta_t)


def _indice_completo_de_timesteps(metadados_janela: pd.DataFrame) -> pd.DataFrame:
    """Constrói o índice completo ``(classe, window_id, timestep)`` de
    TODAS as janelas de uma vez, via `np.repeat` — sem loop Python por
    janela. Uma janela com ``n_timesteps=k`` contribui ``k`` linhas
    (``timestep = 0..k-1``); a ORDEM das janelas de entrada é preservada
    (cada bloco de timesteps de uma janela fica contíguo, na ordem em que
    a janela aparece em `metadados_janela`), mas isso não é uma garantia
    de API — quem consome o resultado de `bin_janelas` não deve depender
    de ordem de linhas.
    """
    n_timesteps_por_janela = np.ceil(
        metadados_janela["delta_t"].astype(float).map(_janela_teto).to_numpy()
    ).astype(int)

    classes_repetidas = np.repeat(metadados_janela["classe"].to_numpy(), n_timesteps_por_janela)
    window_ids_repetidos = np.repeat(metadados_janela["window_id"].to_numpy(), n_timesteps_por_janela)
    delta_ts_repetidos = np.repeat(metadados_janela["delta_t"].to_numpy(), n_timesteps_por_janela)
    # concatena um arange(k) por janela -- equivalente a
    # np.hstack([np.arange(k) for k in n_timesteps_por_janela]), mas sem o
    # loop Python explícito: repeat + subtração do offset acumulado de
    # início de cada bloco.
    offsets_inicio_bloco = np.repeat(np.cumsum(n_timesteps_por_janela) - n_timesteps_por_janela, n_timesteps_por_janela)
    posicoes_globais = np.arange(len(classes_repetidas))
    timesteps = posicoes_globais - offsets_inicio_bloco

    return pd.DataFrame(
        {
            "classe": classes_repetidas,
            "window_id": window_ids_repetidos,
            "timestep": timesteps,
            "delta_t": delta_ts_repetidos,
        }
    )


def _bin_fonte_b_vetorizado(fonte_b: pd.DataFrame, indice_completo: pd.DataFrame) -> pd.Series:
    r"""Bina Fonte B (timestamps brutos) em contagem por `(classe,
    window_id, timestep)` — reusa `contagem_por_timestep` (não reescreve a
    lógica de binning em si, ver docstring do módulo), mas chama uma vez
    por GRUPO `(classe, window_id)` que tem pelo menos um evento em
    `fonte_b`, via `groupby(...).apply(...)` — não uma vez por janela do
    dataset inteiro. Janelas sem nenhum evento em `fonte_b` (a maioria,
    tipicamente) nunca entram nesse groupby — ficam com contagem 0 pelo
    `fillna(0)` do merge externo (`bin_janelas`), sem custo de chamada.

    Medido por profiling (cProfile) antes desta reescrita: o gargalo da
    implementação anterior era a filtragem repetida `fonte_b[(fonte_b.classe
    == classe) & (fonte_b.window_id == window_id)]` dentro de um loop
    Python por janela (comparação `==` sobre coluna `object` inteira a
    cada iteração — ~66% do tempo total num arquivo de 2000 janelas), não
    `contagem_por_timestep` em si — por isso a função de binning em si
    continua exatamente a mesma, só o modo de chamá-la mudou.
    """
    if fonte_b.empty:
        return pd.Series(dtype=int)

    n_timesteps_por_janela = (
        indice_completo.groupby(["classe", "window_id"], sort=False)["timestep"].size()
    )

    linhas_classe = []
    linhas_window_id = []
    linhas_timestep = []
    linhas_contagem = []
    for (classe, window_id), grupo in fonte_b.groupby(["classe", "window_id"], sort=False):
        n_timesteps = int(n_timesteps_por_janela.loc[(classe, window_id)])
        contagem = contagem_por_timestep(grupo["timestamp"].to_numpy(), janela=float(n_timesteps))
        linhas_classe.append(np.full(n_timesteps, classe, dtype=object))
        linhas_window_id.append(np.full(n_timesteps, window_id))
        linhas_timestep.append(np.arange(n_timesteps))
        linhas_contagem.append(contagem)

    indice_resultado = pd.MultiIndex.from_arrays(
        [np.concatenate(linhas_classe), np.concatenate(linhas_window_id), np.concatenate(linhas_timestep)],
        names=["classe", "window_id", "timestep"],
    )
    return pd.Series(np.concatenate(linhas_contagem), index=indice_resultado, name="contagem_b")


def bin_janelas(
    fonte_a: pd.DataFrame,
    fonte_b: pd.DataFrame,
    metadados_janela: pd.DataFrame,
) -> pd.DataFrame:
    r"""Bina Fonte A/B de TODAS as janelas de `metadados_janela` num índice
    comum de timestep por `(classe, window_id)` — formato longo.

    `metadados_janela` continua sendo a lista completa de janelas (mesmo
    princípio do `loader` — nenhuma outra tabela decide quais janelas
    existem): uma janela sem nenhuma linha em `fonte_a`/`fonte_b` produz
    uma série inteiramente zero no índice de timestep, não fica ausente do
    resultado — o índice completo de timesteps (`_indice_completo_de_timesteps`)
    é construído a partir de `metadados_janela`, nunca a partir dos dados
    de `fonte_a`/`fonte_b`, exatamente para preservar essa garantia sob
    vetorização (a forma mais fácil de quebrar isso silenciosamente seria
    construir o índice a partir de `fonte_a`/`fonte_b` e perder as janelas
    sem nenhuma linha ali).

    Parameters
    ----------
    fonte_a : pd.DataFrame
        Saída de `loader.carregar_arquivo`/`carregar_diretorio` — precisa
        ter as colunas ``classe``, ``window_id``, ``timestep``,
        ``n_eventos``, ``delta_t`` (metadado de cenário, usado para
        calcular o teto do índice).
    fonte_b : pd.DataFrame
        Idem, com colunas ``classe``, ``window_id``, ``timestamp``,
        ``delta_t``.
    metadados_janela : pd.DataFrame
        Saída de `loader` — colunas ``classe``, ``window_id``, ``delta_t``
        (e qualquer outra coluna de metadado, repassada por linha de
        timestep, não só por janela).

    Returns
    -------
    pd.DataFrame
        Uma linha por ``(classe, window_id, timestep)``, colunas:
        ``contagem_a`` (int, de `fonte_a.n_eventos` reindexado),
        ``contagem_b`` (int, `fonte_b` binada via `contagem_por_timestep`),
        ``timestep_estruturalmente_impossivel`` (bool — `True` só quando
        ``classe == "positiva"`` e ``delta_t == 0.0`` e ``timestep >= 1``,
        ver docstring do módulo), mais todas as colunas de metadado de
        `metadados_janela` (grupo, split, parâmetros de cenário etc.),
        repetidas por timestep. A ordem das linhas não é uma garantia de
        API — normalize (`sort_values`) antes de comparar/depender de
        ordem.

    Raises
    ------
    ValueError
        Se `metadados_janela` não tiver a coluna `delta_t`.

    Notes
    -----
    Implementação vetorizada: o índice completo de `(classe, window_id,
    timestep)` é construído de uma vez via `np.repeat` (sem loop Python
    por janela); Fonte A é reindexada via `merge` + `fillna(0)`; Fonte B
    é binada via `groupby(...).apply(contagem_por_timestep)` — uma chamada
    por GRUPO com evento (não por janela do dataset inteiro). Medido:
    ~61s → ver relatório da tarefa para o arquivo de 2000 janelas usado
    como referência, e o tempo sobre os 270 arquivos do v3 completo.
    """
    if "delta_t" not in metadados_janela.columns:
        raise ValueError("metadados_janela precisa da coluna 'delta_t' (metadado de cenário do loader).")

    duplicadas = metadados_janela.duplicated(subset=["classe", "window_id"], keep=False)
    if duplicadas.any():
        raise ValueError(
            "metadados_janela contém (classe, window_id) duplicado — bin_janelas espera dados de um "
            "único arquivo/combinação de parâmetros, não um diretório concatenado sem desambiguação "
            "(window_id NÃO é global, é reiniciado em 0 por classe em CADA arquivo — ver "
            "storage.escrever_run_hdf5 e loader.py). Concatenar vários arquivos via "
            "loader.carregar_diretorio antes de chamar bin_janelas produz merges com chave não-única "
            "e pode estourar memória silenciosamente ou gerar dado incorreto sem erro — processe cada "
            "arquivo/combinação separadamente com bin_janelas."
        )

    if metadados_janela.empty:
        return pd.DataFrame(
            columns=["classe", "window_id", "timestep", "contagem_a", "contagem_b", "timestep_estruturalmente_impossivel"]
        )

    indice_completo = _indice_completo_de_timesteps(metadados_janela)

    # Fonte A: merge (não filtro por linha) do índice completo com as
    # colunas de dado de fonte_a; timesteps sem linha correspondente viram
    # NaN pelo merge e são preenchidos com 0 explicitamente.
    fonte_a_reduzida = fonte_a[["classe", "window_id", "timestep", "n_eventos"]]
    indice_completo = indice_completo.merge(
        fonte_a_reduzida, on=["classe", "window_id", "timestep"], how="left"
    )
    indice_completo["contagem_a"] = indice_completo["n_eventos"].fillna(0).astype(int)
    indice_completo = indice_completo.drop(columns=["n_eventos"])

    # Fonte B: binada por grupo (classe, window_id), depois juntada ao
    # índice completo pela mesma chave — timesteps/janelas sem nenhum
    # evento em fonte_b não aparecem no resultado do groupby e ficam 0
    # pelo fillna, igual à Fonte A.
    contagem_b_por_grupo = _bin_fonte_b_vetorizado(fonte_b, indice_completo)
    indice_completo = indice_completo.set_index(["classe", "window_id", "timestep"])
    indice_completo["contagem_b"] = contagem_b_por_grupo
    indice_completo["contagem_b"] = indice_completo["contagem_b"].fillna(0).astype(int)
    indice_completo = indice_completo.reset_index()

    indice_completo["timestep_estruturalmente_impossivel"] = (
        (indice_completo["classe"] == "positiva")
        & (indice_completo["delta_t"] == 0.0)
        & (indice_completo["timestep"] >= 1)
    )

    # Metadados (grupo, split, parâmetros de cenário etc.) anexados via
    # merge por (classe, window_id) -- mesma chave de identificação de
    # janela usada no loader; "delta_t" já está em indice_completo (usado
    # para o teto/marcação acima), não duplicado no merge.
    colunas_metadado = [c for c in metadados_janela.columns if c != "delta_t"]
    resultado = indice_completo.merge(
        metadados_janela[["classe", "window_id"] + [c for c in colunas_metadado if c not in ("classe", "window_id")]],
        on=["classe", "window_id"],
        how="left",
    )

    return resultado


_COLUNAS_COMBINACAO = ("recompensa", "pi", "delta_t", "rho", "beta", "g")
"""Combinação COMPLETA do design fatorial usada para agrupar a referência
— explicitamente SEM `seed` (as seeds são pooladas dentro da mesma
combinação para aumentar a amostra da referência, conforme pedido)."""


def calcular_referencia_negativa_treino(binned: pd.DataFrame) -> pd.DataFrame:
    r"""Referência de z-score/CUSUM: média/desvio-padrão de
    `contagem_a`/`contagem_b` por `(combinação completa × timestep)`,
    sobre as janelas de treino (`split == "train"`) da classe NEGATIVA
    daquela combinação — poolando as seeds disponíveis (agrupamento NÃO
    inclui `seed`, ver `_COLUNAS_COMBINACAO`).

    Parameters
    ----------
    binned : pd.DataFrame
        Saída de `bin_janelas` — precisa ter `classe`, `split`,
        `timestep`, `contagem_a`, `contagem_b` e as colunas de
        `_COLUNAS_COMBINACAO`.

    Returns
    -------
    pd.DataFrame
        Uma linha por `(recompensa, pi, delta_t, rho, beta, g, timestep)`,
        colunas ``media_a``, ``desvio_a``, ``media_b``, ``desvio_b``,
        ``n_janelas`` (quantas janelas negativas de treino contribuíram
        para aquele timestep — útil para avaliar confiabilidade da
        referência). `desvio_a`/`desvio_b` NUNCA são exatamente `0.0` —
        pisado em `_EPSILON_DESVIO_PADRAO_ZERO` quando o desvio amostral
        é `0.0` (ver docstring da constante) — para que `calcular_zscore`
        nunca divida por zero.

    Raises
    ------
    ValueError
        Se não houver nenhuma janela negativa de treino em `binned` (a
        referência ficaria vazia/indefinida para toda combinação).
    """
    negativas_treino = binned[(binned["classe"] == "negativa") & (binned["split"] == "train")]
    if negativas_treino.empty:
        raise ValueError(
            "Nenhuma janela negativa com split='train' encontrada em `binned` — "
            "a referência não pode ser calculada sem pelo menos uma janela."
        )

    colunas_grupo = list(_COLUNAS_COMBINACAO) + ["timestep"]
    agregado = negativas_treino.groupby(colunas_grupo, as_index=False).agg(
        media_a=("contagem_a", "mean"),
        desvio_a=("contagem_a", "std"),
        media_b=("contagem_b", "mean"),
        desvio_b=("contagem_b", "std"),
        n_janelas=("contagem_a", "size"),
    )

    # std com n=1 (uma única janela contribuindo) devolve NaN (ddof=1
    # default do pandas, denominador n-1=0) -- trata junto com o caso
    # "std==0.0 de verdade" (n>1, mas contagem constante): os dois casos
    # significam "não há variância amostral estimável neste timestep",
    # mesmo tratamento (piso em EPSILON), não dois comportamentos
    # diferentes por acidente de fórmula.
    agregado["desvio_a"] = agregado["desvio_a"].fillna(0.0).clip(lower=_EPSILON_DESVIO_PADRAO_ZERO)
    agregado["desvio_b"] = agregado["desvio_b"].fillna(0.0).clip(lower=_EPSILON_DESVIO_PADRAO_ZERO)

    assert not agregado[["media_a", "desvio_a", "media_b", "desvio_b"]].isna().any().any(), (
        "referência com NaN após o tratamento de desvio-padrão — não deveria acontecer"
    )
    assert np.isfinite(agregado[["media_a", "desvio_a", "media_b", "desvio_b"]].to_numpy()).all(), (
        "referência com valor não-finito (inf) — não deveria acontecer"
    )

    return agregado


def calcular_zscore(binned: pd.DataFrame, referencia: pd.DataFrame) -> pd.DataFrame:
    r"""Aplica a referência (`calcular_referencia_negativa_treino`) a
    qualquer conjunto de janelas binadas (`bin_janelas`) — qualquer
    grupo/split, não só treino/negativa — retornando o z-score por
    timestep para `contagem_a` e `contagem_b`.

    Usada tanto pelo CUSUM (M1) quanto pela feature de Camada 2 do M2
    (PLANO §5.3.1, "variação relativa de volume... z-score por janela").

    Parameters
    ----------
    binned : pd.DataFrame
        Saída de `bin_janelas` — qualquer subconjunto (treino/val/teste,
        qualquer grupo).
    referencia : pd.DataFrame
        Saída de `calcular_referencia_negativa_treino`.

    Returns
    -------
    pd.DataFrame
        Mesmas linhas de `binned` (mesma ordem, mesmo índice de posição),
        com duas colunas novas: ``zscore_a`` e ``zscore_b`` —
        ``(contagem - media) / desvio``, calculados usando a linha da
        referência que bate em `(recompensa, pi, delta_t, rho, beta, g,
        timestep)`. `NaN` se a combinação/timestep de uma janela não
        tiver entrada correspondente na referência (não deveria acontecer
        para combinações presentes no dataset original, mas não levanta
        erro — permite aplicar uma referência parcial sem quebrar).
    """
    colunas_chave = list(_COLUNAS_COMBINACAO) + ["timestep"]
    resultado = binned.merge(
        referencia[colunas_chave + ["media_a", "desvio_a", "media_b", "desvio_b"]],
        on=colunas_chave,
        how="left",
    )
    resultado["zscore_a"] = (resultado["contagem_a"] - resultado["media_a"]) / resultado["desvio_a"]
    resultado["zscore_b"] = (resultado["contagem_b"] - resultado["media_b"]) / resultado["desvio_b"]
    return resultado.drop(columns=["media_a", "desvio_a", "media_b", "desvio_b"])
