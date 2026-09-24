r"""Camada agregada de features (Semana 9-10, Dia 4-5, PLANO §5.3.1) —
insumo para M2 (XGBoost+SHAP).

Camada nova e PARALELA à raw/binned (`loader.py` → `binning.py` →
`cusum.py`, usada por M1/CUSUM) — não a substitui, não a modifica, só
consome sua saída (`bin_janelas`) mais `fonte_a`/`fonte_c_*` brutos de
`loader.carregar_arquivo`. Produz UMA LINHA POR JANELA (`classe,
window_id`), não por timestep.

Quatro blocos independentes e combináveis (necessário para a ablação
C1-C7, PLANO §5.4.1 — cada bloco deve poder entrar/sair do vetor de
features de M2 sem re-extração):

- `extrair_features_bloco_a` — Fonte A (desembolso): contagem, volume,
  IET, offset do primeiro evento, indicador de batch, flag de vazio.
- `extrair_features_bloco_b` — Fonte B (oráculo): mesma lista, sem volume
  (Fonte B não tem dimensão de magnitude).
- `extrair_features_coordenacao` — τ_Kendall(A,B) via binning + correlação
  cruzada A↔B em lag τ (Camada 3 do PLANO, poder discriminativo alto).
- `extrair_features_bloco_c` — wrapper genérico de agregação
  (mean/min/max/std/count) sobre `fracao_candidato_alvo`, mesma função
  para `fonte_c_secao` (N=5 hoje) e `fonte_c_municipio`/`fonte_c_estado`
  (N=1 hoje, degenerado).

`montar_vetor_features_agregadas` é a função de topo: recebe um único
arquivo `.h5`, roda `loader.carregar_arquivo` + `binning.bin_janelas`
internamente, junta os quatro blocos e separa o resultado em
`"features"` (vetor de treino do M2) e `"metadados"` (identificação de
janela + parâmetros de cenário + flag de auditoria `ambas_vazias`, nunca
usada como feature — vazamento direto de rótulo, já que identifica
`positiva_inativa` por construção).

**Convenção de NaN**: todas as estatísticas contínuas são `NaN` (não
`0.0`, não um sentinela numérico) quando a fonte está vazia na janela ou
o cálculo é matematicamente indefinido (ex. IET com um único evento,
desvio-padrão com uma única unidade em Fonte C). `fonte_a_vazia`/
`fonte_b_vazia` são as únicas colunas binárias e SÃO features legítimas
do M2 (Camada 1 do PLANO — presença/ausência de ativação de oráculo na
janela); a combinação lógica `ambas_vazias` fica só em metadados.

**Fora de escopo desta camada** (decisão do usuário, não tratar aqui):
calibração de `taxa_fonte_a`/`volume_medio_fonte_a`/`taxa_fonte_b`,
ρ/`prop_racional` em U(𝒜), `g` fixo em `"pool"`, assimetria de tráfego de
fundo entre classes, C_min, threshold de fragmentação (proxy de β) e
skew/momentos de Fonte C.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from src.pipeline.binning import bin_janelas
from src.pipeline.loader import carregar_arquivo

_LAG_MAXIMO_CORRELACAO_CRUZADA = 6
"""Teto do alcance de busca de lag na correlação cruzada A↔B
(`extrair_features_coordenacao`) — suposição v0 sem calibração formal
(mesma categoria de `tau_kendall`/`K_RESULTADO_ALVO`, ver CLAUDE.md). O
lag efetivo usado por janela é `min(n_timesteps - 1,
_LAG_MAXIMO_CORRELACAO_CRUZADA)` — nunca maior que o espaço disponível de
sobreposição da própria janela."""

_COLUNAS_CHAVE_JANELA = ["classe", "window_id"]

_COLUNAS_METADADO_JANELA_AGREGADA = (
    "classe",
    "window_id",
    "split",
    "grupo",
    "contrato_ativado",
    "beta",
    "delta_t",
    "g",
    "pi",
    "recompensa",
    "rho",
    "seed",
)
"""Colunas de metadado repassadas no dict `"metadados"` de
`montar_vetor_features_agregadas` — mesmas colunas de
`loader.COLUNAS_METADADO_CENARIO`/`COLUNAS_METADADO_JANELA` combinadas,
nunca incluídas em `"features"`."""


def _iet_estatisticas(timesteps_com_evento: np.ndarray) -> tuple[float, float, float]:
    """Media/desvio-padrão/coeficiente de variação dos intervalos entre
    timesteps CONSECUTIVOS COM EVENTO (não entre todos os timesteps da
    janela) — `NaN` para os três se houver menos de 2 eventos (intervalo
    indefinido com 0 ou 1 evento; ver ponto de decisão 5, CLAUDE.md/plano:
    `NaN` puro, não piso `EPSILON`, distinto do piso usado em
    `binning._EPSILON_DESVIO_PADRAO_ZERO` — aqui IET não alimenta nenhuma
    divisão posterior que quebraria com `NaN`)."""
    if timesteps_com_evento.size < 2:
        return float("nan"), float("nan"), float("nan")
    intervalos = np.diff(np.sort(timesteps_com_evento)).astype(float)
    media = float(np.mean(intervalos))
    desvio = float(np.std(intervalos, ddof=0))
    cv = desvio / media if media != 0 else float("nan")
    return media, desvio, cv


def _features_fonte_binada(binado_fonte: pd.DataFrame, coluna_contagem: str, prefixo: str) -> pd.DataFrame:
    """Núcleo comum de `extrair_features_bloco_a`/`extrair_features_bloco_b`
    sobre a série binada (contagem, IET, offset, flag de vazio) — não inclui
    volume/razão eventos-volume (só Bloco A tem, ver
    `extrair_features_bloco_a`).

    Parameters
    ----------
    binado_fonte : pd.DataFrame
        Saída de `binning.bin_janelas`, já ordenada implicitamente por
        `groupby` — precisa ter `classe`, `window_id`, `timestep`,
        `coluna_contagem`.
    coluna_contagem : str
        `"contagem_a"` ou `"contagem_b"`.
    prefixo : str
        `"fonte_a"` ou `"fonte_b"` — usado para nomear as colunas de saída.

    Returns
    -------
    pd.DataFrame
        Uma linha por `(classe, window_id)`, colunas
        `{prefixo}_n_eventos_total`, `{prefixo}_iet_media`,
        `{prefixo}_iet_std`, `{prefixo}_iet_cv`,
        `{prefixo}_offset_primeiro_evento`, `{prefixo}_vazia`.
    """
    linhas = []
    for (classe, window_id), grupo in binado_fonte.groupby(_COLUNAS_CHAVE_JANELA, sort=False):
        grupo_ordenado = grupo.sort_values("timestep")
        contagens = grupo_ordenado[coluna_contagem].to_numpy()
        timesteps = grupo_ordenado["timestep"].to_numpy()

        n_eventos_total = int(contagens.sum())
        vazia = n_eventos_total == 0

        timesteps_com_evento = timesteps[contagens > 0]
        if vazia:
            offset_primeiro_evento = float("nan")
            iet_media = iet_std = iet_cv = float("nan")
        else:
            offset_primeiro_evento = float(timesteps_com_evento.min())
            iet_media, iet_std, iet_cv = _iet_estatisticas(timesteps_com_evento)

        linhas.append(
            {
                "classe": classe,
                "window_id": window_id,
                f"{prefixo}_n_eventos_total": n_eventos_total,
                f"{prefixo}_iet_media": iet_media,
                f"{prefixo}_iet_std": iet_std,
                f"{prefixo}_iet_cv": iet_cv,
                f"{prefixo}_offset_primeiro_evento": offset_primeiro_evento,
                f"{prefixo}_vazia": vazia,
            }
        )

    return pd.DataFrame(linhas)


def extrair_features_bloco_a(binado: pd.DataFrame, fonte_a: pd.DataFrame) -> pd.DataFrame:
    r"""Bloco A (Fonte A / desembolso) — uma linha por `(classe,
    window_id)`.

    **Não valida duplicata de `(classe, window_id)` por conta própria** —
    assume que quem chama já garantiu isso (a função de topo
    `montar_vetor_features_agregadas`, via `binning.bin_janelas`, ou um
    teste sintético).

    Parameters
    ----------
    binado : pd.DataFrame
        Saída de `binning.bin_janelas` — usada para contagem/IET/offset
        (reindexada com zeros explícitos por timestep; não tem `volume`,
        só `contagem_a`).
    fonte_a : pd.DataFrame
        `fonte_a` bruto de `loader.carregar_arquivo` — usada só para
        `volume` (coluna ausente em `binado`). Uma janela sem nenhuma
        linha aqui (Fonte A vazia) simplesmente não aparece — tratada como
        `volume_total=NaN` no merge externo.

    Returns
    -------
    pd.DataFrame
        Uma linha por `(classe, window_id)`: `fonte_a_n_eventos_total`
        (int), `fonte_a_volume_total`, `fonte_a_volume_medio`,
        `fonte_a_volume_por_timestep_std`, `fonte_a_volume_por_timestep_min`,
        `fonte_a_volume_por_timestep_max`, `fonte_a_razao_eventos_volume`
        (contínuas-NaN — granularidade POR TIMESTEP, não por evento
        individual: essa granularidade não existe em nenhuma camada
        persistida do gerador), `fonte_a_iet_media`, `fonte_a_iet_std`,
        `fonte_a_iet_cv`, `fonte_a_offset_primeiro_evento` (contínuas-NaN),
        `fonte_a_vazia` (bool).
    """
    base = _features_fonte_binada(binado, "contagem_a", "fonte_a")

    volume_por_janela = (
        fonte_a.groupby(_COLUNAS_CHAVE_JANELA, sort=False)["volume"]
        .agg(fonte_a_volume_total="sum", fonte_a_volume_por_timestep_std=lambda s: s.std(ddof=0), fonte_a_volume_por_timestep_min="min", fonte_a_volume_por_timestep_max="max")
        .reset_index()
    )

    resultado = base.merge(volume_por_janela, on=_COLUNAS_CHAVE_JANELA, how="left")

    resultado["fonte_a_volume_medio"] = np.where(
        resultado["fonte_a_n_eventos_total"] > 0,
        resultado["fonte_a_volume_total"] / resultado["fonte_a_n_eventos_total"],
        float("nan"),
    )
    resultado["fonte_a_razao_eventos_volume"] = np.where(
        resultado["fonte_a_vazia"] | (resultado["fonte_a_volume_total"].fillna(0.0) == 0.0),
        float("nan"),
        resultado["fonte_a_n_eventos_total"] / resultado["fonte_a_volume_total"],
    )

    colunas = [
        "classe",
        "window_id",
        "fonte_a_n_eventos_total",
        "fonte_a_volume_total",
        "fonte_a_volume_medio",
        "fonte_a_volume_por_timestep_std",
        "fonte_a_volume_por_timestep_min",
        "fonte_a_volume_por_timestep_max",
        "fonte_a_razao_eventos_volume",
        "fonte_a_iet_media",
        "fonte_a_iet_std",
        "fonte_a_iet_cv",
        "fonte_a_offset_primeiro_evento",
        "fonte_a_vazia",
    ]
    return resultado[colunas]


def extrair_features_bloco_b(binado: pd.DataFrame) -> pd.DataFrame:
    r"""Bloco B (Fonte B / oráculo) — uma linha por `(classe, window_id)`.

    Mesma lista do Bloco A, sem `volume_*`/`razao_eventos_volume` — Fonte B
    não tem dimensão de magnitude (só timestamps, ver
    `layer2_copula/copula.py`). **Não valida duplicata por conta própria**
    (mesma convenção do Bloco A).

    Parameters
    ----------
    binado : pd.DataFrame
        Saída de `binning.bin_janelas`.

    Returns
    -------
    pd.DataFrame
        Uma linha por `(classe, window_id)`: `fonte_b_n_eventos_total`
        (int), `fonte_b_iet_media`, `fonte_b_iet_std`, `fonte_b_iet_cv`,
        `fonte_b_offset_primeiro_evento` (contínuas-NaN), `fonte_b_vazia`
        (bool).
    """
    return _features_fonte_binada(binado, "contagem_b", "fonte_b")


def _correlacao_cruzada_otima(contagem_a: np.ndarray, contagem_b: np.ndarray) -> tuple[float, float]:
    r"""Correlação cruzada A↔B por lag, sobre as séries binadas de UMA
    janela — retorna `(lag_otimo, correlacao_maxima)`.

    **Convenção de sinal**: `lag > 0` significa "B atrasado em relação a
    A" (desloca `contagem_b` para TRÁS por `lag` posições antes de
    correlacionar com `contagem_a` — coerente com `gerar_fonte_b` ser
    função dos timestamps de A: o settlement, quando existe defasagem,
    vem depois do desembolso).

    Cada lag testado usa só a porção sobreposta das duas séries (sem
    padding com zero artificial, que inflaria/desinflaria a correlação
    perto das bordas do intervalo de lag) — normalizada via
    `np.corrcoef` (equivalente a correlação de Pearson após z-score),
    nunca `np.correlate` bruto (não comparável entre janelas de
    magnitudes diferentes).

    Retorna `(nan, nan)` se `contagem_a`/`contagem_b` não tiverem
    variância suficiente em nenhum lag testado (séries constantes —
    inclui o caso de uma das duas ser toda zero, mas quem chama já
    filtra esse caso via `fonte_a_vazia`/`fonte_b_vazia` antes).
    """
    n = contagem_a.size
    lag_maximo = min(n - 1, _LAG_MAXIMO_CORRELACAO_CRUZADA)

    melhor_lag = float("nan")
    melhor_correlacao = float("nan")
    melhor_abs = -1.0

    for lag in range(-lag_maximo, lag_maximo + 1):
        if lag >= 0:
            a_sobreposta = contagem_a[: n - lag]
            b_sobreposta = contagem_b[lag:]
        else:
            a_sobreposta = contagem_a[-lag:]
            b_sobreposta = contagem_b[: n + lag]

        if a_sobreposta.size < 2 or np.std(a_sobreposta) == 0 or np.std(b_sobreposta) == 0:
            continue

        correlacao = float(np.corrcoef(a_sobreposta, b_sobreposta)[0, 1])
        if np.isnan(correlacao):
            continue

        if abs(correlacao) > melhor_abs:
            melhor_abs = abs(correlacao)
            melhor_lag = float(lag)
            melhor_correlacao = correlacao

    return melhor_lag, melhor_correlacao


def extrair_features_coordenacao(binado: pd.DataFrame) -> pd.DataFrame:
    r"""Bloco de coordenação A↔B (Camada 3 do PLANO §5.3.1) — τ_Kendall +
    correlação cruzada em lag τ, uma linha por `(classe, window_id)`.

    **Não valida duplicata por conta própria** (mesma convenção dos
    demais blocos).

    Parameters
    ----------
    binado : pd.DataFrame
        Saída de `binning.bin_janelas` — precisa ter `contagem_a`,
        `contagem_b`.

    Returns
    -------
    pd.DataFrame
        Uma linha por `(classe, window_id)`: `tau_kendall_ab`,
        `lag_correlacao_cruzada_otimo`, `correlacao_cruzada_maxima`
        (todas contínuas-NaN — `NaN` quando `contagem_a` ou `contagem_b`
        são inteiramente zero na janela: sem variância, τ_Kendall e
        correlação de Pearson são indefinidos).
    """
    linhas = []
    for (classe, window_id), grupo in binado.groupby(_COLUNAS_CHAVE_JANELA, sort=False):
        grupo_ordenado = grupo.sort_values("timestep")
        contagem_a = grupo_ordenado["contagem_a"].to_numpy()
        contagem_b = grupo_ordenado["contagem_b"].to_numpy()

        vazia_a = contagem_a.sum() == 0
        vazia_b = contagem_b.sum() == 0

        if vazia_a or vazia_b:
            tau = float("nan")
            lag_otimo = float("nan")
            correlacao_maxima = float("nan")
        else:
            tau_resultado = kendalltau(contagem_a, contagem_b)
            tau = float(tau_resultado.statistic) if not np.isnan(tau_resultado.statistic) else float("nan")
            lag_otimo, correlacao_maxima = _correlacao_cruzada_otima(contagem_a.astype(float), contagem_b.astype(float))

        linhas.append(
            {
                "classe": classe,
                "window_id": window_id,
                "tau_kendall_ab": tau,
                "lag_correlacao_cruzada_otimo": lag_otimo,
                "correlacao_cruzada_maxima": correlacao_maxima,
            }
        )

    return pd.DataFrame(linhas)


def extrair_features_bloco_c(fonte_c: pd.DataFrame) -> pd.DataFrame:
    r"""Bloco C (resultado eleitoral / Fonte C) — wrapper genérico de
    agregação sobre `fracao_candidato_alvo`, uma linha por `(classe,
    window_id)`.

    Chamável sobre `fonte_c_secao` (N=5 unidades hoje), `fonte_c_municipio`
    ou `fonte_c_estado` (N=1 hoje, degenerado sob a hierarquia default —
    `std` sai `NaN`) sem nenhuma reescrita — mesma função para as três
    granularidades. Fonte C **nunca é vazia** (mesmo com
    `contrato_ativado=False`, carrega o resíduo de adesão sincera já
    documentado no CLAUDE.md) — não há flag `_vazia` correspondente.

    **Não valida duplicata por conta própria** (mesma convenção dos
    demais blocos).

    Parameters
    ----------
    fonte_c : pd.DataFrame
        `fonte_c_secao`/`fonte_c_municipio`/`fonte_c_estado` de
        `loader.carregar_arquivo` — precisa ter `classe`, `window_id`,
        `fracao_candidato_alvo`.

    Returns
    -------
    pd.DataFrame
        Uma linha por `(classe, window_id)`: `fonte_c_media`,
        `fonte_c_min`, `fonte_c_max` (contínuas), `fonte_c_std`
        (contínua-NaN, `NaN` se N=1), `fonte_c_n_unidades` (int).
    """
    agregado = fonte_c.groupby(_COLUNAS_CHAVE_JANELA, sort=False)["fracao_candidato_alvo"].agg(
        fonte_c_media="mean",
        fonte_c_min="min",
        fonte_c_max="max",
        fonte_c_std=lambda s: s.std(ddof=1) if s.size > 1 else float("nan"),
        fonte_c_n_unidades="count",
    )
    return agregado.reset_index()


def montar_vetor_features_agregadas(caminho_arquivo: Path) -> dict[str, pd.DataFrame]:
    r"""Função de topo: monta o vetor de features agregadas de UM arquivo
    `.h5` de produção — carrega, bina e junta os quatro blocos.

    Processa um arquivo por vez (mesmo padrão de `binning.bin_janelas` —
    nunca concatenar via `loader.carregar_diretorio` antes de agregar,
    já que `window_id` não é global, é reiniciado em 0 por classe em cada
    arquivo).

    Parameters
    ----------
    caminho_arquivo : Path
        Caminho de um arquivo `.h5` gerado por `gerar_par_de_classes_real`.

    Returns
    -------
    dict[str, pd.DataFrame]
        Duas chaves, ambas com uma linha por `(classe, window_id)`,
        juntáveis por essa mesma chave:

        - `"features"`: vetor de treino do M2 — todas as colunas dos
          quatro blocos, incluindo `fonte_a_vazia`/`fonte_b_vazia`
          (features legítimas de Camada 1). Mantém `classe`/`window_id`
          só como CHAVE DE JUNÇÃO com `"metadados"` (para permitir
          filtrar por `grupo`/`split` depois) — quem monta a matriz de
          treino do M2 deve excluir essas duas colunas antes de treinar
          (mesmo cuidado já documentado para `COLUNAS_METADADO_JANELA`
          em `loader.py`), já que identificam a janela, não descrevem
          seu conteúdo. `ambas_vazias` nunca aparece aqui.
        - `"metadados"`: `classe`, `window_id`, `split`, `grupo`,
          `contrato_ativado` + parâmetros de cenário
          (`COLUNAS_METADADO_CENARIO`) + `ambas_vazias` (flag de
          auditoria — `fonte_a_vazia & fonte_b_vazia`, identifica
          `positiva_inativa` por construção; NUNCA em `"features"`).
    """
    caminho_arquivo = Path(caminho_arquivo)
    carregado = carregar_arquivo(caminho_arquivo)
    metadados_janela = carregado["metadados_janela"]
    fonte_a = carregado["fonte_a"]
    fonte_b = carregado["fonte_b"]
    fonte_c_secao = carregado["fonte_c_secao"]

    binado = bin_janelas(fonte_a, fonte_b, metadados_janela)

    bloco_a = extrair_features_bloco_a(binado, fonte_a)
    bloco_b = extrair_features_bloco_b(binado)
    bloco_coordenacao = extrair_features_coordenacao(binado)
    bloco_c = extrair_features_bloco_c(fonte_c_secao)

    vetor = (
        metadados_janela[list(_COLUNAS_METADADO_JANELA_AGREGADA)]
        .merge(bloco_a, on=_COLUNAS_CHAVE_JANELA, how="left")
        .merge(bloco_b, on=_COLUNAS_CHAVE_JANELA, how="left")
        .merge(bloco_coordenacao, on=_COLUNAS_CHAVE_JANELA, how="left")
        .merge(bloco_c, on=_COLUNAS_CHAVE_JANELA, how="left")
    )

    vetor["ambas_vazias"] = vetor["fonte_a_vazia"] & vetor["fonte_b_vazia"]

    colunas_features = [
        c
        for c in vetor.columns
        if c not in set(_COLUNAS_METADADO_JANELA_AGREGADA) | {"ambas_vazias"}
    ]
    colunas_metadados = list(_COLUNAS_METADADO_JANELA_AGREGADA) + ["ambas_vazias"]

    return {
        "features": vetor[_COLUNAS_CHAVE_JANELA + colunas_features].reset_index(drop=True),
        "metadados": vetor[colunas_metadados].reset_index(drop=True),
    }
