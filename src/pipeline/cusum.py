r"""M1 — baseline estatístico de detecção (PLANO §5.3.2): CUSUM unilateral
sobre o z-score por timestep já calculado em `binning.calcular_zscore`.

Três peças, nesta ordem de dependência:

1. `cusum_unilateral` — CUSUM unilateral padrão para UMA série de z-score
   (uma janela, uma fonte).
2. `aplicar_cusum_por_janela` — aplica `cusum_unilateral` a `contagem_a` e
   `contagem_b` de TODAS as janelas de um DataFrame binado/com z-score,
   vetorizado por `(classe, window_id)` (groupby, não `iterrows` — mesmo
   padrão de `binning._bin_fonte_b_vetorizado`).
3. `calcular_metricas_m1` — F1/precisão/recall/FPR, SEMPRE estratificado
   por `(grupo, delta_t, recompensa, pi)`, nunca um número agregado único.

Mais `fracao_disparos_no_piso_da_referencia` — função auxiliar para
responder se os disparos do CUSUM estão concentrados em timesteps onde a
referência usava o piso `binning._EPSILON_DESVIO_PADRAO_ZERO` (variância
artificial) em vez de desvio-padrão real estimado.

**Duas decisões de design, resolvidas com o usuário antes da implementação
(não escolhidas em silêncio):**

1. **CUSUM independente por fonte (A e B), combinado por OR.** Dois
   acumuladores separados — um sobre `zscore_a`, outro sobre `zscore_b` —
   cada um com seu próprio disparo; a janela é marcada como detectada
   (`disparo_m1`) se QUALQUER um dos dois cruzar `h`. Preserva
   interpretabilidade de qual fonte disparou (relevante para a ablação
   C1-C7 futura) e evita que uma fonte ruidosa dilua o sinal da outra —
   uma única série combinada (soma/máximo de z-scores antes do CUSUM)
   perderia essa granularidade sem ganho compensador, já que Fonte A e
   Fonte B têm mecanismos de mascaramento/ruído estruturalmente diferentes
   (π mascara só A; ver ponto 3 abaixo).

2. **Timesteps `timestep_estruturalmente_impossivel` são PULADOS, não
   zerados.** O acumulador `S+` congela no valor do último timestep válido
   — timesteps impossíveis não contam nem a favor nem contra a detecção
   (não são "zero informativo": zero informativo seria "não houve evento,
   mas poderia ter havido"; aqui não poderia ter havido, por construção do
   gerador — ver `binning.py`, docstring do módulo, para o mecanismo
   completo). **Consequência que exigiu atenção na implementação:** o
   critério de disparo é "S+ cruzou h em QUALQUER timestep da varredura"
   (flag sticky), nunca "valor de S+ no timestep final" — em
   `positiva/delta_t=0.0`, toda a evidência real está concentrada no único
   timestep válido (`timestep=0`); se o critério fosse lido só do estado
   final após pular os timesteps 1-23 (todos impossíveis), um cruzamento
   em `timestep=0` seria perdido silenciosamente. Testado explicitamente
   (`tests/test_pipeline_cusum.py`).

3. **`calcular_metricas_m1` estratifica por `(delta_t, recompensa, pi)`,
   incluindo `pi` — não só `delta_t × recompensa`.** π mascara
   especificamente Fonte A (Bernoulli, via
   `src.generator.privacidade.mascara_sobrevivencia_pi`) — como CUSUM_A e
   CUSUM_B já rodam separados (decisão 1 acima), omitir `pi` do
   agrupamento esconderia o efeito mais relevante para o TCC: detecção via
   Fonte A degradando conforme π aumenta, enquanto Fonte B
   (estruturalmente irredutível mesmo sob π→1 — ver `docs/adversary_model_draft.tex`,
   §"What Remains Observable Even as π→1") não degrada da mesma forma.
   **`grupo` NÃO entra nesse agrupamento** (correção de design feita
   durante a implementação, ver docstring de `calcular_metricas_m1`): se
   entrasse, cada estrato teria rótulo binário constante e a matriz de
   confusão degeneraria (nunca `vp` e `fp` na mesma linha). Em vez disso,
   dentro de cada `(delta_t, recompensa, pi)`, as negativas DAQUELE MESMO
   estrato são pareadas separadamente com `positiva_ativa` e com
   `positiva_inativa`, produzindo duas linhas por estrato de combinação —
   cada uma com F1/precisão/recall/FPR completos e não-degenerados.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.binning import _EPSILON_DESVIO_PADRAO_ZERO

_COLUNAS_COMBINACAO_METRICAS = ("delta_t", "recompensa", "pi")
"""Eixos de combinação usados por `calcular_metricas_m1` para agrupar
positivas e negativas a comparar entre si — NÃO inclui `grupo` (ver
docstring de `calcular_metricas_m1` para por que `grupo` participa do
resultado, mas não do `groupby`) nem `rho`/`beta`/`g` (constantes no
dataset v3 atual em tudo que não é `g="pool"`; podem ser adicionados no
futuro se o design fatorial variar esses eixos — ver CLAUDE.md, pendência
de `g` fixo)."""

_COLUNAS_METADADO_JANELA_CUSUM = ("classe", "window_id", "grupo", "split", "delta_t", "recompensa", "pi", "rho", "beta", "g", "seed")
"""Colunas de identificação/metadado repassadas por `aplicar_cusum_por_janela`
— mesmas colunas de `loader.COLUNAS_METADADO_JANELA`/`COLUNAS_METADADO_CENARIO`
combinadas, já presentes na saída de `binning.bin_janelas`/`calcular_zscore`."""

_CHAVE_JANELA_CUSUM = ("classe", "window_id", "delta_t", "recompensa", "pi", "rho", "beta", "g", "seed")
"""Chave completa que identifica uma janela SEM ambiguidade quando
`com_zscore` vem de múltiplos arquivos/combinações concatenados —
`(classe, window_id)` sozinho NÃO basta: `window_id` é reiniciado em 0 por
classe em CADA arquivo/combinação (mesmo motivo já documentado em
`binning.bin_janelas`, que agora valida isso explicitamente). Achado real,
não hipotético: a primeira execução de `scripts/m1_cusum_grid_completo.py`
sobre o v3 completo, ANTES desta correção, agrupou por só `(classe,
window_id)` e produziu 2.000 "janelas" em vez das 540.000 esperadas —
janelas de até 135 combinações diferentes que coincidiam em `window_id`
foram misturadas na mesma sequência de acumulador CUSUM, produzindo FPR=1.0
espúrio (sequências de z-score de várias combinações concatenadas quase
sempre cruzam qualquer h razoável). `seed` entra aqui (diferente de
`_COLUNAS_COMBINACAO_METRICAS`, usada só por `calcular_metricas_m1` para
AGRUPAR/comparar depois que cada janela já foi processada individualmente)
porque a MESMA combinação de eixos do design fatorial roda com seeds
distintas, cada uma um arquivo separado, cada um reiniciando `window_id`
em 0 de novo."""


def cusum_unilateral(
    zscore: np.ndarray,
    k: float,
    h: float,
    impossivel: np.ndarray | None = None,
) -> tuple[bool, int | None]:
    r"""CUSUM unilateral (detecção de desvio POSITIVO) sobre uma série de
    z-score de UMA janela/fonte.

    Recorrência padrão: ``S+_t = max(0, S+_{t-1} + zscore_t - k)``,
    ``S+_{-1} = 0``. Disparo quando ``S+_t >= h`` em algum ``t`` — critério
    testado ao longo de TODA a varredura (flag sticky), não só no valor
    final de `S+` (ver decisão de design 2 na docstring do módulo).

    Parameters
    ----------
    zscore : np.ndarray
        Série de z-score por timestep, uma janela/uma fonte (`zscore_a` OU
        `zscore_b` de `binning.calcular_zscore`, nunca as duas juntas —
        ver decisão de design 1).
    k : float
        Slack/sensibilidade — subtraído do z-score a cada timestep antes
        de acumular; controla quanto desvio "tolerar" sem disparar (maior
        `k` = menos sensível a desvios pequenos/sustentados). Parâmetro
        explícito, sem default aqui — calibração de `k`/`h` é decisão de
        quem chama (mesma disciplina de `K_RESULTADO_ALVO`/`tau_kendall`
        em `config.py`/`ParametrosStubGeracao`, ver CLAUDE.md), justificada
        no ponto de uso (`scripts/m1_cusum_grid_completo.py`), não
        escondida aqui.
    h : float
        Limiar de disparo — quando `S+` atinge ou ultrapassa `h`, a janela
        é marcada como detectada por esta fonte.
    impossivel : np.ndarray | None
        Máscara booleana, mesmo tamanho de `zscore` — `True` nos timesteps
        `timestep_estruturalmente_impossivel` (ver `binning.bin_janelas`).
        Esses timesteps são PULADOS na recorrência: `S+` não avança nem
        retrocede neles, o próximo timestep válido continua a partir do
        último `S+` válido. `None` (default) equivale a nenhum timestep
        impossível (todos entram na recorrência normalmente).

    Returns
    -------
    tuple[bool, int | None]
        ``(disparou, timestep_disparo)`` — `disparou` é `True` se `S+`
        cruzou `h` em qualquer timestep válido da varredura;
        `timestep_disparo` é o índice (posição em `zscore`, não um valor
        de tempo derivado) do PRIMEIRO timestep em que o cruzamento
        ocorreu, ou `None` se nunca disparou. Não retorna a trajetória
        completa de `S+` — só o necessário para identificar e localizar o
        primeiro disparo.

    Raises
    ------
    ValueError
        Se `h <= 0` (limiar não-positivo não tem leitura sensata — `S+`
        parte de 0 e só cresce, disparando trivialmente em `t=0` mesmo sem
        nenhum desvio) ou se `impossivel` tiver tamanho diferente de
        `zscore`.
    """
    if h <= 0:
        raise ValueError(f"h deve ser > 0 (recebido {h!r}) — limiar não-positivo dispara trivialmente em t=0.")

    zscore = np.asarray(zscore, dtype=float)
    n = zscore.size

    if impossivel is None:
        impossivel = np.zeros(n, dtype=bool)
    else:
        impossivel = np.asarray(impossivel, dtype=bool)
        if impossivel.size != n:
            raise ValueError(f"impossivel deve ter o mesmo tamanho de zscore ({n}), recebido {impossivel.size}.")

    s_mais = 0.0
    for t in range(n):
        if impossivel[t]:
            continue
        s_mais = max(0.0, s_mais + zscore[t] - k)
        if s_mais >= h:
            return True, t

    return False, None


def _cusum_grupo(grupo: pd.DataFrame, k: float, h: float) -> pd.Series:
    """Aplica `cusum_unilateral` a `contagem_a`/`contagem_b` de UMA janela
    (já ordenada por `timestep` — ver `aplicar_cusum_por_janela`), via
    `zscore_a`/`zscore_b`/`timestep_estruturalmente_impossivel`.

    Colunas de METADADO da janela (`grupo`, `split`, e as próprias colunas
    da chave do `groupby`) não são lidas aqui — `groupby(...).apply(...,
    include_groups=False)` já as remove de `grupo` antes de chegar nesta
    função (evita o `FutureWarning`/comportamento deprecado do pandas de
    operar sobre as colunas de agrupamento). `aplicar_cusum_por_janela`
    reanexa a chave via `reset_index()` depois do `groupby`; `grupo` (o
    subgrupo `positiva_ativa`/`positiva_inativa`/`negativa`) e `split`
    precisam ser lidos de fora desta função por quem chama, já que não
    fazem parte da chave e são removidos junto pelo `include_groups=False`
    — ver `aplicar_cusum_por_janela` para como são reanexados."""
    disparo_a, timestep_disparo_a = cusum_unilateral(
        grupo["zscore_a"].to_numpy(), k=k, h=h, impossivel=grupo["timestep_estruturalmente_impossivel"].to_numpy()
    )
    disparo_b, timestep_disparo_b = cusum_unilateral(
        grupo["zscore_b"].to_numpy(), k=k, h=h, impossivel=grupo["timestep_estruturalmente_impossivel"].to_numpy()
    )
    primeira_linha = grupo.iloc[0]
    dados = {
        "disparo_a": disparo_a,
        "disparo_b": disparo_b,
        "disparo_m1": disparo_a or disparo_b,
        "timestep_disparo_a": timestep_disparo_a,
        "timestep_disparo_b": timestep_disparo_b,
    }
    for coluna in ("grupo", "split"):
        if coluna in grupo.columns:
            dados[coluna] = primeira_linha[coluna]
    return pd.Series(dados)


def aplicar_cusum_por_janela(com_zscore: pd.DataFrame, k: float, h: float) -> pd.DataFrame:
    r"""Aplica CUSUM unilateral (Fonte A e Fonte B, separados — decisão de
    design 1) a CADA janela de `com_zscore`, vetorizado por
    `_CHAVE_JANELA_CUSUM` (não só `(classe, window_id)`).

    **Bug corrigido durante a validação no grid completo do v3, não
    hipotético:** a versão original agrupava só por `(classe, window_id)`
    — insuficiente quando `com_zscore` vem de MÚLTIPLOS arquivos/combinações
    concatenados (`window_id` é reiniciado em 0 por classe em CADA
    arquivo/combinação, mesmo motivo já documentado em
    `binning.bin_janelas`). Rodar assim sobre o v3 completo colapsou
    540.000 janelas em 2.000 grupos (janelas de até 135 combinações
    diferentes coincidindo em `window_id` foram misturadas na mesma
    sequência de acumulador CUSUM) e produziu FPR=1.0 espúrio em todas as
    negativas. Corrigido agrupando por `_CHAVE_JANELA_CUSUM` — inclui
    `seed` e todos os eixos de combinação, não só `pi`/`delta_t`/`recompensa`
    (que `calcular_metricas_m1` usa depois para AGRUPAR janelas já
    processadas, um propósito diferente de desambiguar QUAIS linhas
    pertencem à mesma janela aqui).

    Parameters
    ----------
    com_zscore : pd.DataFrame
        Saída de `binning.calcular_zscore` — precisa ter `classe`,
        `window_id`, `timestep`, `zscore_a`, `zscore_b`,
        `timestep_estruturalmente_impossivel`, mais TODAS as colunas de
        `_CHAVE_JANELA_CUSUM` (`delta_t`, `recompensa`, `pi`, `rho`, `beta`,
        `g`, `seed`) — obrigatórias agora, não opcionais, porque são a
        chave que desambigua janelas de combinações diferentes.
    k, h : float
        Parâmetros de `cusum_unilateral`, mesmos para as duas fontes e
        todas as janelas desta chamada.

    Returns
    -------
    pd.DataFrame
        Uma linha por janela (chave `_CHAVE_JANELA_CUSUM`), colunas
        ``disparo_a``, ``disparo_b``, ``disparo_m1`` (OR de
        `disparo_a`/`disparo_b`), ``timestep_disparo_a``,
        ``timestep_disparo_b`` (int ou `NaN` se não disparou), mais as
        colunas de metadado presentes em `com_zscore` (repetidas da
        primeira linha de cada janela — são constantes dentro da janela,
        por construção de `bin_janelas`).

    Raises
    ------
    ValueError
        Se `com_zscore` estiver vazio, não tiver as colunas obrigatórias,
        ou tiver `(classe, window_id, timestep)` duplicado DENTRO da chave
        completa de combinação — cada timestep deve aparecer exatamente
        uma vez por janela; duplicata ali corromperia silenciosamente a
        sequência do acumulador CUSUM se não fosse rejeitada.
    """
    colunas_obrigatorias = (
        {"classe", "window_id", "timestep", "zscore_a", "zscore_b", "timestep_estruturalmente_impossivel"}
        | set(_CHAVE_JANELA_CUSUM)
    )
    faltando = colunas_obrigatorias - set(com_zscore.columns)
    if faltando:
        raise ValueError(f"com_zscore está sem as colunas obrigatórias: {sorted(faltando)}")
    if com_zscore.empty:
        raise ValueError("com_zscore está vazio — não há nenhuma janela para aplicar CUSUM.")

    chave = list(_CHAVE_JANELA_CUSUM)
    duplicadas = com_zscore.duplicated(subset=chave + ["timestep"], keep=False)
    if duplicadas.any():
        raise ValueError(
            "com_zscore contém (classe, window_id, timestep) duplicado DENTRO da chave completa de "
            "combinação (delta_t, recompensa, pi, rho, beta, g, seed) — cada timestep deveria aparecer "
            "exatamente uma vez por janela. Isso indica dado corrompido/mal formado na entrada (não é o "
            "mesmo problema já coberto por bin_janelas, que valida window_id duplicado ENTRE combinações "
            "diferentes — aqui a ambiguidade persiste mesmo com a chave completa), e corromperia "
            "silenciosamente a sequência do acumulador CUSUM se não fosse rejeitado aqui."
        )

    ordenado = com_zscore.sort_values(chave + ["timestep"])
    resultado = ordenado.groupby(chave, sort=False, group_keys=False).apply(
        lambda grupo: _cusum_grupo(grupo, k=k, h=h), include_groups=False
    )
    resultado = resultado.reset_index()  # reanexa a chave (_CHAVE_JANELA_CUSUM) como colunas
    resultado["timestep_disparo_a"] = resultado["timestep_disparo_a"].astype("Int64")
    resultado["timestep_disparo_b"] = resultado["timestep_disparo_b"].astype("Int64")
    return resultado


def calcular_metricas_m1(previsoes: pd.DataFrame) -> pd.DataFrame:
    r"""F1/precisão/recall/FPR de M1, SEMPRE estratificado por
    `(grupo, delta_t, recompensa, pi)` — nunca um número agregado único
    (ver decisão de design 3 na docstring do módulo, sobre a inclusão de
    `pi`).

    **Correção de design, decidida com o usuário durante a implementação:**
    `grupo` NÃO entra na chave de agrupamento (embora esteja em
    `_COLUNAS_ESTRATO_METRICAS` — usada só para localizar as colunas de
    combinação, não para o `groupby`, ver `_COLUNAS_COMBINACAO_METRICAS`
    abaixo). Se `grupo` entrasse diretamente no `groupby`, cada estrato
    teria rótulo binário CONSTANTE (`positiva_* `→ sempre positivo,
    `negativa` → sempre negativo) e a matriz de confusão degeneraria: um
    estrato "positiva_ativa" só teria `vp`/`fn` (nunca `vn`/`fp`), tornando
    F1/FPR indefiníveis de verdade. Em vez disso, dentro de cada
    `(delta_t, recompensa, pi)`, as janelas **negativas daquele MESMO
    estrato** (nunca um pool geral de negativas de outros `delta_t`/
    `recompensa`/`pi` — mesmo princípio já usado na referência de
    `calcular_referencia_negativa_treino`) são pareadas separadamente com
    `positiva_ativa` e com `positiva_inativa`, produzindo DUAS linhas por
    `(delta_t, recompensa, pi)` — uma por subgrupo positivo — cada uma com
    matriz de confusão completa e não-degenerada. `fpr` sai igual nas duas
    linhas do mesmo `(delta_t, recompensa, pi)` (mesmo conjunto de
    negativas reaproveitado); `vp`/`fn`/`precisao`/`recall`/`f1` mudam
    conforme o subgrupo positivo usado.

    Parameters
    ----------
    previsoes : pd.DataFrame
        Saída de `aplicar_cusum_por_janela` — precisa ter `disparo_m1` e
        `grupo`, `delta_t`, `recompensa`, `pi`.

    Returns
    -------
    pd.DataFrame
        Uma linha por `(subgrupo_positivo, delta_t, recompensa, pi)`, onde
        `subgrupo_positivo` ∈ {"positiva_ativa", "positiva_inativa"} —
        colunas ``n_positivas``, ``n_negativas``, ``vp`` (verdadeiro
        positivo), ``fp``, ``vn``, ``fn``, ``precisao``, ``recall``, ``f1``,
        ``fpr`` (falso positivo / (falso positivo + verdadeiro negativo)).
        Métricas são `NaN` quando o denominador correspondente é zero (ex.
        `precisao` sem nenhum disparo no estrato) — não `0.0` silencioso,
        para não confundir "sem dado" com "desempenho zero". Se um
        `(delta_t, recompensa, pi)` não tiver NENHUMA janela negativa,
        esse estrato é omitido do resultado (não há como calcular
        `fp`/`vn`/`fpr` sem negativas) — não levanta erro, só não aparece.

    Raises
    ------
    ValueError
        Se `previsoes` não tiver as colunas obrigatórias.
    """
    colunas_combinacao = list(_COLUNAS_COMBINACAO_METRICAS)
    faltando = (set(colunas_combinacao) | {"disparo_m1", "grupo"}) - set(previsoes.columns)
    if faltando:
        raise ValueError(f"previsoes está sem as colunas obrigatórias: {sorted(faltando)}")

    linhas = []
    for chave, estrato in previsoes.groupby(colunas_combinacao, sort=False):
        negativas = estrato[estrato["grupo"] == "negativa"]
        if negativas.empty:
            continue  # sem negativas nesse estrato -- fp/vn/fpr indefiníveis, estrato omitido

        fp = int(negativas["disparo_m1"].sum())
        vn = int((~negativas["disparo_m1"]).sum())
        fpr = fp / (fp + vn) if (fp + vn) > 0 else float("nan")

        for subgrupo_positivo in ("positiva_ativa", "positiva_inativa"):
            positivas = estrato[estrato["grupo"] == subgrupo_positivo]
            vp = int(positivas["disparo_m1"].sum())
            fn = int((~positivas["disparo_m1"]).sum())

            precisao = vp / (vp + fp) if (vp + fp) > 0 else float("nan")
            recall = vp / (vp + fn) if (vp + fn) > 0 else float("nan")
            f1 = (
                2 * precisao * recall / (precisao + recall)
                if not np.isnan(precisao) and not np.isnan(recall) and (precisao + recall) > 0
                else float("nan")
            )

            linha = dict(zip(colunas_combinacao, chave))
            linha.update(
                {
                    "subgrupo_positivo": subgrupo_positivo,
                    "n_positivas": len(positivas),
                    "n_negativas": len(negativas),
                    "vp": vp, "fp": fp, "vn": vn, "fn": fn,
                    "precisao": precisao, "recall": recall, "f1": f1, "fpr": fpr,
                }
            )
            linhas.append(linha)

    return pd.DataFrame(linhas)


def fracao_disparos_no_piso_da_referencia(
    previsoes: pd.DataFrame,
    referencia: pd.DataFrame,
) -> pd.DataFrame:
    r"""Item 5 do pedido original — para os disparos de CUSUM, verifica se
    o timestep de disparo (`timestep_disparo_a`/`timestep_disparo_b`) usava
    um desvio-padrão de referência no piso `_EPSILON_DESVIO_PADRAO_ZERO`
    (variância artificial, ver `binning.calcular_referencia_negativa_treino`)
    ou desvio-padrão real estimado — para distinguir sinal real de artefato
    do piso.

    Junta internamente `previsoes` (uma linha por JANELA, com o índice de
    timestep do disparo) com `referencia` (uma linha por
    `combinação × timestep`, com `desvio_a`/`desvio_b`) — o chamador não
    precisa montar esse merge; só precisa passar a MESMA `referencia` usada
    para calcular o z-score que alimentou o CUSUM (`binning.calcular_zscore`).

    Parameters
    ----------
    previsoes : pd.DataFrame
        Saída de `aplicar_cusum_por_janela` — uma linha por janela, com
        `disparo_a`/`disparo_b`, `timestep_disparo_a`/`timestep_disparo_b`
        e as colunas de combinação (`recompensa`, `pi`, `delta_t`, `rho`,
        `beta`, `g`).
    referencia : pd.DataFrame
        Saída de `binning.calcular_referencia_negativa_treino` — mesma
        referência usada para gerar o z-score de `previsoes`.

    Returns
    -------
    pd.DataFrame
        Uma linha por ``fonte`` ∈ {"a", "b"}, colunas ``n_disparos``,
        ``n_disparos_no_piso``, ``fracao_no_piso``. `fracao_no_piso` é
        `NaN` se não houve nenhum disparo daquela fonte (não `0.0`
        silencioso — "sem disparo" é diferente de "0% dos disparos usaram
        o piso").
    """
    colunas_combinacao = ["recompensa", "pi", "delta_t", "rho", "beta", "g"]

    linhas = []
    for fonte in ("a", "b"):
        col_disparo = f"disparo_{fonte}"
        col_timestep = f"timestep_disparo_{fonte}"
        col_desvio_ref = f"desvio_{fonte}"

        disparos = previsoes[previsoes[col_disparo]].copy()
        n_disparos = len(disparos)
        if n_disparos == 0:
            linhas.append({"fonte": fonte, "n_disparos": 0, "n_disparos_no_piso": 0, "fracao_no_piso": float("nan")})
            continue

        disparos["timestep"] = disparos[col_timestep].astype(int)
        juntado = disparos.merge(
            referencia[colunas_combinacao + ["timestep", col_desvio_ref]],
            on=colunas_combinacao + ["timestep"],
            how="left",
        )

        no_piso = juntado[col_desvio_ref] <= _EPSILON_DESVIO_PADRAO_ZERO
        n_no_piso = int(no_piso.sum())
        linhas.append(
            {
                "fonte": fonte,
                "n_disparos": n_disparos,
                "n_disparos_no_piso": n_no_piso,
                "fracao_no_piso": n_no_piso / n_disparos,
            }
        )

    return pd.DataFrame(linhas)
