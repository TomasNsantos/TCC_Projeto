r"""M2 — XGBoost + SHAP (PLANO §5.3.2), Dia 6, Semana 9-10.

Consome a camada agregada de features do Dia 4-5
(`features_agregadas.montar_vetor_features_agregadas`, persistida em
`output/features_agregadas_v3/{features,metadados}.parquet`) e treina o
primeiro detector supervisionado do projeto — uma prova de conceito no
grid completo, não a ablação fatorial C1-C7 (§5.4.1, Semanas 14-17).

Sete peças, nesta ordem de dependência:

1. `carregar_dataset_m2` — lê os dois parquets, confirma alinhamento
   posicional.
2. `preparar_matriz_treino` — monta (X, y) a partir de features/metadados
   já carregados, sem filtrar por split.
3. `treinar_xgboost` — `XGBClassifier` com hiperparâmetros default, `val`
   usado só para early stopping (não é tuning).
4. `calcular_metricas_m2` — F1/precisão/recall/AUROC/FPR, estratificado
   por `(delta_t, recompensa, pi) × subgrupo_positivo` — réplica exata da
   estrutura de `cusum.calcular_metricas_m1`.
5. `amostrar_para_shap` — amostra estratificada por `grupo × pi × delta_t`
   do split de teste, para rodar SHAP sem processar o teste inteiro.
6. `calcular_shap_values` — `shap.TreeExplainer` sobre a amostra.
7. `ranking_importancia_shap` — duas visões do mesmo `shap_values` já
   calculado (agregado e por π), sem custo adicional de tempo.

**Decisões fechadas pelo usuário antes da implementação (plano aprovado
em `1-separar-fonte-quizzical-feather.md`), não reabertas aqui:**

1. **Target binário**: `negativa=0`, `{positiva_ativa, positiva_inativa}=1`.
   `positiva_inativa` é 100% determinística via `fonte_a_vazia`/
   `fonte_b_vazia` (assimetria estrutural do gerador, ver CLAUDE.md,
   "Peculiaridade — positiva_inativa e ausência total em Fonte A/B") —
   aceita como característica do dataset nesta rodada, não suprimida.
2. **Features de entrada**: todas as colunas de `"features"` de
   `features_agregadas_v3`, exceto `_COLUNAS_EXCLUIDAS_FEATURES` (chave
   de junção posicional, nunca dado de treino) e qualquer parâmetro de
   cenário (só em `"metadados"`, nunca em `"features"` — convenção já
   herdada de `loader.py`/`binning.py`/`cusum.py`/`features_agregadas.py`,
   não duplicada aqui).
3. **Lacunas do §5.3.1** (z-score agregado por janela, entropia temporal,
   desvio de IET vs. Poisson, mutual information ativação×resultado por
   `g`) ficam FORA desta rodada — pendência v0, não implementadas aqui.
4. **Split**: usa a coluna `"split"` já existente em `metadados`
   (`train`/`val`/`test`), calculada por `storage.calcular_split` — por
   JANELA INDIVIDUAL dentro de cada arquivo/combinação, não por
   seed/cenário inteiro (auditoria anterior, ver CLAUDE.md). **Nota
   metodológica**: esse split é INTRA-CENÁRIO — train/val/test da mesma
   combinação de parâmetros compartilham a mesma distribuição
   paramétrica. Métricas medidas aqui refletem reconhecer o padrão dado
   que o regime já foi visto no treino, não generalização entre regimes
   nunca vistos (isso é o design fatorial completo, §5.4, Semanas 14-17).
5. **Hiperparâmetros**: `XGBClassifier` com defaults, sem tuning/CV —
   prova de conceito. `val` é usado só para `early_stopping_rounds`
   (decide quando parar de adicionar árvores, não busca hiperparâmetro).
6. **Métricas sempre quebradas por `(delta_t, recompensa, pi) ×
   subgrupo_positivo`** — nunca um número agregado único, mesma
   disciplina de M1.
7. **SHAP via `TreeExplainer`** sobre amostra estratificada
   (`grupo × pi × delta_t`, ~15-20k linhas do teste), não o teste
   completo.
8. **Checagem de dominância em duas etapas** (completa → condicional sem
   `_COLUNAS_CAMADA_1_CRUA`), leitura humana do ranking, sem corte
   numérico automático no código.
9. **Sem helper `juntar_por_janela` nesta rodada** — `features`/
   `metadados` já vêm alinhados posicionalmente de
   `montar_vetor_features_agregadas`, sem merge adicional necessário.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost
from sklearn.metrics import roc_auc_score

_COLUNAS_EXCLUIDAS_FEATURES = ("classe", "window_id")
"""Colunas presentes em `features` só como chave de junção posicional
com `metadados` — nunca entram no vetor de treino."""

_COLUNAS_CAMADA_1_CRUA = (
    "fonte_a_n_eventos_total",
    "fonte_a_volume_total",
    "fonte_a_volume_medio",
    "fonte_a_vazia",
    "fonte_b_n_eventos_total",
    "fonte_b_vazia",
)
"""Etapa 8b (decisão 8): contagem/volume bruto de A/B + as duas flags de
vazio — candidatas a exclusão condicional se dominarem o ranking SHAP.

**`fonte_a_vazia`/`fonte_b_vazia` dominarem o ranking não é o mesmo
achado que `fonte_a_n_eventos_total`/`fonte_a_volume_total`/
`fonte_a_volume_medio`/`fonte_b_n_eventos_total` dominarem** — as duas
flags dominam estruturalmente por causa de `positiva_inativa` (100%
vazia por construção, decisão 1), não por vazamento; as quatro colunas
de contagem/volume dominarem é o sinal que a checagem 8a/8b foi
desenhada para pegar (ver seção "Protocolo de SHAP e checagem de
dominância" do plano). As seis colunas são tratadas como um bloco só
para a decisão de RE-TREINAR (8b); a LEITURA do ranking (qual das duas
categorias domina e por quê) é responsabilidade de quem lê o relatório
final, não deste módulo."""

_COLUNAS_COMBINACAO_METRICAS = ("delta_t", "recompensa", "pi")
"""Mesmos eixos de `cusum._COLUNAS_COMBINACAO_METRICAS` — `grupo` NÃO
entra aqui (seria usado no groupby direto e degeneraria a matriz de
confusão, ver `calcular_metricas_m2`)."""


def carregar_dataset_m2(
    caminho_features: Path, caminho_metadados: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    r"""Carrega `features.parquet`/`metadados.parquet` (saída consolidada
    de `features_agregadas.montar_vetor_features_agregadas` sobre o grid
    completo, ver `scripts/features_agregadas_grid_completo.py`) e
    confirma alinhamento posicional — mesma checagem já validada nesse
    script, agora parte do módulo de produção.

    **Não faz merge por `(classe, window_id)`** — essa chave sozinha NÃO
    é única no dataset consolidado (window_id é reiniciado em 0 por
    classe em cada uma das 270 combinações do grid, ver
    `binning.py`/`cusum.py`/CLAUDE.md — pendência "helper de junção por
    (classe, window_id)"). `features`/`metadados` já vêm alinhados
    posicionalmente (mesmo índice, mesma ordem) por construção de
    `montar_vetor_features_agregadas` — este carregamento só confirma
    essa garantia, nunca reconstrói o alinhamento via merge.

    Parameters
    ----------
    caminho_features, caminho_metadados : Path
        Caminhos dos dois arquivos parquet.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        `(features, metadados)`, intactos (sem juntar) — quem chama
        decide como usar cada um.

    Raises
    ------
    ValueError
        Se os dois DataFrames tiverem tamanhos diferentes, ou se
        `classe`/`window_id` não baterem linha a linha entre eles (sinal
        de que o alinhamento posicional foi quebrado — ex. um dos
        parquets foi reordenado/filtrado independentemente do outro).
    """
    features = pd.read_parquet(Path(caminho_features))
    metadados = pd.read_parquet(Path(caminho_metadados))

    if len(features) != len(metadados):
        raise ValueError(
            f"features ({len(features)} linhas) e metadados ({len(metadados)} linhas) "
            "têm tamanhos diferentes — alinhamento posicional quebrado."
        )
    if not (features["classe"].to_numpy() == metadados["classe"].to_numpy()).all():
        raise ValueError("coluna 'classe' não bate linha a linha entre features e metadados — alinhamento posicional quebrado.")
    if not (features["window_id"].to_numpy() == metadados["window_id"].to_numpy()).all():
        raise ValueError("coluna 'window_id' não bate linha a linha entre features e metadados — alinhamento posicional quebrado.")

    return features, metadados


def preparar_matriz_treino(
    features: pd.DataFrame,
    metadados: pd.DataFrame,
    excluir_colunas: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.Series]:
    r"""Monta `(X, y)` a partir de `features`/`metadados` JÁ ALINHADOS
    posicionalmente (ver `carregar_dataset_m2`) — não filtra por split,
    isso é responsabilidade de quem chama (ex. aplicar uma máscara
    booleana sobre `metadados["split"]` antes ou depois desta função).

    Parameters
    ----------
    features : pd.DataFrame
        Saída de `carregar_dataset_m2`/`montar_vetor_features_agregadas`
        — precisa ter `_COLUNAS_EXCLUIDAS_FEATURES`.
    metadados : pd.DataFrame
        Mesmo índice posicional de `features` — precisa ter `grupo`.
    excluir_colunas : tuple[str, ...]
        Colunas extras a excluir de X, além de
        `_COLUNAS_EXCLUIDAS_FEATURES` — usado pela etapa 8b para remover
        `_COLUNAS_CAMADA_1_CRUA`. Default vazio (etapa 8a, todas as
        features).

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        `X` (features sem as colunas excluídas), `y` (binário: `0` se
        `grupo == "negativa"`, `1` caso contrário — decisão 1, inclui
        `positiva_inativa` no rótulo positivo).

    Raises
    ------
    ValueError
        Se `len(features) != len(metadados)`.
    """
    if len(features) != len(metadados):
        raise ValueError(f"features ({len(features)} linhas) e metadados ({len(metadados)} linhas) com tamanhos diferentes.")

    colunas_excluidas = set(_COLUNAS_EXCLUIDAS_FEATURES) | set(excluir_colunas)
    x = features.drop(columns=[c for c in colunas_excluidas if c in features.columns])
    y = (metadados["grupo"] != "negativa").astype(int)
    y.name = "y"

    return x, y


def treinar_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    early_stopping_rounds: int = 20,
) -> xgboost.XGBClassifier:
    r"""Treina `XGBClassifier` com hiperparâmetros DEFAULT (decisão 5,
    sem tuning/CV) — única exceção é `early_stopping_rounds`, que usa
    `x_val`/`y_val` só para decidir quando parar de adicionar árvores
    (não busca nenhum hiperparâmetro, não conta como tuning). Resolve o
    ponto que a decisão 4 do plano deixou em aberto para `val`.

    Parameters
    ----------
    x_train, y_train : pd.DataFrame, pd.Series
        Matriz/rótulo de treino — saída de `preparar_matriz_treino`
        filtrada por `split == "train"`.
    x_val, y_val : pd.DataFrame, pd.Series
        Idem, filtrada por `split == "val"` — usada só para early
        stopping.
    early_stopping_rounds : int
        Repassado direto ao `XGBClassifier` (default `20`).

    Returns
    -------
    xgboost.XGBClassifier
        Modelo treinado (`eval_metric="logloss"`).
    """
    modelo = xgboost.XGBClassifier(
        eval_metric="logloss",
        early_stopping_rounds=early_stopping_rounds,
    )
    modelo.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    return modelo


def calcular_metricas_m2(previsoes: pd.DataFrame) -> pd.DataFrame:
    r"""F1/precisão/recall/AUROC/FPR de M2, SEMPRE estratificado por
    `(delta_t, recompensa, pi) × subgrupo_positivo` — réplica EXATA da
    estrutura de `cusum.calcular_metricas_m1` (ver docstring daquela
    função para a justificativa completa de por que `grupo` não entra
    diretamente no `groupby`), com AUROC adicionada (requer `y_proba`,
    que M1/CUSUM não tinha por ser um detector sem probabilidade).

    `grupo` NÃO entra na chave de agrupamento — se entrasse, cada estrato
    teria rótulo binário CONSTANTE e a matriz de confusão degeneraria
    (um estrato `positiva_ativa` só teria `vp`/`fn`, nunca `vn`/`fp`).
    Em vez disso, dentro de cada `(delta_t, recompensa, pi)`, as janelas
    negativas DAQUELE MESMO estrato são pareadas separadamente com
    `positiva_ativa` e com `positiva_inativa`, produzindo DUAS linhas por
    estrato — cada uma com matriz de confusão completa e AUROC calculável
    sobre esse subconjunto (negativas do estrato + só o subgrupo
    positivo em questão).

    Parameters
    ----------
    previsoes : pd.DataFrame
        Precisa ter `y_true` (0/1), `y_pred` (0/1), `y_proba` (float,
        probabilidade da classe 1), `grupo`, `delta_t`, `recompensa`,
        `pi`.

    Returns
    -------
    pd.DataFrame
        Uma linha por `(subgrupo_positivo, delta_t, recompensa, pi)`,
        onde `subgrupo_positivo` ∈ {"positiva_ativa", "positiva_inativa"}
        — colunas `n_positivas`, `n_negativas`, `vp`, `fp`, `vn`, `fn`,
        `precisao`, `recall`, `f1`, `auroc`, `fpr`. Métricas são `NaN`
        quando o denominador/pré-requisito correspondente não permite
        cálculo (ex. `auroc` com só uma classe presente no subconjunto) —
        não `0.0` silencioso. Se um `(delta_t, recompensa, pi)` não tiver
        nenhuma janela negativa, esse estrato é omitido (mesmo
        comportamento de `calcular_metricas_m1`).

    Raises
    ------
    ValueError
        Se `previsoes` não tiver as colunas obrigatórias.
    """
    colunas_combinacao = list(_COLUNAS_COMBINACAO_METRICAS)
    faltando = (set(colunas_combinacao) | {"y_true", "y_pred", "y_proba", "grupo"}) - set(previsoes.columns)
    if faltando:
        raise ValueError(f"previsoes está sem as colunas obrigatórias: {sorted(faltando)}")

    linhas = []
    for chave, estrato in previsoes.groupby(colunas_combinacao, sort=False):
        negativas = estrato[estrato["grupo"] == "negativa"]
        if negativas.empty:
            continue  # sem negativas nesse estrato -- fp/vn/fpr indefiníveis, estrato omitido

        fp = int(negativas["y_pred"].sum())
        vn = int((negativas["y_pred"] == 0).sum())
        fpr = fp / (fp + vn) if (fp + vn) > 0 else float("nan")

        for subgrupo_positivo in ("positiva_ativa", "positiva_inativa"):
            positivas = estrato[estrato["grupo"] == subgrupo_positivo]
            vp = int(positivas["y_pred"].sum())
            fn = int((positivas["y_pred"] == 0).sum())

            precisao = vp / (vp + fp) if (vp + fp) > 0 else float("nan")
            recall = vp / (vp + fn) if (vp + fn) > 0 else float("nan")
            f1 = (
                2 * precisao * recall / (precisao + recall)
                if not np.isnan(precisao) and not np.isnan(recall) and (precisao + recall) > 0
                else float("nan")
            )

            subconjunto = pd.concat([negativas, positivas])
            if subconjunto["y_true"].nunique() < 2:
                auroc = float("nan")
            else:
                auroc = float(roc_auc_score(subconjunto["y_true"], subconjunto["y_proba"]))

            linha = dict(zip(colunas_combinacao, chave))
            linha.update(
                {
                    "subgrupo_positivo": subgrupo_positivo,
                    "n_positivas": len(positivas),
                    "n_negativas": len(negativas),
                    "vp": vp, "fp": fp, "vn": vn, "fn": fn,
                    "precisao": precisao, "recall": recall, "f1": f1,
                    "auroc": auroc, "fpr": fpr,
                }
            )
            linhas.append(linha)

    return pd.DataFrame(linhas)


def amostrar_para_shap(
    x_test: pd.DataFrame,
    metadados_test: pd.DataFrame,
    n_amostra: int,
    seed: int,
) -> pd.DataFrame:
    r"""Amostra estratificada por `grupo × pi × delta_t` (decisão 7 do
    plano), ~`n_amostra` linhas do split de teste — evita rodar SHAP
    sobre o teste completo (81.000 linhas no dataset de produção).

    Parameters
    ----------
    x_test : pd.DataFrame
        Matriz de features, já filtrada por `split == "test"` — mesmo
        índice posicional de `metadados_test`.
    metadados_test : pd.DataFrame
        Mesmo índice posicional de `x_test` — precisa ter `grupo`, `pi`,
        `delta_t`.
    n_amostra : int
        Tamanho alvo da amostra final (aproximado — o tamanho real
        depende do arredondamento proporcional por estrato).
    seed : int
        Semente para reprodutibilidade da amostragem.

    Returns
    -------
    pd.DataFrame
        Subconjunto de `x_test` (mesmas colunas), com o índice original
        preservado — quem chama pode usar esse índice para recuperar as
        linhas correspondentes de `metadados_test` via `.loc`.

    Raises
    ------
    ValueError
        Se `len(x_test) != len(metadados_test)`.
    """
    if len(x_test) != len(metadados_test):
        raise ValueError(f"x_test ({len(x_test)} linhas) e metadados_test ({len(metadados_test)} linhas) com tamanhos diferentes.")

    fracao = min(1.0, n_amostra / len(x_test)) if len(x_test) > 0 else 0.0

    chaves_estrato = metadados_test[["grupo", "pi", "delta_t"]].copy()
    chaves_estrato["_posicao"] = np.arange(len(metadados_test))

    amostrada = chaves_estrato.groupby(["grupo", "pi", "delta_t"], sort=False, group_keys=False).apply(
        lambda grupo: grupo.sample(frac=fracao, random_state=seed) if len(grupo) > 0 else grupo,
        include_groups=False,
    )

    posicoes = amostrada["_posicao"].to_numpy() if "_posicao" in amostrada.columns else amostrada.to_numpy().ravel()
    return x_test.iloc[posicoes]


def calcular_shap_values(modelo: xgboost.XGBClassifier, x_amostra: pd.DataFrame) -> shap.Explanation:
    r"""`shap.TreeExplainer(modelo)(x_amostra)` — decisão 7 do plano.

    **Não mede tempo internamente** — quem chama esta função é
    responsável por medir/reportar `time.perf_counter()` ao redor dela
    (mantém a função pura, sem efeito colateral de impressão/log).

    Parameters
    ----------
    modelo : xgboost.XGBClassifier
        Modelo treinado por `treinar_xgboost`.
    x_amostra : pd.DataFrame
        Saída de `amostrar_para_shap` (ou qualquer subconjunto de
        features compatível com o modelo).

    Returns
    -------
    shap.Explanation
    """
    explainer = shap.TreeExplainer(modelo)
    return explainer(x_amostra)


def ranking_importancia_shap(
    shap_values: shap.Explanation,
    metadados_amostra: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    r"""Duas visões do mesmo `shap_values` já calculado — NÃO roda
    `TreeExplainer` de novo, é reorganização do resultado existente, sem
    custo adicional de tempo (emenda aprovada ao plano original).

    Parameters
    ----------
    shap_values : shap.Explanation
        Saída de `calcular_shap_values` — `shap_values.values` tem shape
        `(n_amostra, n_features)` (classificação binária, valores da
        classe positiva) e `shap_values.feature_names` tem os nomes das
        colunas.
    metadados_amostra : pd.DataFrame
        Metadados correspondentes à MESMA amostra passada a
        `calcular_shap_values` (mesmo índice posicional) — precisa ter
        `pi`.

    Returns
    -------
    dict[str, pd.DataFrame]
        - `"agregado"`: colunas `feature`, `shap_medio_abs` — média do
          `|SHAP value|` por feature sobre toda a amostra, ordenada
          decrescente.
        - `"por_pi"`: colunas `pi`, `feature`, `shap_medio_abs` — média
          do `|SHAP value|` por feature, agrupada por `pi`, ordenada por
          `pi` e depois por `shap_medio_abs` decrescente dentro de cada
          `pi`. Atende PLANO §5.3.2 ("a análise de SHAP identifica quais
          features de quais fontes mais contribuem para a detecção em
          cada regime de privacidade π") — o ranking agregado sozinho
          não cobre isso.

    Raises
    ------
    ValueError
        Se `shap_values.values.shape[0] != len(metadados_amostra)`.
    """
    valores = np.asarray(shap_values.values)
    if valores.ndim == 3:
        # Multiclasse/binária com uma dimensão extra de classe -- pega a
        # última classe (positiva, índice 1 em binário 0/1).
        valores = valores[:, :, -1]

    if valores.shape[0] != len(metadados_amostra):
        raise ValueError(
            f"shap_values tem {valores.shape[0]} linhas mas metadados_amostra tem "
            f"{len(metadados_amostra)} — precisam vir da mesma amostra, mesma ordem."
        )

    nomes_features = list(shap_values.feature_names)
    abs_valores = np.abs(valores)

    agregado = pd.DataFrame(
        {"feature": nomes_features, "shap_medio_abs": abs_valores.mean(axis=0)}
    ).sort_values("shap_medio_abs", ascending=False).reset_index(drop=True)

    df_abs = pd.DataFrame(abs_valores, columns=nomes_features)
    df_abs["pi"] = metadados_amostra["pi"].to_numpy()

    por_pi = (
        df_abs.groupby("pi")[nomes_features]
        .mean()
        .reset_index()
        .melt(id_vars="pi", var_name="feature", value_name="shap_medio_abs")
        .sort_values(["pi", "shap_medio_abs"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return {"agregado": agregado, "por_pi": por_pi}
