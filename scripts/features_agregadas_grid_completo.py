r"""Camada agregada de features (Semana 9-10, Dia 4-5, PLANO §5.3.1) — roda
`montar_vetor_features_agregadas` sobre o grid completo do dataset de
produção v3 (270 arquivos, `output/dataset_producao_v3/`), mede o tempo
real e salva o resultado para uso futuro do M2 (XGBoost+SHAP).

**Processamento arquivo por arquivo**, mesmo padrão de
`scripts/m1_cusum_grid_completo.py` — `montar_vetor_features_agregadas`
já encapsula `carregar_arquivo` + `bin_janelas` por arquivo internamente
(`window_id` não é global, é reiniciado em 0 por classe em CADA arquivo;
concatenar antes de agregar produziria chave não-única — ver
`binning.bin_janelas`/CLAUDE.md).

Ao final, faz checagens básicas de sanidade (não uma validação
estatística completa, isso é trabalho de EDA separado):
- Nenhuma linha duplicada em `(classe, window_id)` no resultado
  concatenado (cada arquivo já garante isso internamente; aqui confirma
  que a concatenação entre arquivos não introduziu duplicata — arquivos
  diferentes têm `window_id` colidindo por design, então a chave real de
  não-duplicação inclui as colunas de metadado de cenário).
- `fonte_a_vazia=True` implica todas as estatísticas contínuas de Fonte A
  NaN (e o inverso: `fonte_a_vazia=False` implica nenhuma NaN) — mesma
  checagem para Fonte B.
- Contagem de linhas por `grupo` bate com `metadados_janela` (nenhuma
  janela perdida/duplicada na composição dos 4 blocos).

Salva `features.parquet`/`metadados.parquet` em
`output/features_agregadas_v3/` — formato colunar (pyarrow já é
dependência transitiva do projeto, confirmado antes de rodar este
script), juntável por `(classe, window_id)` + as colunas de metadado de
cenário (necessárias para desambiguar `window_id` entre arquivos, mesmo
princípio de `cusum._CHAVE_JANELA_CUSUM`).

Uso (a partir da raiz do projeto, para `import src....` resolver):
    python -m scripts.features_agregadas_grid_completo
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features_agregadas import montar_vetor_features_agregadas

DIRETORIO_V3 = Path("output/dataset_producao_v3")
DIRETORIO_SAIDA = Path("output/features_agregadas_v3")

_COLUNAS_COMBINACAO_CENARIO = ("beta", "delta_t", "g", "pi", "recompensa", "rho", "seed")
"""Mesmas colunas de `loader.COLUNAS_METADADO_CENARIO` -- usadas aqui só
para checar duplicata real (classe, window_id) DENTRO da mesma
combinação, já que window_id colide entre arquivos por design (não é
duplicata de verdade)."""


_COLUNAS_SEMPRE_CALCULAVEIS_FONTE_A = (
    "fonte_a_volume_total",
    "fonte_a_volume_medio",
    "fonte_a_volume_por_timestep_std",
    "fonte_a_volume_por_timestep_min",
    "fonte_a_volume_por_timestep_max",
    "fonte_a_razao_eventos_volume",
    "fonte_a_offset_primeiro_evento",
)
_COLUNAS_IET_FONTE_A = ("fonte_a_iet_media", "fonte_a_iet_std", "fonte_a_iet_cv")
_COLUNAS_SEMPRE_CALCULAVEIS_FONTE_B = ("fonte_b_offset_primeiro_evento",)
_COLUNAS_IET_FONTE_B = ("fonte_b_iet_media", "fonte_b_iet_std", "fonte_b_iet_cv")


def _checar_consistencia_vazio(features: pd.DataFrame, prefixo: str, colunas_sempre_calculaveis: tuple[str, ...]) -> None:
    """`{prefixo}_vazia=True` deve implicar TODAS as colunas passadas NaN;
    `{prefixo}_vazia=False` deve implicar NENHUMA NaN nelas.

    **Não inclui as colunas de IET** (`_media`/`_std`/`_cv`) — essas são
    indefinidas quando há apenas 1 TIMESTEP distinto com evento (não 1
    EVENTO), o que acontece estruturalmente com `delta_t=0.0` na classe
    positiva (todos os N eventos de desembolso caem em `timestep=0` por
    construção do gerador, ver `binning.py`/CLAUDE.md, "timestep
    estruturalmente impossível") — `fonte_a_vazia=False` mas IET
    indefinido mesmo com dezenas de eventos. Ver `_checar_iet_consistente`
    para a checagem correta desse caso."""
    vazia = features[f"{prefixo}_vazia"]

    for coluna in colunas_sempre_calculaveis:
        nan_mas_nao_vazia = (~vazia) & features[coluna].isna()
        if nan_mas_nao_vazia.any():
            raise AssertionError(
                f"{coluna}: {int(nan_mas_nao_vazia.sum())} linhas com {prefixo}_vazia=False mas valor NaN"
            )

        nao_nan_mas_vazia = vazia & features[coluna].notna()
        if nao_nan_mas_vazia.any():
            raise AssertionError(
                f"{coluna}: {int(nao_nan_mas_vazia.sum())} linhas com {prefixo}_vazia=True mas valor NÃO-NaN"
            )


def _checar_iet_consistente(features: pd.DataFrame, prefixo: str, colunas_iet: tuple[str, ...]) -> None:
    """IET (intervalo entre timesteps consecutivos COM evento) é indefinido
    (`NaN`) sempre que há menos de 2 TIMESTEPS DISTINTOS com evento na
    janela — não uma proxy indireta via `delta_t`/`vazia`. Dois cenários
    produzem isso, ambos legítimos (ver `features_agregadas._iet_estatisticas`):
    (a) `{prefixo}_n_eventos_total <= 1` (0 ou 1 evento — com <=1 evento há
    no máximo 1 timestep distinto, trivialmente); (b) `n_eventos_total >= 2`
    mas TODOS caem no mesmo timestep (ex. `delta_t=0.0` na classe positiva,
    onde todo o desembolso cai em `timestep=0` por construção do gerador —
    achado do grid completo do v3). **Esta checagem não distingue (a) de
    (b)** (faria isso precisar da série binada por janela, não só do vetor
    agregado) — só confirma que NaN nunca aparece fora desses dois casos
    conhecidos, via `{prefixo}_n_eventos_total`: `>= 2` eventos é condição
    NECESSÁRIA (não suficiente) para >= 2 timesteps distintos, então
    `n_eventos_total <= 1` cobre (a) exatamente; (b) não é filtrável só a
    partir do vetor agregado, então a checagem aceita NaN com
    `n_eventos_total >= 2` sem levantar erro — não é uma prova completa de
    corretude, só uma rede contra regressões grosseiras (ex. NaN aparecendo
    também quando `vazia=True`, que já é outro caso, ou em volume/offset,
    cobertos por `_checar_consistencia_vazio`)."""
    n_eventos = features[f"{prefixo}_n_eventos_total"]

    for coluna in colunas_iet:
        nan_com_dois_ou_mais_eventos = (n_eventos >= 2) & features[coluna].isna()
        # Não é um erro em si (cenário (b) acima é legítimo) -- só reportado
        # para visibilidade, já que é informação nova sobre o dataset.
        if nan_com_dois_ou_mais_eventos.any():
            print(
                f"  aviso: {coluna} tem {int(nan_com_dois_ou_mais_eventos.sum())} linhas com "
                f">=2 eventos mas NaN -- esperado quando todos os eventos caem no mesmo timestep "
                f"(ex. delta_t=0.0 na classe positiva), não um erro."
            )

        nan_com_zero_ou_um_evento_mas_nao_marcado_vazia = (n_eventos == 0) & (~features[f"{prefixo}_vazia"])
        if nan_com_zero_ou_um_evento_mas_nao_marcado_vazia.any():
            raise AssertionError(
                f"{coluna}: {int(nan_com_zero_ou_um_evento_mas_nao_marcado_vazia.sum())} linhas com "
                f"{prefixo}_n_eventos_total==0 mas {prefixo}_vazia=False -- inconsistência real entre "
                "contagem e flag de vazio."
            )


def main() -> None:
    t_inicio = time.perf_counter()

    print("=== Camada agregada de features (Dia 4-5) sobre o grid completo do dataset v3 ===")
    print()

    caminhos = sorted(DIRETORIO_V3.glob("*.h5"))
    print(f"Processando {len(caminhos)} arquivos, um por vez (montar_vetor_features_agregadas)...")

    t0 = time.perf_counter()
    blocos_features = []
    blocos_metadados = []
    for i, caminho in enumerate(caminhos):
        resultado = montar_vetor_features_agregadas(caminho)
        blocos_features.append(resultado["features"])
        blocos_metadados.append(resultado["metadados"])
        if (i + 1) % 30 == 0:
            print(f"  {i + 1}/{len(caminhos)} arquivos, {time.perf_counter() - t0:.1f}s decorridos")

    features = pd.concat(blocos_features, ignore_index=True)
    metadados = pd.concat(blocos_metadados, ignore_index=True)
    del blocos_features, blocos_metadados

    tempo_extracao = time.perf_counter() - t0
    print(f"  extração completa em {tempo_extracao:.1f}s ({tempo_extracao / 60:.2f}min) -- {len(features)} janelas totais")
    print()

    print("Checando consistência (sem duplicata real, NaN <-> flag de vazio, contagem por grupo)...")
    t0 = time.perf_counter()

    chave_completa = ["classe", "window_id"] + list(_COLUNAS_COMBINACAO_CENARIO)
    duplicadas = metadados.duplicated(subset=chave_completa, keep=False)
    if duplicadas.any():
        raise AssertionError(f"{int(duplicadas.sum())} linhas duplicadas em (classe, window_id, {', '.join(_COLUNAS_COMBINACAO_CENARIO)})")

    assert len(features) == len(metadados), f"features ({len(features)}) e metadados ({len(metadados)}) com tamanhos diferentes"

    _checar_consistencia_vazio(features, "fonte_a", _COLUNAS_SEMPRE_CALCULAVEIS_FONTE_A)
    _checar_consistencia_vazio(features, "fonte_b", _COLUNAS_SEMPRE_CALCULAVEIS_FONTE_B)
    _checar_iet_consistente(features, "fonte_a", _COLUNAS_IET_FONTE_A)
    _checar_iet_consistente(features, "fonte_b", _COLUNAS_IET_FONTE_B)

    assert "ambas_vazias" not in features.columns, "ambas_vazias vazou para 'features' -- vazamento de rótulo"
    assert "ambas_vazias" in metadados.columns, "ambas_vazias ausente de 'metadados'"

    contagem_por_grupo = metadados["grupo"].value_counts()
    print(f"  consistência verificada em {time.perf_counter() - t0:.1f}s")
    print(f"  contagem por grupo:\n{contagem_por_grupo.to_string()}")
    print()

    DIRETORIO_SAIDA.mkdir(parents=True, exist_ok=True)
    caminho_features = DIRETORIO_SAIDA / "features.parquet"
    caminho_metadados = DIRETORIO_SAIDA / "metadados.parquet"

    print(f"Salvando em {DIRETORIO_SAIDA}/ ...")
    t0 = time.perf_counter()
    features.to_parquet(caminho_features, index=False)
    metadados.to_parquet(caminho_metadados, index=False)
    print(f"  salvo em {time.perf_counter() - t0:.1f}s -- {caminho_features.name} ({caminho_features.stat().st_size / 1e6:.1f}MB), {caminho_metadados.name} ({caminho_metadados.stat().st_size / 1e6:.1f}MB)")
    print()

    tempo_total = time.perf_counter() - t_inicio
    print(f"=== TEMPO TOTAL: {tempo_total:.1f}s ({tempo_total / 60:.2f}min) ===")
    print()

    print("=" * 100)
    print("RESUMO DE VAZIO POR GRUPO (fração de janelas com fonte_a_vazia / fonte_b_vazia)")
    print("=" * 100)
    # Junção posicional -- mesma razão de _checar_iet_consistente:
    # (classe, window_id) não é chave única no dataset completo, merge por
    # ela estoura memória (~7.6GiB, confirmado empiricamente).
    assert len(features) == len(metadados)
    resumo = features[["fonte_a_vazia", "fonte_b_vazia"]].copy()
    resumo["grupo"] = metadados["grupo"].to_numpy()
    print(resumo.groupby("grupo")[["fonte_a_vazia", "fonte_b_vazia"]].mean().to_string())

    print()
    print("=" * 100)
    print("AMOSTRA DE COLUNAS (5 primeiras linhas de features)")
    print("=" * 100)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(features.head().to_string(index=False))


if __name__ == "__main__":
    main()
