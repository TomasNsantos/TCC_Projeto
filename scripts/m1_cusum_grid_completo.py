r"""M1 (baseline estatístico, PLANO §5.3.2) — roda CUSUM sobre o grid
completo do dataset de produção v3 (270 arquivos, `output/dataset_producao_v3/`)
e reporta métricas de detecção estratificadas.

**Processamento arquivo por arquivo, não `loader.carregar_diretorio`
seguido de `bin_janelas` sobre o dataset inteiro concatenado.** Tentativa
anterior (mesma sessão) de rodar `bin_janelas` sobre as 540.000 janelas do
v3 inteiro de uma vez estourou memória (`ArrayMemoryError`, tentando
alocar ~3.36 GiB no meio de um merge que deveria produzir bem menos linhas)
— causa raiz: `window_id` NÃO é global, é reiniciado em 0 por classe em
CADA arquivo (`storage.escrever_run_hdf5`); concatenar 270 arquivos via
`carregar_diretorio` produz `(classe, window_id)` massivamente duplicado
entre combinações diferentes, e o merge interno de `bin_janelas` (por
`(classe, window_id, timestep)`) tratou isso como chave não-única,
produzindo um produto cartesiano parcial entre janelas de arquivos
diferentes que coincidiam nesses valores. `bin_janelas` agora VALIDA isso
explicitamente (levanta `ValueError` se `(classe, window_id)` duplicado em
`metadados_janela`) — este script nunca deveria hit essa validação, porque
processa um arquivo por vez (cada arquivo é uma única combinação, sem
duplicação de `window_id` dentro dele).

Pipeline por arquivo: `loader.carregar_arquivo` → `binning.bin_janelas` →
acumula em uma lista. Depois de processar TODOS os arquivos:
`binning.calcular_referencia_negativa_treino` (uma vez, sobre o acumulado
inteiro, só `split="train"`) → `binning.calcular_zscore` (todos os splits)
→ `cusum.aplicar_cusum_por_janela` (todos os splits) →
`cusum.calcular_metricas_m1` (estratificado por `delta_t × recompensa ×
pi`, duas linhas por estrato — `positiva_ativa`/`positiva_inativa` — ver
`src/pipeline/cusum.py` para a justificativa completa do agrupamento) +
`cusum.fracao_disparos_no_piso_da_referencia`.

**`k`/`h` — escolha explícita, não calibrada ao problema.** Valores
clássicos de CUSUM em controle estatístico de processo (Montgomery,
*Introduction to Statistical Quality Control*): `k = 0.5` (meio
desvio-padrão de deslocamento mínimo a detectar — já em unidades de σ, já
que o CUSUM aqui roda sobre z-score, não sobre a contagem bruta) e `h = 5.0`
(limiar clássico ≈ 4-5σ, trade-off padrão entre ARL — average run length —
sob controle e tempo médio de detecção sob desvio real). **Não são
calibrados especificamente para este dataset/problema** — mesma categoria
de pendência de `K_RESULTADO_ALVO`/`tau_kendall`/`taxa_fonte_a` (ver
CLAUDE.md): valores de literatura geral de CUSUM, usados como ponto de
partida razoável para o M1 "baseline sem treinamento" (PLANO §5.3.2: "sem
parâmetros aprendidos"), não uma escolha ajustada aos dados. Se os
resultados sugerirem que outro `k`/`h` seria mais informativo, isso é uma
decisão a tomar EXPLICITAMENTE depois de ver os resultados, olhando só
`split` em {"train", "val"} — nunca `split="test"` (conforme instrução do
usuário) — não escondida numa escolha silenciosa aqui.

Uso (a partir da raiz do projeto, para `import src....` resolver):
    python -m scripts.m1_cusum_grid_completo
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.pipeline.binning import bin_janelas, calcular_referencia_negativa_treino, calcular_zscore
from src.pipeline.cusum import aplicar_cusum_por_janela, calcular_metricas_m1, fracao_disparos_no_piso_da_referencia
from src.pipeline.loader import carregar_arquivo

DIRETORIO_V3 = Path("output/dataset_producao_v3")

K_CUSUM: float = 0.5
"""Slack/sensibilidade -- valor clássico de CUSUM (0.5σ), não calibrado
especificamente a este problema. Ver docstring do módulo."""

H_CUSUM: float = 5.0
"""Limiar de disparo -- valor clássico de CUSUM (~4-5σ), não calibrado
especificamente a este problema. Ver docstring do módulo."""


def main() -> None:
    t_inicio = time.perf_counter()

    print("=== M1 (CUSUM) sobre o grid completo do dataset de produção v3 ===")
    print(f"k={K_CUSUM}, h={H_CUSUM} (valores clássicos de CUSUM, não calibrados -- ver docstring do script)")
    print()

    caminhos = sorted(DIRETORIO_V3.glob("*.h5"))
    print(f"Processando {len(caminhos)} arquivos, um por vez (carregar_arquivo + bin_janelas)...")

    t0 = time.perf_counter()
    blocos_binados = []
    for i, caminho in enumerate(caminhos):
        carregado = carregar_arquivo(caminho)
        binado = bin_janelas(carregado["fonte_a"], carregado["fonte_b"], carregado["metadados_janela"])
        blocos_binados.append(binado)
        if (i + 1) % 30 == 0:
            print(f"  {i + 1}/{len(caminhos)} arquivos, {time.perf_counter() - t0:.1f}s decorridos")

    binado_completo = pd.concat(blocos_binados, ignore_index=True)
    del blocos_binados
    tempo_binning = time.perf_counter() - t0
    print(f"  binning completo em {tempo_binning:.1f}s ({tempo_binning/60:.2f}min) -- {len(binado_completo)} linhas totais")

    print("Calculando referência negativa-treino (só split='train', sobre o acumulado)...")
    t0 = time.perf_counter()
    referencia = calcular_referencia_negativa_treino(binado_completo)
    print(f"  referência calculada em {time.perf_counter() - t0:.1f}s -- {len(referencia)} linhas (combinação x timestep)")

    print("Calculando z-score (todos os splits)...")
    t0 = time.perf_counter()
    com_zscore = calcular_zscore(binado_completo, referencia)
    del binado_completo
    print(f"  z-score calculado em {time.perf_counter() - t0:.1f}s")

    print("Aplicando CUSUM por janela (todos os splits)...")
    t0 = time.perf_counter()
    previsoes = aplicar_cusum_por_janela(com_zscore, k=K_CUSUM, h=H_CUSUM)
    del com_zscore
    print(f"  CUSUM aplicado em {time.perf_counter() - t0:.1f}s -- {len(previsoes)} janelas avaliadas")

    print("Calculando métricas M1 (estratificadas)...")
    t0 = time.perf_counter()
    metricas = calcular_metricas_m1(previsoes)
    print(f"  métricas calculadas em {time.perf_counter() - t0:.1f}s -- {len(metricas)} estratos")

    print("Calculando fração de disparos no piso da referência (item 5)...")
    t0 = time.perf_counter()
    piso = fracao_disparos_no_piso_da_referencia(previsoes, referencia)
    print(f"  calculado em {time.perf_counter() - t0:.1f}s")

    tempo_total = time.perf_counter() - t_inicio
    print()
    print(f"=== TEMPO TOTAL: {tempo_total:.1f}s ({tempo_total/60:.2f}min) ===")
    print()

    print("=" * 100)
    print("MÉTRICAS M1 ESTRATIFICADAS (subgrupo_positivo x delta_t x recompensa x pi)")
    print("=" * 100)
    colunas_exibicao = ["subgrupo_positivo", "delta_t", "recompensa", "pi", "n_positivas", "n_negativas", "vp", "fp", "vn", "fn", "precisao", "recall", "f1", "fpr"]
    print(metricas[colunas_exibicao].sort_values(["subgrupo_positivo", "delta_t", "recompensa", "pi"]).to_string(index=False))

    print()
    print("=" * 100)
    print("DISPAROS VS. PISO DA REFERÊNCIA (item 5)")
    print("=" * 100)
    print(piso.to_string(index=False))

    print()
    print("=" * 100)
    print("RESUMO AGREGADO POR subgrupo_positivo (média simples sobre estratos, só para visão geral)")
    print("=" * 100)
    print(metricas.groupby("subgrupo_positivo")[["precisao", "recall", "f1", "fpr"]].mean().to_string())


if __name__ == "__main__":
    main()
