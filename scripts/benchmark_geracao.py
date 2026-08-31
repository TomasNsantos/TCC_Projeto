"""Benchmark pontual: mede o custo de gerar uma única janela de cada classe
com os parâmetros stub atuais, e projeta o custo do grid completo.

Ferramenta de diagnóstico, não faz parte do pipeline testado
(`src/pipeline/`) — não tem suíte de testes formal. Existe só para decidir,
com números reais em vez de suposição, se a próxima tarefa (paralelização
de `gerar_par_de_classes_real`) precisa de processos ou threads, e que
tamanho de lote por worker faz sentido.

Uso (a partir da raiz do projeto, para `import src....` resolver):
``python -m scripts.benchmark_geracao``
"""

from __future__ import annotations

import os
import statistics
import time

import numpy as np

from src.generator.adversarial_mode import gerar_cenario_adversarial
from src.generator.layer1_abm import ElectionModel
from src.generator.normal_mode import gerar_cenario_normal
from src.pipeline.config import (
    N_JANELAS_POR_CLASSE_PADRAO,
    ParametrosPopulacionaisStub,
    ParametrosStubGeracao,
)
from src.pipeline.geracao import _seed_sequence_para_int

N_REPETICOES = 20

# Mesma combinação de teste usada em tests/test_pipeline_geracao.py
# (ativa o contrato na maioria das janelas) -- não é um ponto qualquer do
# grid, é o caso já validado como "exercitando o caminho realista", não o
# caso degenerado de contrato nunca ativando.
_PARAMS = {"g": "pool", "delta_t": 20.0, "recompensa": 10.0, "rho": 0.3, "beta": 1}

_POPULACIONAIS = ParametrosPopulacionaisStub(n_agentes=100, n_secoes=4, n_candidatos=3)
_STUB_GERACAO = ParametrosStubGeracao(tau_kendall=0.5, taxa_fonte_a=1.0, volume_medio_fonte_a=1000.0, taxa_fonte_b=1.0)

# Contagem real de runs (combinação x seed já inclusas -- PLANO §5.2.2: a
# própria fórmula do documento é 4(pi) x 3(g) x 3(delta_t) x 3(lambda) x
# 3(rho) x 5(seeds) = 1.620, e o experimento de robustez é 1 x 3(beta) x
# 5(seeds) = 15 -- as seeds já estão multiplicadas em ambos os números, não
# são um eixo adicional para multiplicar de novo aqui. Cada run (uma
# combinação x seed) gera N_JANELAS_POR_CLASSE janelas de cada classe.
N_RUNS_GRID_PRINCIPAL = 1620
N_RUNS_ROBUSTEZ = 15
N_JANELAS_POR_CLASSE = N_JANELAS_POR_CLASSE_PADRAO
N_CLASSES = 2


def _medir_janela_positiva(window_id: int) -> float:
    seed_modelo = np.random.SeedSequence([12345, window_id, 0])
    seed_fonte_b = np.random.SeedSequence([12345, window_id, 1])

    inicio = time.perf_counter()
    modelo = ElectionModel(
        n_agentes=_POPULACIONAIS.n_agentes,
        alpha_beta=_POPULACIONAIS.alpha_beta,
        prop_racional=_POPULACIONAIS.prop_racional,
        n_secoes=_POPULACIONAIS.n_secoes,
        n_candidatos=_POPULACIONAIS.n_candidatos,
        candidato_alvo=_POPULACIONAIS.candidato_alvo,
        prob_conformidade=_POPULACIONAIS.prob_conformidade,
        recompensa=_PARAMS["recompensa"],
        delta_t=_PARAMS["delta_t"],
        rho=_PARAMS["rho"],
        beta=_PARAMS["beta"],
        granularidade=_PARAMS["g"],
        unidade_alvo=None,
        seed=seed_modelo,
    )
    gerar_cenario_adversarial(modelo, _STUB_GERACAO.tau_kendall, _seed_sequence_para_int(seed_fonte_b))
    return time.perf_counter() - inicio


def _medir_janela_negativa(window_id: int) -> float:
    seed_modelo = np.random.SeedSequence([54321, window_id, 0])
    seed_fonte_b = np.random.SeedSequence([54321, window_id, 1])

    inicio = time.perf_counter()
    modelo = ElectionModel(
        n_agentes=_POPULACIONAIS.n_agentes,
        alpha_beta=_POPULACIONAIS.alpha_beta,
        prop_racional=_POPULACIONAIS.prop_racional,
        n_secoes=_POPULACIONAIS.n_secoes,
        n_candidatos=_POPULACIONAIS.n_candidatos,
        candidato_alvo=_POPULACIONAIS.candidato_alvo,
        prob_conformidade=_POPULACIONAIS.prob_conformidade,
        recompensa=0.0,
        delta_t=_PARAMS["delta_t"],
        rho=_PARAMS["rho"],
        beta=_PARAMS["beta"],
        granularidade=_PARAMS["g"],
        unidade_alvo=None,
        seed=seed_modelo,
    )
    seed_fonte_a_negativa = seed_modelo.spawn(1)[0]
    gerar_cenario_normal(
        modelo,
        janela=_PARAMS["delta_t"],
        taxa_fonte_a=_STUB_GERACAO.taxa_fonte_a,
        volume_medio_fonte_a=_STUB_GERACAO.volume_medio_fonte_a,
        taxa_fonte_b=_STUB_GERACAO.taxa_fonte_b,
        random_state_fonte_a=seed_fonte_a_negativa,
        random_state_fonte_b=seed_fonte_b,
    )
    return time.perf_counter() - inicio


def _formatar_duracao(segundos: float) -> str:
    if segundos < 60:
        return f"{segundos:.1f}s"
    if segundos < 3600:
        return f"{segundos / 60:.1f}min"
    if segundos < 86400:
        return f"{segundos / 3600:.1f}h"
    return f"{segundos / 86400:.1f}dias"


def main() -> None:
    print(f"Medindo {N_REPETICOES} repeticoes por classe (params={_PARAMS})...\n")

    tempos_positiva = [_medir_janela_positiva(i) for i in range(N_REPETICOES)]
    tempos_negativa = [_medir_janela_negativa(i) for i in range(N_REPETICOES)]

    media_positiva = statistics.mean(tempos_positiva)
    desvio_positiva = statistics.stdev(tempos_positiva)
    media_negativa = statistics.mean(tempos_negativa)
    desvio_negativa = statistics.stdev(tempos_negativa)

    print("=== Tempo por janela (N={}) ===".format(N_REPETICOES))
    print(f"  Classe positiva: media={media_positiva * 1000:.2f}ms  desvio={desvio_positiva * 1000:.2f}ms")
    print(f"  Classe negativa: media={media_negativa * 1000:.2f}ms  desvio={desvio_negativa * 1000:.2f}ms")

    n_runs = N_RUNS_GRID_PRINCIPAL + N_RUNS_ROBUSTEZ
    n_janelas_totais_por_classe = n_runs * N_JANELAS_POR_CLASSE

    tempo_total_serial = n_janelas_totais_por_classe * (media_positiva + media_negativa)

    n_cpus = os.cpu_count() or 1
    tempo_total_paralelo = tempo_total_serial / n_cpus

    print("\n=== Projecao do grid completo ===")
    print(f"  Runs (combinacao x seed ja inclusas): {N_RUNS_GRID_PRINCIPAL} (grid principal) + {N_RUNS_ROBUSTEZ} (robustez) = {n_runs}")
    print(f"  x {N_JANELAS_POR_CLASSE} janelas/classe x {N_CLASSES} classes")
    print(f"  = {n_janelas_totais_por_classe:,} janelas por classe ({n_janelas_totais_por_classe * N_CLASSES:,} janelas totais)")
    print(f"\n  Tempo serial estimado:  {_formatar_duracao(tempo_total_serial)}  ({tempo_total_serial:,.0f}s)")
    print(f"  N_CPUS detectados (os.cpu_count()): {n_cpus}")
    print(f"  Tempo paralelo hipotetico (/{n_cpus} nucleos, escalonamento ideal): {_formatar_duracao(tempo_total_paralelo)}  ({tempo_total_paralelo:,.0f}s)")


if __name__ == "__main__":
    main()
