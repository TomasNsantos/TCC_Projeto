r"""M3 (LSTM Autoencoder) — Dia 7, Semana 9-10: benchmark de custo
computacional, NÃO implementação de M3 (PLANO §5.3.2: "infraestrutura a
definir com o orientador de IA nas semanas 9-10, antes de iniciar a
implementação"). Este script produz o número concreto (tempo/memória de
UMA época, no shape real dos dados, na máquina real) que essa decisão de
infraestrutura precisa — não valida convergência, não usa dados reais de
treino, não produz um modelo funcional.

Máquina de referência: Windows 11, Intel Core i3-8100T, GPU integrada
Intel UHD 630 (sem CUDA/MPS) — confirmado via smoke test abaixo.

**Shape real dos dados — achado desta etapa, não assumido:** M3 consome
`binning.bin_janelas` (série temporal por janela, colunas `contagem_a`/
`contagem_b`, sem NaN), NÃO a camada agregada do M2
(`features_agregadas.py`, uma linha por janela). O número de timesteps
por janela **varia com `delta_t`** — `binning._janela_teto` usa
`delta_t` diretamente quando `!= 0.0`, e a constante
`JANELA_TRAFEGO_FUNDO_QUANDO_DELTA_T_ZERO=24.0` só quando `delta_t==0.0`.
Verificado empiricamente em arquivos reais do v3 (2.000 janelas cada,
sem variação): `delta_t=0.0` → 24 timesteps; `delta_t=2.0` → 2
timesteps; `delta_t=24.0` → 24 timesteps. O design fatorial (`Δt ∈
{0h, 2h, 24h}`) produz DOIS comprimentos de sequência distintos, não um
único shape — este benchmark mede os dois separadamente.

**Projeção ponderada, não duas linhas alternativas.** Δt é um dos 5
eixos cruzados no fatorial principal (PLANO §5.2.2, linha 249:
`4(π) x 3(g) x 3(Δt) x 3(λ) x 3(ρ) x 5(seeds) = 1.620`) — cada run
pertence a EXATAMENTE UM dos dois shapes medidos, não aos dois
simultaneamente. Corrigido após revisão: a primeira versão deste
script reportava as duas linhas (24 e 2 timesteps) como se fossem
cenários alternativos de melhor/pior caso, cada uma multiplicada pelos
11.340 runs inteiros — superestimando (0,04h→1,09h, contando TODOS os
runs como o shape mais caro) e subestimando (contando todos como o
mais barato) ao mesmo tempo, sem produzir o número único acionável que
a decisão de infraestrutura precisa. Fixando Δt e cruzando os outros 5
eixos: `4x3x3x3x5=540` combinações por nível de Δt — 2 dos 3 níveis
(0h, 24h) caem no shape de 24 timesteps (1.080 combinações), 1 dos 3
(2h) cai no shape de 2 timesteps (540 combinações); a projeção final é
a SOMA ponderada por essa contagem real (confirmada no PLANO, não
assumida), não a soma nem a média das duas linhas medidas.

Uso (a partir da raiz do projeto, para `import src....` resolver):
    python -m scripts.benchmark_m3_custo
"""

from __future__ import annotations

import os
import sys
import time

import psutil
import torch
from torch import nn

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    # Console Windows usa cp1252 por padrão -- não cobre Δ/λ/π/ρ (achado
    # desta correção: a versão anterior deste script já usava esses
    # símbolos em outras strings sem erro só porque nunca chegavam a ser
    # impressas nessa combinação exata; forçar UTF-8 evita depender de
    # sorte de encoding em prints futuros deste script).
    sys.stdout.reconfigure(encoding="utf-8")

N_CANAIS = 2
"""contagem_a, contagem_b -- as duas colunas de binning.bin_janelas."""

HIDDEN_SIZE = 32
"""Tamanho do estado oculto do LSTM -- escolha arbitrária de benchmark,
pequena e plausível, NÃO é hiperparâmetro final de M3."""

BATCH_SIZE = 2000
"""Mesmo número de janelas por classe por arquivo no dataset de
produção v3 (confirmado empiricamente: bin_janelas produz exatamente
2000 grupos (classe, window_id) por arquivo) -- torna a medição
diretamente comparável ao volume real de um arquivo/combinação, em vez
de um valor de batch arbitrário."""

SHAPES_REAIS = {
    "delta_t=0.0 ou 24.0 (24 timesteps)": 24,
    "delta_t=2.0 (2 timesteps)": 2,
}

N_COMBINACOES_FATORIAL = 1_620
"""PLANO §5.2.2 (docs/PLANO TCC ARTIGO V4_2.md:249): design fatorial
principal = 4(π) x 3(g) x 3(Δt) x 3(λ) x 3(ρ) x 5(seeds) = 1.620.
Δt é um dos 3 eixos cruzados nesse produto -- logo, para CADA um dos 3
níveis de Δt, o número de combinações é 1.620 / 3 = 540 (fixando Δt e
cruzando os outros 5 eixos: 4x3x3x3x5=540), não uma fração arbitrária.
Confirmado lendo o PLANO, não assumido."""

N_NIVEIS_DELTA_T = 3
"""Δt ∈ {0h, 2h, 24h} (PLANO §5.2.2, linha 194) -- 3 níveis, cada um
com N_COMBINACOES_FATORIAL / N_NIVEIS_DELTA_T = 540 combinações."""

N_COMBINACOES_POR_NIVEL_DELTA_T = N_COMBINACOES_FATORIAL // N_NIVEIS_DELTA_T
"""540 combinações por nível de Δt -- ver N_COMBINACOES_FATORIAL acima."""

FRACAO_RUNS_POR_SHAPE = {
    "delta_t=0.0 ou 24.0 (24 timesteps)": 2 * N_COMBINACOES_POR_NIVEL_DELTA_T,  # Δt=0h + Δt=24h
    "delta_t=2.0 (2 timesteps)": 1 * N_COMBINACOES_POR_NIVEL_DELTA_T,  # Δt=2h
}
"""Número de combinações do fatorial (antes de multiplicar por C1-C7)
que caem em cada shape -- 2 dos 3 níveis de Δt (0h, 24h) produzem
shape=24; 1 dos 3 níveis (2h) produz shape=2. Soma = 1.620, confere com
N_COMBINACOES_FATORIAL."""

N_COMBINACOES_ABLACAO = 7
"""C1-C7, combinações de fonte na ablação (PLANO §5.3.2/§5.4.1)."""

N_RUNS_DESIGN_FATORIAL = N_COMBINACOES_FATORIAL * N_COMBINACOES_ABLACAO
"""1.620 combinações x 7 combinações de fonte (C1-C7), PLANO §5.3.2."""

N_CORES_LOGICOS = 4
"""Windows 11, Intel Core i3-8100T -- confirmado via os.cpu_count()."""

SEED = 42


class LSTMAutoencoderBrinquede(nn.Module):
    r"""Encoder+decoder LSTM mínimo -- poucas camadas, hidden_size
    pequeno. NÃO é a arquitetura final de M3 (sem validação de
    convergência, sem hiperparâmetros ajustados) -- só precisa ser
    plausível o suficiente para um forward+backward representativo do
    tipo de operação que M3 faria."""

    def __init__(self, n_canais: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(input_size=n_canais, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=n_canais, num_layers=1, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_timesteps = x.shape[1]
        _, (h_n, _) = self.encoder(x)
        # h_n: (num_layers=1, batch, hidden_size) -- vira a entrada do
        # decoder repetida por timestep (padrão comum de autoencoder
        # sequencial: o vetor latente é repetido a cada passo do decoder).
        latente = h_n[-1].unsqueeze(1).repeat(1, n_timesteps, 1)
        reconstrucao, _ = self.decoder(latente)
        return reconstrucao


def _montar_batch_sintetico(n_timesteps: int, n_canais: int, batch_size: int, seed: int) -> torch.Tensor:
    r"""Batch aleatório do shape real confirmado -- NÃO usa dados reais
    do v3. O benchmark mede custo computacional (forward+backward), não
    qualidade de reconstrução -- dado sintético do shape certo já é
    suficiente e evita I/O de HDF5 dentro da medição de tempo."""
    gerador = torch.Generator().manual_seed(seed)
    return torch.rand((batch_size, n_timesteps, n_canais), generator=gerador)


def _rodar_uma_epoca(modelo: nn.Module, batch: torch.Tensor, otimizador: torch.optim.Optimizer) -> tuple[float, float, float]:
    r"""Forward + MSE de reconstrução + backward + step do otimizador,
    UMA vez -- mede tempo (`time.perf_counter()`) e RSS do processo
    (`psutil`, MB) antes/depois. Retorna `(tempo_segundos,
    rss_antes_mb, rss_depois_mb)`."""
    processo = psutil.Process(os.getpid())
    rss_antes_mb = processo.memory_info().rss / (1024 * 1024)

    criterio = nn.MSELoss()

    t0 = time.perf_counter()
    otimizador.zero_grad()
    reconstrucao = modelo(batch)
    loss = criterio(reconstrucao, batch)
    loss.backward()
    otimizador.step()
    tempo_segundos = time.perf_counter() - t0

    rss_depois_mb = processo.memory_info().rss / (1024 * 1024)

    return tempo_segundos, rss_antes_mb, rss_depois_mb


def main() -> None:
    print("=== M3 (LSTM Autoencoder), Dia 7 — benchmark de custo computacional ===")
    print()

    print("=" * 100)
    print("ETAPA 0 — VERIFICAÇÃO DE AMBIENTE")
    print("=" * 100)
    print(f"torch versão: {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.backends.mps.is_available(): {torch.backends.mps.is_available()}")
    print("Estado: torch já estava instalado no venv do projeto (requirements.txt já pinava "
          "torch==2.14.0+cpu antes desta etapa) — nada foi instalado agora.")
    print(f"os.cpu_count(): {os.cpu_count()} (N_CORES_LOGICOS assumido = {N_CORES_LOGICOS})")
    print()

    print("=" * 100)
    print("ETAPA 1 — SHAPE REAL DOS DADOS (binning.bin_janelas, não features_agregadas.py)")
    print("=" * 100)
    print(f"Canais por timestep: {N_CANAIS} (contagem_a, contagem_b — ambos int64, sem NaN)")
    print("Timesteps por janela — VARIA com delta_t, dois valores confirmados empiricamente:")
    for descricao, n_timesteps in SHAPES_REAIS.items():
        print(f"  {descricao} -> shape (n_timesteps, n_canais) = ({n_timesteps}, {N_CANAIS})")
    print(f"Batch size usado no benchmark: {BATCH_SIZE} (= nº de janelas/classe/arquivo real no v3)")
    print()

    print("=" * 100)
    print("ETAPA 2 — BENCHMARK DE UMA ÉPOCA, POR SHAPE")
    print("=" * 100)
    resultados: dict[str, dict[str, float]] = {}
    for descricao, n_timesteps in SHAPES_REAIS.items():
        print(f"-- {descricao} --")
        torch.manual_seed(SEED)
        modelo = LSTMAutoencoderBrinquede(n_canais=N_CANAIS, hidden_size=HIDDEN_SIZE)
        otimizador = torch.optim.Adam(modelo.parameters())
        batch = _montar_batch_sintetico(n_timesteps, N_CANAIS, BATCH_SIZE, SEED)

        tempo_segundos, rss_antes_mb, rss_depois_mb = _rodar_uma_epoca(modelo, batch, otimizador)
        delta_rss_mb = rss_depois_mb - rss_antes_mb

        print(f"  tempo de 1 época: {tempo_segundos:.4f}s")
        print(f"  RSS antes: {rss_antes_mb:.1f}MB | RSS depois: {rss_depois_mb:.1f}MB | delta: {delta_rss_mb:+.1f}MB")
        print()

        resultados[descricao] = {
            "n_timesteps": n_timesteps,
            "tempo_segundos": tempo_segundos,
            "rss_antes_mb": rss_antes_mb,
            "rss_depois_mb": rss_depois_mb,
            "delta_rss_mb": delta_rss_mb,
        }

    print("=" * 100)
    print("ETAPA 3 — PROJEÇÃO PARA O DESIGN FATORIAL COMPLETO")
    print("=" * 100)
    print(f"N_RUNS_DESIGN_FATORIAL = {N_COMBINACOES_FATORIAL} combinações x {N_COMBINACOES_ABLACAO} (C1-C7) = {N_RUNS_DESIGN_FATORIAL}")
    print()
    print("AVISO OBRIGATÓRIO: a projeção abaixo é só o custo de UMA ÉPOCA por run. O número "
          "real de épocas até convergência é DESCONHECIDO nesta etapa — não foi assumido nem "
          "inventado. Para obter o tempo total real, multiplique o número final abaixo pelo "
          "número de épocas até convergência (variável em aberto, decisão de quem definir a "
          "infraestrutura/arquitetura final de M3).")
    print()

    print("Δt é um dos 5 eixos cruzados no fatorial (PLANO §5.2.2: 4(π) x 3(g) x 3(Δt) x 3(λ) x "
          f"3(ρ) x 5(seeds) = {N_COMBINACOES_FATORIAL}) — cada run pertence a EXATAMENTE UM dos "
          "dois shapes medidos, não aos dois. As linhas abaixo não são cenários alternativos "
          "(melhor/pior caso): são a partição real dos runs, ponderada pela contagem exata de "
          "combinações por nível de Δt, não por uma fração assumida.")
    print()

    tempo_total_ponderado_segundos = 0.0
    for descricao, r in resultados.items():
        n_combinacoes_fatorial_deste_shape = FRACAO_RUNS_POR_SHAPE[descricao]
        n_runs_deste_shape = n_combinacoes_fatorial_deste_shape * N_COMBINACOES_ABLACAO
        tempo_deste_shape_segundos = r["tempo_segundos"] * n_runs_deste_shape
        tempo_total_ponderado_segundos += tempo_deste_shape_segundos

        print(f"-- {descricao} --")
        print(f"  {n_combinacoes_fatorial_deste_shape}/{N_COMBINACOES_FATORIAL} combinações do fatorial "
              f"x {N_COMBINACOES_ABLACAO} (C1-C7) = {n_runs_deste_shape} runs neste shape")
        print(f"  {r['tempo_segundos']:.4f}s/época x {n_runs_deste_shape} runs = "
              f"{tempo_deste_shape_segundos:.1f}s ({tempo_deste_shape_segundos / 3600:.3f}h)")
        print()

    print(f"TOTAL PONDERADO (soma das duas partições, cobre os {N_RUNS_DESIGN_FATORIAL} runs uma única "
          f"vez cada): {tempo_total_ponderado_segundos:.1f}s ({tempo_total_ponderado_segundos / 3600:.3f}h) "
          "para 1 época em cada um dos 11.340 runs do design fatorial completo.")
    print()

    print("Projeção de memória / paralelização (nos 4 cores lógicos desta máquina):")
    for descricao, r in resultados.items():
        rss_pico_mb = max(r["rss_antes_mb"], r["rss_depois_mb"])
        n_paralelos_por_memoria = None  # não há teto de memória total conhecido para calcular isso com segurança
        print(f"  {descricao}: RSS pico medido ~{rss_pico_mb:.1f}MB para 1 run "
              f"(batch={BATCH_SIZE}, hidden_size={HIDDEN_SIZE}) -- {N_CORES_LOGICOS} runs em paralelo "
              f"(1 por core lógico) custariam ~{rss_pico_mb * N_CORES_LOGICOS:.1f}MB de RSS combinado, "
              "sem contar overhead de processo/SO por worker.")
    print()
    print("Decisão de rodar serializado vs. paralelo (e se a infraestrutura CPU atual é viável "
          "para o design fatorial completo) NÃO é tomada por este script — cabe ao usuário e ao "
          "orientador de IA, com os números acima como insumo.")


if __name__ == "__main__":
    main()
