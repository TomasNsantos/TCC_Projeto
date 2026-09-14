"""Gera o dataset PRINCIPAL (v2) -- grid completo, n_janelas de produção.

Baseado em ``scripts/gerar_dataset_producao_v1.py`` (mesma grade, mesmos
placeholders de Fonte A/B, mesma recompensa) -- só as duas mudanças abaixo
mudam em relação ao v1, ambas decididas na sessão que gerou o v1 (ver
CLAUDE.md).

Uso (a partir da raiz do projeto, para `import src....` resolver):
``python -m scripts.gerar_dataset_producao_v2``

Diferenças em relação a ``gerar_dataset_producao_v1.py``:
1. ``ParametrosPopulacionaisStub(candidato_alvo=None)`` -- ativa o sorteio
   de candidato-alvo por janela (já implementado e testado em
   `gerar_par_de_classes_real`, commit `0fffdd5`, mas nunca invocado
   corretamente no v1: o script v1 construía `ParametrosPopulacionaisStub()`
   sem overrides, resolvendo para o default `candidato_alvo=0` fixo em
   TODA janela/classe/combinação — achado desta sessão, não um bug do
   mecanismo de sorteio em si).
2. `resultado_alvo` não é mais `0.5` fixo (default antigo de
   `ElectionModel`) — `gerar_par_de_classes_real` (`src/pipeline/geracao.py`)
   agora calcula `K_RESULTADO_ALVO / n_candidatos` automaticamente
   (`src/pipeline/config.py::K_RESULTADO_ALVO`, `k=2.0`). Nenhuma mudança
   necessária NESTE script para isso — é uma mudança na camada de geração,
   não na configuração da grade. Motivação: o v1 tinha ativação binária
   0%/100%/100% em `recompensa ∈ {0.5, 1.0, 1.5}` (não um gradiente) — ver
   CLAUDE.md para o achado completo e a ressalva sobre o que essa mudança
   resolve e o que continua uma limitação conhecida.
3. Output em ``output/dataset_producao_v2/`` e manifesto/MLflow próprios
   (``manifesto_producao_v2.db``) -- nunca reusar o do v1 (parâmetros
   populacionais mudaram: `candidato_alvo` de `0` para `None` — ver aviso
   em `runner.orquestrar` sobre reusar manifesto entre stubs diferentes).

PLACEHOLDERS -- estado de calibração dos parâmetros usados aqui
------------------------------------------------------------------
- ``recompensa`` (eixo da grade, abaixo): JÁ TEM valor decidido, com base
  dupla documentada no comentário ao lado do campo -- não é um chute no
  mesmo sentido dos três abaixo.
- ``resultado_alvo`` (calculado em `geracao.py`, não aqui): decidido por
  smoke test empírico nesta sessão (`k=2.0`), NÃO é calibração formal —
  mesma categoria de pendência dos três abaixo, ver `K_RESULTADO_ALVO`
  em `src/pipeline/config.py`.
- ``TAXA_FONTE_A_PLACEHOLDER``, ``VOLUME_MEDIO_FONTE_A_PLACEHOLDER``,
  ``TAXA_FONTE_B_PLACEHOLDER``: PLACEHOLDERS genuínos, sem base de
  literatura (ver CLAUDE.md, Item 5) -- servem só para a pipeline rodar
  sem erro. **Permanecem com os MESMOS valores do v1 nesta v2** — decisão
  explícita desta sessão: as fontes levantadas (custo-por-voto Kaba 2022,
  cadência Votium) foram julgadas analogias fracas demais para virar
  número no prazo desta rodada. Este dataset (v2) NÃO deve ser considerado
  calibrado nesses três parâmetros — continua pendente para uma v3 futura,
  não confundir "candidato_alvo/resultado_alvo corrigidos" com "todos os
  parâmetros calibrados".
"""

from __future__ import annotations

from pathlib import Path

from src.pipeline.config import (
    N_JANELAS_POR_CLASSE_PADRAO,
    GradeFatorial,
    ParametrosPopulacionaisStub,
    ParametrosStubGeracao,
)
from src.pipeline.geracao import criar_gerador_real
from src.pipeline.manifest import Manifesto
from src.pipeline.runner import orquestrar_paralelo
from src.pipeline.tracking import configurar_mlflow

# --------------------------------------------------------------------------
# Placeholders de Fonte A/B -- NÃO calibrados, sem base de literatura ainda.
# Idênticos aos usados no v1 -- ver docstring do módulo acima para a decisão
# explícita de não recalibrar nesta rodada (item 3 das decisões fechadas).
# --------------------------------------------------------------------------

TAXA_FONTE_A_PLACEHOLDER: float = 1.0
"""PLACEHOLDER -- não calibrado, sem base de literatura ainda (ver
CLAUDE.md, Item 5). Mesmo valor do v1 -- pendência não resolvida nesta v2
também, decisão explícita (fontes levantadas julgadas analogias fracas
demais para virar número neste prazo)."""

VOLUME_MEDIO_FONTE_A_PLACEHOLDER: float = 1.0
"""PLACEHOLDER -- não calibrado, sem base de literatura ainda (ver
CLAUDE.md, Item 5). Mesmo valor do v1 -- pendência não resolvida nesta v2
também."""

TAXA_FONTE_B_PLACEHOLDER: float = 1.0
"""PLACEHOLDER -- não calibrado, sem base de literatura ainda (ver
CLAUDE.md, Item 5). Mesmo valor do v1 -- pendência não resolvida nesta v2
também."""

# tau_kendall já decidido (força-alvo de dependência A<->B na cópula
# Clayton) -- não é placeholder no mesmo sentido dos três acima.
TAU_KENDALL: float = 0.6

# --------------------------------------------------------------------------
# Escala de PRODUÇÃO -- n_janelas de verdade (PLANO §5.2.4).
# --------------------------------------------------------------------------

SEEDS_PRODUCAO: list[int] = [1, 2]
"""Mesmas seeds do v1 -- nenhum valor muda entre as duas rodadas além de
candidato_alvo/resultado_alvo (ver diferenças no docstring do módulo)."""

DIRETORIO_OUTPUT_PRODUCAO = Path("output/dataset_producao_v2")
CAMINHO_MANIFESTO_PRODUCAO = "manifesto_producao_v2.db"
CAMINHO_MLFLOW_PRODUCAO = DIRETORIO_OUTPUT_PRODUCAO / "mlflow_producao_v2.db"


def main() -> None:
    grade = GradeFatorial(
        g=["pool"],
        pi=[0.0, 0.5, 0.75, 0.9, 0.95],
        delta_t=[0.0, 2.0, 24.0],
        # recompensa (proxy de lambda, design fatorial): [0.5, 1.0, 1.5]
        # (baixa/média/alta) -- JÁ DECIDIDO, com base DUPLA e INDEPENDENTE,
        # documentada separadamente porque vêm de fontes diferentes:
        #
        # 1) A RAZÃO entre os níveis (0.5 : 1 : 1.5) é ancorada em Kaba
        #    (2022) -- custo de compra de voto como fração do PIB per
        #    capita, valores empíricos 2,65% / 5,29% / 7,94%, cuja razão é
        #    aproximadamente 0.5 : 1 : 1.5. Isto é o que vem da literatura:
        #    a PROPORÇÃO entre baixa/média/alta, não a magnitude absoluta.
        #
        # 2) O VALOR-ÂNCORA (1.0 para "média") vem da estrutura do próprio
        #    modelo, não do Kaba: com threshold_range=(0.2, 0.8) e
        #    alpha_beta=(2, 2) já travados (ParametrosPopulacionaisStub),
        #    recompensa=1.0 é o ponto de simetria onde
        #    recompensa * E[propensao] == E[utility_threshold] (ambos 0.5),
        #    produzindo ~50% de adesão entre agentes racionais
        #    (VoterAgent.step -- ver agent.py) -- evita saturação em 0%/100%
        #    de adesão, que tornaria o dataset degenerado.
        #
        # As duas fontes são INDEPENDENTES: a razão vem da literatura, o
        # valor absoluto vem da matemática do modelo. Não é uma calibração
        # numérica direta do Kaba -- Kaba não fornece a magnitude absoluta
        # de `recompensa` nesta escala do gerador, só a proporção relativa
        # entre níveis de intensidade adversarial.
        #
        # NOTA (v2): estes três níveis testam "abaixo / próximo / acima do
        # limiar de ativação do contrato" (C_min), não uma escala contínua
        # fraco->forte -- por construção matemática (distribuições de
        # fracao_candidato_alvo bem separadas por nível), não existe k em
        # resultado_alvo=k/n_candidatos que produza um gradiente suave nos
        # três pontos ao mesmo tempo. Ver CLAUDE.md para o achado completo
        # e a implicação para avaliação de M1/M2/M3 por nível de recompensa.
        recompensa=[0.5, 1.0, 1.5],
        rho=[0.0, 0.5, 1.0],
        # beta: campo obrigatório de GradeFatorial, mas expandir_grade()
        # NÃO o cruza -- toda combinação do grid principal recebe beta=1
        # fixo (ver config.py::expandir_grade). [1] aqui é só para
        # satisfazer a assinatura da dataclass, não uma variação real.
        beta=[1],
        seeds=SEEDS_PRODUCAO,
    )

    stub_geracao = ParametrosStubGeracao(
        tau_kendall=TAU_KENDALL,
        taxa_fonte_a=TAXA_FONTE_A_PLACEHOLDER,
        volume_medio_fonte_a=VOLUME_MEDIO_FONTE_A_PLACEHOLDER,
        taxa_fonte_b=TAXA_FONTE_B_PLACEHOLDER,
    )

    # Defaults travados: n_agentes=500, n_candidatos=5, alpha_beta=(2,2),
    # prop_racional=0.9 -- sem overrides. candidato_alvo=None é a MUDANÇA
    # desta v2 em relação ao v1 (ver docstring do módulo, item 1): ativa o
    # sorteio por janela em gerar_par_de_classes_real, em vez do default
    # fixo `candidato_alvo=0` que o v1 usou sem perceber.
    populacionais = ParametrosPopulacionaisStub(candidato_alvo=None)

    DIRETORIO_OUTPUT_PRODUCAO.mkdir(parents=True, exist_ok=True)

    configurar_mlflow(CAMINHO_MLFLOW_PRODUCAO)
    manifesto = Manifesto(CAMINHO_MANIFESTO_PRODUCAO)
    gerador = criar_gerador_real(populacionais, stub_geracao, DIRETORIO_OUTPUT_PRODUCAO)

    print("=== Geração do dataset de produção v2 ===")
    print(f"Output: {DIRETORIO_OUTPUT_PRODUCAO}")
    print(f"Manifesto: {CAMINHO_MANIFESTO_PRODUCAO}")
    print(f"MLflow: {CAMINHO_MLFLOW_PRODUCAO}")
    print(f"n_janelas_por_classe: {N_JANELAS_POR_CLASSE_PADRAO}")
    print(f"seeds: {SEEDS_PRODUCAO}")
    print()

    orquestrar_paralelo(
        grade,
        stub_geracao,
        populacionais,
        n_janelas_por_classe=N_JANELAS_POR_CLASSE_PADRAO,
        manifesto=manifesto,
        gerar_par_de_classes=gerador,
    )

    print("\nConcluído. Verifique os HDF5s em", DIRETORIO_OUTPUT_PRODUCAO)


if __name__ == "__main__":
    main()
