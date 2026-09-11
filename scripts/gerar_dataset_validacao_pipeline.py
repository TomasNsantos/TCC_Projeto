"""Gera um dataset PEQUENO só para validar a pipeline ponta a ponta
(grade -> gerador real -> HDF5), NÃO o dataset principal.

Uso (a partir da raiz do projeto, para `import src....` resolver):
``python -m scripts.gerar_dataset_validacao_pipeline``

Diferenças deliberadas em relação a uma rodada de produção:
- ``N_JANELAS_POR_CLASSE_VALIDACAO`` (10) é muito menor que
  ``N_JANELAS_POR_CLASSE_PADRAO`` (1000, produção) -- só para confirmar que
  o pipeline roda em minutos, não horas.
- ``SEEDS_VALIDACAO`` tem poucas seeds -- não cobre o grid com a
  profundidade estatística que a produção exigiria.
- Output isolado em ``output/dataset_validacao_pipeline_v1/`` e manifesto
  próprio (``manifesto_validacao_pipeline_v1.db``), para nunca ser
  confundido com o dataset oficial nem compartilhar o mesmo arquivo de
  progresso.

Para gerar a versão de produção mais tarde: NÃO existe restrição técnica
para simplesmente trocar ``N_JANELAS_POR_CLASSE_VALIDACAO`` por
``N_JANELAS_POR_CLASSE_PADRAO`` e apontar para um manifesto/diretório de
output novos -- ``run_id`` (ver ``src/pipeline/config.py::run_id``) é
calculado só a partir dos eixos da grade (g, pi, delta_t, recompensa, rho,
beta, seed), não de ``n_janelas_por_classe``, então não há conflito de
identidade entre uma rodada de validação e uma de produção, desde que cada
uma use seu próprio manifesto (a mistura só é perigosa se você REUSAR o
MESMO arquivo de manifesto entre rodadas com parametros_stub/populacionais
diferentes -- ver aviso em ``runner.orquestrar``).

PLACEHOLDERS -- estado de calibração dos parâmetros usados aqui
------------------------------------------------------------------
- ``recompensa`` (eixo da grade, abaixo): JÁ TEM valor decidido, com base
  dupla documentada no comentário ao lado do campo -- não é um chute no
  mesmo sentido dos três abaixo.
- ``TAXA_FONTE_A_PLACEHOLDER``, ``VOLUME_MEDIO_FONTE_A_PLACEHOLDER``,
  ``TAXA_FONTE_B_PLACEHOLDER``: PLACEHOLDERS genuínos, sem base de
  literatura (ver CLAUDE.md, Item 5) -- servem só para a pipeline rodar
  sem erro nesta validação. NÃO usar esses três valores no dataset final
  sem antes calibrá-los.
"""

from __future__ import annotations

from pathlib import Path

from src.pipeline.config import GradeFatorial, ParametrosPopulacionaisStub, ParametrosStubGeracao
from src.pipeline.geracao import criar_gerador_real
from src.pipeline.manifest import Manifesto
from src.pipeline.runner import orquestrar_paralelo

# --------------------------------------------------------------------------
# Placeholders de Fonte A/B -- NÃO calibrados, sem base de literatura ainda.
#
# Achado registrado (ver CLAUDE.md, Item 5): os papers já sintetizados no
# projeto (ZKP-FinPay ~1.200 tx/s, Oasis/Phala ~1.000 TPS, Secret Network
# ~10.000 TPS "teórico, não validado") dão THROUGHPUT AGREGADO DE REDE, não
# valor médio de transação nem taxa por janela/cenário do gerador -- e a
# conversão de TPS de rede para taxa de eventos por janela do gerador ainda
# depende de uma decisão de escala não tomada. Não há, hoje, nenhuma base
# numérica sólida para taxa_fonte_a/volume_medio_fonte_a/taxa_fonte_b.
#
# Valores abaixo escolhidos só para exercitar a pipeline (mesma ordem de
# grandeza de `recompensa`, para não ter tráfego de fundo absurdamente
# maior ou menor que o sinal adversarial) -- é um CHUTE, não uma decisão,
# nem minha nem do usuário. NÃO usar para o dataset final.
# --------------------------------------------------------------------------

TAXA_FONTE_A_PLACEHOLDER: float = 1.0
"""PLACEHOLDER -- não calibrado, sem base de literatura ainda (ver
CLAUDE.md, Item 5). Valor escolhido só para exercitar a pipeline, NÃO usar
para o dataset final."""

VOLUME_MEDIO_FONTE_A_PLACEHOLDER: float = 1.0
"""PLACEHOLDER -- não calibrado, sem base de literatura ainda (ver
CLAUDE.md, Item 5). Valor escolhido só para exercitar a pipeline, NÃO usar
para o dataset final."""

TAXA_FONTE_B_PLACEHOLDER: float = 1.0
"""PLACEHOLDER -- não calibrado, sem base de literatura ainda (ver
CLAUDE.md, Item 5). Valor escolhido só para exercitar a pipeline, NÃO usar
para o dataset final."""

# tau_kendall já decidido (força-alvo de dependência A<->B na cópula
# Clayton) -- não é placeholder no mesmo sentido dos três acima.
TAU_KENDALL: float = 0.6

# --------------------------------------------------------------------------
# Escala de validação -- BEM menor que produção, só para rodar em minutos.
# --------------------------------------------------------------------------

N_JANELAS_POR_CLASSE_VALIDACAO: int = 10
"""Muito menor que N_JANELAS_POR_CLASSE_PADRAO (1000, produção) -- só para
confirmar que a pipeline roda ponta a ponta em minutos, não horas. NÃO usar
para o dataset final."""

SEEDS_VALIDACAO: list[int] = [1, 2]
"""Poucas seeds -- suficiente para exercitar reprodutibilidade/paralelismo
sem cobrir o grid com profundidade estatística de produção."""

DIRETORIO_OUTPUT_VALIDACAO = Path("output/dataset_validacao_pipeline_v1")
CAMINHO_MANIFESTO_VALIDACAO = "manifesto_validacao_pipeline_v1.db"


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
        recompensa=[0.5, 1.0, 1.5],
        rho=[0.0, 0.5, 1.0],
        # beta: campo obrigatório de GradeFatorial, mas expandir_grade()
        # NÃO o cruza -- toda combinação do grid principal recebe beta=1
        # fixo (ver config.py::expandir_grade). [1] aqui é só para
        # satisfazer a assinatura da dataclass, não uma variação real.
        beta=[1],
        seeds=SEEDS_VALIDACAO,
    )

    stub_geracao = ParametrosStubGeracao(
        tau_kendall=TAU_KENDALL,
        taxa_fonte_a=TAXA_FONTE_A_PLACEHOLDER,
        volume_medio_fonte_a=VOLUME_MEDIO_FONTE_A_PLACEHOLDER,
        taxa_fonte_b=TAXA_FONTE_B_PLACEHOLDER,
    )

    # Defaults já travados: n_agentes=500, n_candidatos=5, alpha_beta=(2,2),
    # prop_racional=0.9 -- sem overrides.
    populacionais = ParametrosPopulacionaisStub()

    DIRETORIO_OUTPUT_VALIDACAO.mkdir(parents=True, exist_ok=True)

    manifesto = Manifesto(CAMINHO_MANIFESTO_VALIDACAO)
    gerador = criar_gerador_real(populacionais, stub_geracao, DIRETORIO_OUTPUT_VALIDACAO)

    print("=== Geração de validação de pipeline (NÃO é o dataset final) ===")
    print(f"Output: {DIRETORIO_OUTPUT_VALIDACAO}")
    print(f"Manifesto: {CAMINHO_MANIFESTO_VALIDACAO}")
    print(f"n_janelas_por_classe: {N_JANELAS_POR_CLASSE_VALIDACAO} (produção real: 1000)")
    print(f"seeds: {SEEDS_VALIDACAO}")
    print()

    orquestrar_paralelo(
        grade,
        stub_geracao,
        populacionais,
        n_janelas_por_classe=N_JANELAS_POR_CLASSE_VALIDACAO,
        manifesto=manifesto,
        gerar_par_de_classes=gerador,
    )

    print("\nConcluído. Verifique os HDF5s em", DIRETORIO_OUTPUT_VALIDACAO)


if __name__ == "__main__":
    main()
