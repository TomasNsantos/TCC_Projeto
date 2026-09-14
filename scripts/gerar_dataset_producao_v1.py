"""Gera o dataset PRINCIPAL (v1) -- grid completo, n_janelas de produção.

Baseado em ``scripts/gerar_dataset_validacao_pipeline.py`` (mesma grade,
mesmos placeholders de Fonte A/B, mesma recompensa) -- NENHUM valor
numérico muda em relação à validação, só a escala da execução.

Uso (a partir da raiz do projeto, para `import src....` resolver):
``python -m scripts.gerar_dataset_producao_v1``

Diferenças em relação a ``gerar_dataset_validacao_pipeline.py``:
1. ``n_janelas_por_classe=N_JANELAS_POR_CLASSE_PADRAO`` (1000, PLANO
   §5.2.4) em vez do valor pequeno de validação (10).
2. Output em ``output/dataset_producao_v1/`` -- sufixo ``v1`` sinaliza
   que pode precisar de ``v2`` quando ``taxa_fonte_a``/
   ``volume_medio_fonte_a``/``taxa_fonte_b``/``recompensa`` forem
   recalibrados com literatura completa (ver placeholders abaixo).
3. Manifesto próprio (``manifesto_producao_v1.db``), não reusa o de
   validação -- ``run_id`` não inclui ``n_janelas_por_classe``
   (``src/pipeline/config.py::run_id``), então não há conflito de
   identidade entre os dois manifestos, mas cada rodada precisa do seu
   próprio arquivo de qualquer forma (ver aviso em ``runner.orquestrar``
   sobre reusar manifesto com stub/populacionais diferentes).
4. Chama ``configurar_mlflow()`` explicitamente antes de orquestrar,
   apontando para um arquivo dentro do diretório de output -- evita
   cair no tracking URI default do MLflow (``sqlite:///mlflow.db``,
   relativo ao ``cwd``), que foi a causa do ``mlflow.db`` solto na raiz
   do repo na rodada de validação anterior (ver CLAUDE.md).

PLACEHOLDERS -- estado de calibração dos parâmetros usados aqui
------------------------------------------------------------------
- ``recompensa`` (eixo da grade, abaixo): JÁ TEM valor decidido, com base
  dupla documentada no comentário ao lado do campo -- não é um chute no
  mesmo sentido dos três abaixo.
- ``TAXA_FONTE_A_PLACEHOLDER``, ``VOLUME_MEDIO_FONTE_A_PLACEHOLDER``,
  ``TAXA_FONTE_B_PLACEHOLDER``: PLACEHOLDERS genuínos, sem base de
  literatura (ver CLAUDE.md, Item 5) -- servem só para a pipeline rodar
  sem erro. Este dataset (v1) usa os MESMOS valores provisórios da
  validação -- ele NÃO deve ser considerado o dataset final até esses
  três parâmetros serem recalibrados (daí o sufixo v1/v2).
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
# Idênticos aos usados em gerar_dataset_validacao_pipeline.py -- ver esse
# script para a explicação completa do achado (throughput de rede != taxa
# de eventos por janela do gerador, ver CLAUDE.md Item 5).
#
# Valores abaixo escolhidos só para exercitar a pipeline (mesma ordem de
# grandeza de `recompensa`, para não ter tráfego de fundo absurdamente
# maior ou menor que o sinal adversarial) -- é um CHUTE, não uma decisão,
# nem minha nem do usuário. Este dataset (v1) NÃO deve ser tratado como
# calibrado até estes três valores serem substituídos.
# --------------------------------------------------------------------------

TAXA_FONTE_A_PLACEHOLDER: float = 1.0
"""PLACEHOLDER -- não calibrado, sem base de literatura ainda (ver
CLAUDE.md, Item 5). Mesmo valor usado na validação de pipeline -- este
dataset (v1) precisará de v2 quando isso for recalibrado."""

VOLUME_MEDIO_FONTE_A_PLACEHOLDER: float = 1.0
"""PLACEHOLDER -- não calibrado, sem base de literatura ainda (ver
CLAUDE.md, Item 5). Mesmo valor usado na validação de pipeline -- este
dataset (v1) precisará de v2 quando isso for recalibrado."""

TAXA_FONTE_B_PLACEHOLDER: float = 1.0
"""PLACEHOLDER -- não calibrado, sem base de literatura ainda (ver
CLAUDE.md, Item 5). Mesmo valor usado na validação de pipeline -- este
dataset (v1) precisará de v2 quando isso for recalibrado."""

# tau_kendall já decidido (força-alvo de dependência A<->B na cópula
# Clayton) -- não é placeholder no mesmo sentido dos três acima.
TAU_KENDALL: float = 0.6

# --------------------------------------------------------------------------
# Escala de PRODUÇÃO -- n_janelas de verdade (PLANO §5.2.4), não a escala
# pequena da validação de pipeline.
# --------------------------------------------------------------------------

SEEDS_PRODUCAO: list[int] = [1, 2]
"""Mesmas seeds da validação de pipeline -- nenhum valor muda entre as
duas rodadas, só n_janelas_por_classe (ver diferenças no docstring do
módulo)."""

DIRETORIO_OUTPUT_PRODUCAO = Path("output/dataset_producao_v1")
CAMINHO_MANIFESTO_PRODUCAO = "manifesto_producao_v1.db"
CAMINHO_MLFLOW_PRODUCAO = DIRETORIO_OUTPUT_PRODUCAO / "mlflow_producao_v1.db"


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
        seeds=SEEDS_PRODUCAO,
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

    DIRETORIO_OUTPUT_PRODUCAO.mkdir(parents=True, exist_ok=True)

    configurar_mlflow(CAMINHO_MLFLOW_PRODUCAO)
    manifesto = Manifesto(CAMINHO_MANIFESTO_PRODUCAO)
    gerador = criar_gerador_real(populacionais, stub_geracao, DIRETORIO_OUTPUT_PRODUCAO)

    print("=== Geração do dataset de produção v1 ===")
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
