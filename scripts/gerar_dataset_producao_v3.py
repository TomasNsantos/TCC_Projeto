"""Gera as 90 combinações delta_t=0.0 do dataset de produção v3 -- corrige o
acoplamento delta_t=0.0 => janela de tráfego de fundo (Fonte A/B, classe
negativa) zerada, ver CLAUDE.md (entrada "Bug corrigido (2026-09-17)").

Este script NÃO regenera as 180 combinações delta_t != 0.0 -- essas são
copiadas sem alteração do v2 (confirmadas byte-a-byte idênticas ao código
pós-fix, ver relatório da tarefa de regeneração), evitando ~6h de geração
redundante. A composição final de output/dataset_producao_v3/ (180 arquivos
copiados + 90 novos) é montada por um passo separado, fora deste script.

Uso (a partir da raiz do projeto, para `import src....` resolver):
``python -m scripts.gerar_dataset_producao_v3``

Diferenças em relação a ``gerar_dataset_producao_v2.py``:
1. ``delta_t=[0.0]`` -- só o nível que precisa ser regenerado. Os outros
   dois níveis (2.0, 24.0) não são tocados por este script.
2. Output em ``output/dataset_producao_v3_delta_t_zero/`` (diretório
   intermediário, só os 90 arquivos novos) e manifesto/MLflow próprios
   (``manifesto_producao_v3.db``) -- nunca reusar o do v2 (mesmo aviso já
   documentado em ``runner.orquestrar`` sobre reusar manifesto entre
   stubs/grades diferentes). O manifesto v3 registra só estas 90
   combinações nesta execução; a consolidação com as 180 copiadas do v2
   é feita à parte (ver passo de montagem do dataset).
3. Todo o resto (seeds, placeholders, tau_kendall, população, K_RESULTADO_ALVO)
   é IDÊNTICO ao v2 -- ver ``gerar_dataset_producao_v2.py`` para a
   justificativa completa de cada valor, não repetida aqui.
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

# Idênticos ao v2 -- mesma pendência de calibração (CLAUDE.md, Item 5).
TAXA_FONTE_A_PLACEHOLDER: float = 1.0
VOLUME_MEDIO_FONTE_A_PLACEHOLDER: float = 1.0
TAXA_FONTE_B_PLACEHOLDER: float = 1.0
TAU_KENDALL: float = 0.6

SEEDS_PRODUCAO: list[int] = [1, 2]
"""Mesmas seeds do v2 -- necessário para que os run_ids/HDF5s das 90
combinações novas usem exatamente as mesmas seeds já usadas no v2."""

DIRETORIO_OUTPUT_DELTA_T_ZERO = Path("output/dataset_producao_v3_delta_t_zero")
"""Diretório intermediário só com os 90 arquivos novos (delta_t=0.0) --
não é o diretório final do v3; a montagem consolidada
(output/dataset_producao_v3/) copia os 180 arquivos delta_t!=0.0 do v2
para cá e move estes 90, num passo separado."""

CAMINHO_MANIFESTO_PRODUCAO = "manifesto_producao_v3.db"
CAMINHO_MLFLOW_PRODUCAO = DIRETORIO_OUTPUT_DELTA_T_ZERO / "mlflow_producao_v3.db"


def main() -> None:
    grade = GradeFatorial(
        g=["pool"],
        pi=[0.0, 0.5, 0.75, 0.9, 0.95],
        delta_t=[0.0],  # ÚNICA diferença de eixo em relação ao v2 -- só o nível corrigido.
        recompensa=[0.5, 1.0, 1.5],
        rho=[0.0, 0.5, 1.0],
        beta=[1],
        seeds=SEEDS_PRODUCAO,
    )

    stub_geracao = ParametrosStubGeracao(
        tau_kendall=TAU_KENDALL,
        taxa_fonte_a=TAXA_FONTE_A_PLACEHOLDER,
        volume_medio_fonte_a=VOLUME_MEDIO_FONTE_A_PLACEHOLDER,
        taxa_fonte_b=TAXA_FONTE_B_PLACEHOLDER,
    )

    populacionais = ParametrosPopulacionaisStub(candidato_alvo=None)

    DIRETORIO_OUTPUT_DELTA_T_ZERO.mkdir(parents=True, exist_ok=True)

    configurar_mlflow(CAMINHO_MLFLOW_PRODUCAO)
    manifesto = Manifesto(CAMINHO_MANIFESTO_PRODUCAO)
    gerador = criar_gerador_real(populacionais, stub_geracao, DIRETORIO_OUTPUT_DELTA_T_ZERO)

    print("=== Geração das 90 combinações delta_t=0.0 (dataset de produção v3) ===")
    print(f"Output: {DIRETORIO_OUTPUT_DELTA_T_ZERO}")
    print(f"Manifesto: {CAMINHO_MANIFESTO_PRODUCAO}")
    print(f"MLflow: {CAMINHO_MLFLOW_PRODUCAO}")
    print(f"n_janelas_por_classe: {N_JANELAS_POR_CLASSE_PADRAO}")
    print(f"seeds: {SEEDS_PRODUCAO}")
    print(f"combinacoes esperadas: 1(g) x 5(pi) x 1(delta_t) x 3(recompensa) x 3(rho) x 2(seeds) = 90")
    print()

    orquestrar_paralelo(
        grade,
        stub_geracao,
        populacionais,
        n_janelas_por_classe=N_JANELAS_POR_CLASSE_PADRAO,
        manifesto=manifesto,
        gerar_par_de_classes=gerador,
    )

    print("\nConcluído. Verifique os HDF5s em", DIRETORIO_OUTPUT_DELTA_T_ZERO)


if __name__ == "__main__":
    main()
