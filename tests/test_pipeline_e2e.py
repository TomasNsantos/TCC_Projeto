"""Teste de integração único: roda a mecânica completa do pipeline com o
gerador REAL (não fake), de ponta a ponta — config → orquestrar_paralelo →
gerar_par_de_classes_real (ElectionModel + cópula Clayton) → storage.py
(HDF5) → tracking.py (MLflow).

Todos os testes anteriores de `orquestrar`/`orquestrar_paralelo` injetam um
`gerar_par_de_classes` FAKE (`test_pipeline_runner.py`); os testes de
`gerar_par_de_classes_real` chamam a função isolada, sem passar por
`orquestrar_paralelo` (`test_pipeline_geracao.py`). Este arquivo é o
primeiro a exercitar as duas coisas juntas.

**Gap documentado, não resolvido aqui:** o pedido original desta tarefa
descrevia "8 combinações principais da grade + robustez_beta com 2
combinações extras (beta variando)". Isso não é possível hoje:
`orquestrar`/`orquestrar_paralelo` só aceitam UM `GradeFatorial`, e
`expandir_grade()` sempre força `beta=1` em toda combinação que produz —
`RobustezBeta`/`expandir_grade_robustez` (que de fato variam β) nunca
foram conectados ao runner por nenhuma tarefa anterior. Resolvido rodando
`orquestrar_paralelo()` duas vezes sobre o MESMO manifesto com duas grades
DIFERENTES (variando `rho`, não `beta` — ambas ficam com `beta=1`) somando
os mesmos 10 `run_id`s pedidos — isso já exercita algo real e não coberto
antes (multi-grade sobre o mesmo manifesto), sem fingir uma integração de
robustez de β que não existe. A rede de segurança para o eixo de β mora em
`tests/test_pipeline_geracao.py::test_fragmentacao_beta_chega_ate_o_hdf5`
(chama `gerar_par_de_classes_real` direto, não via `orquestrar_paralelo`).
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
import pytest

from src.pipeline.config import (
    GradeFatorial,
    ParametrosPopulacionaisStub,
    ParametrosStubGeracao,
    expandir_grade,
    run_id,
)
from src.pipeline.geracao import criar_gerador_real
from src.pipeline.manifest import Manifesto
from src.pipeline.runner import orquestrar_paralelo
from src.pipeline.tracking import configurar_mlflow

# Mesmos valores de tests/test_pipeline_geracao.py::_PARAMS_ATIVA_CONTRATO,
# já confirmados (naquele arquivo) como ativando o contrato na maioria das
# janelas -- não invento novos números aqui. delta_t/recompensa ficam fixos
# nesse valor; só g/rho/seed variam para formar o grid pequeno.
_DELTA_T = 20.0
_RECOMPENSA = 10.0

# n_janelas pequeno -- só valida mecânica de encaixe entre camadas
# (config -> runner -> gerador real -> storage -> mlflow), NÃO é
# representativo do volume de produção (N_JANELAS_POR_CLASSE_PADRAO=1000
# em config.py).
_N_JANELAS_POR_CLASSE = 10


@pytest.fixture
def populacionais() -> ParametrosPopulacionaisStub:
    return ParametrosPopulacionaisStub(n_agentes=100, n_secoes=4, n_candidatos=3)


@pytest.fixture
def stub_geracao() -> ParametrosStubGeracao:
    return ParametrosStubGeracao(tau_kendall=0.5, taxa_fonte_a=1.0, volume_medio_fonte_a=1000.0, taxa_fonte_b=1.0)


@pytest.fixture
def grade_principal() -> GradeFatorial:
    """2(g) x 1(delta_t) x 1(recompensa) x 1(rho) x 4(seeds) = 8 combinações."""
    return GradeFatorial(g=["pool", "secao"], delta_t=[_DELTA_T], recompensa=[_RECOMPENSA], rho=[0.3], beta=[1], seeds=[1, 2, 3, 4])


@pytest.fixture
def grade_extra() -> GradeFatorial:
    """1(g) x 1(delta_t) x 1(recompensa) x 1(rho, diferente da principal) x 2(seeds) = 2 combinações.

    rho diferente da grade_principal garante run_ids distintos (run_id não
    inclui beta como eixo de robustez real -- ver docstring do módulo)."""
    return GradeFatorial(g=["pool"], delta_t=[_DELTA_T], recompensa=[_RECOMPENSA], rho=[0.5], beta=[1], seeds=[1, 2])


def _run_ids_da_grade(grade: GradeFatorial) -> list[str]:
    combinacoes = expandir_grade(grade)
    return [run_id({k: v for k, v in c.items() if k != "seed"}, c["seed"]) for c in combinacoes]


def test_pipeline_completo_config_ate_mlflow(
    grade_principal: GradeFatorial,
    grade_extra: GradeFatorial,
    populacionais: ParametrosPopulacionaisStub,
    stub_geracao: ParametrosStubGeracao,
    tmp_path,
) -> None:
    configurar_mlflow(tmp_path / "mlflow.db")
    diretorio_hdf5 = tmp_path / "hdf5"
    manifesto = Manifesto(str(tmp_path / "manifesto.sqlite"))

    # gerador_real (via criar_gerador_real) roda em processos loky separados
    # sob orquestrar_paralelo -- mutações de estado num closure local (ex.
    # um contador) não se propagam de volta ao processo principal (memória
    # separada entre processos, mesmo problema já documentado em
    # test_pipeline_runner.py para FakeGeradorParDeClasses). Por isso a
    # verificação de "zero chamadas novas" na reexecução usa
    # manifesto.pendentes() (estado real, não um contador mutado
    # remotamente), não um Mock/contador em cima do gerador.
    gerador_real = criar_gerador_real(populacionais, stub_geracao, diretorio_hdf5)

    orquestrar_paralelo(
        grade_principal, stub_geracao, populacionais, _N_JANELAS_POR_CLASSE, manifesto, gerador_real, n_jobs=2
    )
    orquestrar_paralelo(
        grade_extra, stub_geracao, populacionais, _N_JANELAS_POR_CLASSE, manifesto, gerador_real, n_jobs=2
    )

    run_ids_esperados = _run_ids_da_grade(grade_principal) + _run_ids_da_grade(grade_extra)
    assert len(run_ids_esperados) == 10
    assert len(set(run_ids_esperados)) == 10, "run_ids das duas grades colidiram -- grid mal desenhado"

    # --- manifesto: todas success ---
    assert manifesto.pendentes() == []
    for rid in run_ids_esperados:
        linha = manifesto.obter(rid)
        assert linha is not None, f"run_id {rid} não registrado no manifesto"
        assert linha["status"] == "success", f"run_id {rid} não terminou success: {linha}"

    # --- HDF5: existe, 6 tabelas, contagens/splits coerentes ---
    tabelas_esperadas = ("fonte_a", "fonte_b", "fonte_c_secao", "fonte_c_municipio", "fonte_c_estado", "metadados_janela")
    for rid in run_ids_esperados:
        caminho = manifesto.obter(rid)["caminho_output"]
        assert caminho is not None
        assert Path(caminho).exists(), f"HDF5 de {rid} não existe em disco: {caminho}"

        metadados = pd.read_hdf(caminho, "metadados_janela")
        for tabela in tabelas_esperadas:
            pd.read_hdf(caminho, tabela)  # levanta KeyError se a tabela não existir

        assert len(metadados) == 2 * _N_JANELAS_POR_CLASSE  # positiva + negativa

        splits_presentes = set(metadados["split"])
        assert splits_presentes <= {"train", "val", "test"}
        # proporção aproximada: com n_janelas=10 por classe a granularidade é
        # grosseira (calcular_split com n=10: boundary train<7, val<8.5,
        # test>=8.5 -- ver storage.py), então comparo presença/ordem de
        # grandeza, não igualdade exata de contagem.
        assert "train" in splits_presentes

    # --- MLflow: uma linha por run_id, status success ---
    df_mlflow = mlflow.search_runs()
    nomes_registrados = set(df_mlflow["tags.mlflow.runName"])
    assert nomes_registrados == set(run_ids_esperados)
    for rid in run_ids_esperados:
        status = df_mlflow.loc[df_mlflow["tags.mlflow.runName"] == rid, "tags.status"].iloc[0]
        assert status == "success"

    # --- reexecução: idempotência, zero chamadas novas ao gerador ---
    # gerador_envenenado sempre levanta -- prova mais forte que um contador
    # mutado (que não sobreviveria à separação de processos do loky, ver
    # nota acima): se _registrar_grade filtrar corretamente por
    # manifesto.pendentes() (todas já "success"), a_rodar fica vazia e
    # NENHUM worker é sequer disparado, então gerador_envenenado nunca roda
    # e nenhuma exceção aparece. Se a resumabilidade regredir e alguma
    # combinação for reprocessada, a exceção propaga através de
    # _rodar_uma_combinacao (que a captura só para marcar "failed", não
    # para escondê-la) e a asserção de status abaixo pega isso.
    def gerador_envenenado(params: dict, seed: int, n_janelas: int) -> dict:
        raise AssertionError("gerador não deveria ser chamado -- combinação já era 'success' no manifesto")

    orquestrar_paralelo(
        grade_principal, stub_geracao, populacionais, _N_JANELAS_POR_CLASSE, manifesto, gerador_envenenado, n_jobs=2
    )
    orquestrar_paralelo(
        grade_extra, stub_geracao, populacionais, _N_JANELAS_POR_CLASSE, manifesto, gerador_envenenado, n_jobs=2
    )

    assert manifesto.pendentes() == []
    for rid in run_ids_esperados:
        linha = manifesto.obter(rid)
        assert linha["status"] == "success", (
            f"run_id {rid} não é mais 'success' após reexecução -- gerador_envenenado deve ter rodado: {linha}"
        )

    manifesto.close()
