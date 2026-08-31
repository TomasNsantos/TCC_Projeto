"""Testes de src/pipeline/tracking.py (configurar_mlflow, registrar_run_mlflow)."""

from __future__ import annotations

import mlflow
import pytest

from src.pipeline.tracking import configurar_mlflow, registrar_run_mlflow


@pytest.fixture(autouse=True)
def tracking_uri_temporario(tmp_path) -> None:
    """Nunca aponta para o banco real do projeto -- `tmp_path` isola cada teste."""
    configurar_mlflow(tmp_path / "mlflow.db")


def test_registrar_run_mlflow_sucesso_e_consultavel() -> None:
    resultado = {
        "status": "success",
        "n_janelas_ok": 2000,
        "n_janelas_falha": 3,
        "n_contrato_nao_ativado": 42,
        "caminho_output": "/data/g-pool_delta_t-2.0000_seed-1.h5",
    }
    params = {"g": "pool", "delta_t": 2.0, "recompensa": 5.0, "rho": 0.0, "beta": 1, "n_agentes": 100}

    registrar_run_mlflow("g-pool_delta_t-2.0000_seed-1", params, seed=1, resultado=resultado)

    df = mlflow.search_runs()
    assert len(df) == 1
    linha = df.iloc[0]

    assert linha["tags.mlflow.runName"] == "g-pool_delta_t-2.0000_seed-1"
    assert linha["params.g"] == "pool"
    assert linha["params.delta_t"] == "2.0"
    assert linha["params.seed"] == "1"
    assert linha["params.n_agentes"] == "100"
    assert linha["metrics.n_janelas_ok"] == 2000
    assert linha["metrics.n_janelas_falha"] == 3
    assert linha["metrics.n_contrato_nao_ativado"] == 42
    assert linha["tags.status"] == "success"
    assert linha["tags.caminho_output"] == "/data/g-pool_delta_t-2.0000_seed-1.h5"


def test_registrar_run_mlflow_falha_e_consultavel_com_status_failed() -> None:
    params = {"g": "secao", "delta_t": 2.0, "recompensa": 5.0, "rho": 0.5, "beta": 1}
    resultado = {"status": "failed", "erro": "RuntimeError: Todas as 10 janelas da classe positiva falharam"}

    registrar_run_mlflow("g-secao_delta_t-2.0000_seed-2", params, seed=2, resultado=resultado)

    df = mlflow.search_runs()
    assert len(df) == 1
    linha = df.iloc[0]

    assert linha["tags.status"] == "failed"
    assert linha["params.erro"] == "RuntimeError: Todas as 10 janelas da classe positiva falharam"
    # métricas de sucesso não devem aparecer numa linha de falha
    assert "metrics.n_janelas_ok" not in df.columns or df.loc[df["tags.status"] == "failed", "metrics.n_janelas_ok"].isna().all()


def test_registrar_run_mlflow_trunca_erro_muito_longo() -> None:
    from src.pipeline.tracking import _MAX_PARAM_VAL_LENGTH

    erro_longo = "x" * (_MAX_PARAM_VAL_LENGTH + 500)
    resultado = {"status": "failed", "erro": erro_longo}

    registrar_run_mlflow("run-erro-longo", {"g": "pool"}, seed=1, resultado=resultado)

    df = mlflow.search_runs()
    linha = df.iloc[0]
    assert len(linha["params.erro"]) == _MAX_PARAM_VAL_LENGTH


def test_registrar_run_mlflow_sucesso_e_falha_tem_mesmas_colunas_de_parametro() -> None:
    """Paridade de parâmetros entre sucesso e falha -- pré-requisito para o
    MLflow servir como painel único (ver runner.py/CLAUDE.md): uma linha
    'failed' precisa ter as mesmas colunas params.* que uma linha
    'success', para permitir filtrar/comparar as duas de forma confiável."""
    params_comuns = {"g": "pool", "delta_t": 2.0, "n_agentes": 100, "tau_kendall": 0.5}

    registrar_run_mlflow("run-sucesso", params_comuns, seed=1, resultado={
        "status": "success", "n_janelas_ok": 10, "n_janelas_falha": 0,
        "n_contrato_nao_ativado": 1, "caminho_output": "a.h5",
    })
    registrar_run_mlflow("run-falha", params_comuns, seed=2, resultado={"status": "failed", "erro": "boom"})

    df = mlflow.search_runs()
    assert len(df) == 2

    colunas_params = {c for c in df.columns if c.startswith("params.") and c != "params.erro"}
    for coluna in colunas_params:
        assert df[coluna].notna().all(), f"coluna {coluna} tem valor faltando -- paridade sucesso/falha quebrada"
