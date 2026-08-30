"""Testes do manifesto de execução do pipeline (SQLite)."""

from __future__ import annotations

import os

import pytest

from src.pipeline.manifest import Manifesto


@pytest.fixture
def caminho_manifesto(tmp_path) -> str:
    return str(tmp_path / "manifesto.sqlite")


def test_registrar_pending_cria_linha_pending(caminho_manifesto: str) -> None:
    manifesto = Manifesto(caminho_manifesto)

    manifesto.registrar_pending("r1", {"g": "secao"}, seed=1)

    linha = manifesto.obter("r1")
    assert linha is not None
    assert linha["status"] == "pending"
    assert linha["seed"] == 1
    manifesto.close()


def test_registrar_pending_nao_sobrescreve_success(caminho_manifesto: str) -> None:
    manifesto = Manifesto(caminho_manifesto)
    manifesto.registrar_pending("r1", {"g": "secao"}, seed=1)
    manifesto.marcar_running("r1")
    manifesto.marcar_success("r1", n_janelas_ok=10, n_janelas_falha=0, n_contrato_nao_ativado=2, caminho_output="out.h5")

    manifesto.registrar_pending("r1", {"g": "outra_coisa"}, seed=99)

    linha = manifesto.obter("r1")
    assert linha["status"] == "success"
    assert linha["n_janelas_ok"] == 10
    assert linha["caminho_output"] == "out.h5"
    manifesto.close()


def test_marcar_running_success_failed_atualizam_campos(caminho_manifesto: str) -> None:
    manifesto = Manifesto(caminho_manifesto)
    manifesto.registrar_pending("r1", {"g": "secao"}, seed=1)

    manifesto.marcar_running("r1")
    assert manifesto.obter("r1")["status"] == "running"

    manifesto.marcar_success("r1", n_janelas_ok=5, n_janelas_falha=1, n_contrato_nao_ativado=3, caminho_output="a.h5")
    linha = manifesto.obter("r1")
    assert linha["status"] == "success"
    assert linha["n_janelas_ok"] == 5
    assert linha["n_janelas_falha"] == 1
    assert linha["n_contrato_nao_ativado"] == 3
    assert linha["caminho_output"] == "a.h5"
    assert linha["erro"] is None

    manifesto.registrar_pending("r2", {"g": "municipio"}, seed=2)
    manifesto.marcar_failed("r2", "algo deu errado")
    linha2 = manifesto.obter("r2")
    assert linha2["status"] == "failed"
    assert linha2["erro"] == "algo deu errado"
    manifesto.close()


def test_pendentes_inclui_tudo_menos_success(caminho_manifesto: str) -> None:
    manifesto = Manifesto(caminho_manifesto)
    manifesto.registrar_pending("r_pending", {}, seed=1)
    manifesto.registrar_pending("r_running", {}, seed=2)
    manifesto.marcar_running("r_running")
    manifesto.registrar_pending("r_failed", {}, seed=3)
    manifesto.marcar_failed("r_failed", "erro")
    manifesto.registrar_pending("r_success", {}, seed=4)
    manifesto.marcar_success("r_success", 1, 0, 0, "x.h5")

    pendentes = set(manifesto.pendentes())

    assert pendentes == {"r_pending", "r_running", "r_failed"}
    manifesto.close()


def test_reabrir_manifesto_preserva_dados(caminho_manifesto: str) -> None:
    manifesto = Manifesto(caminho_manifesto)
    manifesto.registrar_pending("r1", {"g": "secao"}, seed=1)
    manifesto.marcar_success("r1", 1, 0, 0, "x.h5")
    manifesto.close()

    assert os.path.exists(caminho_manifesto)

    manifesto_reaberto = Manifesto(caminho_manifesto)
    linha = manifesto_reaberto.obter("r1")
    assert linha is not None
    assert linha["status"] == "success"
    manifesto_reaberto.close()


def test_obter_run_id_inexistente_retorna_none(caminho_manifesto: str) -> None:
    manifesto = Manifesto(caminho_manifesto)
    assert manifesto.obter("nao_existe") is None
    manifesto.close()
