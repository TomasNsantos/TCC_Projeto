"""Testes da camada de configuração do pipeline (grade fatorial, run_id, seeds)."""

from __future__ import annotations

import numpy as np
import pytest

from src.pipeline.config import (
    GradeFatorial,
    ParametrosStubGeracao,
    RobustezBeta,
    derivar_seeds,
    expandir_grade,
    expandir_grade_robustez,
    run_id,
)


def test_expandir_grade_produz_produto_cartesiano_sem_cruzar_beta() -> None:
    grade = GradeFatorial(
        g=["secao", "municipio"],
        delta_t=[2.0, 24.0],
        recompensa=[1.0, 5.0, 10.0],
        rho=[0.0, 0.5, 1.0],
        beta=[1, 5, 20],
        seeds=[1, 2],
    )

    combinacoes = expandir_grade(grade)

    esperado = len(grade.g) * len(grade.delta_t) * len(grade.recompensa) * len(grade.rho) * len(grade.seeds)
    assert len(combinacoes) == esperado
    assert all(c["beta"] == 1 for c in combinacoes)
    assert len(combinacoes) == len({tuple(sorted(c.items())) for c in combinacoes})


def test_expandir_grade_robustez_cruza_beta_e_seeds_em_torno_da_base() -> None:
    base_config = {"g": "secao", "delta_t": 2.0, "recompensa": 5.0, "rho": 0.5, "beta": 1, "seed": 1}
    robustez = RobustezBeta(base_config=base_config, seeds=[10, 20, 30], beta=[1, 5, 20])

    combinacoes = expandir_grade_robustez(robustez)

    assert len(combinacoes) == len(robustez.beta) * len(robustez.seeds)
    for c in combinacoes:
        assert c["g"] == base_config["g"]
        assert c["delta_t"] == base_config["delta_t"]
        assert c["recompensa"] == base_config["recompensa"]
        assert c["rho"] == base_config["rho"]

    betas_vistos = {c["beta"] for c in combinacoes}
    seeds_vistos = {c["seed"] for c in combinacoes}
    assert betas_vistos == set(robustez.beta)
    assert seeds_vistos == set(robustez.seeds)


def test_robustez_beta_default_e_o_valor_do_plano() -> None:
    robustez = RobustezBeta(base_config={}, seeds=[1])
    assert robustez.beta == [1, 5, 20]


def test_parametros_stub_geracao_exige_todos_os_campos() -> None:
    with pytest.raises(TypeError):
        ParametrosStubGeracao()


def test_run_id_e_deterministico_e_ignora_ordem_das_chaves() -> None:
    params_a = {"g": "secao", "delta_t": 2.0, "rho": 0.5}
    params_b = {"rho": 0.5, "delta_t": 2.0, "g": "secao"}

    assert run_id(params_a, seed=3) == run_id(params_a, seed=3)
    assert run_id(params_a, seed=3) == run_id(params_b, seed=3)


def test_run_id_distingue_params_e_seeds_diferentes() -> None:
    base = {"g": "secao", "delta_t": 2.0}

    assert run_id(base, seed=1) != run_id(base, seed=2)
    assert run_id(base, seed=1) != run_id({"g": "municipio", "delta_t": 2.0}, seed=1)


def test_run_id_formatacao_fixa_de_floats_ignora_erro_de_ponto_flutuante() -> None:
    assert run_id({"x": 0.1 + 0.2}, seed=1) == run_id({"x": 0.3}, seed=1)


def test_derivar_seeds_e_deterministico() -> None:
    pares_1 = derivar_seeds(seed=1, n_janelas=5, classe="positiva")
    pares_2 = derivar_seeds(seed=1, n_janelas=5, classe="positiva")

    valores_1 = [(np.random.default_rng(sm).random(), np.random.default_rng(sb).random()) for sm, sb in pares_1]
    valores_2 = [(np.random.default_rng(sm).random(), np.random.default_rng(sb).random()) for sm, sb in pares_2]

    assert valores_1 == valores_2


def test_derivar_seeds_positiva_e_negativa_nao_colidem() -> None:
    pares_pos = derivar_seeds(seed=1, n_janelas=5, classe="positiva")
    pares_neg = derivar_seeds(seed=1, n_janelas=5, classe="negativa")

    valores_pos = [np.random.default_rng(sm).random() for sm, _ in pares_pos]
    valores_neg = [np.random.default_rng(sm).random() for sm, _ in pares_neg]

    assert valores_pos != valores_neg


def test_derivar_seeds_seed_modelo_distinta_entre_janelas() -> None:
    pares = derivar_seeds(seed=1, n_janelas=20, classe="positiva")

    valores = [np.random.default_rng(sm).random() for sm, _ in pares]

    assert len(set(valores)) == len(valores)


def test_derivar_seeds_classe_invalida_levanta_erro() -> None:
    with pytest.raises(ValueError):
        derivar_seeds(seed=1, n_janelas=5, classe="invalida")
