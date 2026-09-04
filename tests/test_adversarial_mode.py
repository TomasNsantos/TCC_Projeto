"""Testes do gerador de topo da classe positiva (modo adversarial)."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

import src.generator.adversarial_mode.cenario as cenario_module
from src.generator.layer1_abm import ElectionModel
from src.generator.adversarial_mode import gerar_cenario_adversarial


def test_contrato_nao_ativa_fonte_b_vazia_e_copula_nao_chamada(monkeypatch: pytest.MonkeyPatch) -> None:
    """gerar_fonte_b nao deve ser chamada quando o contrato nao ativa -- nao
    e so o resultado que precisa ficar vazio, a chamada em si nao pode
    acontecer (nao desperdicar RNG numa copula sem dado de entrada)."""
    mock_gerar_fonte_b = Mock(side_effect=AssertionError("gerar_fonte_b nao deveria ser chamada"))
    monkeypatch.setattr(cenario_module, "gerar_fonte_b", mock_gerar_fonte_b)

    modelo = ElectionModel(n_agentes=200, prop_racional=1.0, recompensa=0.0, resultado_alvo=0.9, seed=1)

    cenario = gerar_cenario_adversarial(modelo, tau_kendall=0.5)

    assert cenario.contrato_ativado is False
    assert cenario.fonte_b.size == 0
    mock_gerar_fonte_b.assert_not_called()


def test_contrato_ativa_formato_e_tamanho_compativel_com_beta() -> None:
    modelo = ElectionModel(
        n_agentes=200,
        prop_racional=1.0,
        recompensa=10.0,
        threshold_range=(0.0, 0.5),
        resultado_alvo=0.1,
        beta=3,
        seed=1,
    )

    cenario = gerar_cenario_adversarial(modelo, tau_kendall=0.6, random_state_fonte_b=2)

    assert cenario.contrato_ativado is True
    assert list(cenario.fonte_a.columns) == ["timestep", "n_eventos", "volume"]

    n_agentes_pagos = sum(1 for a in modelo.agents if a.aderiu)
    assert len(modelo.eventos_desembolso) == n_agentes_pagos * 3
    assert cenario.fonte_b.size == len(modelo.eventos_desembolso)


def test_modelo_ja_executado_levanta_erro() -> None:
    modelo = ElectionModel(n_agentes=50, resultado_alvo=0.0, seed=1)
    modelo.run()

    with pytest.raises(ValueError):
        gerar_cenario_adversarial(modelo, tau_kendall=0.5)


def test_reprodutibilidade_com_mesma_seed() -> None:
    kwargs = {
        "n_agentes": 200,
        "prop_racional": 1.0,
        "recompensa": 10.0,
        "threshold_range": (0.0, 0.5),
        "resultado_alvo": 0.1,
        "seed": 7,
    }
    modelo_1 = ElectionModel(**kwargs)
    modelo_2 = ElectionModel(**kwargs)

    cenario_1 = gerar_cenario_adversarial(modelo_1, tau_kendall=0.6, random_state_fonte_b=3)
    cenario_2 = gerar_cenario_adversarial(modelo_2, tau_kendall=0.6, random_state_fonte_b=3)

    assert cenario_1.contrato_ativado == cenario_2.contrato_ativado
    assert cenario_1.fonte_a.equals(cenario_2.fonte_a)
    assert np.array_equal(cenario_1.fonte_b, cenario_2.fonte_b)
    assert cenario_1.resultado_por_secao.equals(cenario_2.resultado_por_secao)
    assert cenario_1.resultado_por_municipio.equals(cenario_2.resultado_por_municipio)
    assert cenario_1.resultado_por_estado.equals(cenario_2.resultado_por_estado)


_KWARGS_PI = {
    "n_agentes": 200,
    "prop_racional": 1.0,
    "recompensa": 10.0,
    "threshold_range": (0.0, 0.5),
    "resultado_alvo": 0.1,
    "beta": 3,
    "seed": 7,
}


def test_pi_zero_reproduz_comportamento_anterior_independente_de_random_state_pi() -> None:
    modelo_1 = ElectionModel(pi=0.0, **_KWARGS_PI)
    modelo_2 = ElectionModel(pi=0.0, **_KWARGS_PI)

    cenario_sem_seed_pi = gerar_cenario_adversarial(modelo_1, tau_kendall=0.6, random_state_fonte_b=3)
    cenario_com_seed_pi = gerar_cenario_adversarial(
        modelo_2, tau_kendall=0.6, random_state_fonte_b=3, random_state_pi=999
    )

    assert cenario_sem_seed_pi.fonte_a.equals(cenario_com_seed_pi.fonte_a)
    assert np.array_equal(cenario_sem_seed_pi.fonte_b, cenario_com_seed_pi.fonte_b)


def test_random_state_pi_muda_fonte_a_mas_nunca_fonte_b() -> None:
    """Teste central desta tarefa: pi vaza para fonte_a (esperado) mas NUNCA
    para fonte_b, mesmo variando random_state_pi entre chamadas -- trava a
    garantia atraves do pipeline de cenario completo, nao so dentro de
    ElectionModel.fonte_a_eventos_fronteira isolada."""
    modelo_1 = ElectionModel(pi=0.9, **_KWARGS_PI)
    modelo_2 = ElectionModel(pi=0.9, **_KWARGS_PI)

    cenario_1 = gerar_cenario_adversarial(modelo_1, tau_kendall=0.6, random_state_fonte_b=3, random_state_pi=1)
    cenario_2 = gerar_cenario_adversarial(modelo_2, tau_kendall=0.6, random_state_fonte_b=3, random_state_pi=2)

    assert cenario_1.contrato_ativado is True
    assert cenario_2.contrato_ativado is True
    assert not cenario_1.fonte_a.equals(cenario_2.fonte_a)

    assert np.array_equal(cenario_1.fonte_b, cenario_2.fonte_b)


def test_pi_alto_reduz_linhas_de_fonte_a() -> None:
    modelo_pi_zero = ElectionModel(pi=0.0, **_KWARGS_PI)
    modelo_pi_alto = ElectionModel(pi=0.9, **_KWARGS_PI)

    cenario_pi_zero = gerar_cenario_adversarial(modelo_pi_zero, tau_kendall=0.6, random_state_fonte_b=3)
    cenario_pi_alto = gerar_cenario_adversarial(
        modelo_pi_alto, tau_kendall=0.6, random_state_fonte_b=3, random_state_pi=1
    )

    assert cenario_pi_alto.fonte_a["n_eventos"].sum() < cenario_pi_zero.fonte_a["n_eventos"].sum()
