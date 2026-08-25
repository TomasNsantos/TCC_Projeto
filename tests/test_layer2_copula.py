"""Testes da Camada 2 (cópula Clayton), isolada da Camada 1."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kendalltau

from src.generator.layer2_copula import aplicar_batching, gerar_fonte_b

JANELA = 100.0


def _fonte_a_continua(n: int, seed: int) -> np.ndarray:
    """Timestamps contínuos (sem empates) para não atenuar o tau_Kendall empírico."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, JANELA, size=n)


def test_fonte_b_vazia_quando_fonte_a_vazia() -> None:
    """Ausência de sinal espúrio: sem eventos de Fonte A, não há Fonte B gerada."""
    fonte_b = gerar_fonte_b(np.array([]), janela=JANELA, tau_kendall=0.6, random_state=1)

    assert fonte_b.size == 0


def test_modo_normal_produz_independencia() -> None:
    """Sanity Check 4: tau_Kendall(A, B) < 0.1 em modo normal (tau alvo = 0)."""
    fonte_a = _fonte_a_continua(2000, seed=1)

    fonte_b = gerar_fonte_b(fonte_a, janela=JANELA, tau_kendall=0.0, random_state=2)

    tau_empirico, _ = kendalltau(fonte_a, fonte_b)
    assert abs(tau_empirico) < 0.1


def test_modo_adversarial_recupera_tau_alvo() -> None:
    """A conversão theta = 2*tau/(1-tau) deve reproduzir o tau_Kendall alvo."""
    fonte_a = _fonte_a_continua(5000, seed=1)
    tau_alvo = 0.5

    fonte_b = gerar_fonte_b(fonte_a, janela=JANELA, tau_kendall=tau_alvo, random_state=3)

    tau_empirico, _ = kendalltau(fonte_a, fonte_b)
    assert tau_empirico == pytest.approx(tau_alvo, abs=0.05)


def test_dependencia_positiva_cresce_com_tau_alvo() -> None:
    fonte_a = _fonte_a_continua(3000, seed=1)

    tau_baixo, _ = kendalltau(fonte_a, gerar_fonte_b(fonte_a, JANELA, 0.2, random_state=4))
    tau_alto, _ = kendalltau(fonte_a, gerar_fonte_b(fonte_a, JANELA, 0.8, random_state=4))

    assert tau_baixo < tau_alto


def test_tau_fora_do_intervalo_valido_levanta_erro() -> None:
    fonte_a = _fonte_a_continua(10, seed=1)

    with pytest.raises(ValueError):
        gerar_fonte_b(fonte_a, JANELA, tau_kendall=1.0, random_state=1)

    with pytest.raises(ValueError):
        gerar_fonte_b(fonte_a, JANELA, tau_kendall=-0.1, random_state=1)


def test_fonte_b_respeita_janela_de_observacao() -> None:
    fonte_a = _fonte_a_continua(1000, seed=1)

    fonte_b = gerar_fonte_b(fonte_a, JANELA, tau_kendall=0.6, random_state=5)

    assert fonte_b.min() >= 0
    assert fonte_b.max() <= JANELA


def test_batching_fragmenta_em_beta_sub_eventos_na_janela_delta_t_beta() -> None:
    timestamps = np.array([10.0, 20.0, 30.0])
    delta_t = 10.0
    beta = 5

    fragmentados = aplicar_batching(timestamps, delta_t=delta_t, beta=beta, random_state=1)

    assert fragmentados.size == timestamps.size * beta
    for original in timestamps:
        janela = fragmentados[(fragmentados >= original) & (fragmentados < original + delta_t / beta)]
        assert janela.size == beta


def test_batching_com_beta_1_nao_altera_timestamps() -> None:
    timestamps = np.array([1.0, 2.0, 3.0])

    fragmentados = aplicar_batching(timestamps, delta_t=10.0, beta=1, random_state=1)

    assert np.array_equal(fragmentados, timestamps)
