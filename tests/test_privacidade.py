"""Testes de src/generator/privacidade.py (mascara_sobrevivencia_pi)."""

from __future__ import annotations

import numpy as np
import pytest

from src.generator.privacidade import mascara_sobrevivencia_pi


def test_pi_zero_retorna_tudo_true_sem_consumir_random_state() -> None:
    resultado_sem_seed = mascara_sobrevivencia_pi(50, pi=0.0, random_state=None)
    resultado_com_seed = mascara_sobrevivencia_pi(50, pi=0.0, random_state=42)

    assert resultado_sem_seed.dtype == bool
    assert resultado_sem_seed.all()
    # seeds diferentes (None vs 42) produzem o mesmo resultado -- prova de
    # que o RNG nunca foi consultado quando pi=0.0.
    assert np.array_equal(resultado_sem_seed, resultado_com_seed)


def test_pi_um_retorna_tudo_false() -> None:
    resultado = mascara_sobrevivencia_pi(50, pi=1.0, random_state=1)
    assert resultado.dtype == bool
    assert not resultado.any()


def test_n_eventos_zero_retorna_array_vazio_sem_erro() -> None:
    resultado = mascara_sobrevivencia_pi(0, pi=0.5, random_state=1)
    assert resultado.dtype == bool
    assert resultado.shape == (0,)


def test_fracao_de_sobreviventes_proxima_de_um_menos_pi() -> None:
    resultado = mascara_sobrevivencia_pi(10_000, pi=0.3, random_state=7)
    fracao_true = resultado.mean()
    assert 0.68 <= fracao_true <= 0.72


@pytest.mark.parametrize("pi_invalido", [-0.1, 1.1])
def test_pi_fora_do_intervalo_levanta_erro(pi_invalido: float) -> None:
    with pytest.raises(ValueError, match="pi"):
        mascara_sobrevivencia_pi(10, pi=pi_invalido, random_state=1)


def test_reprodutibilidade_mesma_seed_produz_mesmo_array() -> None:
    resultado_1 = mascara_sobrevivencia_pi(200, pi=0.4, random_state=42)
    resultado_2 = mascara_sobrevivencia_pi(200, pi=0.4, random_state=42)
    assert np.array_equal(resultado_1, resultado_2)
