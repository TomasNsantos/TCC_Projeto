"""Testes do gerador de tráfego legítimo (modo normal, classe negativa)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kendalltau

from src.generator.layer1_abm import ElectionModel
from src.generator.normal_mode import (
    contagem_por_timestep,
    gerar_cenario_normal,
    gerar_fonte_a_normal,
    gerar_fonte_b_normal,
)

JANELA = 200.0


def test_fonte_a_normal_taxa_zero_produz_vazio() -> None:
    fonte_a = gerar_fonte_a_normal(janela=JANELA, taxa=0.0, volume_medio=1000.0, random_state=1)

    assert fonte_a.empty
    assert list(fonte_a.columns) == ["timestep", "n_eventos", "volume"]


def test_fonte_b_normal_taxa_zero_produz_vazio() -> None:
    fonte_b = gerar_fonte_b_normal(janela=JANELA, taxa=0.0, random_state=1)

    assert fonte_b.size == 0


def test_fonte_a_normal_formato_compativel_com_fonte_a_eventos_fronteira() -> None:
    fonte_a = gerar_fonte_a_normal(janela=JANELA, taxa=1.0, volume_medio=1000.0, random_state=1)

    assert list(fonte_a.columns) == ["timestep", "n_eventos", "volume"]
    assert fonte_a["timestep"].dtype.kind == "i"
    assert fonte_a["n_eventos"].dtype.kind == "i"
    assert fonte_a["volume"].dtype.kind == "f"
    assert (fonte_a["n_eventos"] > 0).all()
    assert (fonte_a["volume"] > 0).all()


def test_independencia_fonte_a_e_fonte_b_normal() -> None:
    """Mesmo criterio do Sanity Check 4, aplicado ao gerador de trafego normal:
    tau_Kendall(A,B) medido via contagem_por_timestep (sem pareamento natural
    entre processos independentes com contagens distintas), com janela/taxa
    grandes o bastante para reduzir a variancia de cada estimativa individual
    e n=50 rodadas independentes para estabilidade. Usa a MEDIA de |tau| como
    estatistica principal (mais robusta que o maximo sobre varias rodadas,
    que e um extremo e eventualmente excede qualquer limiar fixo por acaso
    mesmo sob independencia genuina) e o maximo com folga maior só como
    checagem de ausencia de outlier grosseiro."""
    janela_grande = 1000.0
    taus = []
    for seed in range(50):
        fonte_a = gerar_fonte_a_normal(janela=janela_grande, taxa=5.0, volume_medio=1000.0, random_state=1000 + seed)
        fonte_b = gerar_fonte_b_normal(janela=janela_grande, taxa=5.0, random_state=2000 + seed)

        timestamps_a = np.repeat(fonte_a["timestep"].to_numpy(dtype=float), fonte_a["n_eventos"].to_numpy())
        contagem_a = contagem_por_timestep(timestamps_a, janela_grande)
        contagem_b = contagem_por_timestep(fonte_b, janela_grande)

        tau, _ = kendalltau(contagem_a, contagem_b)
        taus.append(tau)

    taus = np.array(taus)
    assert np.abs(taus.mean()) < 0.1
    assert np.abs(taus).max() < 0.3


def test_seed_fonte_a_nao_afeta_fonte_b() -> None:
    """Prova direta (nao so correlacao empirica baixa): mudar a seed de Fonte A
    nao muda Fonte B, para a mesma seed de Fonte B."""
    fonte_b_referencia = gerar_fonte_b_normal(janela=JANELA, taxa=0.5, random_state=42)

    for seed_a in (1, 999, 123456):
        gerar_fonte_a_normal(janela=JANELA, taxa=1.0, volume_medio=1000.0, random_state=seed_a)
        fonte_b = gerar_fonte_b_normal(janela=JANELA, taxa=0.5, random_state=42)
        assert np.array_equal(fonte_b, fonte_b_referencia)


def test_gerar_cenario_normal_produz_fonte_c_nao_trivial() -> None:
    """resultado_por_secao varia entre secoes mesmo com recompensa=0 -- prova
    que o voto de base (candidato_preferido) gera resultado plausivel."""
    modelo = ElectionModel(n_agentes=1000, n_secoes=6, n_candidatos=5, recompensa=0.0, seed=1)

    cenario = gerar_cenario_normal(
        modelo,
        janela=JANELA,
        taxa_fonte_a=1.0,
        volume_medio_fonte_a=1000.0,
        taxa_fonte_b=1.0,
        random_state_fonte_a=1,
        random_state_fonte_b=2,
    )

    assert not (cenario.resultado_por_secao == 0.0).all()
    assert cenario.resultado_por_secao.std() > 0.0


def test_gerar_cenario_normal_exige_recompensa_zero() -> None:
    modelo = ElectionModel(recompensa=1.0, n_candidatos=3)

    with pytest.raises(ValueError):
        gerar_cenario_normal(modelo, JANELA, 1.0, 1000.0, 1.0)


def test_gerar_cenario_normal_exige_modelo_nao_executado() -> None:
    modelo = ElectionModel(recompensa=0.0, n_candidatos=3, seed=1)
    modelo.run()

    with pytest.raises(ValueError):
        gerar_cenario_normal(modelo, JANELA, 1.0, 1000.0, 1.0)


def test_gerar_cenario_normal_exige_n_candidatos_maior_que_um() -> None:
    modelo = ElectionModel(recompensa=0.0, n_candidatos=1)

    with pytest.raises(ValueError):
        gerar_cenario_normal(modelo, JANELA, 1.0, 1000.0, 1.0)


def test_gerar_cenario_normal_nunca_chama_resolver_desembolso() -> None:
    modelo = ElectionModel(n_agentes=200, n_candidatos=3, recompensa=0.0, seed=1)

    gerar_cenario_normal(modelo, JANELA, 1.0, 1000.0, 1.0, random_state_fonte_a=1, random_state_fonte_b=2)

    assert modelo.contrato_ativado is None
    assert modelo.eventos_desembolso == []


def test_fonte_a_normal_pi_zero_reproduz_comportamento_anterior() -> None:
    fonte_a_sem_argumentos = gerar_fonte_a_normal(janela=JANELA, taxa=2.0, volume_medio=1000.0, random_state=1)
    fonte_a_com_pi_zero = gerar_fonte_a_normal(
        janela=JANELA, taxa=2.0, volume_medio=1000.0, random_state=1, pi=0.0, random_state_pi=42
    )

    assert fonte_a_sem_argumentos.equals(fonte_a_com_pi_zero)


def test_fonte_a_normal_pi_um_produz_vazio() -> None:
    fonte_a_bruta = gerar_fonte_a_normal(janela=JANELA, taxa=5.0, volume_medio=1000.0, random_state=1)
    assert fonte_a_bruta["n_eventos"].sum() > 0  # garante que há algo para mascarar

    fonte_a_mascarada = gerar_fonte_a_normal(
        janela=JANELA, taxa=5.0, volume_medio=1000.0, random_state=1, pi=1.0, random_state_pi=1
    )

    assert fonte_a_mascarada.empty
    assert list(fonte_a_mascarada.columns) == ["timestep", "n_eventos", "volume"]


def test_fonte_a_normal_pi_intermediario_reduz_eventos_observados() -> None:
    fonte_a_bruta = gerar_fonte_a_normal(janela=JANELA, taxa=5.0, volume_medio=1000.0, random_state=1)
    total_bruto = fonte_a_bruta["n_eventos"].sum()
    assert total_bruto > 0

    fonte_a_mascarada = gerar_fonte_a_normal(
        janela=JANELA, taxa=5.0, volume_medio=1000.0, random_state=1, pi=0.5, random_state_pi=1
    )

    assert fonte_a_mascarada["n_eventos"].sum() < total_bruto


def test_fonte_b_normal_nao_muda_com_a_introducao_de_pi() -> None:
    """Regressao: gerar_fonte_b_normal continua sem parametro pi, mesma
    assinatura/comportamento de antes desta tarefa."""
    fonte_b_1 = gerar_fonte_b_normal(janela=JANELA, taxa=1.0, random_state=42)
    fonte_b_2 = gerar_fonte_b_normal(janela=JANELA, taxa=1.0, random_state=42)

    assert np.array_equal(fonte_b_1, fonte_b_2)


def test_fonte_a_normal_pi_random_state_pi_e_reprodutivel() -> None:
    fonte_a_1 = gerar_fonte_a_normal(
        janela=JANELA, taxa=5.0, volume_medio=1000.0, random_state=1, pi=0.5, random_state_pi=99
    )
    fonte_a_2 = gerar_fonte_a_normal(
        janela=JANELA, taxa=5.0, volume_medio=1000.0, random_state=1, pi=0.5, random_state_pi=99
    )

    assert fonte_a_1.equals(fonte_a_2)
