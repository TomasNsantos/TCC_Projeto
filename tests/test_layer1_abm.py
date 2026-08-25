"""Testes da Camada 1 (ABM), isolada da Camada 2."""

from __future__ import annotations

import numpy as np
import pytest

from src.generator.layer1_abm import ElectionModel


def test_parametros_populacionais_sao_configuraveis() -> None:
    model = ElectionModel(n_agentes=37, alpha_beta=(5.0, 1.0), prop_racional=0.5, seed=1)

    assert len(model.agents) == 37
    assert model.n_agentes == 37


def test_reprodutibilidade_com_mesma_seed() -> None:
    kwargs = dict(n_agentes=100, recompensa=5.0, resultado_alvo=0.0, delta_t=20.0, rho=0.5, seed=123)
    m1 = ElectionModel(**kwargs)
    m2 = ElectionModel(**kwargs)
    m1.run()
    m2.run()
    m1.resolver_desembolso()
    m2.resolver_desembolso()

    assert m1.eventos_adesao == m2.eventos_adesao
    assert m1.eventos_desembolso == m2.eventos_desembolso
    assert m1.fonte_c_resultado().equals(m2.fonte_c_resultado())


def test_agente_racional_nao_adere_sem_incentivo() -> None:
    """Sanity Check 1 (nível ABM): sem incentivo, agentes racionais não geram adesão nem desembolso."""
    model = ElectionModel(n_agentes=200, prop_racional=1.0, recompensa=0.0, resultado_alvo=0.1, seed=1)
    model.run()
    model.resolver_desembolso()

    assert model.fonte_c_resultado_agregado() == 0.0
    assert (model.fonte_c_resultado() == 0.0).all()
    assert model.eventos_adesao == []
    assert model.contrato_ativado is False
    assert model.eventos_desembolso == []


def test_incentivo_suficiente_ativa_contrato_e_gera_desembolso() -> None:
    model = ElectionModel(
        n_agentes=200,
        prop_racional=1.0,
        recompensa=10.0,
        threshold_range=(0.0, 0.5),
        resultado_alvo=0.1,
        seed=1,
    )
    model.run()
    model.resolver_desembolso()

    assert model.fonte_c_resultado_agregado() > 0.0
    assert model.contrato_ativado is True
    assert len(model.eventos_desembolso) > 0


def test_resultado_alvo_zero_ativa_contrato_mesmo_sem_adesao() -> None:
    """Caso de fronteira documentado: alvo=0.0 sempre ativa (0.0 >= 0.0), mesmo com 0 agentes pagos."""
    model = ElectionModel(n_agentes=100, prop_racional=1.0, recompensa=0.0, resultado_alvo=0.0, seed=1)
    model.run()
    model.resolver_desembolso()

    assert model.contrato_ativado is True
    assert model.eventos_desembolso == []


def test_resolver_desembolso_duas_vezes_levanta_erro() -> None:
    model = ElectionModel(n_agentes=50, resultado_alvo=0.0, seed=1)
    model.run()
    model.resolver_desembolso()

    with pytest.raises(RuntimeError):
        model.resolver_desembolso()


def test_rho_invalido_levanta_erro() -> None:
    with pytest.raises(ValueError):
        ElectionModel(rho=1.5)
    with pytest.raises(ValueError):
        ElectionModel(rho=-0.1)


def test_fonte_a_eventos_fronteira_agrega_desembolso_por_timestep() -> None:
    model = ElectionModel(
        n_agentes=200,
        recompensa=5.0,
        threshold_range=(0.0, 0.5),
        resultado_alvo=0.0,
        delta_t=20.0,
        seed=2,
    )
    model.run()
    model.resolver_desembolso()

    fonte_a = model.fonte_a_eventos_fronteira()

    assert list(fonte_a.columns) == ["timestep", "n_eventos", "volume"]
    assert fonte_a["n_eventos"].sum() == len(model.eventos_desembolso)
    assert (fonte_a["n_eventos"] > 0).all()
    assert (fonte_a["volume"] == fonte_a["n_eventos"] * model.recompensa).all()


def test_volume_nao_e_apenas_contagem_de_eventos() -> None:
    """Item 2: volume (monetário) deve divergir numericamente de n_eventos quando recompensa != 1.0."""
    model = ElectionModel(
        n_agentes=200,
        recompensa=3.0,
        threshold_range=(0.0, 0.5),
        resultado_alvo=0.0,
        delta_t=20.0,
        seed=2,
    )
    model.run()
    model.resolver_desembolso()

    fonte_a = model.fonte_a_eventos_fronteira()

    assert not fonte_a.empty
    assert (fonte_a["volume"] != fonte_a["n_eventos"]).any()
    assert np.allclose(fonte_a["volume"], fonte_a["n_eventos"] * 3.0)


def test_fonte_c_granularidade_por_secao_pondera_para_o_agregado() -> None:
    """Item 3: a média ponderada (por agentes-por-seção) das frações por seção reconcilia com o agregado."""
    model = ElectionModel(
        n_agentes=300,
        n_secoes=4,
        recompensa=10.0,
        threshold_range=(0.0, 0.3),
        seed=3,
    )
    model.run()

    por_secao = model.fonte_c_resultado()
    agregado = model.fonte_c_resultado_agregado()

    contagens = np.bincount([a.secao for a in model.agents], minlength=model.n_secoes)
    media_ponderada = (por_secao.to_numpy() * contagens).sum() / contagens.sum()

    assert media_ponderada == pytest.approx(agregado)


def test_fonte_c_secao_vazia_retorna_zero_nao_nan() -> None:
    model = ElectionModel(n_agentes=5, n_secoes=50, seed=1)
    model.run()

    resultado = model.fonte_c_resultado()

    assert len(resultado) == 50
    assert not resultado.isna().any()


def test_prob_conformidade_invalido_levanta_erro() -> None:
    with pytest.raises(ValueError):
        ElectionModel(prob_conformidade=1.5)
    with pytest.raises(ValueError):
        ElectionModel(prob_conformidade=-0.1)


def test_prob_conformidade_menor_que_um_reduz_resultado_conforme() -> None:
    model = ElectionModel(
        n_agentes=300,
        prop_racional=1.0,
        recompensa=10.0,
        threshold_range=(0.0, 0.3),
        prob_conformidade=0.5,
        seed=7,
    )
    model.run()

    assert model.fonte_c_resultado_agregado() <= model.fracao_adesao()


def test_prob_conformidade_um_reproduz_comportamento_anterior() -> None:
    """Retrocompatibilidade: com prob_conformidade=1.0 (default), conforme == bruta."""
    model = ElectionModel(
        n_agentes=300,
        prop_racional=1.0,
        recompensa=10.0,
        threshold_range=(0.0, 0.3),
        prob_conformidade=1.0,
        seed=7,
    )
    model.run()

    assert model.fonte_c_resultado_agregado() == model.fracao_adesao()


def test_prob_conformidade_zero_zera_resultado_conforme_sem_erro() -> None:
    """Caso de fronteira legitimo: ninguem que aderiu vota como prometido."""
    model = ElectionModel(
        n_agentes=100,
        prop_racional=1.0,
        recompensa=10.0,
        threshold_range=(0.0, 0.3),
        prob_conformidade=0.0,
        seed=1,
    )
    model.run()

    assert model.fracao_adesao() > 0.0
    assert model.fonte_c_resultado_agregado() == 0.0


def test_desembolso_paga_agentes_aderidos_nao_conformes() -> None:
    """O oraculo verifica R agregado, nao conformidade individual (ballot secrecy)."""
    model = ElectionModel(
        n_agentes=200,
        prop_racional=1.0,
        recompensa=10.0,
        threshold_range=(0.0, 0.3),
        prob_conformidade=0.5,
        resultado_alvo=0.0,
        seed=3,
    )
    model.run()
    model.resolver_desembolso()

    agentes_pagos = {uid for _, uid in model.eventos_desembolso}
    agentes_aderidos = {a.unique_id for a in model.agents if a.aderiu}
    agentes_nao_conformes_pagos = {
        a.unique_id for a in model.agents if a.aderiu and not a.votou_conforme and a.unique_id in agentes_pagos
    }

    assert agentes_pagos == agentes_aderidos
    assert len(agentes_nao_conformes_pagos) > 0


def test_rho_alto_concentra_desembolso_mais_que_rho_baixo() -> None:
    """Item 4: rho maior produz clustering temporal mensuravelmente mais apertado."""
    kwargs = dict(
        n_agentes=300,
        prop_racional=1.0,
        recompensa=10.0,
        threshold_range=(0.0, 0.3),
        resultado_alvo=0.0,
        delta_t=50.0,
        seed=7,
    )
    m_rho0 = ElectionModel(rho=0.0, **kwargs)
    m_rho05 = ElectionModel(rho=0.5, **kwargs)
    m_rho1 = ElectionModel(rho=1.0, **kwargs)
    for m in (m_rho0, m_rho05, m_rho1):
        m.run()
        m.resolver_desembolso()

    std0 = np.std([t for t, _ in m_rho0.eventos_desembolso])
    std05 = np.std([t for t, _ in m_rho05.eventos_desembolso])
    std1 = np.std([t for t, _ in m_rho1.eventos_desembolso])

    assert std0 > std05 > std1

    sigma_esperado = ElectionModel._SIGMA_FRACAO_DELTA_T * kwargs["delta_t"]
    assert std1 == pytest.approx(sigma_esperado, rel=0.5)


def test_n_candidatos_menor_que_um_levanta_erro() -> None:
    with pytest.raises(ValueError):
        ElectionModel(n_candidatos=0)


def test_candidato_alvo_fora_do_intervalo_levanta_erro() -> None:
    with pytest.raises(ValueError):
        ElectionModel(n_candidatos=3, candidato_alvo=3)
    with pytest.raises(ValueError):
        ElectionModel(n_candidatos=3, candidato_alvo=-1)


def test_n_candidatos_um_reproduz_fracao_conforme_de_antes() -> None:
    """Retrocompatibilidade: com n_candidatos=1 (default), o resultado ignora
    completamente candidato_preferido e reduz-se a aderiu&votou_conforme."""
    model = ElectionModel(
        n_agentes=300,
        prop_racional=1.0,
        recompensa=10.0,
        threshold_range=(0.0, 0.3),
        prob_conformidade=0.7,
        seed=7,
    )
    model.run()

    esperado = sum(1 for a in model.agents if a.aderiu and a.votou_conforme) / model.n_agentes
    assert model.fonte_c_resultado_agregado() == pytest.approx(esperado)


def test_n_candidatos_maior_que_um_conta_voto_de_base_sem_incentivo() -> None:
    """Sem qualquer adesao (recompensa=0), o resultado do candidato-alvo nao
    e zero quando ha mais de um candidato: agentes que nunca aderiram ainda
    'votam' via candidato_preferido, aproximando 1/n_candidatos."""
    n_candidatos = 6
    model = ElectionModel(
        n_agentes=3000,
        prop_racional=1.0,
        recompensa=0.0,
        n_candidatos=n_candidatos,
        seed=1,
    )
    model.run()

    assert model.fonte_c_resultado_agregado() == pytest.approx(1 / n_candidatos, abs=0.03)
    # contraste explicito com n_candidatos=1, que da exatamente 0.0 (ja coberto
    # por test_agente_racional_nao_adere_sem_incentivo, repetido aqui para
    # deixar o contraste visivel lado a lado)
    model_um_candidato = ElectionModel(n_agentes=3000, prop_racional=1.0, recompensa=0.0, seed=1)
    model_um_candidato.run()
    assert model_um_candidato.fonte_c_resultado_agregado() == 0.0


def test_granularidade_trivial_hierarquia_bate_com_agregado() -> None:
    """secoes_por_municipio=n_secoes, municipios_por_estado=1 (os defaults)
    colapsam municipio e estado num unico grupo, igual ao agregado do pool."""
    model = ElectionModel(
        n_agentes=300,
        n_secoes=5,
        n_candidatos=4,
        prop_racional=1.0,
        recompensa=5.0,
        threshold_range=(0.0, 0.5),
        seed=2,
    )
    model.run()

    agregado = model.fonte_c_resultado_agregado()
    assert model.n_municipios == 1
    assert model.n_estados == 1
    assert model.resultado_eleitoral_por_municipio().iloc[0] == pytest.approx(agregado)
    assert model.resultado_eleitoral_por_estado().iloc[0] == pytest.approx(agregado)


def test_hierarquia_nao_trivial_pondera_corretamente() -> None:
    """12 secoes / 4 por municipio -> 3 municipios / 3 por estado -> 1 estado.
    A media ponderada por seção deve reconciliar com município e com o agregado."""
    model = ElectionModel(
        n_agentes=600,
        n_secoes=12,
        secoes_por_municipio=4,
        municipios_por_estado=3,
        n_candidatos=3,
        prop_racional=1.0,
        recompensa=5.0,
        threshold_range=(0.0, 0.5),
        seed=5,
    )
    model.run()

    assert model.n_municipios == 3
    assert model.n_estados == 1

    por_secao = model.resultado_eleitoral_por_secao()
    contagem_secao = np.bincount([a.secao for a in model.agents], minlength=model.n_secoes)
    media_ponderada_secao = (por_secao.to_numpy() * contagem_secao).sum() / contagem_secao.sum()
    assert media_ponderada_secao == pytest.approx(model.fonte_c_resultado_agregado())

    por_municipio = model.resultado_eleitoral_por_municipio()
    contagem_municipio = np.bincount([a.municipio for a in model.agents], minlength=model.n_municipios)
    media_ponderada_municipio = (por_municipio.to_numpy() * contagem_municipio).sum() / contagem_municipio.sum()
    assert media_ponderada_municipio == pytest.approx(model.fonte_c_resultado_agregado())

    assert model.resultado_eleitoral_por_estado().iloc[0] == pytest.approx(model.fonte_c_resultado_agregado())


def test_granularidade_invalida_levanta_erro() -> None:
    with pytest.raises(ValueError):
        ElectionModel(granularidade="pais", unidade_alvo=0)


def test_unidade_alvo_obrigatoria_quando_granularidade_nao_pool() -> None:
    with pytest.raises(ValueError):
        ElectionModel(granularidade="secao")
    with pytest.raises(ValueError):
        ElectionModel(granularidade="municipio")
    with pytest.raises(ValueError):
        ElectionModel(granularidade="estado")


def test_unidade_alvo_fora_do_intervalo_levanta_erro() -> None:
    with pytest.raises(ValueError):
        ElectionModel(n_secoes=5, granularidade="secao", unidade_alvo=5)
    with pytest.raises(ValueError):
        ElectionModel(n_secoes=5, granularidade="secao", unidade_alvo=-1)


def test_resolver_desembolso_granularidade_pool_e_retrocompativel() -> None:
    model = ElectionModel(
        n_agentes=200,
        prop_racional=1.0,
        recompensa=10.0,
        threshold_range=(0.0, 0.5),
        resultado_alvo=0.1,
        seed=1,
    )
    model.run()
    model.resolver_desembolso()

    assert model.contrato_ativado is True


def test_resolver_desembolso_respeita_granularidade_e_unidade_alvo() -> None:
    """Mesma configuracao, exceto granularidade: pool ativa, secao=1 (isolada,
    com resultado bem abaixo do agregado) nao ativa -- prova que a
    granularidade e de fato respeitada, nao so validada."""
    kwargs = dict(
        n_agentes=150,
        n_secoes=5,
        alpha_beta=(0.5, 0.5),
        prop_racional=1.0,
        recompensa=3.0,
        threshold_range=(0.0, 3.0),
        resultado_alvo=0.4,
        seed=1,
    )

    modelo_pool = ElectionModel(granularidade="pool", **kwargs)
    modelo_pool.run()
    modelo_pool.resolver_desembolso()

    modelo_secao = ElectionModel(granularidade="secao", unidade_alvo=1, **kwargs)
    modelo_secao.run()
    modelo_secao.resolver_desembolso()

    assert modelo_pool.contrato_ativado is True
    assert modelo_secao.contrato_ativado is False


def test_beta_um_reproduz_comportamento_anterior() -> None:
    """Retrocompatibilidade: beta=1 (default) e aplicar_batching no-op exato."""
    kwargs = dict(
        n_agentes=200,
        prop_racional=1.0,
        recompensa=10.0,
        threshold_range=(0.0, 0.3),
        resultado_alvo=0.0,
        delta_t=50.0,
        rho=0.5,
        seed=7,
    )
    modelo_default = ElectionModel(**kwargs)
    modelo_beta1 = ElectionModel(beta=1, **kwargs)
    for m in (modelo_default, modelo_beta1):
        m.run()
        m.resolver_desembolso()

    assert modelo_default.eventos_desembolso == modelo_beta1.eventos_desembolso

    agentes_pagos = [a.unique_id for a in modelo_beta1.agents if a.aderiu]
    assert len(modelo_beta1.eventos_desembolso) == len(agentes_pagos)

    fonte_a = modelo_beta1.fonte_a_eventos_fronteira()
    assert (fonte_a["volume"] == fonte_a["n_eventos"] * modelo_beta1.recompensa).all()


def test_beta_maior_que_um_fragmenta_eventos_por_agente() -> None:
    modelo = ElectionModel(
        n_agentes=200,
        prop_racional=1.0,
        recompensa=10.0,
        threshold_range=(0.0, 0.3),
        resultado_alvo=0.0,
        delta_t=50.0,
        beta=5,
        seed=7,
    )
    modelo.run()
    modelo.resolver_desembolso()

    agentes_pagos = [a.unique_id for a in modelo.agents if a.aderiu]
    assert len(modelo.eventos_desembolso) == len(agentes_pagos) * 5

    timestamps_por_agente: dict[int, list[float]] = {}
    for timestamp, unique_id in modelo.eventos_desembolso:
        timestamps_por_agente.setdefault(unique_id, []).append(timestamp)

    janela_fragmento = modelo.delta_t / modelo.beta
    for timestamps in timestamps_por_agente.values():
        assert len(timestamps) == 5
        assert max(timestamps) - min(timestamps) < janela_fragmento


def test_beta_nao_muda_volume_monetario_agregado() -> None:
    """A fragmentacao muda so a distribuicao temporal, nunca o total pago."""
    kwargs = dict(
        n_agentes=200,
        prop_racional=1.0,
        recompensa=10.0,
        threshold_range=(0.0, 0.3),
        resultado_alvo=0.0,
        delta_t=50.0,
        seed=7,
    )
    modelo_beta1 = ElectionModel(beta=1, **kwargs)
    modelo_beta5 = ElectionModel(beta=5, **kwargs)
    for m in (modelo_beta1, modelo_beta5):
        m.run()
        m.resolver_desembolso()

    volume_beta1 = modelo_beta1.fonte_a_eventos_fronteira()["volume"].sum()
    volume_beta5 = modelo_beta5.fonte_a_eventos_fronteira()["volume"].sum()

    assert volume_beta1 == pytest.approx(volume_beta5)


def test_beta_invalido_levanta_erro() -> None:
    with pytest.raises(ValueError):
        ElectionModel(beta=0)
    with pytest.raises(ValueError):
        ElectionModel(beta=-1)
