"""Modelo da Camada 1 (ABM): simulação da população de eleitores.

A simulação roda em duas fases temporais distintas:

Fase 1 (campanha, ``run()``): agentes decidem adesão (enrollment) ao
incentivo ao longo de ``n_steps``. Adesão é uma decisão pré-resultado — não é,
por si só, um observável de Fonte A.

Fase 2 (resolução, ``resolver_desembolso()``): chamada explicitamente após
``run()``. Verifica se o resultado agregado bate com o alvo do adversário
(proxy de "oráculo confirma R") e, se sim, gera um evento de desembolso por
agente aderido, com timestamp na janela pós-resultado ``[0, delta_t]``. A
Fonte A observável (``fonte_a_eventos_fronteira``) vem desses eventos de
desembolso — não dos eventos de adesão — porque o adversary model só libera
pagamento após confirmação do oráculo (`if oracle.result == R: transfer(...)`),
e o plano descreve a Fonte A como tendo "pico concentrado em janela
pós-resultado" (PLANO §5.1.2/§5.2.2).

A Fonte C (resultado eleitoral por seção) é gerada diretamente pela simulação
de agentes e não passa pelo acoplamento via cópula (Camada 2).
"""

from __future__ import annotations

import math

import mesa
import numpy as np
import pandas as pd

from src.generator.layer1_abm.agent import VoterAgent
from src.generator.layer2_copula import aplicar_batching

_GRANULARIDADES_VALIDAS = ("pool", "secao", "municipio", "estado")


class ElectionModel(mesa.Model):
    """Simulação de uma população de eleitores sob um esquema de incentivo.

    Parameters
    ----------
    n_agentes : int
        Tamanho da população participante (parâmetro populacional). Valores
        finais pendentes de consenso com orientadores — ver CLAUDE.md; o
        default aqui é apenas um valor de desenvolvimento.
    alpha_beta : tuple[float, float]
        Parâmetros (α, β) da distribuição Beta da propensão dos agentes
        (parâmetro populacional, não fixado definitivamente).
    prop_racional : float
        Fração de agentes que decide por utilidade esperada determinística;
        o restante decide de forma ruidosa (parâmetro populacional). Não
        confundir com ρ (grau de coordenação, parâmetro do design fatorial).
    recompensa : float
        Magnitude do incentivo oferecido pelo contrato, constante durante a
        simulação. Proxy da intensidade adversarial λ neste esqueleto.
    threshold_range : tuple[float, float]
        Intervalo (min, max) de onde ``utility_threshold`` é amostrado
        uniformemente por agente.
    n_steps : int
        Número de passos de decisão simulados na campanha (Fase 1, janela de
        adesão).
    n_secoes : int
        Número de seções eleitorais simuladas (granularidade g=seção da
        Fonte C).
    secoes_por_municipio : int | None
        Quantas seções compõem cada município, para a granularidade
        g=município. ``None`` (default) usa ``n_secoes`` — ou seja, colapsa
        tudo num único município, reproduzindo o comportamento anterior à
        introdução da hierarquia. Atribuição determinística
        (``secao // secoes_por_municipio``), sem sorteio.
    municipios_por_estado : int | None
        Quantos municípios compõem cada estado, para a granularidade
        g=estado. ``None`` (default) usa o número de municípios resultante —
        ou seja, colapsa tudo num único estado. Atribuição determinística
        (``municipio // municipios_por_estado``), sem sorteio.
    n_candidatos : int
        Número de candidatos concorrendo (parâmetro populacional v0, mesma
        categoria de ``n_agentes``/``alpha_beta``/``prop_racional`` — não
        fixado definitivamente, ver CLAUDE.md). Default ``1`` é
        retrocompatibilidade estrita: com ``n_candidatos=1`` o voto de base
        (``VoterAgent.candidato_preferido``) nunca é consultado, e o
        resultado eleitoral reduz-se exatamente à fração de adesão conforme
        de antes desta funcionalidade — ver ``_voto_e_candidato_alvo``. Este
        gerador não implementa alocação de cadeiras (quociente eleitoral,
        D'Hondt, sobras partidárias) — decisão deliberada, complexidade
        desproporcional ao propósito do gerador; ``resultado_alvo`` é reusado
        como a fração de votos válidos que o candidato-alvo precisa atingir,
        o que já permite simular um quociente pequeno de sistema
        proporcional sem formalizar a regra de alocação.
    candidato_alvo : int
        Índice do candidato do adversário, em ``[0, n_candidatos)``.
    resultado_alvo : float
        Limiar de fração de votos válidos que o resultado do candidato-alvo,
        na granularidade e unidade escolhidas (ver ``granularidade``/
        ``unidade_alvo``), precisa atingir para o contrato ativar ("oráculo
        confirma que o resultado agregado R bate com o esperado pelo
        adversário"). Com ``n_candidatos=1`` isso reduz-se à fração de
        adesão conforme, como antes desta funcionalidade.
    granularidade : str
        Nível territorial que ``resolver_desembolso()`` verifica para
        decidir ativação: ``"pool"`` (default, todo ``n_agentes``,
        retrocompatível), ``"secao"``, ``"municipio"`` ou ``"estado"``.
        Suposição v0: o gerador mira uma única unidade territorial por vez
        (ver ``unidade_alvo``) — ataques simultâneos em múltiplas unidades
        ficam para trabalho futuro. Não afeta quem recebe desembolso: o
        pagamento continua sendo a todo agente aderido no pool inteiro,
        independentemente da unidade-alvo (ver ``resolver_desembolso``).
    unidade_alvo : int | None
        Índice da seção/município/estado específico verificado quando
        ``granularidade != "pool"``. Obrigatório nesse caso (``ValueError``
        se ``None``); ignorado quando ``granularidade == "pool"``.
    delta_t : float
        Duração da janela pós-resultado em que o desembolso ocorre (mesma
        grandeza do Δt do design fatorial; em unidades de timestep por
        enquanto).
    rho : float
        Grau de coordenação (ρ), em ``[0, 1]``. Controla a concentração
        temporal dos eventos de desembolso (Fase 2) dentro de
        ``[0, delta_t]`` — não afeta a decisão de adesão (Fase 1). Ver
        ``resolver_desembolso`` para o mecanismo.
    beta : int
        Fragmentação (β) do design fatorial, em ``[1, ∞)`` — não é
        parâmetro populacional, mesma categoria de ``delta_t``/``rho``.
        Cada evento de desembolso é fragmentado em β sub-eventos via
        ``layer2_copula.aplicar_batching``, reduzindo a amplitude do pico
        observável em Fonte A por fator ``1/β`` sem alterar o total
        monetário pago por agente (ver ``fonte_a_eventos_fronteira``).
        Default ``1`` é retrocompatibilidade estrita: ``aplicar_batching``
        é no-op exato em β≤1 (nem consome números aleatórios), então nada
        muda em relação ao comportamento anterior a este parâmetro.
    prob_conformidade : float
        Probabilidade, em ``[0, 1]``, de um agente aderido efetivamente votar
        conforme prometido (``VoterAgent.votou_conforme``, sorteada uma única
        vez por agente no momento da adesão). Suposição populacional v0 sem
        valor calibrado — o default ``1.0`` foi escolhido especificamente
        para reproduzir o comportamento anterior a este parâmetro (nenhuma
        desistência modelada), não como uma estimativa da conformidade real;
        ver CLAUDE.md. Eixo de heterogeneidade distinto de ``prop_racional``
        (rege a adesão) e de ``rho`` (rege o timing do desembolso) — os três
        não devem ser conflados.
    seed : int | None
        Semente para reprodutibilidade.
    """

    _SIGMA_FRACAO_DELTA_T: float = 0.02
    """Fração de delta_t usada como desvio-padrão do ramo concentrado de rho
    (placeholder v0, pendente de calibração com orientadores)."""

    def __init__(
        self,
        n_agentes: int = 100,
        alpha_beta: tuple[float, float] = (2.0, 2.0),
        prop_racional: float = 0.9,
        recompensa: float = 1.0,
        threshold_range: tuple[float, float] = (0.2, 0.8),
        n_steps: int = 50,
        n_secoes: int = 5,
        secoes_por_municipio: int | None = None,
        municipios_por_estado: int | None = None,
        n_candidatos: int = 1,
        candidato_alvo: int = 0,
        resultado_alvo: float = 0.5,
        granularidade: str = "pool",
        unidade_alvo: int | None = None,
        delta_t: float = 10.0,
        rho: float = 0.0,
        beta: int = 1,
        prob_conformidade: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho deve estar em [0, 1].")
        if beta < 1:
            raise ValueError("beta deve ser >= 1.")
        if not 0.0 <= prob_conformidade <= 1.0:
            raise ValueError("prob_conformidade deve estar em [0, 1].")
        if n_candidatos < 1:
            raise ValueError("n_candidatos deve ser >= 1.")
        if not 0 <= candidato_alvo < n_candidatos:
            raise ValueError("candidato_alvo deve estar em [0, n_candidatos).")
        if granularidade not in _GRANULARIDADES_VALIDAS:
            raise ValueError(f"granularidade deve ser um de {_GRANULARIDADES_VALIDAS}.")
        if granularidade != "pool" and unidade_alvo is None:
            raise ValueError("unidade_alvo é obrigatório quando granularidade != 'pool'.")

        super().__init__(rng=seed)
        self.n_agentes = n_agentes
        self.recompensa = recompensa
        self.n_steps = n_steps
        self.n_secoes = n_secoes
        self.secoes_por_municipio = secoes_por_municipio or n_secoes
        self.n_municipios = math.ceil(n_secoes / self.secoes_por_municipio)
        self.municipios_por_estado = municipios_por_estado or self.n_municipios
        self.n_estados = math.ceil(self.n_municipios / self.municipios_por_estado)
        self.n_candidatos = n_candidatos
        self.candidato_alvo = candidato_alvo
        self.resultado_alvo = resultado_alvo
        self.granularidade = granularidade
        self.unidade_alvo = unidade_alvo
        self.delta_t = delta_t
        self.rho = rho
        self.beta = beta
        self.prob_conformidade = prob_conformidade

        if granularidade != "pool":
            n_unidades = {"secao": n_secoes, "municipio": self.n_municipios, "estado": self.n_estados}[granularidade]
            if not 0 <= unidade_alvo < n_unidades:
                raise ValueError(f"unidade_alvo deve estar em [0, {n_unidades}) para granularidade={granularidade!r}.")

        self.eventos_adesao: list[tuple[int, int]] = []  # (timestep, unique_id)
        self.eventos_desembolso: list[tuple[float, int]] = []  # (timestamp, unique_id)
        self.contrato_ativado: bool | None = None
        self._fase2_executada = False

        alpha, beta_shape = alpha_beta
        propensoes = self.rng.beta(alpha, beta_shape, size=n_agentes)
        thresholds = self.rng.uniform(*threshold_range, size=n_agentes)
        racional_flags = self.rng.random(n_agentes) < prop_racional
        secoes = self.rng.integers(0, n_secoes, size=n_agentes)
        municipios = secoes // self.secoes_por_municipio
        estados = municipios // self.municipios_por_estado
        candidatos_preferidos = self.rng.integers(0, n_candidatos, size=n_agentes)

        VoterAgent.create_agents(
            self,
            n_agentes,
            propensao=list(propensoes),
            utility_threshold=list(thresholds),
            racional=list(racional_flags),
            secao=list(secoes),
            municipio=list(municipios),
            estado=list(estados),
            candidato_preferido=list(candidatos_preferidos),
        )

    def step(self) -> None:
        """Executa um passo de decisão de adesão (Fase 1, ordem embaralhada)."""
        self.agents.shuffle_do("step")
        for agent in self.agents:
            if agent.aderiu and agent.timestep_adesao == self.steps:
                self.eventos_adesao.append((self.steps, agent.unique_id))

    def run(self, n_steps: int | None = None) -> None:
        """Executa a Fase 1 (campanha) por ``n_steps`` passos (default: ``self.n_steps``).

        Não resolve o desembolso — chame ``resolver_desembolso()`` explicitamente
        depois de encerrar a campanha.
        """
        for _ in range(n_steps if n_steps is not None else self.n_steps):
            self.step()

    def resolver_desembolso(self) -> None:
        """Fase 2: resolve ativação do contrato e gera eventos de desembolso.

        Deve ser chamado uma única vez, explicitamente, após ``run()``.
        Levanta ``RuntimeError`` se chamado mais de uma vez, para não
        corromper ``eventos_desembolso`` com uma segunda amostragem.

        ``contrato_ativado`` é decidido comparando ``resultado_alvo`` contra
        o resultado do candidato-alvo na granularidade/unidade configuradas
        (``self.granularidade``/``self.unidade_alvo``): com
        ``granularidade="pool"`` (default), usa ``fonte_c_resultado_agregado()``
        sobre o pool inteiro — retrocompatível. Com ``granularidade`` em
        ``{"secao","municipio","estado"}``, usa o resultado da unidade
        específica ``unidade_alvo`` nesse nível
        (``resultado_eleitoral_por_secao()[unidade_alvo]``, etc.). Suposição
        v0: só uma unidade territorial por vez — ataques simultâneos em
        múltiplas unidades ficam para trabalho futuro.

        Nota de escopo — a granularidade afeta só a condição de ativação, não
        quem recebe desembolso: a lista de pagos (abaixo) continua sendo todo
        agente aderido no pool inteiro, independentemente de
        ``unidade_alvo`` — restringir o pagamento à unidade-alvo reabriria a
        questão de "população visada vs. pool inteiro" que o CLAUDE.md já
        fechou deliberadamente (ver nota de escopo em
        ``fonte_c_resultado_agregado``).

        Se ativar, gera um evento de desembolso por agente aderido, com
        timestamp em ``[0, delta_t]`` amostrado conforme ρ (ver
        ``_amostrar_timestamps_desembolso``) — ρ atua só aqui, não na Fase 1.
        Cada um desses timestamps é então fragmentado em β sub-eventos via
        ``layer2_copula.aplicar_batching`` (usando o mesmo ``self.rng``
        compartilhado do modelo, para manter tudo dentro de um único stream
        reprodutível): o `unique_id` do agente original é repetido β vezes
        (``np.repeat``), na mesma ordem de agrupamento que ``aplicar_batching``
        usa internamente, então cada fragmento continua associado ao agente
        correto em ``eventos_desembolso``. Com β=1 (default),
        ``aplicar_batching`` é no-op exato (nem consome números aleatórios) —
        `eventos_desembolso` continua um evento por agente pago, igual a
        antes deste parâmetro existir.

        Nota de escopo — desembolso independe de conformidade: a lista de
        agentes pagos é construída a partir de ``agent.aderiu`` sozinho, não
        de ``agent.aderiu and agent.votou_conforme``. Isso é proposital, não
        um descuido: o oráculo só verifica o resultado agregado R (ver
        ``fonte_c_resultado_agregado``), não pode auditar votos individuais
        sem violar o sigilo do voto, e portanto não tem como distinguir, no
        momento do desembolso, quem efetivamente votou como prometido de quem
        desistiu após aderir. Desistência pode coexistir com pagamento
        realizado — consistente com a nota do adversary model
        (`docs/adversary_model_draft.tex`) de que condicionar o pagamento
        apenas ao resultado agregado não impede, por si só, desistência ou
        free-riding entre os participantes pagos.
        """
        if self._fase2_executada:
            raise RuntimeError("resolver_desembolso() já foi executado para este modelo.")
        self._fase2_executada = True

        if self.granularidade == "pool":
            resultado_r = self.fonte_c_resultado_agregado()
        elif self.granularidade == "secao":
            resultado_r = self.resultado_eleitoral_por_secao().loc[self.unidade_alvo]
        elif self.granularidade == "municipio":
            resultado_r = self.resultado_eleitoral_por_municipio().loc[self.unidade_alvo]
        else:  # "estado"
            resultado_r = self.resultado_eleitoral_por_estado().loc[self.unidade_alvo]

        self.contrato_ativado = bool(resultado_r >= self.resultado_alvo)
        if not self.contrato_ativado:
            return

        agentes_pagos = [agent.unique_id for agent in self.agents if agent.aderiu]
        if not agentes_pagos:
            return

        timestamps = self._amostrar_timestamps_desembolso(len(agentes_pagos))
        timestamps = aplicar_batching(timestamps, delta_t=self.delta_t, beta=self.beta, random_state=self.rng)
        ids_repetidos = np.repeat(agentes_pagos, self.beta)
        self.eventos_desembolso = list(zip(timestamps.tolist(), ids_repetidos.tolist()))

    def _amostrar_timestamps_desembolso(self, n: int) -> np.ndarray:
        """Amostra ``n`` timestamps de desembolso em ``[0, delta_t]`` conforme ρ.

        Mistura Bernoulli(ρ): cada evento é sorteado do ramo concentrado
        (Normal em torno do centro da janela ``delta_t / 2``, recortada em
        ``[0, delta_t]``) com probabilidade ρ, ou do ramo uniforme
        independente com probabilidade ``1 - rho``. Escolhida por bater
        exatamente nos extremos ρ=0 (uniforme puro) e ρ=1 (concentrado puro),
        ao contrário de uma dispersão continuamente decrescente, que não
        atinge nenhum dos dois extremos exatamente.

        O instante-alvo do ramo concentrado é fixado no centro da janela
        (``delta_t / 2``), não sorteado por simulação: como o ramo uniforme
        também tem média ``delta_t / 2``, isso elimina o termo cruzado de
        variância entre componentes da mistura (diferença de médias ao
        quadrado), que apareceria se o instante-alvo fosse sorteado longe do
        centro — sem essa correção, ρ intermediário poderia produzir std
        *maior* que ρ=0, quebrando a monotonicidade esperada de "mais
        coordenação ⇒ mais concentração temporal". Decisão de modelagem v0,
        pendente de validação.

        Os dois ramos são sempre amostrados como vetores completos de tamanho
        ``n`` (mesmo quando ρ=0 ou ρ=1, quando um deles é descartado), para
        manter o número de draws do RNG compartilhado (``self.rng``)
        previsível e independente de quanto ρ vale — importante para
        reprodutibilidade e para não introduzir bugs de ordem de iteração.

        Decisão de modelagem v0: a dispersão do ramo concentrado é uma fração
        fixa de ``delta_t`` (``_SIGMA_FRACAO_DELTA_T``), não calibrada.
        """
        uniforme = self.rng.uniform(0, self.delta_t, size=n)
        if self.rho == 0.0:
            return uniforme

        t_alvo = self.delta_t / 2
        sigma = self._SIGMA_FRACAO_DELTA_T * self.delta_t
        concentrado = np.clip(self.rng.normal(t_alvo, sigma, size=n), 0, self.delta_t)
        if self.rho == 1.0:
            return concentrado

        usa_concentrado = self.rng.random(n) < self.rho
        return np.where(usa_concentrado, concentrado, uniforme)

    def fonte_a_eventos_fronteira(self) -> pd.DataFrame:
        """Observável da Fonte A: timestamp, contagem e volume monetário do desembolso.

        Reflete a Fase 2 (pós-resultado), não a adesão (Fase 1) — ver
        docstring do módulo. ``volume`` é ``n_eventos * (self.recompensa /
        self.beta)``, não ``n_eventos * self.recompensa``: cada um dos β
        sub-eventos fragmentados de um agente pago carrega ``recompensa/beta``,
        para que o total pago por agente ao longo de todos os seus
        fragmentos continue exatamente ``recompensa`` — a fragmentação
        (evasão do adversário) muda só a distribuição temporal e a
        granularidade da contagem, nunca o valor total pago. Com β=1
        (default), reduz-se exatamente a ``n_eventos * recompensa``, como
        antes deste parâmetro existir. É essa divisão que faz a razão
        n_eventos/volume finalmente carregar sinal discriminativo
        independente sob β>1 (o "indicador de batch" do PLANO §5.3.1) — com
        β=1 a razão ainda é constante, como sempre foi.

        Returns
        -------
        pd.DataFrame
            Colunas ``timestep`` (int, floor do timestamp de desembolso),
            ``n_eventos`` (contagem de desembolsos no timestep) e ``volume``
            (magnitude monetária agregada no timestep).
        """
        if not self.eventos_desembolso:
            return pd.DataFrame(
                {
                    "timestep": pd.Series(dtype=int),
                    "n_eventos": pd.Series(dtype=int),
                    "volume": pd.Series(dtype=float),
                }
            )

        timesteps = [int(np.floor(t)) for t, _ in self.eventos_desembolso]
        contagem = (
            pd.Series(timesteps)
            .value_counts()
            .sort_index()
            .rename_axis("timestep")
            .reset_index(name="n_eventos")
        )
        contagem["volume"] = contagem["n_eventos"] * (self.recompensa / self.beta)
        return contagem

    def _fracoes_por_grupo(self, mascara: np.ndarray, grupos: np.ndarray, n_grupos: int) -> pd.Series:
        """Fração de agentes satisfazendo ``mascara`` (bool, por agente), por grupo.

        Grupos sem agentes atribuídos recebem fração ``0.0`` por convenção
        (não ``NaN``). ``grupos`` é qualquer array de rótulos inteiros por
        agente (seção, município ou estado) — generaliza o que antes era
        específico de seção, reusado pelas três granularidades e pela adesão
        bruta.
        """
        denominador = np.bincount(grupos, minlength=n_grupos).astype(float)
        numerador = np.bincount(grupos, weights=mascara.astype(float), minlength=n_grupos)

        fracoes = np.where(denominador == 0, 0.0, numerador / np.where(denominador == 0, 1, denominador))
        return pd.Series(fracoes, index=pd.RangeIndex(n_grupos))

    def _voto_e_candidato_alvo(self) -> np.ndarray:
        """Máscara booleana por agente: ``True`` se o voto final é o candidato-alvo.

        Com ``n_candidatos == 1`` (default), reduz-se exatamente a
        ``aderiu and votou_conforme`` — ``candidato_preferido`` nunca é
        consultado. Esse branch não é uma otimização: é necessário para
        retrocompatibilidade. Sem ele, todo agente teria
        ``candidato_preferido == 0`` (único valor possível em
        ``range(1)``), coincidindo com o default ``candidato_alvo == 0`` e
        fazendo TODO agente — mesmo quem nunca aderiu — contar como voto no
        candidato-alvo.

        Com ``n_candidatos > 1``, o voto final de cada agente é
        ``candidato_alvo`` se ``aderiu and votou_conforme``, senão
        ``candidato_preferido`` (voto de base, categórico uniforme, v0 —
        sem preferência ideológica/demográfica).
        """
        if self.n_candidatos == 1:
            return np.array([agent.aderiu and agent.votou_conforme for agent in self.agents])

        votos = np.array(
            [
                self.candidato_alvo if (agent.aderiu and agent.votou_conforme) else agent.candidato_preferido
                for agent in self.agents
            ]
        )
        return votos == self.candidato_alvo

    def resultado_eleitoral_por_secao(self) -> pd.Series:
        """Observável da Fonte C: fração de votos do candidato-alvo por seção (g=seção).

        Ver nota de escopo v0 sobre R (pool = eleitorado inteiro, sem base
        separada) em ``fonte_c_resultado_agregado``.

        Returns
        -------
        pd.Series
            Índice ``0..n_secoes-1``, valores em ``[0, 1]``, nome
            ``"fracao_candidato_alvo"``.
        """
        mascara = self._voto_e_candidato_alvo()
        grupos = np.array([agent.secao for agent in self.agents])
        return self._fracoes_por_grupo(mascara, grupos, self.n_secoes).rename("fracao_candidato_alvo")

    def resultado_eleitoral_por_municipio(self) -> pd.Series:
        """Observável da Fonte C: fração de votos do candidato-alvo por município (g=município).

        Município = ``secao // secoes_por_municipio``, determinístico (sem
        sorteio). Ver nota de escopo v0 sobre R em ``fonte_c_resultado_agregado``.

        Returns
        -------
        pd.Series
            Índice ``0..n_municipios-1``, valores em ``[0, 1]``, nome
            ``"fracao_candidato_alvo"``.
        """
        mascara = self._voto_e_candidato_alvo()
        grupos = np.array([agent.municipio for agent in self.agents])
        return self._fracoes_por_grupo(mascara, grupos, self.n_municipios).rename("fracao_candidato_alvo")

    def resultado_eleitoral_por_estado(self) -> pd.Series:
        """Observável da Fonte C: fração de votos do candidato-alvo por estado (g=estado).

        Estado = ``municipio // municipios_por_estado``, determinístico (sem
        sorteio). Ver nota de escopo v0 sobre R em ``fonte_c_resultado_agregado``.

        Returns
        -------
        pd.Series
            Índice ``0..n_estados-1``, valores em ``[0, 1]``, nome
            ``"fracao_candidato_alvo"``.
        """
        mascara = self._voto_e_candidato_alvo()
        grupos = np.array([agent.estado for agent in self.agents])
        return self._fracoes_por_grupo(mascara, grupos, self.n_estados).rename("fracao_candidato_alvo")

    def fonte_c_resultado(self) -> pd.Series:
        """Alias de retrocompatibilidade para ``resultado_eleitoral_por_secao()``.

        Nome mantido pela API anterior a esta funcionalidade; "resultado" já
        significava "por seção" desde antes de existir hierarquia de
        granularidade. Com ``n_candidatos == 1`` (default), o valor é
        idêntico ao de antes (fração de adesão conforme por seção) — só o
        nome da ``Series`` mudou de ``"fracao_conforme"`` para
        ``"fracao_candidato_alvo"``, generalização não observada pelos testes
        existentes (que checam valores, não nomes).
        """
        return self.resultado_eleitoral_por_secao()

    def fonte_c_resultado_agregado(self) -> float:
        """Fração populacional de votos do candidato-alvo (não ponderada por seção).

        É este valor que ``resolver_desembolso()`` usa como R quando
        ``granularidade="pool"`` (default). Com ``n_candidatos == 1``,
        reduz-se exatamente à fração de adesão conforme de antes desta
        funcionalidade — ver ``_voto_e_candidato_alvo``. Equivale à média
        simples por agente, não à média das frações por seção/município/
        estado — só coincidem se os grupos tiverem o mesmo número de
        agentes, o que não é garantido pela atribuição uniforme aleatória.
        Sempre calculado sobre o pool inteiro, independente da hierarquia de
        granularidade (``secoes_por_municipio``/``municipios_por_estado``) —
        só coincide numericamente com ``resultado_eleitoral_por_estado()``
        quando a hierarquia colapsa tudo num único estado (default).

        IMPORTANTE — decisão de escopo v0: este método calcula R apenas sobre
        o pool de agentes que se autosselecionaram como alvo potencial do
        incentivo (todo n_agentes), não sobre um eleitorado total mais amplo
        que incluiria não-visados. Essa é uma simplificação deliberada, não
        um artefato esquecido: um CSC não tem mecanismo de
        targeting/intermediário, então não há como modelar "quem é alvo" de
        forma distinta de "quem se autosselecionou". A implicação é que a
        normalização de C_min por "tamanho médio do grupo por seção" (PLANO
        §5.4.4), quando for operacionalizada nas Semanas 5-6, precisará
        decidir se "grupo" = n_agentes (tautológico com o pool inteiro
        simulado, como está hoje) ou se será necessário revisitar essa
        decisão. Registrado como pendência para a operacionalização de
        C_min, não para este momento.
        """
        return float(self._voto_e_candidato_alvo().sum()) / self.n_agentes

    def fracao_adesao_por_secao(self) -> pd.Series:
        """Fração de adesão BRUTA por seção (``aderiu``, sem gate de conformidade).

        Não é um observável de Fonte C: adesão bruta é estado interno
        pré-resultado (Fase 1), nunca publicamente observável na arquitetura
        do adversary model — daí não levar o prefixo ``fonte_c_``/
        ``resultado_eleitoral_``. Útil para comparar contra
        ``resultado_eleitoral_por_secao()`` e quantificar o efeito de
        ``prob_conformidade``.

        Returns
        -------
        pd.Series
            Índice ``0..n_secoes-1``, valores em ``[0, 1]``, nome
            ``"fracao_adesao_bruta"``.
        """
        mascara = np.array([agent.aderiu for agent in self.agents])
        grupos = np.array([agent.secao for agent in self.agents])
        return self._fracoes_por_grupo(mascara, grupos, self.n_secoes).rename("fracao_adesao_bruta")

    def fracao_adesao(self) -> float:
        """Fração populacional de adesão BRUTA (``aderiu``, sem gate de conformidade).

        Não é um observável de Fonte C (ver ``fracao_adesao_por_secao``).
        Com ``n_candidatos == 1``, sempre ``>= fonte_c_resultado_agregado()``
        (igualdade estrita quando ``prob_conformidade == 1.0`` —
        retrocompatibilidade). Com ``n_candidatos > 1`` essa desigualdade
        **não** é garantida: agentes que nunca aderiram ainda podem "votar"
        no candidato-alvo por coincidência de voto de base
        (``candidato_preferido``), então ``fonte_c_resultado_agregado()``
        pode superar ``fracao_adesao()`` — é justamente esse efeito que torna
        o resultado sensível a `n_candidatos` mesmo sem incentivo (ver
        testes e o notebook de validação).
        """
        n_aderiram = sum(1 for agent in self.agents if agent.aderiu)
        return n_aderiram / self.n_agentes
