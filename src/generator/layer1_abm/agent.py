"""Agente eleitor da Camada 1 (ABM)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mesa

if TYPE_CHECKING:
    from src.generator.layer1_abm.model import ElectionModel


class VoterAgent(mesa.Agent):
    """Eleitor com propensão heterogênea a aceitar o incentivo econômico.

    Parameters
    ----------
    model : ElectionModel
        Modelo ao qual o agente pertence.
    propensao : float
        Propensão individual a aceitar o incentivo, amostrada de Beta(α, β)
        pelo modelo.
    utility_threshold : float
        Recompensa mínima (ponderada pela propensão) para que o agente adira.
    racional : bool
        Se ``True``, decide por comparação determinística de utilidade esperada.
        Se ``False``, decide de forma ruidosa (não segue estritamente a utilidade).
    secao : int
        Identificador da seção eleitoral à qual o agente pertence (granularidade
        g=seção da Fonte C).
    municipio : int
        Identificador do município (granularidade g=município), derivado
        deterministicamente de ``secao`` pelo modelo (``secao // secoes_por_municipio``,
        sem sorteio).
    estado : int
        Identificador do estado (granularidade g=estado), derivado
        deterministicamente de ``municipio`` pelo modelo
        (``municipio // municipios_por_estado``, sem sorteio).
    candidato_preferido : int
        Candidato em que o agente vota na ausência de qualquer influência do
        incentivo (voto de base), sorteado uma única vez na criação —
        categórica uniforme sobre ``range(n_candidatos)``, suposição v0 (sem
        preferência ideológica/demográfica). Só é consultado quando
        ``model.n_candidatos > 1``; com ``n_candidatos == 1`` o voto de base
        nunca entra no cálculo do resultado (ver ``ElectionModel._voto_e_candidato_alvo``).

    Attributes
    ----------
    votou_conforme : bool
        Conformidade com o incentivo prometido, sorteada uma única vez no
        momento da adesão via Bernoulli(``model.prob_conformidade``) — não é
        um parâmetro do construtor. Default ``False`` para agentes que nunca
        aderem (sem sentido definido nesse caso; sempre combinar com
        ``aderiu`` ao ler este atributo). Eixo de heterogeneidade distinto de
        ``racional`` (rege a decisão de adesão) e de ``rho`` do modelo (rege
        o timing do desembolso, não a decisão do agente) — não confundir.
    """

    def __init__(
        self,
        model: "ElectionModel",
        propensao: float,
        utility_threshold: float,
        racional: bool,
        secao: int,
        municipio: int,
        estado: int,
        candidato_preferido: int,
    ) -> None:
        super().__init__(model)
        self.propensao = propensao
        self.utility_threshold = utility_threshold
        self.racional = racional
        self.secao = secao
        self.municipio = municipio
        self.estado = estado
        self.candidato_preferido = candidato_preferido
        self.aderiu = False
        self.timestep_adesao: int | None = None
        self.votou_conforme = False

    def step(self) -> None:
        """Avalia, em um passo, se o agente adere ao incentivo oferecido."""
        if self.aderiu:
            return

        if self.racional:
            aceita = self.model.recompensa * self.propensao >= self.utility_threshold
        else:
            # Comportamento ruidoso: não segue estritamente a utilidade esperada.
            aceita = self.model.rng.random() < self.propensao

        if aceita:
            self.aderiu = True
            self.timestep_adesao = self.model.steps
            # Sorteio único de conformidade: desistência é possível mesmo após
            # adesão (ver resolver_desembolso() e nota sobre Wang et al. no
            # adversary model — desistência entre pagos é vulnerabilidade real
            # do mecanismo, não algo a ignorar).
            self.votou_conforme = self.model.rng.random() < self.model.prob_conformidade
