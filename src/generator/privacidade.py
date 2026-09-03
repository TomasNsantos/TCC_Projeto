"""Mecanismo de privacidade compartilhado entre classe positiva e negativa.

π é o nível de privacidade da arquitetura de blockchain simulada (PLANO
§5.1.2) — fração de informação inacessível ao observador externo. Na
prática, cada evento gerado pelo simulador (Fonte A) tem probabilidade π de
ficar oculto do detector, sorteada independentemente por evento
(mascaramento estocástico Bernoulli(π)). O mecanismo em si já foi decidido
com o orientador; este módulo só concentra a peça compartilhada — sem ele,
a mesma lógica teria que ser duplicada entre a Fonte A da classe positiva
(via `ElectionModel`) e a Fonte A de fundo da classe negativa (via
`normal_mode`), que não têm nenhuma outra dependência estrutural entre si.
"""

from __future__ import annotations

import numpy as np

RandomState = int | np.random.Generator | None


def mascara_sobrevivencia_pi(n_eventos: int, pi: float, random_state: RandomState = None) -> np.ndarray:
    r"""Sorteia, por evento, se ele sobrevive à ocultação de privacidade π.

    Cada evento sobrevive (é observado pelo detector) independentemente
    com probabilidade ``1 - π`` — ``rng.random(n_eventos) >= pi``. π alto
    esconde mais eventos do observador externo, reproduzindo uma
    arquitetura de blockchain mais privada.

    Parameters
    ----------
    n_eventos : int
        Número de eventos a mascarar.
    pi : float
        Nível de privacidade, em ``[0, 1]`` (PLANO §5.1.2). ``pi=0``
        reproduz observabilidade total (nenhum evento é ocultado);
        ``pi=1`` oculta todos os eventos.
    random_state : int | np.random.Generator | None
        Semente para reprodutibilidade. Não é consultado quando ``pi=0``
        (ver nota abaixo).

    Returns
    -------
    np.ndarray
        Array booleano de tamanho ``n_eventos``. ``True`` = evento
        sobrevive (observado); ``False`` = evento ocultado pela
        privacidade.

    Raises
    ------
    ValueError
        Se ``pi`` não estiver em ``[0, 1]`` (mesma convenção de validação
        de ``rho`` em ``ElectionModel.__init__``).

    Notes
    -----
    ``pi=0.0`` é retrocompatibilidade estrita: retorna
    ``np.ones(n_eventos, dtype=bool)`` SEM criar
    ``np.random.default_rng(random_state)`` nem consumir nenhum número do
    RNG — mesmo princípio já usado por ``aplicar_batching`` com ``beta=1``
    em ``layer2_copula/copula.py`` ("nem consome números aleatórios").
    Isso garante que, enquanto π não estiver em uso em nenhum lugar do
    gerador (default ``pi=0.0``), nenhum código existente muda de
    comportamento, mesmo que receba um ``random_state`` "errado".
    """
    if not 0.0 <= pi <= 1.0:
        raise ValueError("pi deve estar em [0, 1].")

    if n_eventos == 0:
        return np.ones(0, dtype=bool)

    if pi == 0.0:
        return np.ones(n_eventos, dtype=bool)

    rng = np.random.default_rng(random_state)
    return rng.random(n_eventos) >= pi
