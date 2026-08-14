# Contexto do projeto

TCC sobre detecção de incentivos econômicos adversariais em eleições via
smart contracts com privacidade nativa. Especificação completa em
`docs/PLANO_TCC_ARTIGO_V4_1.md` — leia antes de propor qualquer mudança
estrutural.

## Fase atual
Implementação do gerador sintético: Camada 1 (ABM de agentes eleitores,
via Mesa) e Camada 2 (estrutura de dependência via cópula Clayton,
biblioteca `copulas`).

## Convenções
- Parâmetros populacionais (n_agentes, alpha_beta, prop_racional) são
  configuráveis, não hardcoded — ainda não fixados definitivamente
  (pendente de validação com orientadores).
- Parâmetros do gerador (π, g, Δt, λ, ρ, β) são distintos dos populacionais
  e fazem parte do design fatorial — não confundir os dois grupos.
- ρ (grau de coordenação) e prop_racional são parâmetros DIFERENTES —
  não devem ser conflados.
- Testes em `tests/` devem cobrir os Sanity Checks descritos no plano
  (§5.2.3) assim que a Camada 1 estiver funcional.

## Estilo
- Código Python com type hints
- Docstrings estilo NumPy
- Sem comentários óbvios; comentar apenas decisões não triviais