# Lista fechada de pendências — Dia 9, Semana 9-10

Consolidação de todas as pendências registradas ao longo da Semana 9-10
(CLAUDE.md, `docs/consolidacao_dia8_semana9-10.md`) mais as confirmações
factuais feitas neste dia. Cada item tem uma marcação:

- **[SÓ ORIENTADOR]** — decisão que só avança com consenso dos orientadores.
- **[AUTOSSUFICIENTE]** — Tom pode resolver sozinho, sem depender deles.
- **[BAIXA PRIORIDADE]** — já registrado como "variar se der tempo" ou
  equivalente; não bloqueia nada agora.

Não é a redação de §5.3 — essa fica adiada até as pendências de
gerador/detector/eleitor abaixo serem resolvidas com os orientadores.

---

## Verificações factuais que embasam as marcações abaixo

Confirmadas por leitura de código/dataset real nesta etapa, não por
anotações anteriores:

- **Seeds no dataset v3: 2 (`{1, 2}`), não 5.** Contagem real nos 270
  arquivos de `output/dataset_producao_v3/`. O PLANO (linha 250: "5(seeds)
  = 1.620") define 5 — diverge do dataset real gerado.
- **π no dataset v3: 5 níveis (`{0.0, 0.5, 0.75, 0.9, 0.95}`), não 4.**
  O PLANO (linha 192: `π ∈ {0.50, 0.75, 0.90, 0.95}`) lista 4 — o dataset
  real tem um nível adicional (π=0,00) não documentado no PLANO.
- **`RobustezBeta`/`expandir_grade_robustez`: continuam desconectados.**
  `src/pipeline/runner.py:128` chama só `expandir_grade`, que sempre
  força `beta=1` (`config.py:307-328`). `expandir_grade_robustez`
  (`config.py:331`) existe e funciona corretamente, mas não é chamada em
  nenhum lugar de `scripts/`/`runner.py`/`geracao.py` — confirmado por
  busca no repositório inteiro. Nada mudou desde o registro mais antigo.
- **Assimetria de tráfego de fundo entre classes — confirmada.**
  `src/pipeline/geracao.py:304-309` (classe positiva, via
  `gerar_cenario_adversarial`): Fonte A/B vêm só do CSC, sem nenhum
  parâmetro de taxa/volume de Poisson. `geracao.py:352-361` (classe
  negativa, via `gerar_cenario_normal`): recebe explicitamente
  `taxa_fonte_a`, `volume_medio_fonte_a`, `taxa_fonte_b` — os 3
  parâmetros de tráfego de fundo Poisson que a positiva nunca vê.

---

## [SÓ ORIENTADOR]

**1. Checagem de dominância redesenhada (Fonte C + Camada 1 crua juntas).**
A Etapa 8b original do plano do Dia 6 (excluir só contagem/volume bruto
de A/B, mantendo Fonte C) foi desenhada antes do achado de dominância de
Fonte C. A ablação C4 (Dia 6, Prompt 3.5) mostrou que remover Fonte C só
desloca a dominância para `fonte_b_n_eventos_total` — não resolve,
revela outro problema estrutural. Precisa ser redesenhada considerando
os dois achados juntos (Fonte C E Camada 1 crua de A/B dominando em
sequência). Depende diretamente da decisão sobre heterogeneidade de
apoio orgânico (item 2 abaixo) para ter um dataset onde valha a pena
redesenhar essa checagem.

**2. Heterogeneidade de apoio orgânico entre candidatos.** Pendência de
gerador. Causa raiz diagnosticada no Dia 6:
`src/generator/layer1_abm/model.py:259` sorteia o voto de base de cada
agente uniformemente entre candidatos (`rng.integers(0, n_candidatos,
...)`), sem nenhum peso de popularidade — confirmado como suposição v0
deliberada e já coberta por teste
(`test_n_candidatos_maior_que_um_conta_voto_de_base_sem_incentivo`).
`fonte_c_media` na classe negativa varia só 0,0005 entre as duas seeds
do grid (ruído amostral, não heterogeneidade estrutural), tornando
`fonte_c_media` um proxy quase-determinístico de "houve CSC ativo".
Caminho de mudança identificado, não implementado: trocar `rng.integers`
uniforme por `rng.choice` com pesos de popularidade (ex. Dirichlet por
seed/cenário) como novo parâmetro do `ElectionModel`. Parametrização
exata não decidida.

**3. Risco ao critério de sucesso falsificável (§5.4.5) — dois
confundidores empilhados, não um só.** C3 isolado (só Fonte C) parece
resolver o problema de detecção quase inteiramente dentro de C7
(A+B+C) — contrariando a premissa de que a integração multimodal é
NECESSÁRIA, não só suficiente. Dois mecanismos de design podem estar
inflando essa separabilidade, independentemente um do outro:
  - (a) Popularidade uniforme entre candidatos (item 2 acima) — já
    diagnosticada no Dia 6 como causa raiz da dominância de
    `fonte_c_media`.
  - (b) **Assimetria de tráfego de fundo entre classes** (confirmada
    nesta etapa, ver seção de verificações acima): a classe negativa
    recebe ruído de fundo Poisson independente em Fonte A/B; a classe
    positiva (ativa ou inativa) NUNCA recebe esse ruído — Fonte A/B na
    positiva vem exclusivamente de eventos ligados ao CSC. Isso significa
    que, independentemente do mecanismo (a), a mera PRESENÇA de qualquer
    evento em Fonte A/B já carrega informação estrutural sobre a classe,
    porque as duas classes são geradas por processos categoricamente
    diferentes (positiva = só CSC; negativa = CSC nunca + ruído de fundo
    sempre). Os dois mecanismos — (a) e (b) — não foram isolados um do
    outro: não se sabe se a dominância observada no Dia 6 viria de
    qualquer um isoladamente, ou se exige os dois juntos.

Este é um achado preliminar de prova de conceito (split intra-cenário,
um único regime de treino/teste, 2 seeds) — não uma conclusão
definitiva. Precisa constar como risco explícito para discussão com os
orientadores antes do design fatorial completo (§5.4, Semanas 14-17).

**4. Placeholders de calibração sem valor fixado.** `taxa_fonte_a`,
`volume_medio_fonte_a`, `taxa_fonte_b` (ordem de grandeza não calibrada —
risco documentado no PLANO §5.2.4 de o detector aprender a separar
classes só pelo volume total); ρ/`prop_racional` em U(𝒜) sem direção
formalizada na utilidade do adversário; `g` fixo em `"pool"` no dataset
de produção (nenhuma variação real de granularidade eleitoral testada);
`C_min` (custo mínimo, §5.4.4) não operacionalizado.

**5. π: 5 níveis reais vs. 4 do PLANO.** Confirmado nesta etapa (ver
Verificações acima). Decisão necessária: atualizar o PLANO para refletir
o dado real (documentar π=0,00 como nível adicional), ou remover π=0,00
do grid de produção (implicaria re-geração/filtragem do dataset v3
existente). Decisão de conteúdo/metodologia, não técnica.

---

## [AUTOSSUFICIENTE]

**6. Lacunas do §5.3.1 na camada agregada (Dia 4-5).** z-score agregado
por janela, entropia temporal, desvio de IET vs. Poisson, mutual
information ativação×resultado por `g` — pendência v0 documentada desde
o Dia 4-5, implementável sem decisão externa.

**7. Projeção de custo de M3 precisa de repetição controlada (Dia 7).**
O benchmark (`scripts/benchmark_m3_custo.py`) mostrou variação de ~3,4x
no tempo de uma época entre duas execuções do mesmo shape, atribuída a
contenda de CPU (diagnosticado e confirmado, não bug de fórmula). Rodar
5-10 execuções por shape, reportar mediana + faixa, antes de considerar
o número final para levar ao orientador de IA.

**8. Seeds: estender o grid de 2 para mais seeds.** Confirmado nesta
etapa (ver Verificações acima) que `GradeFatorial.seeds` já aceita
qualquer lista — tecnicamente uma extensão aditiva, sem decisão de
consenso necessária para o CÓDIGO. (O custo de rodar a geração/M1/M2 de
novo sobre seeds adicionais é uma questão de tempo de projeto, mas não
de decisão técnica externa.)

**9. Conectar `RobustezBeta`/`expandir_grade_robustez` ao runner.**
Confirmado nesta etapa (ver Verificações acima) que a função já existe e
funciona corretamente — o gap é só de CONEXÃO ao runner de produção, não
de implementação. Escrever esse código de conexão não exige consenso dos
orientadores. **Ressalva**: RODAR o experimento de robustez de β de fato
pressupõe decidir `RobustezBeta.base_config` ("a configuração de melhor
desempenho", PLANO §5.4.3) — que depende de M1/M2/M3 completos existirem
primeiro. Isso é um bloqueio de pré-requisito técnico (outras pendências
precisam resolver antes), não um bloqueio de decisão externa — por isso
mantido como `[AUTOSSUFICIENTE]`.

Relacionado: Sanity Check 3 (§5.2.3, degradação monotônica do F1
conforme β aumenta) continua não testável sem essa conexão — ver item 11.

**10. Divergências texto (PLANO) vs. código real — correções de texto.**
  - Split não é temporal — é um corte determinístico por índice de
    `window_id` dentro de cada arquivo/seed (`storage.calcular_split`),
    não por timestamp real. Já documentado como "intra-cenário" nas
    etapas anteriores.
  - Δt não afeta Fonte C, apesar de o PLANO sugerir que deveria (a
    granularidade/ativação do CSC é decidida sobre o pool/unidade-alvo,
    não sobre a janela de divulgação Δt).
  - Cabeçalho do PLANO ainda diz "V4.1" mas o arquivo real do repositório
    é `docs/PLANO TCC ARTIGO V4_2.md`.
  - **Novos, confirmados nesta etapa:** seeds reais = 2 (PLANO diz 5);
    níveis de π reais = 5 (PLANO lista 4, falta π=0,00).

**11. Sanity Checks pendentes de revisitar a nível de detector.** SC3
depende do item 9 (conexão de `RobustezBeta`) e de M1/M2/M3 completos
para ter algo a medir. SC1/SC4 só foram verificados a nível de gerador
(dados sintéticos em si) — nunca revisitados a nível de detector, agora
que M1/M2 existem e já produziram métricas reais. Revisitar não exige
decisão externa, só rodar/conferir contra os modelos já implementados.

---

## [BAIXA PRIORIDADE]

**12. Sensibilidade populacional não testada.** `n_agentes` só testado
em 500 (falta o cenário de sensibilidade com 5.000); `prop_racional` só
testado em 0,9. Já registrado anteriormente como "variar se der tempo" —
não bloqueia nada do que está em andamento.

---

## Fora de escopo (registro de que não foi esquecido, não pendência desta etapa)

**13. McNemar / C_min.** Fora de escopo até o design fatorial completo
(§5.4, Semanas 14-17). Não tratado nem no Dia 6, nem no Dia 7, nem
nesta lista.

---

*Lista fechada para a Semana 9-10 — base para discussão com os
orientadores antes de prosseguir. A redação de §5.3 fica adiada até os
itens `[SÓ ORIENTADOR]` acima serem resolvidos.*
