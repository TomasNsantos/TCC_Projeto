# Contexto do projeto

TCC sobre detecção de incentivos econômicos adversariais em eleições via
smart contracts com privacidade nativa. Especificação completa em
`docs/PLANO_TCC_ARTIGO_V4_1.md` — leia antes de propor qualquer mudança
estrutural.

## Fase atual
Implementação do gerador sintético: Camada 1 (ABM de agentes eleitores,
via Mesa), Camada 2 (estrutura de dependência via cópula Clayton, biblioteca
`copulas`) e o gerador de modo normal/classe negativa
(`src/generator/normal_mode/`, Poisson homogêneo independente).

## Convenções
- Parâmetros populacionais (n_agentes, alpha_beta, prop_racional,
  prob_conformidade, n_candidatos) são configuráveis, não hardcoded — ainda
  não fixados definitivamente (pendente de validação com orientadores).
- Parâmetros do gerador (π, g, Δt, λ, ρ, β) são distintos dos populacionais
  e fazem parte do design fatorial — não confundir os dois grupos.
- ρ (grau de coordenação), prop_racional e prob_conformidade são três eixos
  de heterogeneidade DIFERENTES — não devem ser conflados: prop_racional
  rege a decisão de adesão (Fase 1), ρ rege o timing do desembolso (Fase 2),
  prob_conformidade rege se um agente aderido efetivamente vota como
  prometido (também Fase 1, mas ortogonal a prop_racional).
- Testes em `tests/` devem cobrir os Sanity Checks descritos no plano
  (§5.2.3) assim que a Camada 1 estiver funcional.
- **Mecanismo de conformidade (v0):** `ElectionModel.prob_conformidade` +
  `VoterAgent.votou_conforme` modelam desistência entre agentes aderidos —
  cada agente aderido sorteia conformidade uma única vez via
  Bernoulli(prob_conformidade). Default `1.0` foi escolhido só por
  retrocompatibilidade (reproduz o comportamento anterior a este parâmetro,
  sem desistência), não é uma estimativa calibrada do valor real — pendente
  de validação com orientadores, como os demais parâmetros populacionais
  desta lista. O desembolso (Fase 2) continua pagando todo agente aderido,
  independente de conformidade: o oráculo só verifica o resultado agregado
  R, não pode auditar votos individuais sem violar sigilo do voto.
- **Decisão de escopo — R é calculado só sobre o pool visado:**
  `fonte_c_resultado()`/`fonte_c_resultado_agregado()` (proxy de R) usam
  como denominador `n_agentes` inteiro, não um eleitorado total mais amplo
  que incluiria não-visados — não há população-base separada no gerador.
  Decisão deliberada, não pendência esquecida: um CSC (criminal smart
  contract) é permissionless e autônomo, sem intermediário decidindo quem é
  alvo — o único filtro de participação é a autosseleção via utilidade do
  próprio agente (já implementada em `VoterAgent.step()`). Modelar uma
  segunda população estruturalmente separada reintroduziria implicitamente
  um avaliador de suscetibilidade, contrariando a própria motivação do CSC
  de eliminar esse intermediário (ver `docs/adversary_model_draft.tex`,
  Def. 1/§I-A). Implicação pendente: a normalização de C_min por "tamanho
  médio do grupo por seção" (PLANO §5.4.4), ao ser operacionalizada nas
  Semanas 5-6, precisará decidir se "grupo" = n_agentes (tautológico com o
  pool simulado, como está hoje) ou se essa decisão precisa ser revisitada.
- **Escolha de candidato (v0):** `VoterAgent.candidato_preferido` (voto de
  base, sorteado uma vez na criação, categórica uniforme sobre
  `range(n_candidatos)` — sem preferência ideológica/demográfica, fora de
  escopo) só é consultado quando `ElectionModel.n_candidatos > 1`; com o
  default `n_candidatos=1` o voto de base nunca entra no cálculo do
  resultado (`ElectionModel._voto_e_candidato_alvo`), reproduzindo
  exatamente o comportamento anterior a esta funcionalidade — retrocompatibilidade
  estrita, não um valor calibrado. Decisão deliberada: o gerador NÃO
  implementa alocação de cadeiras (quociente eleitoral, D'Hondt, sobras
  partidárias) — complexidade desproporcional ao propósito do gerador;
  `resultado_alvo` é reusado como a fração de votos válidos que o
  candidato-alvo precisa atingir, o que já permite simular um quociente
  pequeno de sistema proporcional sem formalizar a regra de alocação. Nota
  de orientação (não é default de construtor): rodadas futuras explorando o
  efeito de sistemas proporcionais devem testar `n_candidatos` na faixa de
  6-10.
- **Hierarquia de granularidade seção→município→estado (v0):** `município =
  seção // secoes_por_municipio`, `estado = município // municipios_por_estado`
  — atribuição determinística (sem sorteio), calculada uma vez na criação
  dos agentes. Defaults (`secoes_por_municipio=None`→`n_secoes`,
  `municipios_por_estado=None`→`n_municipios` computado) colapsam tudo num
  único município e num único estado, retrocompatível com o comportamento
  anterior a esta hierarquia (só seção existia). `resultado_eleitoral_por_secao/
  municipio/estado()` compartilham a mesma lógica de voto
  (`_voto_e_candidato_alvo`); `fonte_c_resultado_agregado()` continua sendo
  o agregado do pool inteiro, independente da hierarquia — só coincide com
  `resultado_eleitoral_por_estado()` quando a hierarquia default colapsa tudo
  num estado só.
- **Limitação v0 — uma unidade territorial-alvo por vez:**
  `ElectionModel.resolver_desembolso()` decide ativação checando o resultado
  de uma única seção/município/estado (`granularidade`/`unidade_alvo`,
  default `granularidade="pool"` = comportamento anterior, todo o pool). O
  gerador não modela ataques simultâneos em múltiplas unidades territoriais
  — fica para trabalho futuro. A granularidade afeta só a condição de
  ativação: o desembolso (Fase 2) continua pagando todo agente aderido no
  pool inteiro, independente de `unidade_alvo` — não restringe pagamento à
  unidade-alvo, para não reabrir a decisão de escopo de R acima.
- **Gerador de modo normal / classe negativa (v0):**
  `src/generator/normal_mode/` (`gerar_fonte_a_normal`, `gerar_fonte_b_normal`,
  `gerar_cenario_normal`) produz a classe negativa dos datasets de detecção
  (PLANO §5.2.4) — Fonte A/B como processos de Poisson homogêneos
  independentes (fluxos de RNG separados, sem cópula, sem superposição de
  múltiplos tipos de contrato — decisão v0 deliberada) e Fonte C de uma
  `ElectionModel` real com `recompensa=0` (eleição de fato acontece, só sem
  CSC por trás). **Pendência explícita:** `taxa`/`volume_medio` de Fonte A/B
  não têm valor calibrado e não têm default nas funções — a ordem de
  grandeza correta só pode ser decidida depois que os níveis de λ
  (baixa/média/alta) do design fatorial tiverem valores numéricos no
  gerador, o que ainda não aconteceu. Não adivinhar um valor "razoável"
  antes disso — risco explícito do PLANO §5.2.4 é o detector aprender a
  separar as classes só pelo volume total se a ordem de grandeza for
  escolhida sem cuidado.
- **Gerador de modo adversarial / classe positiva (v0):**
  `src/generator/adversarial_mode/` (`gerar_cenario_adversarial`) é a função
  de composição de topo para a classe positiva, simétrica a
  `gerar_cenario_normal()` — antes dela, essa composição (Fase 1 → Fase 2 →
  timestamps brutos de `eventos_desembolso` → `gerar_fonte_b`) só existia
  manualmente repetida em `tests/test_integration.py`. Recebe uma
  `ElectionModel` já configurada e ainda não executada (mesmo padrão de
  `gerar_cenario_normal`), roda `run()` + `resolver_desembolso()`, e só
  chama `gerar_fonte_b` se `contrato_ativado` — sem essa checagem, a cópula
  rodaria sem dado de entrada. **`tau_kendall` é parâmetro obrigatório, sem
  default:** controla a força-alvo de dependência A↔B na cópula Clayton
  (Camada 2) — suposição v0 sem valor calibrado, mesma convenção de
  `taxa`/`volume_medio` em `normal_mode/trafego.py`. Não confundir com ρ:
  `tau_kendall` (aqui) controla a dependência estatística entre Fonte A e
  Fonte B; ρ (`ElectionModel`) controla só a concentração temporal do
  timing de desembolso dentro da Fase 2 — dois eixos diferentes, um não
  deriva do outro.
- **Sanity Check 1 (vazio) ≠ modo normal (tráfego de fundo) — dois cenários
  de teste diferentes, um não substitui o outro:** Sanity Check 1
  (`test_sanity_check_1_lambda_zero_nao_gera_sinal_espurio`) testa o caso
  degenerado "nada acontece" (`recompensa=0`, sem CSC ativo nem tráfego de
  fundo — Fonte A/B ficam vazias por construção). O gerador de modo normal
  (`gerar_cenario_normal`) testa o caso realista "há tráfego de fundo
  plausível e eleição de verdade, só não há CSC" — por isso exige
  `n_candidatos > 1` (ver `_voto_e_candidato_alvo`): com `n_candidatos=1` e
  `recompensa=0`, o resultado por seção degenera para exatamente zero em
  todo lugar, reproduzindo o cenário do Sanity Check 1 em vez do cenário
  realista que esta função existe para gerar.
- **Pendência metodológica — duas formas de medir τ_Kendall(A,B), ainda não
  unificadas:** `test_modo_normal_produz_independencia`
  (`tests/test_layer2_copula.py`) calcula τ_Kendall por pareamento direto dos
  timestamps brutos — só funciona porque a cópula Clayton gera Fonte B
  pareada 1:1 com Fonte A por construção. O gerador de modo normal
  (`src/generator/normal_mode/trafego.py::contagem_por_timestep`) usa
  binning por janela deslizante, porque Fonte A/B independentes têm
  contagens de eventos diferentes, sem pareamento natural. As duas
  abordagens medem grandezas relacionadas mas não idênticas. Quando a
  Camada 3 de feature engineering for implementada (Semanas 9-10, PLANO
  §5.3.1 — "τ_Kendall(A,B) estimado por janela deslizante"), será preciso
  escolher UM método canônico que funcione igual para classe positiva e
  negativa; o pareamento direto que a Sanity Check 4 usa hoje é um artefato
  conveniente da geração sintética acoplada, não necessariamente
  representativo de como o feature será calculado sobre dados observados
  quaisquer. Pendência registrada, não resolvida agora.
- **Fragmentação β integrada à Fase 2:** `ElectionModel.beta` (parâmetro do
  design fatorial, categoria de `delta_t`/`rho` — não populacional; default
  `1`) conecta `layer2_copula.aplicar_batching` a `resolver_desembolso()`.
  Cada evento de desembolso é fragmentado em β sub-eventos, cada um levando
  `recompensa/beta` em `fonte_a_eventos_fronteira()` (o total pago por
  agente não muda com a fragmentação, só a distribuição temporal e a
  granularidade da contagem). β=1 é retrocompatibilidade estrita:
  `aplicar_batching` é no-op exato nesse caso (nem consome números
  aleatórios). Isso destrava gerar Fonte A com β>1 a partir do fluxo real do
  modelo — pré-requisito do Sanity Check 3 (§5.2.3, degradação monotônica do
  F1 conforme β aumenta), que continua não implementado (depende do
  detector, que ainda não existe).
- **Redução de amplitude por β não é monotônica — achado documentado, não
  corrigido:** o pico de volume de Fonte A NÃO cai monotonicamente conforme
  β aumenta — cai abruptamente de β=1 para β=2, depois volta a subir
  gradualmente até β=20 (formato de V), com os parâmetros do notebook
  `validacao_visual_batching_granularidade_visao_geral.ipynb` (seção 1,
  células 2-7): pico=1040.0 em β=1, mínimo de 180.0 em β=2, subindo de novo
  até 898.5 em β=20 — quase de volta ao nível de β=1. Causa: a janela de
  fragmentação usada por `aplicar_batching` é `delta_t/beta` — ENCOLHE
  conforme β cresce, ao contrário da intuição "mais fragmentos = mais
  dispersão". Em β baixo, o salto de "nenhuma dispersão" (β=1) para uma
  janela de tamanho `delta_t/2` (β=2) é abrupto e dispersa bastante; em β
  alto, a janela fica tão estreita que os sub-eventos fragmentados voltam a
  se reconcentrar perto do timestamp original, revertendo parte do efeito
  de evasão. Não é um bug de implementação — `aplicar_batching` faz
  exatamente o que `copula.py` documenta e o que `tests/test_layer2_copula.py`
  verifica; é a fórmula `delta_t/beta` para a janela em si que é uma
  suposição v0 nunca validada com os orientadores, decisão de modelagem, não
  erro de código. **Implicação pendente, não decidida aqui:** a expectativa
  do Sanity Check 3 (§5.2.3, degradação monotônica do F1 conforme β aumenta)
  pode não se sustentar se o detector usar amplitude de pico como sinal
  principal, já que essa amplitude em si não é monotônica em β. Duas saídas
  possíveis, nenhuma escolhida: (a) mudar a fórmula da janela de fragmentação
  (ex.: janela fixa independente de β) — decisão que envolve o Prof.
  Alexandre, pois mexe na representação do mecanismo de evasão do adversário
  no smart contract; ou (b) manter a fórmula atual e reformular o critério
  do Sanity Check 3 para usar features potencialmente mais robustas a esse
  efeito (ex.: razão eventos/volume, o "indicador de batch" já citado no
  PLANO §5.3.1, ou contagem de sub-eventos por timestep), que podem continuar
  monotônicas mesmo quando a amplitude bruta não é.
- **Sanity Check 2 (ρ=1.0, λ máximo) — limite conhecido, caracterizado por
  teste, não corrigido:** τ_Kendall(A,B) não atinge o limiar de 0.4 do PLANO
  nessa configuração (τ empírico ≈ 0.134 com os parâmetros do notebook
  `validacao_visual_desistencia.ipynb`, células 9-13). Causa: ρ=1.0 colapsa
  o desvio-padrão de Fonte A para perto de
  `_SIGMA_FRACAO_DELTA_T * delta_t` (≈4, contra ≈57 em ρ=0), e τ_Kendall,
  estatística de postos, perde poder discriminativo quando uma das
  variáveis tem variância própria muito pequena — mesmo com o θ interno da
  cópula pedindo dependência forte. Agora coberto por
  `test_sanity_check_2_rho_alto_caracteriza_limite_conhecido` e
  `test_tau_kendall_cai_monotonicamente_conforme_rho_aumenta`
  (`tests/test_integration.py`) — guardas de regressão do comportamento
  *conhecido*, não validação de que o Sanity Check 2 "passa". Não alterar
  `_SIGMA_FRACAO_DELTA_T`/`_amostrar_timestamps_desembolso` para forçar esse
  teste a "passar" sem validação com os orientadores. Pendência em aberto,
  não resolvida por esses testes: reformular o critério do Sanity Check 2 em
  termos de F1 do detector completo (§5.2.3) ou de features de amplitude de
  pico/entropia (§5.3.1), já que ambos dependem do pipeline de detecção
  ainda não implementado.
- **`gerar_par_de_classes_real` (`src/pipeline/geracao.py`) — fecha o loop
  do pipeline (grade → gerador real → HDF5), cinco decisões v0 tomadas ao
  ligar peças de tarefas anteriores que não previam exatamente como as
  outras ficariam:**
  1. Mapeamento `g` → `granularidade`/`unidade_alvo` de `ElectionModel`:
     `g` vira `granularidade` direto; quando `!= "pool"`, `unidade_alvo=0`
     — stub v0 (sempre válido, mas não é escolha definitiva de qual
     seção/município/estado mirar).
  2. `resultado_alvo`/`threshold_range` de `ElectionModel` não são campos
     de `GradeFatorial`/`ParametrosPopulacionaisStub` — ficam nos defaults
     da própria classe (`0.5`/`(0.2, 0.8)`). Limite de escopo desta camada
     de config, não esquecimento; se precisarem variar por combinação da
     grade, `GradeFatorial` precisa ganhar esses campos numa tarefa futura.
  3. **Achado de compatibilidade:** `copulas.bivariate.clayton.Clayton`
     (usada por `gerar_fonte_b` quando `tau_kendall != 0`) só aceita
     `int`/`np.random.RandomState` para `random_state` — rejeita
     `SeedSequence` e `np.random.Generator` diretamente, mais restrita que
     `gerar_fonte_b`'s próprio type hint (`RandomState = int | Generator |
     None`) sugere. Nenhum teste anterior pegou isso porque nenhum usava
     `SeedSequence` como seed até este pipeline existir
     (`config.derivar_seeds` produz `SeedSequence`). Contorno local em
     `geracao.py` (`_seed_sequence_para_int`, via
     `SeedSequence.generate_state`) — `layer2_copula/copula.py` não foi
     alterado.
  4. `derivar_seeds` só produz 2 seeds por janela (`seed_modelo`,
     `seed_fonte_b`) mas a classe negativa precisa de uma terceira fonte
     de aleatoriedade independente para `gerar_cenario_normal`'s
     `random_state_fonte_a`. Reusar `seed_modelo` diretamente (como uma
     primeira leitura da especificação desta tarefa sugeria) acoplaria
     deterministicamente o RNG do `ElectionModel` (Fonte C) ao tráfego de
     Fonte A — dois `np.random.default_rng` semeados com o MESMO valor
     compartilham o stream de bits do PCG64 subjacente, mesmo chamando
     métodos diferentes. Corrigido antes de codar: a seed de Fonte A da
     classe negativa é derivada localmente via `seed_modelo.spawn(1)[0]`
     — mesmo mecanismo de `SeedSequence.spawn` já usado no resto do
     pipeline, sem alterar o contrato de `derivar_seeds`.
  5. `window_id` em `escrever_run_hdf5` é por classe (`0..n_janelas-1` em
     cada uma), não um índice global — a coluna `classe` junto com
     `window_id` identifica uma janela unicamente nas 6 tabelas do HDF5.
  Achado adicional de implementação (PyTables, não do gerador): `pd.HDFStore.put(...,
  format="table")` com uma tabela de 0 linhas não escreve o grupo (falha
  silenciosa — `store.keys()` fica vazio, leitura posterior levanta
  `KeyError`). `escrever_run_hdf5` usa `format="fixed"` só para tabelas
  vazias (raro — só se nenhuma janela do run teve nenhum evento naquela
  fonte), `"table"` no caso normal.
- **`orquestrar_paralelo` (`src/pipeline/runner.py`) — paralelização por
  combinação×seed via `joblib.Parallel`/`backend="loky"`, não por janela
  individual.** Decisão apoiada em medição, não suposição: o benchmark de
  uma tarefa anterior (`scripts/benchmark_geracao.py`) mostrou ~7ms/janela,
  desvio baixo (~7% da média), sem cauda longa — o gargalo do grid completo
  (1.635 runs × 2.000 janelas) é volume de chamadas, não latência por
  janela; paralelizar dentro de uma combinação pagaria overhead de processo
  sem ganho correspondente. Cada worker (`_rodar_uma_combinacao`) roda uma
  combinação inteira (2.000 janelas, processadas serialmente por dentro,
  como já era) e SÓ RETORNA `(run_id, resultado)` — nunca escreve no
  `Manifesto` diretamente. Dois motivos, ambos no docstring da função: (1)
  `Manifesto` já documentava (antes desta tarefa) que não é
  thread-safe/process-safe para escrita concorrente — workers `loky` são
  processos separados, sem acesso à conexão SQLite já aberta no processo
  principal; abrir uma conexão própria por worker sobre o mesmo arquivo
  arriscaria `database is locked` sob escrita concorrente do SQLite; (2)
  mesmo que funcionasse, violaria essa garantia já assumida em outros
  lugares. `orquestrar_paralelo` agrega a lista completa de resultados de
  `Parallel` e escreve tudo serialmente, no processo principal, só depois —
  mesmo padrão two-phase (marcar `"running"` antes, `"success"`/`"failed"`
  depois) que `orquestrar` já usa, garantindo que uma interrupção no meio
  do lote paralelo deixe o manifesto num estado retomável (mesma lógica de
  "running órfã" já testada para a versão serial). `backend="loky"`
  (processos) em vez de threads: escolha estrutural, não reativa a um bug
  observado — `ElectionModel`/geradores usam RNGs com estado interno por
  instância; não há estado global compartilhado hoje, mas threads no mesmo
  processo Python teriam risco estrutural de introduzir esse acoplamento no
  futuro sem nenhum aviso de tipo ou teste que pegasse isso, enquanto
  processos separados isolam isso por construção. `n_jobs` default
  (`None` → `max(1, os.cpu_count() - 1)`) nunca cai abaixo de 1, mesmo se
  `os.cpu_count()` devolver `None` (ambientes onde a contagem não é
  detectável) ou a máquina tiver 1 núcleo só. Verificado empiricamente, não
  só assumido: `test_orquestrar_paralelo_usa_processos_separados_de_verdade`
  (`tests/test_pipeline_runner.py`) extrai o PID de cada worker (via
  `os.getpid()` embutido no `caminho_output` do fake — mutações de estado
  no fake dentro de um worker `loky` não se propagam de volta ao processo
  de teste, memória separada entre processos) e confirma que aparecem
  múltiplos PIDs distintos, todos diferentes do processo de teste — prova
  de que `n_jobs=2` roda em processos reais, não uma regressão silenciosa
  para execução serial disfarçada.
- **Tracking MLflow (`src/pipeline/tracking.py`) — sistema paralelo ao
  `Manifesto`, não um substituto.** `configurar_mlflow`/`registrar_run_mlflow`
  dão um painel consultável (`mlflow.search_runs()`) dos parâmetros e
  métricas de cada `run_id` do lote, com backend SQLite local — mas
  `Manifesto` continua sendo a única fonte de verdade para resumabilidade
  (`pendentes()`); os dois sistemas são escritos a partir do mesmo lugar em
  `orquestrar`/`orquestrar_paralelo`, mas não se comunicam entre si.
  **Decisão deliberada — sucesso E falha geram run do MLflow, seguindo
  sugestão do prompt da tarefa:** se só sucessos aparecessem, o MLflow
  daria uma vista parcial do lote (combinações que falharam simplesmente
  não apareceriam), forçando quem olha o painel a cruzar com o SQLite mesmo
  assim — contrariando o propósito de painel único. Falhas recebem tag
  `status="failed"` e a mensagem de erro como parâmetro (truncado no limite
  real de `mlflow.log_param`, `mlflow.utils.validation.MAX_PARAM_VAL_LENGTH`
  — 6000 chars na versão instalada, não um valor adivinhado; truncamento
  explícito em vez de deixar o comportamento da lib decidir, que levantaria
  erro de validação em vez de truncar silenciosamente). `caminho_output` é
  gravado como TAG, não como artifact — os HDF5s são grandes demais para
  copiar para dentro do storage do MLflow e já têm seu próprio local
  (`diretorio_output` do gerador real).
  **Paridade de parâmetros entre sucesso e falha, verificada explicitamente
  em teste (`test_registrar_run_mlflow_sucesso_e_falha_tem_mesmas_colunas_de_parametro`,
  `tests/test_pipeline_tracking.py`):** os dois branches (`orquestrar` e o
  laço serial de `orquestrar_paralelo`) passam o MESMO dict mesclado
  (`params_completos` — eixos da grade + `vars(parametros_stub)` +
  `vars(populacionais)`) para `registrar_run_mlflow`, nunca uma versão
  parcial no branch de falha. Sem essa paridade, uma linha
  `status="failed"` em `mlflow.search_runs()` teria colunas de parâmetro
  faltando/`NaN` que uma linha `status="success"` tem, impedindo
  filtrar/comparar as duas de forma confiável — quebraria o objetivo
  declarado de painel único. Em `orquestrar_paralelo`, isso exigiu uma
  mudança em `_rodar_uma_combinacao`: o worker agora devolve
  `params_completos` dentro do dict de resultado (não só implícito no
  escopo do worker), porque o laço serial que escreve no manifesto/MLflow
  depois de `Parallel` retornar só tem `resultados` em mãos, não as
  combinações originais.
  **Escrita do MLflow segue a mesma regra "sempre serial, processo
  principal" já documentada para o `Manifesto`:** nenhum worker `loky` abre
  sessão MLflow própria — mesmo raciocínio de `orquestrar_paralelo`
  (workers são processos separados, sem acesso ao estado do processo
  principal; abrir recursos por worker arriscaria inconsistência sem
  ganho correspondente).
  `orquestrar`/`orquestrar_paralelo` NÃO recebem parâmetro novo de URI do
  MLflow — assume-se que `configurar_mlflow()` já foi chamada por quem
  monta o pipeline antes de invocar qualquer uma das duas, mesma
  responsabilidade de setup que já cabia a quem constrói `Manifesto(caminho)`.
  **Achado de ambiente, não do código do projeto:** o tracking URI default
  do MLflow (quando `configurar_mlflow` não é chamado) é
  `sqlite:///mlflow.db`, relativo ao `cwd` — não `./mlruns/` como se poderia
  supor de versões antigas da lib. Os testes em `test_pipeline_runner.py`
  usam uma fixture `autouse=True` (`_tracking_uri_isolado`) apontando para
  `tmp_path` especificamente para evitar que rodar a suíte crie um
  `mlflow.db` poluindo a raiz do repositório.

## Estilo
- Código Python com type hints
- Docstrings estilo NumPy
- Sem comentários óbvios; comentar apenas decisões não triviais