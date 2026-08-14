# PLANO TCC/ARTIGO V4.1

# Detecção de Incentivos Econômicos Adversariais em Eleições via Smart Contracts com Privacidade Nativa

> **Versão 4.1 Final** — documento de referência pré-reunião com orientadores
> 
> 
> *Changelog v4.1:* Mesa removido de OE3, C3, §5.2.2 e cronograma; McNemar corrigido ("três quartos ≥ 3/4"); nota de Fontes derivadas em §5.4.1; nota GPU (M3) em §5.3.2; nota parâmetros ABM em §5.2.2; alerta de datas no cronograma.
> 

## Integração de Fontes Heterogêneas Indiretas sob Observabilidade Limitada

### Plano de TCC/Artigo — Versão 3 (Refatorada)

---

## 1. Problema

Blockchains programáveis com privacidade nativa — implementando zero-knowledge proofs, execução off-chain e oráculos descentralizados — permitem a construção de contratos inteligentes capazes de distribuir incentivos econômicos condicionados a eventos públicos verificáveis, como resultados eleitorais agregados.

Esse mecanismo apresenta três propriedades que o distinguem da compra de voto tradicional e o tornam especialmente preocupante. Primeiro, resolve o **commitment problem** clássico: na compra de voto convencional, o comprador não pode garantir que o eleitor não deserte após receber o pagamento, pois o voto é secreto. No modelo baseado em smart contract, o pagamento só é executado se o resultado eleitoral agregado verificável on-chain for coerente com o comportamento esperado — o contrato substitui a confiança pela execução automática. Segundo, o esquema é **anônimo e descentralizado**: participantes não são identificáveis na camada de execução, e não há intermediário controlável. Terceiro, é **operacionalmente escalável**: um único contrato pode coordenar incentivos para milhares de eleitores sem interação manual.

A consequência direta para detecção é que métodos clássicos de anomaly detection em blockchain — baseados em análise de grafos de transação, rastreamento de carteiras ou inspeção de estado de contratos — tornam-se estruturalmente inviáveis. A privacidade nativa não é um obstáculo contornável: ela elimina por design os dados de entrada sobre os quais esses métodos operam.

Esse cenário representa uma instância inédita do **Open Challenge 2** identificado por Shevchuk et al. (2025) em revisão sistemática de 363 artigos sobre anomaly detection em blockchain: a integração de fontes heterogêneas indiretas não como enriquecimento opcional da análise, mas como única estratégia estruturalmente viável quando dados transacionais diretos são inacessíveis.

---

## 2. Pergunta de Pesquisa

Dado que a privacidade nativa elimina o acesso a dados transacionais diretos, em que medida a integração de fontes heterogêneas indiretas — eventos de fronteira entre camadas, interações com oráculos e padrões temporais correlacionados a eventos eleitorais externos — permite detectar comportamento adversarial associado à compra de votos via smart contracts, e quais são os limites estruturais dessa detectabilidade sob diferentes níveis de observabilidade e capacidades adversariais?

---

## 3. Objetivo Geral

Desenvolver e avaliar um framework de detecção de padrões adversariais em ambientes blockchain com privacidade nativa, baseado exclusivamente em fontes observáveis indiretas, e caracterizar formalmente os limites estruturais de detectabilidade em função das variáveis de observabilidade do sistema e das capacidades do adversário — incluindo a quantificação do custo mínimo de ataque sob diferentes regimes de detectabilidade.

---

## 4. Objetivos Específicos

**OE1.** Construir um modelo de ameaça formal para o cenário adversarial eleitoral, especificando: adversário com orçamento limitado, função de utilidade e estratégias de evasão realistas — incluindo fragmentação de transações (batching); mecanismo de incentivo condicional via oráculo que resolve o commitment problem; garantias de privacidade da arquitetura subjacente; e superfícies de observação disponíveis ao detector.

**OE2.** Definir formalmente o modelo de observabilidade do sistema, estabelecendo uma taxonomia das fontes indiretas disponíveis, suas propriedades estatísticas individuais e estrutura de dependência sob comportamento adversarial versus normal, como função do nível de privacidade π.

**OE3.** Implementar um gerador de dados sintéticos em duas camadas complementares: modelagem agent-based para geração de populações heterogêneas de eleitores com propensão configurável a aceitar incentivo, e cópula paramétrica para preservação da estrutura de dependência entre fontes observáveis — com validação por sanity checks sob condições extremas. A escolha das bibliotecas de implementação será definida em acordo com os orientadores durante as semanas 5–8.

**OE4.** Projetar e implementar um pipeline de detecção com três abordagens complementares: baseline estatístico sobre séries temporais, modelo supervisionado com interpretabilidade via SHAP, e modelo não supervisionado baseado em redes neurais recorrentes para captura de dependências temporais.

**OE5.** Executar avaliação experimental em dois níveis: (a) design fatorial principal variando nível de privacidade, granularidade eleitoral, atraso de divulgação, intensidade adversarial e coordenação; (b) experimento de robustez adversarial variando capacidade de batching — com estudo de ablação completo de combinações de fontes e critério de sucesso falsificável.

**OE6.** Caracterizar os limites estruturais de detectabilidade: identificar a fronteira empírica e o upper bound teórico da detectabilidade em função de π, e quantificar o custo mínimo de ataque — valor normalizado pago pelo adversário para atingir P(ativação do contrato) > 0,5 — em função das variáveis do sistema.

---

## 5. Metodologia

### 5.1 Parte 1 — Modelo de Ameaça e Observabilidade Formal

**Duração:** Semanas 1–4 (~160h)

### 5.1.1 Modelo de Ameaça Formal

O modelo de ameaça seguirá o formato adversary model padrão em venues de segurança de sistemas (IEEE S&P, USENIX Security), não frameworks de engenharia como STRIDE. Será composto por quatro elementos:

**Definição do adversário A:**

```
Adversário A:
  Objetivo:     maximizar P(ativação do contrato | resultado R)
                minimizando custo B e detectabilidade
  Orçamento:    B unidades monetárias (limitado, racional)
  Função util.: U(eleitor_i) = recompensa × P(resultado_R) - custo_oportunidade
  Conhecimento: arquitetura pública da blockchain, calendário eleitoral,
                interface pública do oráculo, estrutura de agregação dos
                resultados eleitorais
  Capacidades:
    ├── Deploy de smart contract com ZK-proof na camada de execução privada
    ├── Interação com oráculo descentralizado para ingestão de resultado R
    ├── Movimentação de ativos entre L1 e L2 via bridge
    └── Fragmentação de transações (batching): β ∈ {1, 5, 20} saques
        por evento, para reduzir amplitude dos picos observáveis na Fonte A
  Limitações:   não controla o oráculo, não controla a divulgação
                de resultados eleitorais, não conhece o algoritmo do detector
```

**Resolução do commitment problem:**

O contrato resolve o problema clássico de credibilidade do suborno. Na compra de voto tradicional, o eleitor pode receber pagamento e votar de forma diferente — o comprador não tem mecanismo de enforcement. No modelo baseado em smart contract com oráculo, a execução do pagamento é condicionada ao resultado agregado verificável: `if oracle.result == R: transfer(pool, reward_per_voter)`. O eleitor individual não controla o resultado agregado, mas sua participação em massa — coordenada pelo incentivo financeiro — é o mecanismo de comprometimento. Isso justifica o caráter escalável e a resistência à deserção individual.

**Definição das garantias de privacidade:**

Nível de privacidade π ∈ [0,1]: fração de informação interna inacessível ao observador externo. Em π = 0 (blockchain pública transparente), todos os dados de transação são observáveis. Em π → 1 (privacidade completa), apenas eventos de fronteira e interações de oráculo são detectáveis.

**Propriedades de segurança alvo do detector:**

O detector D fornece garantia de detecção com probabilidade ≥ δ dado comportamento adversarial com intensidade λ > λ*(π), com taxa de falso positivo ≤ α em comportamento normal.

### 5.1.2 Modelo de Observabilidade

Definição formal da função de observabilidade O(π) que mapeia nível de privacidade para conjunto de superfícies observáveis:

```
Fonte A — Eventos de fronteira (L1):
  Observável: timestamp, volume agregado de entrada/saída entre camadas
  Inacessível: identidade dos participantes internos, lógica do contrato,
               valores individuais
  Resolução temporal: bloco (~segundos a minutos)
  Impacto do batching: β fragmentações reduzem amplitude dos picos por
                       fator 1/β, aumentando o limiar de detecção

Fonte B — Interações com oráculo:
  Observável: timestamp de ativação, endereço do contrato de oráculo na L1
  Inacessível: qual resultado ativou, qual ação interna foi tomada
  Resolução temporal: evento discreto (episódico)

Fonte C — Dados eleitorais externos:
  Observável: resultado agregado por granularidade g
              g ∈ {seção individual, município, estado}
  Resolução temporal: variável (Δt de divulgação oficial)

Fontes derivadas — Padrões de coordenação:
  Sinal: correlação temporal entre A, B e C
  Poder discriminativo: mais alto que qualquer fonte isolada
  Requisito: janela de observação suficiente para estimação estatística
```

---

### 5.2 Parte 2 — Gerador de Dados Sintéticos

**Duração:** Semanas 5–8 (~160h)

### 5.2.1 Justificativa Formal da Necessidade de Dados Sintéticos

O uso de dados sintéticos é estruturalmente necessário, não apenas metodologicamente conveniente, por três razões:

1. Ataques do tipo descrito não possuem registros históricos documentados em blockchain com privacidade nativa
2. Dados de ataques reais seriam inacessíveis por design — a privacidade que torna o ataque difícil de detectar também torna impossível coletar dados supervisionados reais
3. A geração sintética permite controle preciso das variáveis do design fatorial, condição necessária para a caracterização dos limites de detectabilidade

### 5.2.2 Arquitetura em Duas Camadas do Gerador

O gerador opera em duas camadas complementares com responsabilidades distintas:

**Camada 1 — Simulação de População de Agentes:**

Responsável pela geração da população de eleitores com heterogeneidade realista:

```
Estrutura conceitual do agente eleitor:
propensao:           amostrado de Beta(α, β) configurável
utility_threshold: valor mínimo de recompensa para aceitar
racional:            segue utilidade esperada (maioria) ou
comportamento ruidoso (minoria)

Parâmetros populacionais configuráveis:
  n_agentes:    [50, 500, 5.000]   (tamanho do grupo participante)
  alpha_beta:   [distribuição Beta configurável por cenário]
  prop_racional:[0.7, 0.9, 1.0]   (fração de agentes racionais)
```

A simulação por etapas gera: padrões de adesão ao incentivo, resultados eleitorais emergentes por seção e o comportamento agregado que se manifesta nas fontes observáveis. A implementação utilizará Python com bibliotecas a serem definidas em acordo com os orientadores durante as semanas 5–8.

*Ju**stificativa da abordagem agent-based:*** A Camada 1 adota simulação agent-based em lugar de modelos paramétricos mais simples (como um modelo Beta-Binomial para taxas de adesão) por uma razão metodológica central: o sinal de detecção do framework depende dos **padrões temporais** dos eventos nas fontes observáveis, não apenas do volume agregado. Um modelo Beta-Binomial capturaria corretamente a distribuição marginal de aceitação do incentivo, mas não a dinâmica sequencial pela qual agentes heterogêneos — com distinção explícita entre racionais e ruidosos — decidem ao longo do tempo. Essa dinâmica é o que produz o clustering temporal realista na Fonte A (picos de eventos de fronteira em janelas pré e pós-divulgação eleitoral). Substituí-la por um processo de chegada simples com taxa calibrada subestimaria a dificuldade real do problema de detecção, comprometendo a validade externa dos experimentos.

***Nota** — parâmetros populacionais não-fatoriais:* n_agentes, alpha_beta e prop_racional não fazem parte do design fatorial principal. Devem ser fixados por consenso com os orientadores nas semanas 5–6, com análise de sensibilidade para ao menos dois valores de n_agentes.

**Camada 2 — Estrutura de Dependência entre Fontes (Cópula Paramétrica):**

Responsável por preservar a correlação entre as fontes A, B e C — não garantida apenas pela camada de simulação de agentes:

```
Modo adversarial:
  - Eventos da Fonte A: pico concentrado em janela pós-resultado
  - Ativações da Fonte B: correlacionadas com divulgação via cópula Clayton
  - Dados da Fonte C: resultado eleitoral emergente da simulação
  - Estrutura de dependência A↔B: τ_Kendall > 0.4 no modo adversarial
  - Fragmentação: β saques por evento, distribuídos em janela de [0, Δt/β]

Modo normal (baseline):
  - Processos de Poisson independentes para A e B
  - τ_Kendall ≈ 0 (independência entre fontes)
```

**Parâmetros de controle do gerador:**

```
π ∈ {0.50, 0.75, 0.90, 0.95}    (nível de privacidade)
g ∈ {seção, município, estado}    (granularidade eleitoral)
Δt ∈ {0h, 2h, 24h}              (atraso de divulgação)
λ ∈ {baixa, média, alta}          (intensidade adversarial)
ρ ∈ {0.0, 0.5, 1.0}             (grau de coordenação)
β ∈ {1, 5, 20}                   (fragmentação — experimento de robustez)
```

### 5.2.3 Validação do Gerador por Sanity Checks

O gerador será validado sob condições extremas antes da geração dos datasets de produção:

```
Sanity Check 1 — Ausência total de ataque (λ = 0):
  Esperado: nenhum detector supera baseline aleatório (F1 ≈ 0.5 ± ε)
  Critério de falha: qualquer detector com F1 > 0.6 indica vazamento
                     de sinal artificial no gerador

Sanity Check 2 — Ataque máximo e coordenação perfeita (λ = máx, ρ = 1.0):
  Esperado: todos os detectores com C7 atingem F1 > 0.9
  Critério de falha: F1 < 0.8 indica ausência de sinal discriminativo
                     nas features — problema na camada de geração

Sanity Check 3 — Batching mínimo (β = 1) vs. máximo (β = 20):
  Esperado: degradação monotônica do F1 conforme β aumenta
  Critério de falha: degradação não-monotônica indica inconsistência
                     no modelo de fragmentação

Sanity Check 4 — Independência das fontes no modo normal:
  Esperado: τ_Kendall(A, B) < 0.1 nos datasets de comportamento normal
  Critério de falha: correlação espúria indica contaminação entre modos
```

### 5.2.4 Construção dos Datasets e Protocolo de Split

**Composição dos datasets:**

Para cada combinação do design fatorial principal (π × g × Δt × λ × ρ), serão gerados:

- 1.000 janelas temporais de comportamento adversarial (classe positiva)
- 1.000 janelas temporais de comportamento normal (classe negativa)
- 5 sementes aleatórias distintas (para análise de variância entre runs)

A classe negativa é explicitamente construída e balanceada para evitar positivity bias — cenários sem ataque recebem o mesmo tratamento de feature engineering que os cenários adversariais.

**Protocolo de split com separação temporal:**

```
Split: 70% treino / 15% validação / 15% teste

Separação temporal obrigatória:
  - Janelas de treino precedem cronologicamente as de validação e teste
  - Janelas de teste correspondem ao período "pós-eleição" simulado
  - Nenhuma janela de treino sobrepõe temporalmente as de teste
  (previne data leakage de padrões sazonais ou tendências temporais)

Total de datasets gerados:
  Fatorial principal: 4(π) × 3(g) × 3(Δt) × 3(λ) × 3(ρ) × 5(seeds) = 1.620 datasets
  Robustez (β):       1 configuração base × 3(β) × 5(seeds) = 15 datasets adicionais
```

---

### 5.3 Parte 3 — Pipeline de Detecção

**Duração:** Semanas 9–13 (~200h)

### 5.3.1 Feature Engineering sob Observabilidade Limitada

As features são construídas exclusivamente a partir das fontes observáveis indiretas, organizadas em três camadas de poder discriminativo:

**Camada 1 — Sinais de presença (poder discriminativo baixo):**

- Contagem de eventos de fronteira por janela temporal
- Presença/ausência de ativação de oráculo na janela
- Volume agregado de entrada/saída por período
- Indicador de batch: razão entre número de eventos e volume total na Fonte A

**Camada 2 — Sinais de padrão (poder discriminativo médio):**

- Variação relativa de volume em relação à média histórica (z-score por janela)
- Densidade temporal de ativações de oráculo
- Razão entrada/saída em janelas pré e pós-divulgação eleitoral
- Entropia temporal dos eventos por janela
- Amplitude normalizada dos picos: sensível à fragmentação por β

**Camada 3 — Sinais de coordenação (poder discriminativo alto):**

- Correlação cruzada entre Fontes A e B em lag temporal τ
- τ_Kendall(A, B) estimado por janela deslizante
- Coeficiente de concentração temporal relativo ao evento eleitoral (Fonte C)
- Mutual information entre ativações e resultado eleitoral por granularidade g
- Desvio da distribuição de intervalos entre eventos vs. Poisson esperado

### 5.3.2 Modelos de Detecção

**M1 — Baseline estatístico:**

Detecção por limiar em séries temporais usando z-score e CUSUM sobre volume da Fonte A. Sem treinamento, sem parâmetros aprendidos — serve como lower bound de desempenho e como referência para o critério de sucesso falsificável. Representa o estado da prática em sistemas de monitoramento simples.

**M2 — Modelo supervisionado (XGBoost + SHAP):**

Gradient Boosted Trees treinado sobre features das três camadas com rótulos de treino. Interpretabilidade via SHAP values — requisito crescente em venues de segurança para justificar decisões do detector. A análise de SHAP identifica quais features de quais fontes mais contribuem para a detecção em cada regime de privacidade π.

**M3 — Modelo não supervisionado (LSTM Autoencoder):**

Autoencoder com encoder e decoder LSTM treinado exclusivamente em dados normais. Detecção via limiar de erro de reconstrução. Não requer dados rotulados de ataque — realista para o cenário operacional onde ataques reais são raros. Captura dependências temporais entre eventos que features estáticas não capturam.

*Nota de implementação:* O design fatorial com ablação gera 1.620 × 7 = 11.340 runs de M3. Esse volume requer acesso a GPU. Infraestrutura a definir com o orientador de IA nas semanas 9–10, antes de iniciar a implementação.

**Justificativa da tríade:** Os três modelos cobrem o espectro de suposições sobre disponibilidade de dados rotulados — crítico para aplicabilidade real. Uma autoridade eleitoral implementando o sistema pode ou não ter exemplos históricos de ataque.

---

### 5.4 Parte 4 — Avaliação Experimental e Caracterização de Limites

**Duração:** Semanas 14–17 (~160h)

### 5.4.1 Estudo de Ablação por Combinação de Fontes

Para cada modelo M1, M2, M3, avaliar as sete combinações de fontes:

```
C1: Apenas Fonte A (eventos de fronteira)
C2: Apenas Fonte B (interações com oráculo)
C3: Apenas Fonte C (dados eleitorais externos)
C4: A + B
C5: A + C
C6: B + C
C7: A + B + C (integração completa)
```

Hipótese central a testar: C7 supera significativamente C1, C2 e C3 isolados — evidência empírica de que a integração multimodal é necessária (não apenas útil) para detecção sob privacidade nativa. Essa é a resposta empírica direta ao Open Challenge 2 de Shevchuk et al. (2025).

*Nota — Fontes derivadas na ablação:* Os padrões de coordenação (A↔B↔C) definidos em §5.1.2 não aparecem como combinação isolada na ablação porque, por definição, exigem pelo menos duas fontes primárias. Em C7, os sinais da Camada 3 (correlação cruzada, τ_Kendall, mutual information entre fontes) capturam implicitamente essa coordenação.

### 5.4.2 Design Fatorial Principal

Para cada combinação (π × g × Δt × λ × ρ), aplicar os três modelos nas sete combinações de fontes. Para cada configuração, medir o conjunto de métricas definido na seção 5.4.4. Análise de variância (ANOVA fatorial) para identificar quais variáveis têm efeito significativo na detectabilidade e suas interações.

### 5.4.3 Experimento de Robustez Adversarial (Batching)

Experimento separado do fatorial principal, fixando a configuração de melhor desempenho do experimento anterior:

```
Configuração base: melhor (π, g, Δt, λ, ρ) do fatorial principal
Variável independente: β ∈ {1, 5, 20}
Hipótese: degradação monotônica do F1 conforme β aumenta
Objetivo: quantificar o custo de evasão do adversário via batching
          e identificar limiar de β acima do qual a detecção colapsa
```

Este experimento responde à pergunta operacional: *"Se o adversário souber que os eventos de fronteira são observados, quanto esforço de fragmentação é necessário para tornar o ataque indetectável?"*

### 5.4.4 Métricas de Avaliação

Para cada configuração do design fatorial e do experimento de robustez:

**Métricas de desempenho do detector:**

1. **F1-score** — métrica principal de comparação entre abordagens
2. **Precisão e Recall** — decomposição do F1 para análise de trade-offs
3. **AUROC** — desempenho independente de limiar
4. **Taxa de falso positivo** — crítica para aplicabilidade operacional

**Métrica de segurança:**

1. **Custo mínimo do ataque** — valor normalizado pago pelo adversário para atingir P(ativação do contrato) > 0,5, em função de π, λ, g e β:

```
C_min(π, λ, g, β) = min{B : P(contrato ativado | B, π, λ, g, β) > 0.5}

Normalizado por: número de seções-alvo × tamanho médio do grupo por seção

Interpretação: custo unitário de atingir resultado eleitoral esperado em uma seção
               abaixo do qual o ataque não é economicamente viável
```

Esta métrica conecta o desempenho de detecção a uma afirmação de segurança acionável: para cada regime (π, g, Δt), existe um limiar C_min abaixo do qual o ataque é indetectável E economicamente inviável — essa interseção define o espaço de "segurança prática".

### 5.4.5 Critério de Sucesso Falsificável

O trabalho atinge seu objetivo central se:

> *"A integração completa de fontes (C7) supera a melhor fonte isolada (max(C1, C2, C3)) em F1-score com p < 0,05 pelo teste de McNemar em pelo menos três quartos (≥ 3/4) das configurações de nível de privacidade π testadas, para pelo menos um dos três modelos de detecção."*
> 

**Teste de McNemar:** escolhido por ser adequado para comparação de classificadores em dados emparelhados, sem exigência de normalidade — apropriado para os datasets sintéticos gerados.

**Interpretação:** se o critério for atingido, o trabalho fornece evidência empírica de que a integração multimodal é necessária para detecção sob privacidade nativa — resposta direta e falsificável ao Open Challenge 2. Se não for atingido, caracteriza os limites do framework proposto com a mesma precisão.

### 5.4.6 Caracterização dos Limites Estruturais de Detectabilidade

**Limite empírico — fronteira de detectabilidade:**

Para cada nível de privacidade π, identificar λ*(π): valor mínimo de intensidade adversarial abaixo do qual nenhum modelo, mesmo com C7, produz F1 acima do baseline por margem estatisticamente significativa. A curva λ*(π) é a **fronteira de detectabilidade** — resultado central do trabalho.

**Upper bound teórico — limite de informação:**

Calcular a mutual information I(C7; Y) entre o conjunto completo de fontes e o label adversarial, em função de π. Quando I cai abaixo de um limiar crítico ε, nenhum detector pode superar o acaso independentemente do modelo — bound teórico intransponível.

**Curva de custo-detectabilidade:**

Plotar C_min em função de π para os diferentes valores de g — a curva de segurança prática que responde: *"Quão caro deve ser o ataque para que o adversário fique acima do limiar de detecção?"* Este resultado tem implicação direta para policy: identifica as condições de divulgação de resultados (g e Δt) que maximizam o custo de ataque para um dado nível de privacidade.

**Impacto das variáveis institucionais:**

Identificar os valores críticos g* e Δt* além dos quais a detectabilidade colapsa para cada π — produzindo recomendações acionáveis: *"Divulgação com granularidade inferior a g* ou atraso superior a Δt* torna o ataque indetectável pelo framework proposto com custo C_min."*

---

## 6. Cronograma

| Semana | Período | Atividade | Horas |
| --- | --- | --- | --- |
| 1–2 | Jul 11–Jul 24 | Literatura complementar; rascunho do adversary model | 80h |
| 3–4 | Jul 25–Ago 07 | Adversary model completo (commitment problem + batching); validação co-orientador | 80h |
| 5–6 | Ago 08–Ago 21 | Arquitetura e implementação do gerador (camada de agentes + cópula); definição dos parâmetros populacionais com orientadores | 80h |
| 7–8 | Ago 22–Set 04 | Sanity checks; geração dos datasets; protocolo de split | 80h |
| 9–10 | Set 05–Set 18 | Feature engineering; implementação M1 e M2 (XGBoost + SHAP); estimativa de custo computacional de M3 e definição de infraestrutura | 80h |
| 11–13 | Set 19–Out 09 | Implementação M3 (LSTM Autoencoder); experimentos iniciais; início da redação por seções concluídas | 120h |
| 14–15 | Out 10–Out 23 | Design fatorial principal; ablação de fontes; experimento de robustez (β); custo mínimo (C_min) | 80h |
| 16 | Out 24–Out 30 | Limites de detectabilidade (λ*(π)); redação final TCC + rascunho do artigo; preparação da defesa | 40h¹ |

> ¹ Semana de alta intensidade — análise dos limites, escrita e preparação ocorrem em paralelo. A redação deve ser iniciada progressivamente a partir da semana 11, à medida que seções são concluídas.
> 

**Total:** 640h em 16 semanas de calendário (Jul 11–Out 30).(

OBS: Estimativa mínima, assumindo 40h/semana.Somando preparação antes de 11 de julho + tempo extra alocado em fases critícas, total pode ultrapassar +800h

**Pós-defesa:**

| Período | Atividade | Deadline |
| --- | --- | --- |
| Out 31–Nov 09 | Conversão para formato IEEE S&P; análise de robustez finalizada | **10/11 — IEEE S&P 2027** |
| Nov 10–Jan 13 | Incorporação de reviews; ampliação se necessário | **14/01 — IEEE ICBC 2027** |
| Jan 14–Jan 25 | Adaptação para USENIX Security | **26/01 — USENIX Security 2027** |

---

## 7. Contribuições Esperadas

**C1 — Modelo de ameaça formal para compra de votos via smart contracts com privacidade nativa**

Primeira especificação formal do cenário adversarial incluindo: resolução do commitment problem via execução condicional em smart contract, estratégias realistas de evasão por batching, e definição rigorosa das garantias de privacidade e superfícies de observação. Responde a lacuna identificada nos 363 artigos revisados por Shevchuk et al. (2025).

**C2 — Framework de observabilidade indireta em blockchain com privacidade**

Taxonomia formal das fontes observáveis disponíveis quando dados transacionais diretos são inacessíveis, com caracterização de suas propriedades estatísticas individuais, estrutura de dependência, e sensibilidade à capacidade de batching do adversário.

**C3 — Gerador sintético em duas camadas com validação formal**

Gerador parametrizável combinando simulação agent-based para populações heterogêneas de eleitores com cópula paramétrica para preservação da estrutura de dependência entre fontes — validado por sanity checks sob condições extremas e com protocolo de split temporal. Artefato open-source para reprodutibilidade conforme política de open science da USENIX Security.

**C4 — Evidência empírica da necessidade de integração multimodal**

Estudo de ablação completo (7 combinações de fontes × 3 modelos × design fatorial) demonstrando empiricamente que C7 supera fontes isoladas com significância estatística — resposta direta e falsificável ao Open Challenge 2 de Shevchuk et al. (2025) com critério McNemar especificado.

**C5 — Caracterização dos limites estruturais e curva de custo-detectabilidade**

Fronteira formal λ*(π) entre o espaço de parâmetros onde detecção é viável e onde colapsa, combinada com a curva C_min(π, g, Δt) de custo mínimo do ataque — produzindo recomendações concretas para reguladores eleitorais sobre granularidade e atraso de divulgação de resultados.

---

## 8. Considerações Éticas

O estudo será conduzido exclusivamente em ambiente simulado, com foco em análise defensiva. Nenhum dado real de eleitores, transações ou sistemas eleitorais será utilizado de forma operacional. Todos os dados gerados são sintéticos e não permitem identificação de pessoas reais.

O gerador de dados sintéticos e o código de detecção serão disponibilizados publicamente para reprodutibilidade, em conformidade com a política de open science da USENIX Security e as práticas recomendadas pelo IEEE S&P.

Os resultados têm caráter acadêmico e visam contribuir para o fortalecimento da segurança de processos eleitorais, não para a exploração de vulnerabilidades. A pesquisa não envolve seres humanos, dados pessoais ou sistemas computacionais reais.