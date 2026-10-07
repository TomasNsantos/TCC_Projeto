# Consolidação Dia 8 — Semana 9-10

**Escopo:** Dias 3 (M1/CUSUM), 6 (M2/XGBoost+SHAP) e 7 (benchmark de
custo M3). Material factual para discussão com os orientadores — **não**
é a lista fechada de pendências nem o texto de §5.3 (Dia 9). Sem
McNemar (§5.4.5) nesta etapa — fora de escopo até o design fatorial
completo (§5.4, Semanas 14-17).

---

## 1. Achados de Δt=0 pós-fix

Contexto: a auditoria de 16/09 documentou que, no dataset pré-fix, a
classe negativa tinha 100% das janelas vazias em Fonte A/B quando
`delta_t=0.0` — separador trivial de classe. O bug foi corrigido no v3
(commit `782b941`). Esta seção confirma se `Δt=0` ainda se comporta de
forma anomalamente diferente dos outros dois níveis, agora com M1 e M2
rodando sobre o dataset corrigido.

### 1.1 M1 (CUSUM) — reexecutado nesta etapa (`scripts/m1_cusum_grid_completo.py`, determinístico, mesmos parâmetros do Dia 3)

**Achado principal: M1 não distingue `Δt=0.0` de `Δt=24.0` — as duas
produzem métricas IDÊNTICAS, célula a célula**, em todos os 15
estratos de `recompensa × π` comuns aos dois níveis. Isso não é
coincidência nem sinal residual do bug de `Δt=0` — é consequência de
`binning._janela_teto`: ambos os níveis usam o mesmo teto de janela
(24 timesteps), então o CUSUM opera sobre séries binadas idênticas em
tamanho e distribuição de referência.

`Δt=2.0`, em contraste, tem FPR consistentemente muito menor (0,0015 a
0,01, contra 0,126 a 0,239 em Δt∈{0,24}) — resultado de uma janela de
observação mais curta, não de um artefato do fix.

| subgrupo | Δt | recompensa | π | precisão | recall | F1 | FPR |
|---|---|---|---|---|---|---|---|
| positiva_ativa | 0.0 | 0.5 | 0.00 | 0,4312 | 1,0 | 0,6025 | 0,1260 |
| positiva_ativa | 24.0 | 0.5 | 0.00 | 0,4312 | 1,0 | 0,6025 | 0,1260 |
| positiva_ativa | 2.0 | 0.5 | 0.00 | 0,9845 | 1,0 | 0,9922 | 0,0015 |
| positiva_ativa | 0.0 | 1.0 | 0.00 | 0,8881 | 1,0 | 0,9407 | 0,1260 |
| positiva_ativa | 24.0 | 1.0 | 0.00 | 0,8881 | 1,0 | 0,9407 | 0,1260 |
| positiva_ativa | 2.0 | 1.0 | 0.00 | 0,9985 | 1,0 | 0,9993 | 0,0015 |
| positiva_ativa | 0.0 | 0.5 | 0.95 | 0,2855 | 1,0 | 0,4442 | 0,2390 |
| positiva_ativa | 24.0 | 0.5 | 0.95 | 0,2855 | 1,0 | 0,4442 | 0,2390 |
| positiva_ativa | 2.0 | 0.5 | 0.95 | 0,9317 | 1,0 | 0,9646 | 0,0070 |

`positiva_inativa`: `vp=0` em TODOS os estratos, para os três níveis de
Δt — CUSUM nunca detecta essa subclasse (esperado: Fonte A/B vazias por
construção quando `contrato_ativado=False`). `precisao=0.000000` (não
`NaN`) sempre que `fp>0` no mesmo estrato (fórmula `vp/(vp+fp)` com
`vp=0`); `recall`/`F1` ficam `NaN` quando `n_positivas=0` (não há
`positiva_inativa` amostrada em `recompensa≥1.0`, mesmo padrão já
documentado no Dia 6).

Tempo de reexecução: 475,5s (7,93min) para o grid completo — dentro da
faixa já observada para M1.

### 1.2 M2 (XGBoost) — reaproveitado do Dia 6 (sessão anterior), sem reexecução

**`Δt=0.0`: F1/AUROC/precisão/recall = 1,000 em TODOS os 5 níveis de
π, para os dois modelos (C7 completo e C4 sem Fonte C).** Nenhuma
degradação, nem mesmo em C4/π=0,95 (onde `Δt=2.0` já mostrava
precisão caindo para 0,491). `Δt=24.0` também dá 1,000 em todos os
casos — os dois níveis se comportam de forma idêntica entre si (mesmo
padrão de M1: teto de janela compartilhado).

`Δt=2.0` é o único nível onde há qualquer degradação observável em C4,
crescente com π (precisão de 0,933 em π=0,00 a 0,491 em π=0,95).

| Modelo | Δt | recompensa | π | precisão | recall | F1 | AUROC |
|---|---|---|---|---|---|---|---|
| C7 (completo) | 0.0 | 0.5 | 0,00–0,95 | 1,000 | 1,000 | 1,000 | 1,000 |
| C4 (sem Fonte C) | 0.0 | 0.5 | 0,00–0,95 | 1,000 | 1,000 | 1,000 | 1,000 |
| C4 (sem Fonte C) | 2.0 | 0.5 | 0,00 | 0,875 | 1,000 | 0,933 | 1,000 |
| C4 (sem Fonte C) | 2.0 | 0.5 | 0,95 | 0,491 | 1,000 | 0,659 | 1,000 |
| C7 (completo) | 2.0 | 0.5 | 0,95 | 0,933 | 1,000 | 0,966 | 1,000 |

### 1.3 Leitura conjunta — sem ambiguidade quanto ao ponto central, com uma ressalva

O padrão é consistente entre M1 e M2: **`Δt=0` não é mais um caso
patológico pós-fix — comporta-se de forma idêntica a `Δt=24`**, tanto
no baseline estatístico (M1) quanto no detector supervisionado (M2). O
fix do bug de `Δt=0` (commit `782b941`) cumpriu o que se propunha:
eliminar o separador trivial de classe documentado na auditoria de
16/09.

**Ressalva que os dados não permitem resolver aqui:** `Δt=0` e `Δt=24`
serem idênticos não é evidência de que `Δt=0` "voltou ao normal" no
sentido de "se comporta como qualquer outro nível intermediário" — é
evidência de que os dois compartilham o mesmo teto de janela
(`_janela_teto`, 24 timesteps para ambos) por construção do gerador.
Não há, nos dados desta consolidação, um terceiro ponto de comparação
independente que devesse ser diferente de `Δt=0`/`Δt=24` mas não é —
então não dá para separar "o fix funcionou" de "o teto de janela
compartilhado torna os dois indistinguíveis por design". As duas
leituras são consistentes com os números; qual delas é a explicação
completa não está resolvido nesta etapa.

---

## 2. Dominância de Fonte C (Dia 6)

O modelo completo (C7, A+B+C) treinado no grid completo do dataset v3
apresenta F1/AUROC=1,000 para `positiva_ativa` em todos os 5 níveis de
π, sem nenhuma degradação — inclusive sob mascaramento máximo de Fonte
A (π=0,95). O ranking SHAP (`TreeExplainer`, amostra estratificada de
18.000 linhas) explica esse resultado: `fonte_c_media` domina
sozinha, com SHAP médio absoluto de 6,898890 — cerca de 7 vezes o
valor da segunda feature mais importante (`fonte_b_n_eventos_total`,
0,952816). Das 26 features do vetor, 13 têm SHAP exatamente 0,000000,
incluindo **todo o Bloco de coordenação** (`tau_kendall_ab`,
`lag_correlacao_cruzada_otimo`, `correlacao_cruzada_maxima`) — o
modelo não usa nenhum sinal de correlação temporal A↔B, mesmo esse
sendo o mecanismo central que o Bloco de coordenação foi desenhado
para capturar.

Uma ablação pontual (C4, A+B sem Fonte C — não o estudo C1-C7 completo,
que fica para §5.4) mostrou que remover Fonte C não resolve o padrão:
a dominância **migra**, não desaparece. `fonte_b_n_eventos_total` passa
a SHAP=10,260477 (mais dominante ainda que `fonte_c_media` era),
seguida de `fonte_a_n_eventos_total` (2,003586) — ambas features de
Camada 1 (contagem bruta), não de coordenação. O Bloco de coordenação
permanece em SHAP=0,000000 mesmo sem Fonte C disponível.

A causa raiz foi diagnosticada em investigação separada no Dia 6:
`model.py:259` sorteia o voto de base de cada agente de forma
**uniforme e independente** entre candidatos (`rng.integers(0,
n_candidatos, ...)`), sem nenhum parâmetro de popularidade diferencial
— confirmado como suposição v0 deliberada, já coberta por teste que a
trata como comportamento esperado
(`test_n_candidatos_maior_que_um_conta_voto_de_base_sem_incentivo`).
Verificado empiricamente no v3: `fonte_c_media` na classe negativa
varia só 0,0005 entre as duas seeds do grid, com desvio dentro-de-seed
de ~0,0205 — ruído amostral, não heterogeneidade estrutural entre
cenários. Isso torna `fonte_c_media` um proxy quase-determinístico de
"houve CSC ativo", sem exigir que o modelo combine informação de
múltiplas fontes.

**Risco explícito ao critério de sucesso falsificável do TCC (§5.4.5):**
o critério exige que C7 (A+B+C) supere significativamente
max(C1,C2,C3) — ou seja, que a integração multimodal seja NECESSÁRIA,
não só suficiente. Os achados acima sugerem que C3 isolado (só Fonte
C) já resolve o problema de detecção quase inteiramente dentro de C7,
contrariando essa premissa. **Isto é um achado preliminar de prova de
conceito, não uma conclusão definitiva**: o split usado é intra-cenário
(train/val/test da mesma combinação de parâmetros — não testa
generalização entre regimes nunca vistos), o regime de treino é único
(hiperparâmetros XGBoost default, sem tuning), e a causa raiz já
identificada (homogeneidade de apoio orgânico) pode estar inflando
artificialmente essa separabilidade neste dataset específico. Precisa
constar como risco explícito para discussão com os orientadores antes
do design fatorial completo — não deve ser resolvido nem silenciado
unilateralmente nesta etapa.

---

## 3. Lista de pendências (material bruto — não é a lista fechada do Dia 9)

**a. Checagem de dominância redesenhada (Fonte C + Camada 1 crua
juntas).** A Etapa 8b original do plano do Dia 6 (excluir só
contagem/volume bruto de A/B, mantendo Fonte C) foi desenhada antes do
achado de dominância de Fonte C — como a ablação C4 mostrou que
remover Fonte C só desloca a dominância para `fonte_b_n_eventos_total`
(não resolve, revela outro problema estrutural), a Etapa 8b como
planejada não testaria o problema real. Precisa ser redesenhada
considerando os dois achados juntos. Não executada.

**b. Heterogeneidade de apoio orgânico entre candidatos.** Pendência
de gerador — caminho de implementação identificado
(`model.py:259`, trocar `rng.integers` uniforme por `rng.choice` com
pesos de popularidade, ex. amostrados de uma Dirichlet por
seed/cenário), mas parametrização não decidida. Fica para decisão
explícita com os orientadores.

**c. Lacunas do §5.3.1 não implementadas na camada agregada (Dia 4-5).**
z-score agregado por janela, entropia temporal, desvio de IET vs.
Poisson, mutual information ativação×resultado por `g` — pendência v0
documentada desde o Dia 4-5, mantida sem alteração.

**d. Risco ao critério de sucesso falsificável (§5.4.5)** — ver Seção 2
acima.

**e. Placeholders de calibração ainda sem valor fixado:** `taxa_fonte_a`,
`volume_medio_fonte_a`, `taxa_fonte_b` (ordem de grandeza não calibrada,
risco documentado no PLANO §5.2.4 de o detector aprender a separar
classes só pelo volume total); ρ/`prop_racional` em U(𝒜) sem direção
formalizada na utilidade do adversário; `g` fixo em `"pool"` no dataset
de produção (nenhuma variação real de granularidade eleitoral testada
ainda); `C_min` (custo mínimo, §5.4.4) não operacionalizado.

**f. Projeção de custo de M3 precisa de repetição controlada antes de
ser considerada final (Dia 7).** O benchmark
(`scripts/benchmark_m3_custo.py`) mostrou variação de ~3,4x no tempo
de uma época entre duas execuções do mesmo shape, atribuída a contenda
de CPU do sistema, não a bug de fórmula (diagnosticado e confirmado
no Dia 7). A projeção atual (~0,227h a ~0,738h para 1 época em
todo o design fatorial, dependendo de qual execução se usa como
referência) precisa de 5-10 repetições por shape, reportando
mediana + faixa, antes de ser levada como número final ao orientador
de IA. Adicionalmente: qualquer mudança futura no gerador (ex. item b
acima) pode alterar o shape/volume de dados que M3 consumiria,
tornando a projeção atual sujeita a revisão por esse motivo também.
Não corrigido nesta etapa — só registrado.

**g. McNemar / C_min — fora de escopo até o design fatorial completo
(§5.4, Semanas 14-17).** Não tratados nesta consolidação nem no Dia 6/7.

---

*Material bruto para discussão com os orientadores — não é a lista
fechada de pendências (Dia 9) nem o texto de §5.3 (Dia 9).*
