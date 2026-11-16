# Decisões de desenho do experimento (versão definitiva)

Última atualização: 12/nov/2026. Cada decisão traz o motivo e os números que a sustentam.
Itens marcados com **(a confirmar)** são hipóteses ou escolhas ainda em aberto.
---

## 1. Métrica
- Escolhida: area_under_roc_curve (AUC), em todas as bases, inclusive multiclasse
- Métrica principal fixa: AUC.
- Sensibilidade: acurácia (`predictive_accuracy`), só para verificar se o resultado do
  recomendador se mantém; não substitui a AUC nem entra no resultado principal.
- O ranking das famílias dentro de cada base não depende de a AUC ser binária ou
  multiclasse. A escala só pesa nos usos de valor absoluto: o limiar de 0,70 das bases
  fracas e o arrependimento médio em AUC entre bases. Por isso os resultados também
  serão estratificados entre binárias e multiclasse (seção 15).
- Cobertura: em 3 tasks (3, 6 e 11), AUC e acurácia têm a mesma cobertura de runs e flows
- Verificação do cálculo (task 11, 3 classes, 5 runs): o valor do OpenML coincide com a AUC
  one-vs-rest ponderada pela frequência das classes, calculada sobre as predições
  agrupadas (maior diferença: 4,6e-7). A média simples entre classes fica abaixo em todos os
  runs (de 0,001 a 0,16)
- Limitações: verificado numa única task (3 classes), em 5 runs; a ponderação por frequência
  de classe possui parte da sensibilidade ao desbalanceamento
- Runs com AUC exatamente 0,5 (predição constante): verificado, ver seção 7
  (média de 5,34% dos runs por task, máximo 31,56%)
- Evidência: 01_preparacao_dados.ipynb, Seção 2 (proporção de AUC == 0,5). 
- Evidência: 01_preparacao_dados.ipynb, Seção 6 (verificação da AUC multiclasse)
- Composição das 60 tasks: 31 binárias e 29 multiclasse (de 3 a 26 classes); os resultados do recomendador serão reportados também estratificados entre binárias e multiclasse.

## 2. Fonte das bases
- Suíte OpenML-CC18 (72 tasks de classificação curadas)
- Justificativa: suíte padrão da literatura, procedimento de avaliação uniforme
  (as 72 tasks usam 10-fold cross-validation)
- Evidência: 01_preparacao_dados.ipynb, Seção 1.

## 3. Filtros de seleção
- Instâncias: 500 a 50.000 (custo do pymfe na semana 2)
- Features: no máximo ~500
- Valores faltantes: aceitos (tratados no pré-processamento das metacaracterísticas)
- Procedimento de avaliação: único (10-fold, confirmado)
- Resultado dos filtros de tamanho: 72 -> 60 tasks (6 com mais de 50.000 instâncias,
  10 com mais de 500 features, 0 com menos de 500 instâncias)

## 4. Flows, bibliotecas e famílias
- Bibliotecas: sklearn e weka, com implementações mantidas separadas; o mlr (flows do R)
  entrou como bloco próprio, com prefixo `mlr:`. Os flows `automlbenchmark_*` ficaram fora por serem AutoML. Flows sem prefixo
  de biblioteca no nome (12 flows, 133 runs, 20 tasks; ex.: smo, j48, iBK, A1DE,
  hyperpipes) entram como weka pelo nome do algoritmo; a origem não pôde ser confirmada
  nos metadados (os genéricos naivebayes, multilayerperceptron e randomforest trazem a
  dependência "pocopnml"). 40 desses runs chegam às tabelas finais, 1 por célula (em 40
  células, no máximo 17% dos runs da célula), então o efeito esperado é desprezível
- Unidade de classificador: família do algoritmo de aprendizado, sem os envelopes de busca
  de hiperparâmetros e de pré-processamento
  - sklearn: estimador final do Pipeline; nos SearchCV (Grid, Randomized e Halving), o
    estimador interno
  - weka: família mais externa, exceto FilteredClassifier, MultiSearch e MultiScheme, que
    são descartados para ficar com o classificador interno (Bagging, AdaBoostM1 e
    AttributeSelectedClassifier permanecem como família). O CVParameterSelection
    (também de busca de hiperparâmetros) não é descartado, mas não entra nas 22
    famílias finais
  - mlr: sufixos .preproc / .imputed / .tuned / .dummied normalizados
  - removido o sufixo de versão, ex.: "(33)"
- Baixar o bruto sem filtrar; o agrupamento acontece depois
- Evidência: 01_preparacao_dados.ipynb, Seção 3.

## 5. Agregação
- Nível: um valor por (task, família de classificador)
- **Regra principal atual: máximo esperado de k = 3 runs sem reposição por célula (`max_sub3`)**,
  que controla o viés do número de runs (seção 9)
- Sensibilidade: máximo completo (`tab_max`) e mediana (`tab_med`)
- Mínimo de 3 runs por célula; abaixo disso, valor faltante (calibrado, seção 8)
- Motivo: poucos flows concentram muitos runs (350 runs de um único pipeline na task 3)
- Evidência: 01_preparacao_dados.ipynb, Seção 4.

---

## 6. Extração dos runs
- Só a métrica `area_under_roc_curve` foi extraída (constante `FUNCAO` em
  `src/extracao_runs.py`)
- `list_evaluations` do OpenML dá erro com size acima de 1000, então a página é de 1000
- A ordem da listagem padrão não segue run_id nem valor, e o `sort_order` ordena só pelo
  valor da AUC. Por isso foram baixados **todos** os runs das 60 tasks, em duas fases
  (tasks com até 70.000 runs primeiro, as maiores depois), com script em segundo plano e
  retomada
- Fase 1: 42 tasks, 777.822 runs. Fase 2: 18 tasks grandes,
  2.846.703 runs
- Total: **3.624.525 runs**, 0 duplicados (por task e entre tasks), 0 nulos, 0 valores
  fora de [0,1]. Em todas as 60 tasks o total baixado supera a contagem aproximada da API
  em 1 a 1.000 runs (a contagem arredonda para milhares)
- Por task: mediana de ~21.500 runs, mínimo 4.957, máximo 415.117 (task 31)
- Exemplo: task 3 tem 173.304 runs e 927 flows distintos (mais de 17 minutos para baixar)
- Dados em `data/runs/task_<id>/` (páginas csv.gz e arquivo COMPLETA). As colunas são
  run_id, setup_id, flow_id, value e upload_time; o nome do flow vem de `flows.csv`
- Evidência: 01_preparacao_dados.ipynb, Seção 1 e Seção 2.

---

## 7. Limpeza dos runs (v3)
Problema: valores constantes perto de 0,5 repetidos em centenas ou milhares de runs
(predição constante). Concentram-se nos flows do mlr (glmnet 6766 e 5970, rpart 5636 e
5859, xgboost 6767). A degeneração é da combinação base × configuração, não do flow
(só 47 de 6.172 setups com >= 5 runs têm mais de 90% dos runs em [0,49; 0,51]).

Passos:
1. Remover runs com AUC exatamente 0,5: 218.506 (6,0%)
2. Regra por task: dentro de cada task, valor arredondado em 4 casas entre 0,47 e 0,53
   com >= 200 runs e >= 0,5% da task; remove runs a até 0,0005 dele. 127 pares em 54
   tasks, 823.307 runs. União com o passo 1: 3.624.525 -> 2.800.528 (v2)
3. Segunda passada com 3 casas: 55 pares extras detectados em 22 tasks (24.792 runs). A task 3560 (mediana 0,547; 2.981 runs) ficou de fora por ser ambígua, resultando em 21.811 runs efetivamente removidos. v3 = v2 menos esses extras: 2.778.717 runs (845.808 removidos no total, 23,3%); menor n por task = 4.679
- Resíduo explicado: task 3022 (vowel, 11 classes) mantém 8,96% (1.498 runs) com AUC < 0,5 espalhados em flows do sklearn (BernoulliNB, GradientBoosting, DecisionTree, SVC), confirmando que a regra preserva classificadores genuinamente ruins sem pico constante.
- Mecanismo dos picos < 0,5: verificado na task 3903 que a AUC teórica de um preditor constante usando a prior de treino por fold (com predições agrupadas) é 0,499252, coincidindo com o valor modal observado em 47.315 runs (0,4993); a inspeção direta das predições (ARFF) de 5 runs removidos nessa task confirmou uma única probabilidade e uma única classe predita por fold.
- Arquivos: `runs_consolidado.parquet` (sem limpeza), `runs_limpos_v3.parquet`

Validação da regra (nas 22 famílias finais):
- 501.597 runs removidos, com valores entre 0,485 e 0,527 (média 0,497); nenhum acima
  de 0,53; 2.364 (0,47%) acima de 0,51 e 542 (0,11%) acima de 0,52
- Os 542 acima de 0,52 se concentram em 10 células das bases 125920, 3560 e 219 (97%).
  Nas duas primeiras, que são fracas, a janela da regra pode se sobrepor ao desempenho
  genuíno
- Base 219 (flows removidos acima de 0,52 e efeito no max_sub3): dos 80 runs removidos
  acima de 0,52, 72 são do RandomForest do sklearn; esse efeito é nulo (dif = +0,0002,
  2,1% dos runs da célula removidos; weka:randomforest sem mudança). O efeito da limpeza
  na 219 vem de outras famílias: sklearn:GradientBoosting (limpo 0,869 contra bruto
  0,744, 55% removidos), mlr:rpart (0,737 contra 0,690, 57%) e sklearn:DecisionTree
  (0,935 contra 0,918, 20%). Os runs removidos dessas três famílias estão todos em
  valores únicos ou quase únicos em torno de 0,5: GradientBoosting, 1.270 runs em 0,5000
  (635), 0,4999 (420) ou 0,5001 (215); mlr:rpart, 7.515 runs, todos em 0,4999;
  DecisionTree, 418 runs, todos em 0,4999. São predições constantes. Portanto, na 219 a
  limpeza remove predições constantes e não desempenho real
- Checagem por nome de flow (dummy|zeror): 28 flows (27 do weka, em que ZeroR é o
  classificador final ou o base de um wrapper, e 1 sklearn DummyClassifier), 667 runs nas
  60 tasks. A limpeza removeu 99,1% (661); a AUC mediana desses flows fica entre 0,4957 e 0,5000 
  (exceto o flow 7050, com 0,9116), coerente com predição constante.
- Sobraram 6 runs (flows 56, 589, 1190, 7849, 7050 e 43339). O 7050 (weka.LWL_ZeroR, AUC
  0,91) não é constante: ZeroR é só o base do LWL. Quatro dos outros cinco (task 11, dois
  runs, task 31 e task 14954; AUC de 0,494 a 0,497) parecem predições constantes isoladas
  que a regra por valor não pega, porque ela exige repetição; o quinto, um weka.ZeroR(1)
  na task 3021 com AUC 0,962, é anômalo e não foi explicado. Nenhum pertence às 22
  famílias finais (weka:grading, weka:zeror e sklearn:DummyClassifier ficam abaixo do
  corte de 70%), então não afetam as tabelas
- Esses 661 runs são menos de 0,1% dos 845.808 removidos pela regra por valor: a
  identificação por nome de flow não substitui a regra por valor. Ela também erra nos dois
  sentidos (falso positivo no LWL_ZeroR; os picos do mlr não aparecem por nome)

Efeito da limpeza (comparação com e sem limpeza; em todos os casos, "ranking" é a ordem
das famílias dentro de cada base e a correlação é de Spearman, uma por base, nas 60 bases):
- **No max_sub3: relevante**: limpo e bruto calculados com a esperança exata do máximo de
  3 runs (`emax_exact`). 168 das 1.114 células (15,1%) mudam mais que o
  limiar de materialidade fixo LIM = 0,005, quase sempre com limpo >= bruto (máx. 0,298).
  Por biblioteca: mlr 21 de 90, sklearn 135 de 514, weka 12 de 510. Correlação entre os rankings 
  com e sem limpeza: mediana 0,977 (mín. 0,59, base 125920)
- **Na mediana (`tab_med`)**: diferença limpo menos bruto nas 1.114 células: média 0,020, máximo 0,49 (maior
  que o do max_sub3, 0,298). Correlação entre os rankings com e sem limpeza: mediana 0,971
  (mín. 0,64, base 10101)
- **No máximo completo (`tab_max`): praticamente nulo**: 1 célula em 1.114
  (weka:randomforest, base 3560, 0,5186 contra 0,5202); correlação entre os rankings com e
  sem limpeza = 1,0 nas 60 bases. Os runs degenerados raramente são o melhor run de uma
  célula; o que a limpeza muda é o que a esperança de 3 runs ou a mediana encontram
- **Mecanismo (vale para as três agregações)**: a diferença cresce com a fração de runs
  removida da célula. 747 das 1.114 células não perdem runs e não mudam, então no máximo
  367 (33%) podem mudar. No max_sub3, a diferença tem Spearman 0,99 com a fração removida;
  nas 31 células com mais de 60% removidos a diferença média é 0,158 (máx. 0,298), sobretudo em mlr:rpart (7 das 10 maiores)
  e sklearn:GradientBoostingClassifier. Na mediana, quando
  mais da metade dos runs da célula é removida, a mediana bruta cai na janela da regra
  (valor constante) e a limpa recupera o valor real: na célula de maior diferença (task 28,
  sklearn:GradientBoosting, 61,2% removidos) ela vai de 0,49998 a 0,9905
- **Leitura conjunta**: o máximo completo mede o melhor run submetido e quase não depende
  da limpeza; max_sub3 e mediana medem o desempenho de runs típicos e dependem dela. Por
  isso a diferença entre `tab_max` e `tab_sub3` mistura dois efeitos (o viés de volume e
  tuning, e os runs degenerados), e a limpeza é o que deixa a comparação interpretável
- Decisão: limpeza mantida e tratada como parte do método. O max_sub3 limpo mede o
  máximo esperado de 3 runs **não degenerados**. Sensibilidades: `tab_sub3_sem_limpeza`,
  `tab_med` e `tab_max`
- Evidência: 01_preparacao_dados.ipynb, Seção 2 e Seção 6.

## 8. Famílias, mínimo de runs por célula e corte de cobertura
- 657 famílias distintas antes do corte. Removidos 2.377 runs sem família (0,09%):
  2.351 de bibliotecas fora de sklearn, weka e mlr (1.811 de flows `classif.*` antigos
  sem prefixo; o resto, AutoML e wrappers como openmlpimp, arbok/TPOT, pipelineopt, keras
  e bsplitz) e 26 dos flows `automlbenchmark_*`. Sobram **2.776.340 runs** (`runs_v3_familia.parquet`)
- Distribuição de runs por célula: mediana 1 no weka e 2 no mlr e no sklearn; p90 de 15
  no weka e acima de 1.300 no mlr e no sklearn
- O mínimo de runs por célula e o corte de cobertura foram calibrados juntos.
  Famílias (cobertura 60% / 65% / 70% / 75% / 80%):
  - min 1: 110 / 96 / 90 / 63 / 27 (células vazias de 22% a 5%)
  - min 3: 27 / 26 / 22 / 19 / 10
  - min 5: 17 / 16 / 15 / 15 / 8
  - min 10: 14 / 13 / 13 / 13 / 8
- **Escolha: mínimo de 3 runs por célula e presença em >= 70% das bases (>= 42 de 60)**
  -> **22 famílias** (11 weka, 9 sklearn, 2 mlr), 15,6% de células vazias
  - sklearn: AdaBoost, DecisionTree, FKCEigenPro, GaussianNB, GradientBoosting,
    HistGradientBoosting, LogisticRegression, RandomForest, SVC
  - weka: adaboostm1, attributeselectedclassifier, bagging, ibk, j48, logistic, lwl,
    multilayerperceptron, naivebayes, randomforest, reptree
  - mlr: naiveBayes, rpart
- Motivo: com min 1 a comparação mede o orçamento de tuning, não a família (o weka
  entra com ~1 run por célula e o mlr e o sklearn com centenas); o min 3 preserva todos
  os tipos clássicos (árvores, ensembles, linear, SVM, Bayes, vizinhos, redes)
- Alternativas avaliadas na calibração: min 1 (90 famílias), min 5 (15 famílias, sem Naive Bayes) e corte de
  75% (19 famílias; GaussianNB, mlr:naiveBayes e weka:lwl, com 42 a 43 bases, ficam na borda)
- Famílias abaixo do corte ficam fora por cobertura. Ex.: xgboost do mlr (22 das 60 bases)
  é limitação registrada, sem tratamento especial
- sklearn:FKCEigenPro: pacote sklearn_extra (`fast_kernel`), 50.604 runs, 60 bases,
  3 flows (imputação, com ou sem padronização, e uma versão com a classe renomeada
  FKC_EigenPro -> FKCEigenPro), 14.913 setups; agrupada como uma família
- Evidência: 01_preparacao_dados.ipynb, Seção 3.

## 9. Viés do número de runs, agregação exata e sensibilidades
- O máximo cresce com o número de runs (n) da célula: correlação de Spearman entre n e o
  máximo por base com mediana de 0,46 (geral 0,17), contra 0,07 com a mediana (geral 0,05)
  -> viés de amostragem, não só qualidade
- **max_sub3**: para cada célula com n >= 3, esperança exata do máximo de k = 3 runs
  sorteados sem reposição, calculada analiticamente sobre os valores ordenados da célula
  (`emax_exact`: soma de v(i) · C(i-1, 2) / C(n, 3)). Mitiga o viés de volume de runs sem
  introduzir ruído de simulação ou dependência de semente (restam apenas a incerteza da
  própria amostra de runs disponível no OpenML e o limiar de materialidade fixo LIM = 0,005)
- Distância de AUC entre famílias vizinhas no ranking de cada base (1.054 distâncias na
  `tab_sub3`): mediana 0,0029, média 0,0083, quartis 0,00082 e 0,0088, máx. 0,17. Cerca de
  um quarto dos pares vizinhos está separado por menos de 0,00082 em AUC (abaixo do limiar
  de materialidade de 0,005); a ordem estrita entre famílias quase empatadas é instável
- Correlação de ranking por base entre max e max_sub3: mediana 0,73 (média 0,72, mín.
  0,36, máx. 0,94)
- Robustez do k: max_sub3 contra max_sub10 com mediana 0,93 (média 0,91, mín. 0,66,
  máx. 1,0), usando só as 823 das 1.114 células com n >= 10; **mantido k = 3** para não
  perder células de famílias com poucos runs
- Movimento de ranking (max -> max_sub3, média por família): SVC -10,2;
  GradientBoosting -7,0; attributeselectedclassifier -3,5; RandomForest +0,1;
  DecisionTree +0,2; FKCEigenPro +0,8; sobem sobretudo famílias com poucos runs
  (até +3,25, em weka:logistic). Interpretação: o max_sub3 também penaliza famílias
  sensíveis a hiperparâmetros (SVC, GradientBoosting), não só o viés de volume
- Concordância entre `max` e `max_sub3`: a família vencedora (1º lugar) é a mesma em 26,7%
  das bases (16 de 60). Olhando para o pódio (top-3), as duas métricas compartilham em média
  1,65 das 3 famílias (o pódio é idêntico em 11 bases e totalmente diferente em 8). Na posição
  média do ranking (onde 1º é o melhor), `HistGradientBoosting` lidera tanto no `max` (3,34º)
  quanto no `max_sub3` (2,88º), seguido pelo `RandomForest` (3,53º e 3,40º). Já famílias
  sensíveis a tuning despencam ao sair do máximo completo para o `max_sub3`: `SVC` cai de
  4,54º para 14,75º e `GradientBoosting` vai de 6,17º para 13,19º
- Arrependimento em AUC: escolher pelo max_sub3 e avaliar no máximo perde em média
  0,0032 (mediana 0,0008; máx. 0,028); o inverso perde em média 0,016 (mediana 0,0080;
  máx. 0,126)
- **Decisão: max_sub3 exato (k = 3) é a métrica de agregação principal; a mediana (`tab_med`)
  e o máximo completo (`tab_max`) são as sensibilidades.** O recomendador é avaliado nas três
- Evidência: 01_preparacao_dados.ipynb, Seção 3, Seção 4 e Seção 6.

## 10. Tabelas base × família
- `tab_sub3` (60 × 22), `tab_max` (sensibilidade), `tab_med` (mediana) e `tab_n` (nº de
  runs por célula), em parquet e csv; `celulas_longa.parquet`
- 206 células faltantes de 1.320 (15,6%), as de n < 3
- `tab_sub3_sem_limpeza` (60 × 22, em parquet): `max_sub3` calculado com `emax_exact`
  sobre os runs sem limpeza (brutos), diretamente comparável à `tab_sub3` sem diferença de
  sorteio ou semente
- Todas as tabelas usam os runs limpos (v3), exceto `tab_sub3_sem_limpeza`. A comparação
  do recomendador com e sem limpeza é feita só no max_sub3; o máximo completo e a mediana
  entram sempre limpos (a mediana bruta mede a degeneração, não a família, como na
  task 28: 0,9905 limpa contra 0,49998 bruta)
- `nucleo_sub3`, `tab_sub3_sem_fracas` e `nucleo_sub3_sem_fracas` (seções 11 e 12)
- Evidência: 01_preparacao_dados.ipynb, Seção 4, Seção 5 e Seção 6.

## 11. Células faltantes
- Distribuição: weka 150 dos 206, mlr 30, sklearn 26. As 6 famílias do sklearn sem
  faltantes garantem pelo menos 6 famílias em comum entre quaisquer duas bases
- **Estrutura de bloco**: em 14 das 60 bases o weka e o mlr estão quase ausentes
  (8 a 11 de 11 famílias do weka); nas 46 restantes o weka está completo (0 a 1 NaN)
  - Bases sem weka: 14952, 14954, 125920, 125922, 146800, 146817, 146819, 146820, 146821,
    146822, 146824, 167119, 167140, 167141
  - Divisão temporal: as 46 têm primeiro run entre abr/2014 e abr/2016; as 14, entre
    out/2016 e abr/2018 (mediana 21/02/2018), sem sobreposição. O corte é pela data dos
    runs, não pelo task_id (14952 e 14954 têm task_id baixo, mas pertencem ao bloco tardio)
  - Volume: 97,8% dos runs do weka e 99,9% dos do mlr (102.859 de 102.942) são de até 2017; as 14 bases receberam
    104 dos 18.987 runs do weka (0,5%) e 2.057 dos 102.942 do mlr (2%)
  - Causa provável: o lote de tarefas de 2018 surgiu depois da atividade dessas
    bibliotecas
  - Nas bases 14952 e 14954 faltam 10 das 11 famílias weka, sklearn:GaussianNB,
    sklearn:GradientBoostingClassifier e mlr:naiveBayes (na 14954 também mlr:rpart). Das
    células contadas, GaussianNB, GradientBoosting e j48 têm 1 ou 2 runs (abaixo do mínimo
    de 3) e mlr:naiveBayes tem 0; as demais famílias weka faltantes não foram contadas
- **Checagem das 14 bases fora do núcleo** (Mann-Whitney, 14 fora contra 46 no núcleo,
  descritores sklearn): instâncias, features, classes, melhor AUC, mediana e amplitude
  sem diferença relevante (prob_sup entre 0,51 e 0,61; p > 0,22). Proporção de
  categóricas: mediana 0,155 fora contra 0,059 no núcleo (p = 0,224), tendência fraca.
  O número total de runs é bem menor fora (prob_sup 0,05), efeito mecânico por faltar
  weka e mlr
- **Decisão**:
  - **Resultado principal:** núcleo sem bases fracas, **45 bases × 17 famílias**, sem NaN
  - **Sensibilidade:** 58 bases × 22 famílias (`tab_sub3_sem_fracas`), **sem imputação**: o
    gabarito de cada base é o ranking das famílias observadas nela; o Spearman entre bases
    usa só as famílias em comum (mínimo de 5; nenhum par ficou abaixo, o mínimo observado
    é 6). Resultados no total e por grupo (bases completas e bases só-sklearn)
  - **Descartada:** imputação das células, que inventaria AUC na variável que o
    recomendador tenta prever, e com falta em bloco o viés seria sistemático
- Composição do núcleo (17 famílias, sem nenhuma do mlr): sklearn AdaBoost, DecisionTree,
  FKCEigenPro, GradientBoosting, HistGradientBoosting, RandomForest, SVC; weka
  adaboostm1, attributeselectedclassifier, bagging, ibk, j48, logistic,
  multilayerperceptron, naivebayes, randomforest, reptree. Em relação às 22 famílias, o
  núcleo perde 5: mlr:naiveBayes, mlr:rpart, sklearn:GaussianNB, sklearn:LogisticRegression
  e weka:lwl
- Número de faltantes **não** entra como metacaracterística (não existe para uma base
  nova); serve só como diagnóstico
- Evidência: 01_preparacao_dados.ipynb, Seção 5 e Seção 6.

## 12. Bases fracas e saturação
- Bases fracas: 3560 (melhor máximo 0,599; todas as famílias entre 0,52 e 0,60) e
  125920 (0,665; só 11 famílias). A terceira mais baixa é a 23, com 0,749
- **Regra (definida antes de rodar o recomendador)**: excluir as bases cujo melhor
  máximo é < 0,70. Gera `tab_sub3_sem_fracas` (58 × 22) e `nucleo_sub3_sem_fracas`
  (45 × 17; só a 3560 sai do núcleo, porque a 125920 já está fora dele). Não se corta
  nada pelo resultado
- Saturação: 28 das 60 bases têm melhor máximo > 0,99, com amplitude mediana de 0,073
  entre as famílias. Em 25% das bases há 5 ou mais famílias a menos de 0,001 do melhor
  (máx. 15). Empates tratados com postos médios (padrão do Spearman)
- Bases com correlação de ranking baixa entre dado limpo e bruto: 125920 (0,59),
  23 (0,83), 29 (0,86), 3902 (0,89) e 2074 (0,89)
- Evidência: 01_preparacao_dados.ipynb, Seção 4, Seção 5 e Seção 6.

## 13. Metacaracterísticas (pymfe) e plano inicial da Fase 2

### O que já está definido (testado na Fase 1)
- **Grupos e codificação no `pymfe`**: grupos `general`, `statistical` e `info-theory`
  (73 metacaracterísticas extraídas), com `transform_cat="one-hot-full"` para garantir
  invariância à renomeação das categorias (nas 22 bases com atributos categóricos,
  codificações como `gray` ou ordinal alteram de 24 a 27 metacaracterísticas ao permutar
  rótulos)
- **Filtro de estabilidade**: descarte de `attr_conc.mean` e `attr_conc.sd` (variam entre
  rodadas idênticas), restando 71 metacaracterísticas determinísticas
- **Métricas de avaliação do recomendador**:
  - **Principal:** **arrependimento em AUC** (diferença entre a AUC da melhor família da
    base e a AUC da família recomendada), complementado pela taxa de acerto no top-1 e top-3
  - **Secundária:** correlação de Spearman do ranking completo (mantida para diagnóstico
    global, mas não como métrica principal porque 28 das 60 bases são saturadas acima de
    0,99 e um quarto dos pares vizinhos difere por menos de 0,00082 em AUC, abaixo do
    limiar de materialidade de 0,005)
- **Variantes de tabelas a avaliar**:
  1. **Principal:** `nucleo_sub3_sem_fracas` (45 × 17, dados limpos, `max_sub3`)
  2. **Sensibilidade de cobertura (sem imputação):** `tab_sub3_sem_fracas` (58 × 22),
     reportada no total e estratificada (bases completas vs. bases só-`sklearn`); o
     baseline usa posto normalizado pelo nº de famílias observadas na base
  3. **Sensibilidade de agregação:** mediana (`tab_med`) e máximo completo (`tab_max`)
  4. **Sensibilidade de limpeza:** `tab_sub3_sem_limpeza` (comparada à versão limpa)
  5. **Sensibilidade de bases fracas:** `tab_sub3` (60 × 22, incluindo as bases 3560 e 125920)

### Plano preliminar para a Fase 2 (a consolidar após extração completa e estudo do modelo)
- **Faltantes nas bases originais (9 de 60 bases)**: partir da imputação simples por base
  antes do `MFE.fit` (mediana nos numéricos, moda nos categóricos) + inclusão da proporção
  original de faltantes como metacaracterística; avaliar métodos alternativos (ex.:
  `KNNImputer`) na etapa de modelagem
- **NaNs residuais do `pymfe` (ex.: `cat_to_num`, `sd_ratio`, `can_cor.sd`)**: regra final
  a definir após rodar a extração nas 60 bases (descarte de metacaracterísticas com alta
  proporção de NaN ou variância zero, e imputação/padronização ajustadas estritamente no
  treino de cada dobra do LODO para evitar vazamento de dados)
- **Configuração do recomendador ($k$-NN) e baseline**: validação *leave-one-dataset-out*
  (LODO) comparando $k$-NN sobre metacaracterísticas padronizadas (valores ímpares como
  $k \in \{1, 3, 5, 7\}$ e/ou ponderação por distância, a refinar na Fase 2) contra o
  baseline de melhor ranking médio no treino
- Evidência: `02_pymfe_testes.ipynb` (testes preliminares de codificação categórica,
  instabilidade de `attr_conc` e comportamento com atributos categóricos/faltantes).

## 14. Limitações a declarar no relatório
- Os runs do OpenML não vêm de um experimento controlado: mesmo controlando o volume com
  o max_sub3 (k = 3), a métrica mistura a qualidade intrínseca da família com quais
  configurações alguém optou por submeter. Além disso, o max_sub3 penaliza famílias
  sensíveis a hiperparâmetros, enquanto o máximo completo favorece famílias com mais
  runs (a mediana tab_med será avaliada como sensibilidade)
- A AUC multiclasse do OpenML é one-vs-rest ponderada pela frequência das classes e não é
  comparável em escala com a AUC binária (e herda sensibilidade ao desbalanceamento). O
  ranking dentro de cada base não depende disso; os usos de valor absoluto sim (limiar
  de 0,70 das bases fracas, arrependimento). Mitigação: estratificação binárias/multiclasse
  e sensibilidade com acurácia (seção 15)
- O resultado depende da regra de limpeza (afeta ~15% das células no max_sub3: 168 de
  1.114 células)
- Famílias populares abaixo do corte de cobertura (xgboost, lightgbm) ficam fora
- Faltantes em bloco: 14 bases (runs de out/2016 a abr/2018) têm quase apenas famílias do
  sklearn, então o conjunto de candidatos difere entre os grupos. O resultado principal
  usa o núcleo de 45 bases e perde 5 famílias (mlr:naiveBayes, mlr:rpart,
  sklearn:GaussianNB, sklearn:LogisticRegression e weka:lwl); as bases fora do núcleo
  têm mais categóricas, com evidência fraca
- Duas bases fracas (3560 e 125920) e muitas saturadas: o ranking nelas é pouco informativo
- O weka entra com poucos runs por célula (mediana 1), então seu máximo é instável
- Famílias meta agrupam estimadores internos diferentes
- O ranking depende da amostra finita de runs disponível no OpenML: embora o `emax_exact`
  elimine o ruído de sorteio, células com poucos runs (como n = 3) ainda refletem apenas
  as submissões existentes, e pares separados por menos que o limiar de materialidade
  (LIM = 0,005) podem trocar de posição com novos runs
- Flows sem prefixo de biblioteca no nome (40 runs nas tabelas, 1 por célula) foram
  assumidos como weka pelo nome do algoritmo, sem confirmação nos metadados
- Os limiares da regra de limpeza (200 runs, 0,5% da task, ±0,0005, janela de 0,47 a
  0,53) não tiveram análise de sensibilidade; o efeito total da limpeza é delimitado pela
  comparação com e sem limpeza (`tab_sub3_sem_limpeza`)

## 15. Pendências
- **Extração das metacaracterísticas (`pymfe`):** calcular os grupos `general`,
  `statistical` e `info-theory` (com codificação `one-hot-full` e descarte de
  `attr_conc`) para as 60 bases, incluindo a fração original de valores faltantes como
  metacaracterística.
- **Pré-processamento das metacaracterísticas:** definir o tratamento definitivo dos
  `NaN` residuais, infinitos e colunas constantes (avaliando a sensibilidade com
  `KNNImputer`) e aplicar padronização/normalização antes do cálculo de distâncias.
- **Baseline de ranking médio:** implementar o recomendador de referência baseado no
  ranking médio das famílias no conjunto de treino.
- **Recomendador $k$-NN e validação LODO:** implementar a busca por vizinhos (valores
  ímpares como $k \in \{1, 3, 5, 7\}$ e/ou ponderação por distância) e estruturar o
  protocolo *leave-one-dataset-out* (garantindo que a base de teste não entre no treino,
  na imputação nem na padronização).
- **Métricas de avaliação (Seção 13):** avaliar o recomendador no núcleo principal
  (`45 × 17`) usando arrependimento em AUC e acerto no top-1/top-3 como métricas
  principais, e correlação de Spearman como métrica secundária.
- **Comparação binária vs. multiclasse:** estratificar a
  avaliação do recomendador entre bases binárias (31) e multiclasse (29) para verificar
  diferenças de comportamento ou escala entre os dois cenários.
- **Sensibilidade com acurácia:** extrair a tabela de
  desempenho com `predictive_accuracy` para as 60 tasks e rodar o recomendador como
  sensibilidade, depois do resultado principal com a AUC.
  - Extração: parametrizar `FUNCAO` em `src/extracao_runs.py` (`--funcao`); avaliar
    limitar a extração aos flows das 22 famílias, conferindo antes que o filtro por
    flow do `list_evaluations` funciona.
  - Limpeza: reaproveitar os `run_id` removidos na limpeza da AUC (v3), porque a regra
    por valor 0,47 a 0,53 não vale para acurácia (o preditor constante tem acurácia
    igual à proporção da classe majoritária). Antes, conferir que os `run_id` coincidem
    nas 60 bases.
  - Comparar: Spearman por base entre os rankings de AUC e de acurácia, e as métricas do
    recomendador nas duas.
- **Avaliação nas sensibilidades já construídas na Fase 1:** rodar o recomendador sobre as
  tabelas alternativas já geradas (`tab_sub3_sem_fracas` 58 × 22 sem imputação, `tab_med`
  com mediana, `tab_max` com máximo completo, `tab_sub3_sem_limpeza` sem a limpeza e
  `tab_sub3` 60 × 22 com as bases fracas).