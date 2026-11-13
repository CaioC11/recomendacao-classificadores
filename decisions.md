# Decisões de desenho do experimento (versão definitiva)

Última atualização: 12/nov/2026. Cada decisão traz o motivo e os números que a sustentam.
Itens marcados com **(a confirmar)** são hipóteses ou escolhas ainda em aberto. A seção 16
lista o que mudou em relação à versão anterior.

---

## 1. Métrica
- Escolhida: area_under_roc_curve (AUC), em todas as bases, inclusive multiclasse
- Uma métrica fixa de desempenho (pedido da Juscimara); descartado usar duas métricas
  (principal + secundária) para o desempenho dos classificadores
- Cobertura: em 3 tasks (3, 6 e 11), AUC e acurácia têm a mesma cobertura de runs e flows
- Verificação do cálculo (task 11, 3 classes, 1 run): o valor do OpenML (0.936468)
  coincide com a AUC média ponderada pela frequência das classes, calculada sobre as
  predições agrupadas (0.936468). A média simples entre classes deu 0.842 e a
  ponderada por fold deu 0.9370
- Limitações: verificado em um único run; a ponderação por frequência de classe herda
  parte da sensibilidade ao desbalanceamento
- Runs com AUC exatamente 0,5 (predição constante): verificado, ver seção 7
  (média de 5,34% dos runs por task, máximo 31,56%)
- Evidência: 01_selecao_bases.ipynb, Seção 2 (exec 21 e 23, proporção de AUC == 0,5). A verificação multiclasse da task 11 foi feita em teste avulso fora deste notebook.
- Composição das 60 tasks: 31 binárias e 29 multiclasse (de 3 a 26 classes); os resultados do recomendador serão reportados também estratificados entre binárias e multiclasse.

## 2. Fonte das bases
- Suíte OpenML-CC18 (72 tasks de classificação curadas)
- Justificativa: suíte padrão da literatura, procedimento de avaliação uniforme
  (as 72 tasks usam 10-fold cross-validation)
- Evidência: 01_selecao_bases.ipynb, Seção 1 (exec 1–6).

## 3. Filtros de seleção
- Instâncias: 500 a 50.000 (custo do pymfe na semana 2)
- Features: no máximo ~500
- Valores faltantes: aceitos (tratados no pré-processamento das metacaracterísticas)
- Procedimento de avaliação: único (10-fold, confirmado)
- Resultado dos filtros de tamanho: 72 -> 60 tasks (6 com mais de 50.000 instâncias,
  10 com mais de 500 features, 0 com menos de 500 instâncias)
- Cobertura de flows (amostra de até 1000 runs por task): mínimo 13, mediana 107,5,
  máximo 193; 58 tasks com pelo menos 30 flows; abaixo de 30: texture (125922, 13 flows)
  e dresses-sales (125920, 14 flows); car (146821) tem exatamente 30
- **As 60 tasks foram mantidas**: a meta inicial de 20 a 40 tasks foi abandonada e a
  regra de >= 30 flows não foi usada como corte. A cobertura passou a ser controlada
  pela presença das famílias (seção 8)

## 4. Flows, bibliotecas e famílias
- Bibliotecas: sklearn e weka, com implementações mantidas separadas; o mlr (flows do R)
  entrou como bloco próprio, com prefixo `mlr:` (a decisão original só previa sklearn e
  weka). Os flows `automlbenchmark_*` ficaram fora por serem AutoML
- Unidade de classificador: família do classificador final
  - sklearn: estimador final do Pipeline e estimador interno nos SearchCV
  - weka: família mais externa nos wrappers (ex.: Bagging, AdaBoostM1)
  - mlr: sufixos .preproc / .imputed / .tuned / .dummied normalizados
  - removido o sufixo de versão, ex.: "(33)"
- Baixar o bruto sem filtrar; o agrupamento acontece depois
- Risco: nomes de módulos sklearn antigos sugerem runs de versões antigas da biblioteca
- Famílias meta (bagging, adaboostm1, attributeselectedclassifier) agrupam estimadores
  internos variados; a família é "o estimador externo"
- Evidência: 01_selecao_bases.ipynb, Seção 3 (exec 35–42 e 52).

## 5. Agregação (revisada em 06/nov/2026)
- Nível: um valor por (task, família de classificador)
- **Regra principal atual: máximo por célula, com controle do viés do número de runs**
  (seção 9). A decisão original (30/out) era a mediana, com o máximo como sensibilidade;
  Caio passou a preferir o máximo
- Sensibilidade: mediana e média dos 5 melhores
- Mínimo de 3 runs por célula; abaixo disso, valor faltante (calibrado, seção 8)
- Motivo: poucos flows concentram muitos runs (350 runs de um único pipeline na task 3)
- Limitações: o máximo cresce com o número de runs (viés de volume) e a mediana depende
  de quais configurações foram submetidas
- Evidência: 01_selecao_bases.ipynb, Seção 4 (exec 53–56).

---

## 6. Extração dos runs
- `list_evaluations` do OpenML dá erro com size acima de 1000, então a página é de 1000
- A listagem padrão não é monotônica em run_id nem em valor; `sort_order` ordena pelo
  valor da AUC. Por isso foram baixados **todos** os runs das 60 tasks, em duas fases
  (tasks pequenas primeiro, grandes depois), com script em segundo plano e retomada
- Fase 1: 42 tasks, 777.822 runs, 97 minutos, 0 falhas. Fase 2: 18 tasks grandes
- Total: **3.624.525 runs**, 0 duplicados, 0 nulos, 0 valores fora de [0,1]
- Por task: mediana de ~21.000 runs, mínimo ~4.000, máximo ~415.000 (task 31)
- Exemplo: task 3 tem 173.304 runs e 927 flows distintos (~15 min para baixar)
- Dados em `data/runs/task_<id>/` (páginas csv.gz e arquivo COMPLETA). As colunas são
  run_id, setup_id, flow_id, value e upload_time; o nome do flow vem de `flows.csv`
- Evidência: 01_selecao_bases.ipynb, Seção 1 (exec 7–16) e Seção 2 (exec 17–22).

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
- Mecanismo dos picos < 0,5: verificado na task 3903 que a AUC teórica de um preditor constante usando a prior de treino por fold (com predições agrupadas) é 0,499252, coincidindo com o valor modal observado em 47.315 runs (0,4993).
- Arquivos: `runs_consolidado.parquet` (sem limpeza), `runs_limpos_v3.parquet`

Validação da regra (nas 22 famílias finais):
- 501.597 runs removidos, com valores entre 0,485 e 0,527 (média 0,497); nenhum acima
  de 0,53; 2.364 (0,47%) acima de 0,51 e 542 (0,11%) acima de 0,52
- Os 542 acima de 0,52 se concentram em 10 células das bases 125920, 3560 e 219 (97%).
  Nas duas primeiras, que são fracas, a janela da regra se sobrepõe ao desempenho
  genuíno

Efeito da limpeza:
- **No máximo completo: praticamente nulo**: 1 célula em 1.114 (weka:randomforest,
  base 3560, 0,5186 contra 0,5202); correlação de ranking por base = 1,0 nas 60 bases
- **No max_sub3: relevante**: 173 das 1.114 células (15,5%) mudam acima do ruído
  (p95 = 0,0048, medido com R = 200, ver seção 9), quase sempre com limpo >= bruto
  (máx. 0,31). Por biblioteca: mlr 22 de 90, sklearn 135 de 514, weka 16 de 510.
  Correlação de ranking por base: mediana 0,976 (mín. 0,69, base 125920). Na mediana
  por célula: 0,971 (mín. 0,64)
- Mecanismo: a diferença cresce com a fração de runs removida da célula (Spearman
  0,76). 747 das 1.114 células não perdem runs e não mudam; nas 31 com mais de 60%
  removidos a diferença média é 0,157 (máx. 0,31). As maiores diferenças são em
  mlr:rpart (8 das 10 maiores) e sklearn:GradientBoostingClassifier
- Decisão: limpeza mantida e tratada como parte do método. O max_sub3 limpo mede o
  máximo esperado de 3 runs **funcionais**. Sensibilidade: `tab_sub3_sem_limpeza`
- Evidência: 01_selecao_bases.ipynb, Seção 2 (exec 21–33) e Seção 6 (exec 74–78).

## 8. Famílias, mínimo de runs por célula e corte de cobertura
- 657 famílias distintas antes do corte. Removidos 2.377 runs sem família (0,09%):
  AutoML e wrappers (openmlpimp, arbok/TPOT, pipelineopt, keras, bsplitz) e flows
  `classif.*` antigos sem prefixo. Sobram **2.776.340 runs** (`runs_v3_familia.parquet`)
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
- Sensibilidades: min 1 (90 famílias), min 5 (15 famílias, sem Naive Bayes) e corte de
  75% (19 famílias; GaussianNB, mlr:naiveBayes e weka:lwl, com 42 a 43 bases, ficam na borda)
- Famílias abaixo do corte ficam fora por cobertura. Ex.: xgboost do mlr (22 das 60 bases)
  é limitação registrada, sem tratamento especial
- sklearn:FKCEigenPro: pacote sklearn_extra (`fast_kernel`), 50.604 runs, 60 bases,
  3 flows (imputação, com ou sem padronização, e uma versão com a classe renomeada
  FKC_EigenPro -> FKCEigenPro), 14.913 setups; agrupada como uma família
- Evidência: 01_selecao_bases.ipynb, Seção 3 (exec 45–51).

## 9. Viés do número de runs, ruído e agregação principal
- O máximo cresce com o número de runs: correlação de Spearman entre n e o máximo por
  base com mediana de 0,46 (geral 0,17), contra 0,07 com a mediana (geral 0,05)
  -> viés de amostragem, não só qualidade
- **max_sub3**: para cada célula com n >= 3, sorteia 3 runs sem reposição, toma o máximo
  e repete 100 vezes (semente 42); o valor é a média desses máximos
  (máximo esperado de 3 runs aleatórios)
- Ruído do sorteio, duas medições:
  - R = 200, entre duas sementes (tabelas de comparação com e sem limpeza): mediana
    0,0003, p95 0,0048, máx. 0,028. É o limiar usado em "mudam acima do ruído" (seção 7).
   Confirmado: diferença absoluta entre duas sementes (1 e 2), R = 200 (01_selecao_bases.ipynb, Seção 5, exec 73, e Seção 6, exec 74 e 76).
  - Na `tab_sub3` (R = 100, 5 sementes, 1.114 células): desvio-padrão por célula entre
    sementes com mediana 0,00066, média 0,0015, p95 0,0059 e máx. 0,0217; maior
    amplitude (máx. - mín.) numa célula 0,058. As duas medições usam estatísticas e
    valores de R diferentes e não são diretamente comparáveis
- Teto de Spearman por ruído do sorteio (correlação entre o ranking de duas sementes,
  por base, 60 bases): mediana 0,992, média 0,9895, quartis 0,985 e 0,997, mín. 0,936,
  máx. 1,000. O sorteio quase não altera o ranking das famílias. Isso mede só o ruído do
  sorteio, não a incerteza da própria amostra de runs
- Distância de AUC entre famílias vizinhas no ranking de cada base (1.054 distâncias na
  `tab_sub3`): mediana 0,0030, média 0,0084, quartis 0,00078 e 0,0087, máx. 0,18. Cerca de
  um quarto dos pares vizinhos está a menos de ~0,0008, da ordem do ruído; a ordem entre
  famílias quase empatadas é instável
- Correlação de ranking por base entre max e max_sub3: mediana 0,74 (média 0,72, mín.
  0,36, máx. 0,94)
- Robustez do k: max_sub3 contra max_sub10 com mediana 0,925 (média 0,90, mín. 0,67,
  máx. 1,0), usando só as 823 das 1.114 células com n >= 10; **mantido k = 3** para não
  perder células de famílias com poucos runs
- Movimento de ranking (max -> max_sub3, média por família): SVC -10,1;
  GradientBoosting -6,7; attributeselectedclassifier -3,3; RandomForest +0,1;
  DecisionTree +0,3; FKCEigenPro +0,7; sobem sobretudo famílias com poucos runs
  (até +3,2). Interpretação: o max_sub3 também penaliza famílias sensíveis a
  hiperparâmetros (SVC, GradientBoosting), não só o viés de volume
- Concordância entre métricas: o melhor (top-1) coincide em 25% das bases; sobreposição
  do top-3 média de 1,6 de 3 (0 em 7 bases, 3 em 11). HistGradientBoosting é o melhor
  em posto médio nas duas (3,34 e 2,83), seguido do RandomForest (3,53 e 3,43); SVC vai
  de 4,54 a 14,68 e GradientBoosting de 6,17 a 12,89
- Arrependimento em AUC: escolher pelo max_sub3 e avaliar no máximo perde em média
  0,003 (mediana 0,0007; máx. 0,028); o inverso perde 0,016 (mediana 0,007; máx. 0,127)
- **Decisão: max_sub3 (k = 3) é a métrica de agregação principal; o máximo completo é a
  sensibilidade** (busca extensa de hiperparâmetros). O recomendador é avaliado nas duas
- Opcional: a esperança exata do máximo de 3 runs (soma de v(i) · C(i-1, 2) / C(n, 3) sobre
  os valores ordenados) elimina o sorteio; só vale a pena se a tabela for refeita
- Evidência: 01_selecao_bases.ipynb, Seção 4 (exec 53–57), Seção 5 (exec 73) e Seção 6 (exec 79–81 e 90–93).

## 10. Tabelas base × família (`dados/tabelas/`)
- `tab_sub3` (60 × 22), `tab_max` (sensibilidade), `tab_med` (mediana) e `tab_n` (nº de
  runs por célula), em parquet e csv; `celulas_longa.parquet`
- 206 células faltantes de 1.320 (15,6%), as de n < 3
- `tab_sub3_sem_limpeza` (max_sub3 do dado bruto, semente 1, R = 200) e
  `tab_sub3_limpo_mesma_semente` (contraparte com a mesma semente), para comparar
  limpeza sem diferença de sorteio. Não misturar com `tab_sub3` (semente 42, R = 100)
- `nucleo_sub3`, `tab_sub3_sem_fracas` e `nucleo_sub3_sem_fracas` (seções 11 e 12)
- Evidência: 01_selecao_bases.ipynb, Seção 4 (exec 63–64), Seção 5 (exec 72) e Seção 6 (exec 77 e 82–83).

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
    bibliotecas **(não verificada)**
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
- **Decisão (confirmada em 07/nov/2026)**:
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
- Outras sensibilidades: corte de 75% (19 famílias)
- Evidência: 01_selecao_bases.ipynb, Seção 5 (exec 65–72) e Seção 6 (exec 84–89).

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
- Bases com correlação de ranking baixa entre dado limpo e bruto: 125920 (0,69),
  23 (0,81), 29, 28 e 3902 (a 29 e a 28 sem explicação ainda)
- Evidência: 01_selecao_bases.ipynb, Seção 4 (exec 60–62) e Seção 6 (exec 76 e 82).

## 13. Metacaracterísticas (pymfe) e avaliação (semana 2)
- pymfe com grupos general, statistical e info-theory (73 metacaracterísticas);
  codificação das categóricas `one-hot-full` (invariante à renomeação das categorias: com
  `gray` ou codificação ordinal, 24 a 27 das 73 mudam ao renomear categorias); `attr_conc`
  descartada (instável entre rodadas idênticas); imputação por mediana (numéricos) e moda
  (categóricos), por base, **provisória**. 22 das 60 bases têm categóricas e 9 têm
  valores faltantes
- NaN residuais do pymfe (2 ou 3 por base: cat_to_num, sd_ratio e, em algumas, can_cor.sd):
  tratamento **ainda não decidido (a confirmar)**, a decidir com o pymfe rodado nas 60
  bases. O k-NN não aceita NaN, então precisa estar fechado antes do recomendador
- Planejado: fração de valores faltantes como metacaracterística e sensibilidade com
  um imputador melhor (ex.: kNN-imputer) nas 9 bases com faltantes
- Recomendador: k-NN sobre metacaracterísticas, k = 1, 3, 5, 10, com
  leave-one-dataset-out; baseline: melhor ranking médio
- Métricas de avaliação: Spearman, posição real da família recomendada e arrependimento
  em AUC (diferença entre o melhor da base e a família recomendada). **A métrica
  principal ainda não está decidida (a confirmar)**; deve ser escolhida antes de rodar o
  k-NN. O plano anterior tinha o Spearman como principal. Evidências até agora:
  - o ruído do sorteio não limita o Spearman (teto 0,992);
  - o ranking muda bastante entre max e max_sub3 (mediana 0,74) e um quarto dos pares
    vizinhos está no nível do ruído; o arrependimento em AUC é tolerante a trocas entre
    famílias quase empatadas e mede quanta AUC se perde ao seguir a recomendação
- Variantes a rodar, por prioridade **(a confirmar)**: `nucleo_sub3_sem_fracas` (principal);
  `tab_sub3_sem_fracas` (58 × 22, sensibilidade); `tab_sub3` (60 bases); versões com o
  máximo completo (`tab_max`) no núcleo e nas 58; `tab_sub3_sem_limpeza`
- Reportar no total e por grupo (bases completas e bases só-sklearn) nas variantes com
  faltantes. Para o baseline nessas variantes, usar posto normalizado pelo nº de
  famílias da base ou o núcleo (o posto bruto mistura bases de 7 a 22 famílias)
- Evidência: 02_pymfe_testes.ipynb (testes preliminares de codificação, instabilidade de attr_conc e atributos categóricos/faltantes).

## 14. Limitações a declarar no relatório
- Os runs do OpenML não vêm de um experimento controlado: mesmo controlando o volume com o max_sub3 (k = 3), a métrica mistura a qualidade intrínseca da família com quais configurações alguém optou por submeter. Além disso, o max_sub3 penaliza famílias sensíveis a hiperparâmetros, enquanto o máximo completo favorece famílias com mais runs (a mediana tab_med será avaliada como sensibilidade).
- O resultado depende da regra de limpeza (afeta ~15% das células no max_sub3)
- Famílias populares abaixo do corte de cobertura (xgboost, lightgbm) ficam fora
- Faltantes em bloco: 14 bases (runs de out/2016 a abr/2018) só têm famílias do
  sklearn, então o conjunto de candidatos difere entre os grupos. O resultado principal
  usa o núcleo de 45 bases e perde 5 famílias (mlr:naiveBayes, mlr:rpart,
  sklearn:GaussianNB, sklearn:LogisticRegression e weka:lwl); as bases fora do núcleo
  têm mais categóricas, com evidência fraca
- Duas bases fracas (3560 e 125920) e muitas saturadas: o ranking nelas é pouco informativo
- O weka entra com poucos runs por célula (mediana 1), então seu máximo é instável
- Famílias meta agrupam estimadores internos diferentes
- O ranking pode mudar com outros runs: o teto de Spearman só mede o ruído do sorteio

## 15. Pendências
- Decidir a métrica principal de avaliação do recomendador (seção 13)
- Decidir o tratamento dos NaN residuais do pymfe e a imputação definitiva (semana 2)
- Rodar o pymfe nas 60 bases, a fração de faltantes como metacaracterística e a
  sensibilidade com kNN-imputer (semana 2)
- Curva do máximo esperado em função de k (1, 3, 10, 30) por família, como medida de
  sensibilidade ao tuning (opcional)
- Comparar o max_sub3 com a esperança exata do máximo de 3 runs (opcional)
- Confirmar a causa dos faltantes em bloco (a divisão temporal já está medida)
- Contar os runs das demais famílias weka faltantes em 14952 e 14954 (opcional)
- Investigar o resíduo da task 3022 e as correlações baixas das bases 29 e 28
- Corrigir a decisão de aceitar flows do mlr no relatório (a decisão 4 original só
  previa sklearn e weka)