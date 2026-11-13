# Recomendação de Classificadores

Pipeline experimental de meta-aprendizado para recomendação de famílias de classificadores sobre a suíte **OpenML-CC18**, utilizando metacaracterísticas estruturais das bases (`pymfe`) e avaliação por *leave-one-dataset-out* (LODO).

**Autor:** Caio

---

## Visão geral do experimento

O experimento está dividido em duas etapas principais:

1. **Preparação dos dados e construção das tabelas base × família (concluída):**
   - **Bases:** 60 *tasks* da suíte OpenML-CC18 (31 binárias e 29 multiclasse), selecionadas entre 500 e 50.000 instâncias, até 500 atributos e avaliação uniforme via *10-fold cross-validation*.
   - **Extração e limpeza:** 3.624.525 *runs* extraídos via API do OpenML (métrica `area_under_roc_curve`), reduzidos a **2.778.717 *runs*** após a remoção de preditores constantes degenerados concentrados próximos a $\text{AUC} = 0{,}5$.
   - **Famílias e agregação:** agrupamento pela família do estimador final com corte de $\ge 3$ *runs* por célula e presença em $\ge 70\%$ das bases (**22 famílias** de `sklearn`, `weka` e `mlr`). O desempenho por célula é agregado pelo máximo esperado de $k = 3$ tentativas aleatórias (`max_sub3`, $R = 100$), controlando o viés de volume de *tuning*.
   - **Matrizes de avaliação:** remoção prévia de 2 bases fracas ($\text{AUC}_{\max} < 0{,}70$), definindo o **núcleo sem faltantes (`45 × 17`)** como cenário principal e a tabela completa sem imputação (**`58 × 22`**) como análise de sensibilidade.

2. **Metacaracterísticas e recomendador (em andamento):**
   - Extração de metacaracterísticas (`general`, `statistical` e `info-theory`) via `pymfe` com codificação `one-hot-full`.
   - Avaliação de recomendador baseado em $k$-NN sobre metacaracterísticas via validação *leave-one-dataset-out*.

Toda a justificativa metodológica, testes de diagnóstico e análises de sensibilidade estão documentados em [`decisions.md`](decisions.md).

---

## Estrutura do repositório

```text
recomendacao-classificadores/
├── decisions.md                         #Registro definitivo das decisões e evidências numéricas
├── requirements.txt                     #Dependências do ambiente Python
├── notebooks/
│   ├── 01_preparacao_dados.ipynb        #Fase 1: seleção, limpeza, famílias, agregação e núcleo
│   └── 02_pymfe_testes.ipynb            #Fase 2: testes de codificação e diagnóstico do pymfe
└── src/
    └── extracao_runs.py                 #Script CLI com retomada automática para download dos runs
```

> **Nota sobre os dados:** Os arquivos brutos e intermediários (`data/runs/`, arquivos `.parquet` e tabelas `.csv`) não são versionados no Git (`.gitignore`) devido ao volume (~3,6 milhões de registros), mas podem ser integralmente reproduzidos a partir do script e dos notebooks acima.

---

## Como reproduzir

### 1. Configuração do ambiente

```bash
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Extração dos runs do OpenML (opcional)

Caso não possua a pasta `data/runs/` localmente, execute o extrator a partir da raiz do repositório (o script suporta interrupção e retomada automática):

```bash
python src/extracao_runs.py --fase 1     # Tasks com até 70.000 runs
python src/extracao_runs.py --fase 2     # Tasks maiores
```

### 3. Execução dos notebooks

1. **`notebooks/01_preparacao_dados.ipynb`:** consolida os *runs*, aplica a limpeza `v3`, agrupa as 22 famílias, calcula `max_sub3` e gera as matrizes finais em `data/tabelas/` (`nucleo_sub3_sem_fracas.parquet`, `tab_sub3_sem_fracas.parquet`, `tab_max.parquet`, `tab_med.parquet` e `tab_sub3_sem_limpeza.parquet`). *(As células iniciais de consulta à API na Seção 1 podem ser puladas caso `cobertura_tasks.csv` e `contagem_runs.csv` já estejam em `data/`.)*
2. **`notebooks/02_pymfe_testes.ipynb`:** executa os testes preliminares de extração e estabilidade das metacaracterísticas com `pymfe`.