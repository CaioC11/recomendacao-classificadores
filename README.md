# Recomendação de Classificadores via Meta-Aprendizado (OpenML-CC18)

Experimento de meta-aprendizado para recomendação de famílias de algoritmos de classificação sobre a suíte **OpenML-CC18**, utilizando histórico público de execuções (*runs*) avaliadas por duas métricas: área sob a curva ROC (`area_under_roc_curve`) e acurácia (`predictive_accuracy`); e metacaracterísticas extraídas com `pymfe`. Os resultados do recomendador são reportados nas duas métricas, também estratificados entre bases binárias (31) e multiclasse (29).

> **Estado atual:** a Fase 1 (seleção, limpeza e tabelas base × família) está concluída em AUC. A extração da acurácia, as tabelas equivalentes e o recomendador são a Fase 2.

## Estrutura do Repositório

```text
recomendacao-classificadores/
├── README.md
├── decisions.md                            #Registro completo das decisões
├── requirements.txt                        #Dependências
├── .gitignore
├── src/
│   └── extracao_runs.py                    #Script CLI
├── notebooks/
│   ├── 01_preparacao_dados.ipynb           #Seleção, limpeza v3, famílias, agregação max_sub3 e diagnósticos
│   └── 02_pymfe_testes.ipynb               #Testes de estabilidade, codificação categórica e faltantes no pymfe
└── data/
    ├── cobertura_tasks.csv                 #Cobertura inicial de runs e flows nas 60 tasks selecionadas
    ├── contagem_runs.csv                   #Estimativa de volume de runs por task usada na extração
    ├── flows.csv                           #Catálogo dos 2.794 flows extraídos (flow_id e flow_name)
    ├── familias.csv                        #Mapeamento das 657 famílias, cobertura e flag das 22 finais
    └── tabelas/
        ├── celulas_longa.parquet           #Tabela em formato longo das 1.114 células válidas (n >= 3)
        ├── nucleo_sub3_sem_fracas.{csv,parquet}   #Matriz principal: 45 bases × 17 famílias (sem NaN, max_sub3)
        ├── nucleo_sub3.{csv,parquet}              #Núcleo completo antes do corte de bases fracas (46 × 17)
        ├── tab_sub3.{csv,parquet}                 #Matriz completa limpa em max_sub3 (60 × 22, 15,6% NaN)
        ├── tab_sub3_sem_fracas.{csv,parquet}      #Sensibilidade sem imputação e sem bases fracas (58 × 22)
        ├── tab_sub3_sem_limpeza.{csv,parquet}     #Sensibilidade em max_sub3 calculada sobre os runs brutos (60 × 22)
        ├── tab_max.{csv,parquet}                  #Sensibilidade agregada pelo máximo completo da célula (60 × 22)
        ├── tab_med.{csv,parquet}                  #Sensibilidade agregada pela mediana da célula (60 × 22)
        └── tab_n.{csv,parquet}                    #Contagem de runs válidos por célula (60 × 22)
```

> **Arquivos locais ignorados no Git (`.gitignore`):** os 3.655 arquivos brutos compactados em `data/runs/task_<id>/pagina_*.csv.gz`, os `.parquet` intermediários consolidados em `data/` (`runs_consolidado.parquet`, `runs_limpos*.parquet`, `runs_v3_familia*.parquet`) e `data/extracao.log` não são versionados devido ao tamanho.Todas as tabelas abaixo são de AUC. As equivalentes de acurácia (principal e sensibilidades) serão geradas na Fase 2, com os mesmos `run_id` da limpeza da AUC. A lista de bases fracas (`3560` e `125920`, definida por `AUC_max < 0,70`) vale para as duas métricas.

## Tabelas Base × Família (`data/tabelas/`)

| Arquivo | Dimensão | Descrição | Uso no experimento |
| :--- | :---: | :--- | :--- |
| `nucleo_sub3_sem_fracas` | 45 × 17 | `max_sub3` (runs limpos), sem células faltantes e sem bases fracas (`AUC_max < 0,70`) | **Tabela principal** de avaliação do recomendador |
| `tab_sub3_sem_fracas` | 58 × 22 | `max_sub3` (runs limpos) para todas as 22 famílias, excluindo as bases `3560` e `125920` | Sensibilidade de cobertura sem imputação |
| `tab_sub3` | 60 × 22 | `max_sub3` (runs limpos) nas 60 bases e 22 famílias (1.114 células observadas; 206 `NaN`) | Sensibilidade incluindo bases fracas |
| `nucleo_sub3` | 46 × 17 | Submatriz sem `NaN` antes da remoção da base fraca `3560` | Referência intermediária do núcleo |
| `tab_sub3_sem_limpeza` | 60 × 22 | `max_sub3` exato calculado sobre os *runs* brutos (sem a limpeza `v3`) | Sensibilidade ao efeito da limpeza |
| `tab_max` | 60 × 22 | Máximo observado de AUC por célula (runs limpos) | Sensibilidade de agregação (viés de volume/tuning) |
| `tab_med` | 60 × 22 | Mediana observada de AUC por célula (runs limpos) | Sensibilidade de agregação (desempenho típico) |
| `tab_n` | 60 × 22 | Número de *runs* válidos em cada célula (`n >= 3`) | Diagnóstico de volume amostral |

Todas as 8 matrizes estão disponíveis em `.parquet` e `.csv` (indexadas por `task_id`).

## Como Reproduzir

### 1. Ambiente

Crie e ative um ambiente virtual Python (testado em Python 3.14) e instale as dependências:

```bash
python -m venv .venv
#Linux/macOS: source .venv/bin/activate
#Windows (PowerShell): .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2. Extração dos runs brutos (opcional — tabelas já versionadas)

Para baixar localmente todas as páginas de avaliações (`area_under_roc_curve`) das 60 *tasks* em `data/runs/`:

```bash
python src/extracao_runs.py --fase 1       # 42 tasks menores (<= 70.000 runs)
python src/extracao_runs.py --fase 2       # 18 tasks maiores (> 70.000 runs)
#Ou em execução única:
python src/extracao_runs.py --fase todas --dados data
```

Opções disponíveis via `python src/extracao_runs.py --help`:
- `--fase {1,2,todas}`: etapa de extração (`1`, `2` ou `todas`; padrão: `1`).
- `--task TASK`: baixa apenas uma *task* específica para teste.
- `--dados DADOS`: diretório base de dados (padrão: `data`).
- `--limite-fase1 LIMITE_FASE1`: corte de volume de *runs* entre as fases 1 e 2 (padrão: `70000`).

> Na Fase 1 só a AUC foi extraída. A extração da acurácia (parametrizar `FUNCAO` com `--funcao`) é pendência da Fase 2 (ver `decisions.md`, seção 15).

### 3. Preparação das matrizes e diagnósticos

- **`notebooks/01_preparacao_dados.ipynb`**:
  - Com `data/runs/` presente localmente, execute a partir da **Seção 2 (`Integridade e limpeza`)** até o fim para regenerar todos os `.parquet` intermediários e as tabelas em `data/tabelas/` sem refazer as consultas de paginação da API na Seção 1.
- **`notebooks/02_pymfe_testes.ipynb`**:
  - Pode ser executado diretamente (*Restart & Run All*) para reproduzir os testes de invariância categórica (`one-hot-full`), determinismo (`attr_conc`) e imputação de faltantes com `pymfe`.

## Documentação Metodológica

Todas as justificativas quantitativas, verificações de integridade, análises de sensibilidade e limitações registradas antes da modelagem estão detalhadas em [`decisions.md`](decisions.md).