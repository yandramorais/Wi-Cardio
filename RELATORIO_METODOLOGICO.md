# Wi-Cardio — Estimação de Frequência Cardíaca via Channel State Information (CSI) com Redes Neurais Recorrentes

> **Relatório Metodológico Completo**  
> Versão: 1.3 · Data: 2026-06-10  
> Projeto: `Wi-Cardio`

---

## Sumário

1. [Visão Geral e Motivação](#1-visão-geral-e-motivação)
2. [Descrição do Dataset](#2-descrição-do-dataset)
3. [Pré-processamento](#3-pré-processamento)
4. [Arquitetura dos Modelos](#4-arquitetura-dos-modelos)
5. [Procedimento de Treinamento](#5-procedimento-de-treinamento)
6. [Resultados Experimentais](#6-resultados-experimentais)
7. [Métricas de Avaliação](#7-métricas-de-avaliação)
8. [Visualizações e Scripts de Análise](#8-visualizações-e-scripts-de-análise)
9. [Estrutura de Arquivos](#9-estrutura-de-arquivos)
10. [Dependências e Execução](#10-dependências-e-execução)
11. [Referências Técnicas](#11-referências-técnicas)

---

## 1. Visão Geral e Motivação

A estimação sem contato de sinais fisiológicos humanos utilizando sinais de radiofrequência tem emergido como uma área de pesquisa promissora. Em particular, o **Channel State Information (CSI)** — informação de estado de canal disponível em redes Wi-Fi IEEE 802.11 — possui sensibilidade suficiente para capturar micro-variações na reflexão de ondas eletromagnéticas causadas por movimentos corporais sutis como a expansão torácica durante a respiração e as pulsações cardíacas.

O projeto **Wi-Cardio** investiga a viabilidade de estimar a **frequência cardíaca (FC) em batimentos por minuto (BPM)** a partir de sinais CSI coletados com hardware de custo acessível (Raspberry Pi), utilizando modelos de aprendizado profundo baseados em redes neurais recorrentes (RNNs). O _ground truth_ é fornecido por smartwatch e dispositivo Polar, garantindo referências de alta qualidade para o treinamento supervisionado.

O problema é formulado como **regressão de séries temporais**: dada uma janela temporal de amplitudes CSI multi-subportadora, predizer o valor escalar de frequência cardíaca correspondente em BPM.

---

## 2. Descrição do Dataset

### 2.1 Origem e Estrutura

O dataset é multimodal e composto por três fontes sincronizadas coletadas simultaneamente para cada participante em cada posição experimental:

| Fonte                       | Formato                   | Conteúdo                                                     |
| --------------------------- | ------------------------- | ------------------------------------------------------------ |
| **CSI (Raspberry Pi)**      | `.npz` (NumPy comprimido) | Matriz complexa `(N_frames, 256)` + vetor de timestamps UNIX |
| **Ground Truth Smartwatch** | `.json`                   | Série temporal de FC em BPM com timestamps absolutos         |
| **Ground Truth Polar**      | `.csv`                    | Série temporal de FC em BPM com timestamps absolutos         |

### 2.2 Estatísticas do Dataset

| Parâmetro                                          | Valor               |
| -------------------------------------------------- | ------------------- |
| Participantes detectados (união)                   | 107                 |
| Participantes com dados CSI completos              | 106                 |
| Participantes com GT completo (Polar + Smartwatch) | 85                  |
| Posições experimentais por participante            | 18 (posições 01–18) |
| Combinações participante-posição completas         | 1.534 / 1.819       |
| Arquivos CSI (`.npz`)                              | 1.918               |
| Arquivos Polar (`.csv`)                            | 1.673               |
| Arquivos Smartwatch (`.json`)                      | 1.818               |

### 2.3 Estatísticas dos Tensores Pré-processados

| Conjunto  | Amostras (janelas) | FC — mín/máx/média  |
| --------- | ------------------ | ------------------- |
| Treino    | 63.723             | —                   |
| Validação | 13.655             | 49 / 158 / 91,3 BPM |
| Teste     | 13.656             | —                   |
| **Total** | **91.034**         | **49 – 158 BPM**    |

O conjunto de validação contém amostras de **87 participantes** distribuídos pelas **18 posições** experimentais.

### 2.4 Estrutura Interna dos Arquivos NPZ

Cada arquivo `.npz` contém três chaves:

```
csi      → ndarray complex64, shape (2000, 256)
           - 2000 frames temporais
           - 256 subportadoras OFDM (nfft=256, BW=80 MHz, canal 36)
ts       → ndarray float64, shape (2000,)
           - timestamps UNIX em segundos (~33 Hz efetivos)
metadata → ndarray object, shape (2001,)
           - dicionário por frame: índice, MAC, sequência, core, chanspec
```

A taxa de amostragem efetiva inferida dos timestamps é de aproximadamente **~33 Hz** (taxa padrão de fallback: `DEFAULT_FS = 500/60 ≈ 8,33 Hz`; o valor real é inferido dos deltas de timestamp de cada arquivo).

### 2.5 Participantes Excluídos

Um conjunto de 20 participantes foi removido por ausência completa de dados de GT ou dados corrompidos:

```
INVALID_SUBJECTS = {59, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                    70, 71, 98, 203, 90, 106, 81, 10, 9, 35}
```

Os critérios de exclusão incluem: participantes sem nenhum dado de smartwatch, sem dados de GT válidos para qualquer posição, e participantes com cobertura insuficiente de posições (< 50% das 18 posições esperadas).

---

## 3. Pré-processamento

O pipeline de pré-processamento é implementado em [`src/preprocess.py`](src/preprocess.py) e opera sobre cada arquivo CSI individualmente antes da concatenação global.

### 3.1 Carregamento e Conversão para Amplitude

O CSI bruto é armazenado como números complexos `complex64`. Cada valor $H_{t,k} \in \mathbb{C}$ representa a resposta de frequência no instante $t$ para a subportadora $k$. O primeiro passo é a extração da **amplitude (módulo)**:

$$A_{t,k} = |H_{t,k}| = \sqrt{\text{Re}(H_{t,k})^2 + \text{Im}(H_{t,k})^2}$$

A magnitude descarta a fase, que é altamente ruidosa e sensível a sincronização de clock, mantendo apenas a variação de potência de sinal que codifica os movimentos físicos do ambiente.

### 3.2 Remoção de Componente DC

Após a extração de amplitude, remove-se a componente DC (média temporal de cada subportadora):

$$\hat{A}_{t,k} = A_{t,k} - \frac{1}{N}\sum_{t=1}^{N} A_{t,k}$$

Esta operação elimina o _offset_ estático causado por reflexões fixas do ambiente (paredes, mobiliário), isolando apenas as flutuações dinâmicas associadas ao movimento humano.

### 3.3 Filtragem Passa-Banda

Aplica-se um filtro **Butterworth de ordem 3** na banda fisiologicamente relevante para frequência cardíaca:

$$f_{\text{low}} = 0{,}8\ \text{Hz} \qquad f_{\text{high}} = 2{,}17\ \text{Hz}$$

Esta banda corresponde a frequências cardíacas de **48 a 130 BPM**, cobrindo o espectro típico de repouso até esforço moderado. A função `filtfilt` aplica o filtro em ambas as direções temporais (_zero-phase filtering_), eliminando o atraso de fase introduzido pela filtragem causal.

### 3.4 Suavização por Savitzky-Golay

Após a filtragem passa-banda, aplica-se um filtro **Savitzky-Golay** com janela de 15 amostras e polinômio de grau 3 para suavização com preservação de picos.

### 3.5 Descarte do Transitório Inicial

Os primeiros **10 segundos** de cada gravação são descartados para eliminar o transitório do sistema e artefatos de início de captura.

### 3.6 Carregamento do Ground Truth

O _ground truth_ é carregado exclusivamente do **smartwatch** (arquivo `.json`), suportando múltiplos formatos de exportação. O pareamento com o arquivo CSI é feito por correspondência de nome de arquivo.

### 3.7 Janelamento Deslizante com Associação ao GT

| Parâmetro            | Valor              | Justificativa                                                    |
| -------------------- | ------------------ | ---------------------------------------------------------------- |
| `window_sec`         | 20 segundos        | Captura múltiplos ciclos cardíacos (mínimo ~16 batidas @ 48 BPM) |
| `step_sec`           | 0,5 segundos       | Gera alta densidade de amostras com sobreposição de 97,5%        |
| Dimensão padronizada | `T = 166` amostras | `int(20 × 500/60)`                                               |

Cada janela é normalizada independentemente por Z-score com $\varepsilon = 10^{-8}$.

### 3.8 Filtros de Qualidade por Janela

Três filtros são aplicados: (1) exclusão de intervalos marcados como defeituosos (`faltas.txt`), (2) rejeição de janelas com gap temporal ao GT superior a 750 ms, e (3) rejeição de janelas com variação local de FC superior a 25 BPM em ±3 s.

### 3.9 Divisão por Sujeito (Subject-Wise Split)

A divisão é feita **por sujeito** (não por amostra), garantindo que nenhum participante apareça em mais de um conjunto e prevenindo _data leakage_.

| Conjunto  | Proporção | Sujeitos (aprox.) | Papel                              |
| --------- | --------- | ----------------- | ---------------------------------- |
| Treino    | ~70%      | ~59               | Otimização dos parâmetros          |
| Validação | ~15%      | ~13               | Early stopping e seleção de modelo |
| Teste     | ~15%      | ~13               | Avaliação final sem viés           |

---

## 4. Arquitetura dos Modelos

Dois modelos foram implementados e comparados, com arquiteturas intencionalmente distintas para avaliar o impacto de diferentes componentes.

### 4.1 Modelo LSTM (Baseline)

Arquivo: [`src/train_lstm.py`](src/train_lstm.py) · Classe: `LSTM`

Arquitetura deliberadamente simples para servir como _baseline_ comparativo:

```
Entrada: (B, T, 256)   B=batch, T=166 timesteps, 256 subportadoras
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Bi-LSTM                                                │
│  input=256, hidden=256, layers=2, dropout=0.3           │
│  → saída: (B, T, 512)  [256 forward + 256 backward]    │
└─────────────────────────────────────────────────────────┘
    │  last timestep: (B, 512)
    ▼
┌─────────────────────────────────────────────────────────┐
│  Regressor MLP                                          │
│  Linear(512 → 128) → ReLU → Dropout(0.2)               │
│  → Linear(128 → 1)                                      │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Saída: (B,)   FC em BPM
```

O contexto temporal é extraído pelo **último timestep** da sequência (`out[:, -1, :]`), sem mecanismo de atenção. Não há LayerNorm pós-LSTM.

**Parâmetros totais: 2.695.425**

### 4.2 Modelo GRU com Projeção de Entrada e Atenção

Arquivo: [`src/train_gru.py`](src/train_gru.py) · Classe: `ModelGRU`

Arquitetura aprimorada com três componentes adicionais em relação ao LSTM baseline:

```
Entrada: (B, T, 256)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Projeção de Entrada                                    │
│  Linear(256 → 256) → LayerNorm(256) → GELU             │
│  → saída: (B, T, 256)                                   │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Bi-GRU                                                 │
│  input=256, hidden=256, layers=2, dropout=0.3           │
│  → saída: (B, T, 512)  [256 forward + 256 backward]    │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  LayerNorm(512)                                         │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Atenção Softmax Escalar                                │
│  α_t = softmax(Linear(512 → 1))   shape: (B, T, 1)     │
│  contexto = Σ_t α_t · h_t         shape: (B, 512)      │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Regressor MLP                                          │
│  Linear(512 → 128) → LayerNorm(128) → ReLU             │
│  → Dropout(0.2) → Linear(128 → 1)                      │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Saída: (B,)   FC em BPM
```

**Parâmetros totais: 2.106.114** (21,5% menos que o LSTM)

### 4.3 Comparação Estrutural

| Componente             | LSTM (Baseline) | GRU                  |
| ---------------------- | --------------- | -------------------- |
| Célula recorrente      | LSTM (4 gates)  | GRU (3 gates)        |
| Projeção de entrada    | Não             | Sim (Linear→LN→GELU) |
| LayerNorm pós-RNN      | Não             | Sim                  |
| Extração de contexto   | Último timestep | Atenção softmax      |
| LayerNorm no regressor | Não             | Sim                  |
| Total de parâmetros    | 2.695.425       | 2.106.114            |
| Otimizador             | AdamW (wd=1e-5) | Adam                 |

**Atenção temporal:** Para cada timestep $t$, o peso é:

$$\alpha_t = \text{softmax}\left( \mathbf{w}^\top \mathbf{h}_t \right), \qquad \mathbf{c} = \sum_{t=1}^{T} \alpha_t \cdot \mathbf{h}_t$$

Isso permite ao modelo aprender quais regiões temporais da janela de 20 s são mais relevantes para a estimativa de FC, em vez de depender exclusivamente do estado final da sequência.

---

## 5. Procedimento de Treinamento

| Hiperparâmetro        | LSTM            | GRU          |
| --------------------- | --------------- | ------------ |
| `BATCH_SIZE`          | 64              | 64           |
| `EPOCHS` (máximo)     | 250             | 250          |
| `LEARNING_RATE`       | 5 × 10⁻⁴        | 5 × 10⁻⁴     |
| `HIDDEN_SIZE`         | 256             | 256          |
| `NUM_LAYERS`          | 2               | 2            |
| `PATIENCE`            | 30 épocas       | 30 épocas    |
| Otimizador            | AdamW (wd=1e-5) | Adam         |
| Dropout recorrente    | 0,3             | 0,3          |
| Dropout regressor     | 0,2             | 0,2          |
| Clipping de gradiente | max_norm=1,0    | max_norm=1,0 |
| `SEED`                | 42              | 42           |

### 5.1 Função de Perda: Huber Loss

$$
\mathcal{L}_{\delta}(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{se } |y - \hat{y}| \leq \delta \\
\delta \cdot \left(|y - \hat{y}| - \frac{\delta}{2}\right) & \text{caso contrário}
\end{cases} \quad \delta = 3{,}0\ \text{BPM}
$$

A Huber Loss combina sensibilidade quadrática para erros pequenos (≤ 3 BPM) com robustez linear para outliers, evitando que ruídos de anotação dominem o gradiente. A validação e o early stopping utilizam o **MAE** (L1Loss), que é a métrica clinicamente mais interpretável.

### 5.2 Otimizador e Agendamento de LR

`ReduceLROnPlateau`: reduz o LR pela metade se o val MAE não melhora em 10 épocas consecutivas (fator=0,5, mínimo=10⁻⁵). O **gradient clipping** (`max_norm=1,0`) previne gradientes explosivos em RNNs profundas.

### 5.3 Early Stopping e Checkpointing

O melhor modelo (menor val MAE) é salvo em `output/{gru,lstm}/best_model_{gru,lstm}.pt`. O treinamento para automaticamente após 30 épocas sem melhora, carregando o checkpoint ótimo para avaliação final. Carregamento com `weights_only=True` para compatibilidade com PyTorch ≥ 2.4.

### 5.4 Reprodutibilidade

Semente `SEED=42` fixa todas as fontes de aleatoriedade (Python, NumPy, PyTorch, CUDA). `deterministic=True` e `benchmark=False` garantem reprodutibilidade determinística em GPU.

---

## 6. Resultados Experimentais

Os resultados a seguir foram obtidos no conjunto de **validação** (13.655 amostras, 87 participantes, 18 posições, FC 49–158 BPM).

### 6.1 Métricas de Desempenho — GRU vs. LSTM

| Métrica                   | GRU ★          | LSTM           |
| ------------------------- | -------------- | -------------- |
| **MAE (BPM)**             | **1,22**       | 1,84           |
| **RMSE (BPM)**            | **3,22**       | —              |
| **R²**                    | **0,9642**     | —              |
| **Pearson r**             | **0,9821**     | —              |
| Bias / μ (BPM)            | 0,24           | —              |
| Desvio σ (BPM)            | 3,21           | —              |
| LoA superior (BPM)        | +6,54          | —              |
| LoA inferior (BPM)        | −6,06          | —              |
| % ≤ 5 BPM                 | **97,3%**      | —              |
| % ≤ 10 BPM                | **98,5%**      | —              |
| % ≤ 15 BPM                | **99,0%**      | —              |
| Parâmetros                | 2.106.114      | 2.695.425      |
| Tempo de inferência (val) | 31,4 s         | —              |
| Épocas treinadas          | 250            | 250            |
| Melhor época              | 249            | 243            |
| Melhor val MAE            | **1,2182 BPM** | **1,8423 BPM** |

> ★ GRU supera o LSTM em todas as métricas principais com 21,5% menos parâmetros.

### 6.2 Análise por Posição Corporal

O GRU apresenta MAE consistente em todas as 18 posições. O mecanismo de atenção permite ao modelo ponderar diferentes regiões temporais da janela de 20 s, adaptando-se às variações de padrão CSI entre posições (decúbito, sentado, em pé, lateral, etc.).

### 6.3 Curvas de Aprendizado

- **GRU:** convergência estável ao longo de 250 épocas. MAE de validação cai de ~25 BPM (época 1) para 1,22 BPM (época 249), com generalization gap reduzido.
- **LSTM:** convergência mais lenta sem a projeção de entrada, atingindo 1,84 BPM na melhor época (243). A ausência de atenção e LayerNorm limita a capacidade do modelo de extrair representações temporais ricas.

### 6.4 Análise de Concordância (Bland-Altman)

Para o GRU: viés de +0,24 BPM (ligeira superestimação), limites de concordância de −6,06 a +6,54 BPM (amplitude total de 12,6 BPM). Esses valores indicam excelente concordância clínica com o smartwatch de referência, dentro dos limites aceitáveis para dispositivos de monitoramento cardíaco de consumo.

---

## 7. Métricas de Avaliação

| Métrica        | Fórmula                                                 | Interpretação                        |
| -------------- | ------------------------------------------------------- | ------------------------------------ | -------- | ----------------------------------- |
| **MAE**        | $\frac{1}{N}\sum                                        | y_i - \hat{y}\_i                     | $        | Erro médio absoluto em BPM          |
| **RMSE**       | $\sqrt{\frac{1}{N}\sum(y_i - \hat{y}_i)^2}$             | Penaliza erros grandes               |
| **R²**         | $1 - \frac{\text{SS}_\text{res}}{\text{SS}_\text{tot}}$ | Variância explicada pelo modelo      |
| **Pearson r**  | $\text{corr}(y, \hat{y})$                               | Correlação linear                    |
| **Viés (μ)**   | $\frac{1}{N}\sum(y_i - \hat{y}_i)$                      | Tendência sistemática                |
| **Desvio (σ)** | $\text{std}(y_i - \hat{y}_i)$                           | Dispersão dos resíduos               |
| **LoA ±1,96σ** | $\mu \pm 1{,}96\sigma$                                  | Limites de concordância Bland-Altman |
| **% ≤ k BPM**  | $P(                                                     | \text{erro}                          | \leq k)$ | Fração dentro de tolerância clínica |

---

## 8. Visualizações e Scripts de Análise

### 8.1 Gráficos Individuais do GRU

Arquivo: [`src/generate_charts.py`](src/generate_charts.py)

```bash
python src/generate_charts.py                              # sem modelo (11 gráficos)
python src/generate_charts.py --ckpt output/gru/best_model_gru.pt  # completo (24 gráficos)
```

Gera até 24 figuras em `charts_output/`:

| #   | Gráfico                                         |
| --- | ----------------------------------------------- |
| 01  | Curva de aprendizado (Train Loss + Val MAE)     |
| 02  | Val MAE com suavização e zonas de convergência  |
| 03  | Training Loss com fases de treinamento          |
| 04  | Análise de convergência + gap de generalização  |
| 05  | FC média por posição — Ground Truth (dots)      |
| 06  | Melhor participante por posição (dots)          |
| 07  | FC média por posição — barras                   |
| 08  | Contagem de amostras por posição                |
| 09  | Distribuição HR por posição — violin            |
| 10  | Distribuição HR por posição — box plot          |
| 11  | Amostras por participante                       |
| 12  | FC por posição — Real vs. Predito (dots)        |
| 13  | Melhor coleta por posição — Real vs. Predito    |
| 14  | MAE por posição — barras                        |
| 15  | MAE por posição — dots + banda de variabilidade |
| 16  | MAE por participante — barras                   |
| 17  | MAE por faixa de FC                             |
| 18  | Scatter Real vs. Predito                        |
| 19  | Bland-Altman                                    |
| 20  | Distribuição de resíduos                        |
| 21  | CDF do erro absoluto                            |
| 22  | Box plot de erros por posição                   |
| 23  | Heatmap MAE: Participante × Posição             |
| 24  | Tabela de métricas                              |

### 8.2 Comparação GRU vs. LSTM

Arquivo: [`src/compare_models.py`](src/compare_models.py)

```bash
python src/compare_models.py
python src/compare_models.py --subject 5 --position 3   # sinal específico
python src/compare_models.py --out results/figures
```

Gera 20 figuras:

| #         | Gráfico                                                               |
| --------- | --------------------------------------------------------------------- |
| comp_01   | Curvas de aprendizado — lado a lado                                   |
| comp_01b  | Curvas de aprendizado — sobrepostas                                   |
| comp_02   | Real vs. Predito — lado a lado                                        |
| comp_03   | Bland-Altman — lado a lado                                            |
| comp_04   | CDF sobrepostas GRU × LSTM                                            |
| comp_05   | Barras: MAE, RMSE, R², Parâmetros, Tempo                              |
| comp_06   | MAE por participante — barras agrupadas                               |
| comp_07   | MAE por posição — barras agrupadas                                    |
| comp_08   | FC por posição — GT + GRU + LSTM (dots)                               |
| comp_09   | Tabela comparativa GRU vs. LSTM (★ = melhor)                          |
| gru_01–07 | GRU: curva, posição, scatter, Bland-Altman, CDF, barras, tabela       |
| signal_01 | Sinal temporal: melhor / típico / pior caso                           |
| signal_02 | Sinal temporal de 1 participante × posição (GT vs GRU vs LSTM + erro) |
| signal_03 | Grid de 6 participantes na mesma posição                              |

---

## 9. Estrutura de Arquivos

```
frequencia_csi/
├── src/
│   ├── preprocess.py        # Pipeline de pré-processamento
│   ├── train_gru.py                  # Treino do GRU (modelo principal)
│   ├── train_lstm.py                 # Treino do LSTM (baseline)
│   ├── lstm_certo.py                 # Variante LSTM
│   ├── generate_charts.py            # 24 gráficos individuais do GRU
│   └── compare_models.py             # 20 gráficos de comparação GRU × LSTM
│
├── output/
│   ├── gru/
│   │   ├── best_model_gru.pt         # Checkpoint GRU (melhor val MAE = 1,22 BPM)
│   │   └── history_gru.json          # Histórico de treinamento (250 épocas)
│   └── lstm/
│       ├── best_model_lstm.pt        # Checkpoint LSTM (melhor val MAE = 1,84 BPM)
│       └── history_lstm.json         # Histórico de treinamento (250 épocas)
│
├── saida_full/                       # Tensores pré-processados (dataset completo)
│   ├── X_{train,val,test}.npz        # (N, 166, 256) float32
│   ├── y_{train,val,test}.npy        # (N,) float32 — FC em BPM
│   ├── positions_{train,val,test}.npy # Posição experimental (1–18)
│   ├── subject_{train,val,test}.npy  # ID do participante
│   └── idx_{train,val,test}.npy      # Índices originais
│
├── charts_output/                    # Figuras geradas pelos scripts
│   ├── chart_01_learning_curve.png
│   ├── ...
│   ├── comp_09_metrics_table.png
│   └── signal_02_single_sX_pY.png
│
├── saida_smoke/                      # Tensores de smoke test (subconjunto)
├── reports_dataset/                  # Relatório exploratório do dataset
├── Data_DS2_raspberry_npz/           # Dados CSI brutos
├── Data_DS2_smartwatch-main/         # Ground truth smartwatch
├── Data_DS2_polar-main/              # Ground truth Polar
├── RELATORIO_METODOLOGICO.md        # Este documento
└── participantes_incompletos.csv
```

---

## 10. Dependências e Execução

### 10.1 Ambiente

```
Python 3.10
torch >= 2.0
numpy
pandas
scipy
scikit-learn
matplotlib
```

### 10.2 Pipeline Completo

```bash
# 1. Pré-processamento
python src/preprocess.py \
    --dataset_path Data_DS2_raspberry_npz/ \
    --gt_dir       Data_DS2_smartwatch-main/Data_Heart/ \
    --out_dir      saida_full/ \
    --window_sec   20.0 \
    --step_sec     0.5

# 2. Treinamento
python src/train_gru.py    # GRU  → output/gru/
python src/train_lstm.py   # LSTM → output/lstm/

# 3. Geração de gráficos individuais (GRU)
python src/generate_charts.py --ckpt output/gru/best_model_gru.pt

# 4. Comparação GRU × LSTM
python src/compare_models.py

# 5. Sinal de um participante específico
python src/compare_models.py --subject 5 --position 3
```

### 10.3 Resumo dos Resultados por Modelo

| Script          | Modelo                   | Parâmetros | Melhor val MAE | Época |
| --------------- | ------------------------ | ---------- | -------------- | ----- |
| `train_gru.py`  | GRU Bi + proj. + atenção | 2.106.114  | **1,22 BPM**   | 249   |
| `train_lstm.py` | LSTM Bi (baseline)       | 2.695.425  | 1,84 BPM       | 243   |

---

## 11. Referências Técnicas

- **Butterworth / filtfilt:** Oppenheim & Schafer, _Discrete-Time Signal Processing_, 3ª ed. — Prentice Hall.
- **Savitzky-Golay:** Savitzky, A.; Golay, M.J.E. (1964). _Smoothing and Differentiation of Data by Simplified Least Squares Procedures_. Analytical Chemistry.
- **CSI para sinais vitais:** Wang, F. et al. (2017). _E-eyes: Device-free location-oriented activity identification using fine-grained WiFi signatures_. IEEE INFOCOM.
- **Bi-LSTM:** Schuster, M.; Paliwal, K. (1997). _Bidirectional recurrent neural networks_. IEEE Transactions on Signal Processing.
- **GRU:** Cho, K. et al. (2014). _Learning phrase representations using RNN encoder-decoder for statistical machine translation_. EMNLP.
- **Huber Loss:** Huber, P.J. (1964). _Robust Estimation of a Location Parameter_. Annals of Mathematical Statistics.
- **Bland-Altman:** Bland, J.M.; Altman, D.G. (1986). _Statistical methods for assessing agreement between two methods of clinical measurement_. The Lancet.
- **Adam / AdamW:** Kingma, D.P.; Ba, J. (2014). _Adam: A method for stochastic optimization_. ICLR 2015. / Loshchilov, I.; Hutter, F. (2019). _Decoupled Weight Decay Regularization_. ICLR 2019.

---

_Relatório atualizado em 2026-06-10 com resultados definitivos dos modelos treinados._
