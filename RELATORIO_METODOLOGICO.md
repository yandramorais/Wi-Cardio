# PulseFi — Estimação de Frequência Cardíaca via Channel State Information (CSI) com Redes Neurais Recorrentes

> **Relatório Metodológico Completo**  
> Versão: 1.1 · Data: 2026-06-08  
> Projeto: `frequencia_csi`

---

## Sumário

1. [Visão Geral e Motivação](#1-visão-geral-e-motivação)
2. [Descrição do Dataset](#2-descrição-do-dataset)
3. [Pré-processamento](#3-pré-processamento)
   - 3.1 [Carregamento e Conversão para Amplitude](#31-carregamento-e-conversão-para-amplitude)
   - 3.2 [Remoção de Componente DC](#32-remoção-de-componente-dc)
   - 3.3 [Filtragem Passa-Banda](#33-filtragem-passa-banda)
   - 3.4 [Suavização por Savitzky-Golay](#34-suavização-por-savitzky-golay)
   - 3.5 [Descarte do Transitório Inicial](#35-descarte-do-transitório-inicial)
   - 3.6 [Carregamento do Ground Truth](#36-carregamento-do-ground-truth)
   - 3.7 [Janelamento Deslizante com Associação ao GT](#37-janelamento-deslizante-com-associação-ao-gt)
   - 3.8 [Filtros de Qualidade por Janela](#38-filtros-de-qualidade-por-janela)
   - 3.9 [Divisão por Sujeito (Subject-Wise Split)](#39-divisão-por-sujeito-subject-wise-split)
4. [Arquitetura dos Modelos](#4-arquitetura-dos-modelos)
   - 4.1 [Modelo LSTM Bidirecional com Atenção](#41-modelo-lstm-bidirecional-com-atenção)
   - 4.2 [Modelo GRU Bidirecional com Atenção](#42-modelo-gru-bidirecional-com-atenção)
   - 4.3 [Comparação Estrutural LSTM vs GRU](#43-comparação-estrutural-lstm-vs-gru)
5. [Procedimento de Treinamento](#5-procedimento-de-treinamento)
   - 5.1 [Função de Perda: Huber Loss](#51-função-de-perda-huber-loss)
   - 5.2 [Otimizador e Agendamento de LR](#52-otimizador-e-agendamento-de-lr)
   - 5.3 [Early Stopping e Checkpointing](#53-early-stopping-e-checkpointing)
   - 5.4 [Reprodutibilidade](#54-reprodutibilidade)
6. [Métricas de Avaliação](#6-métricas-de-avaliação)
7. [Estrutura de Arquivos](#7-estrutura-de-arquivos)
8. [Dependências e Execução](#8-dependências-e-execução)
9. [Referências Técnicas](#9-referências-técnicas)

---

## 1. Visão Geral e Motivação

A estimação sem contato de sinais fisiológicos humanos utilizando sinais de radiofrequência tem emergido como uma área de pesquisa promissora. Em particular, o **Channel State Information (CSI)** — informação de estado de canal disponível em redes Wi-Fi IEEE 802.11 — possui sensibilidade suficiente para capturar micro-variações na reflexão de ondas eletromagnéticas causadas por movimentos corporais sutis como a expansão torácica durante a respiração e as pulsações cardíacas.

O projeto **PulseFi** investiga a viabilidade de estimar a **frequência cardíaca (FC) em batimentos por minuto (BPM)** a partir de sinais CSI coletados com hardware de custo acessível (Raspberry Pi), utilizando modelos de aprendizado profundo baseados em redes neurais recorrentes (RNNs). O *ground truth* é fornecido por smartwatch e dispositivo Polar, garantindo referências de alta qualidade para o treinamento supervisionado.

O problema é formulado como **regressão de séries temporais**: dada uma janela temporal de amplitudes CSI multi-subportadora, predizer o valor escalar de frequência cardíaca correspondente em BPM.

---

## 2. Descrição do Dataset

### 2.1 Origem e Estrutura

O dataset é multimodal e composto por três fontes sincronizadas coletadas simultaneamente para cada participante em cada posição experimental:

| Fonte | Formato | Conteúdo |
|---|---|---|
| **CSI (Raspberry Pi)** | `.npz` (NumPy comprimido) | Matriz complexa `(N_frames, 256)` + vetor de timestamps UNIX |
| **Ground Truth Smartwatch** | `.json` | Série temporal de FC em BPM com timestamps absolutos |
| **Ground Truth Polar** | `.csv` | Série temporal de FC em BPM com timestamps absolutos |

### 2.2 Estatísticas do Dataset

| Parâmetro | Valor |
|---|---|
| Participantes detectados (união) | 107 |
| Participantes com dados CSI completos | 106 |
| Participantes com GT completo (Polar + Smartwatch) | 85 |
| Posições experimentais por participante | 17 (posições 01–17) |
| Combinações participante-posição completas | 1.534 / 1.819 |
| Arquivos CSI (`.npz`) | 1.918 |
| Arquivos Polar (`.csv`) | 1.673 |
| Arquivos Smartwatch (`.json`) | 1.818 |

### 2.3 Estrutura Interna dos Arquivos NPZ

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

A taxa de amostragem efetiva inferida dos timestamps é de aproximadamente **~33 Hz** (500 frames / 60 segundos → `DEFAULT_FS = 500/60 ≈ 8.33 Hz` como fallback; o valor real é inferido dos deltas de timestamp de cada arquivo).

### 2.4 Participantes Excluídos

Um conjunto de 20 participantes foi removido por ausência completa de dados de GT ou dados sabidamente corrompidos:

```
INVALID_SUBJECTS = {59, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                    70, 71, 98, 203, 90, 106, 81, 10, 9, 35}
```

Os critérios de exclusão incluem: participantes sem nenhum dado de smartwatch (`61, 64–71`), sem dados de GT válidos para qualquer posição (`203`), e participantes com cobertura insuficiente de posições (< 50% das 17 posições esperadas).

---

## 3. Pré-processamento

O pipeline de pré-processamento é implementado em [`src/preprocess-pulseFi.py`](src/preprocess-pulseFi.py) e opera sobre cada arquivo CSI individualmente antes da concatenação global.

### 3.1 Carregamento e Conversão para Amplitude

O CSI bruto é armazenado como números complexos `complex64`. Cada valor $H_{t,k} \in \mathbb{C}$ representa a resposta de frequência no instante $t$ para a subportadora $k$. O primeiro passo é a extração da **amplitude (módulo)**:

$$A_{t,k} = |H_{t,k}| = \sqrt{\text{Re}(H_{t,k})^2 + \text{Im}(H_{t,k})^2}$$

```python
def csi_to_amplitude(x: np.ndarray):
    if np.iscomplexobj(x):
        return np.abs(x)
    return x
```

A magnitude descarta a fase, que é altamente ruidosa e sensível a sincronização de clock, mantendo apenas a variação de potência de sinal que codifica os movimentos físicos do ambiente.

### 3.2 Remoção de Componente DC

Após a extração de amplitude, remove-se a componente DC (média temporal de cada subportadora):

$$\hat{A}_{t,k} = A_{t,k} - \frac{1}{N}\sum_{t=1}^{N} A_{t,k}$$

```python
def remove_dc(x: np.ndarray):
    return x - np.mean(x, axis=0)
```

Essa operação elimina o *offset* estático causado por reflexões fixas do ambiente (paredes, mobiliário), isolando apenas as flutuações dinâmicas associadas ao movimento humano. Sem essa etapa, as variações fisiológicas de pequena amplitude seriam obscurecidas pelo pedestal de energia estática.

### 3.3 Filtragem Passa-Banda

Aplica-se um filtro **Butterworth de ordem 3** na banda de frequências fisiologicamente relevante para frequência cardíaca:

$$f_{\text{low}} = 0.8\ \text{Hz} \qquad f_{\text{high}} = 2.17\ \text{Hz}$$

Essa banda corresponde a frequências cardíacas de **48 a 130 BPM**, cobrindo o espectro típico de repouso até esforço moderado.

```python
BANDPASS = (0.8, 2.17)
FILTER_ORDER = 3

def bandpass_filter(x, fs, low, high, order=FILTER_ORDER):
    b, a = butter(order, [low, high], btype="band", fs=fs)
    y = np.empty_like(x)
    for i in range(x.shape[1]):
        y[:, i] = filtfilt(b, a, x[:, i])
    return y
```

A função `filtfilt` aplica o filtro em ambas as direções temporais (*zero-phase filtering*), eliminando o atraso de fase introduzido pela filtragem causal. Cada uma das 256 subportadoras é filtrada independentemente, preservando as correlações inter-subportadora enquanto atenua as componentes fora da banda cardíaca.

A taxa de amostragem efetiva `fs_eff` é inferida diretamente dos deltas de timestamp de cada arquivo:

```python
def infer_fs(ts):
    dt = np.mean(np.diff(ts))
    return 1.0 / dt
```

Se a inferência falhar (timestamps inválidos ou ausentes), usa-se o valor padrão `DEFAULT_FS = 500/60 ≈ 8.33 Hz`.

### 3.4 Suavização por Savitzky-Golay

Após a filtragem passa-banda, aplica-se um filtro **Savitzky-Golay** para suavização adicional com preservação de picos:

$$\text{parâmetros: janela} = 15\ \text{amostras},\ \text{polinômio de grau}\ 3$$

```python
def savgol_smooth(x, window=9, poly=3):
    # build_windows chama com window=15, poly=3
    if x.shape[0] < window:
        return x
    if x.ndim == 1:
        return savgol_filter(x, window, poly)
    y = np.empty_like(x)
    for i in range(x.shape[1]):
        y[:, i] = savgol_filter(x[:, i], window, poly)
    return y
```

O filtro Savitzky-Golay ajusta um polinômio local a cada ponto da série, oferecendo suavização com preservação da forma dos picos e vales — característica fundamental para manter as oscilações rítmicas associadas ao ciclo cardíaco sem distorcer sua morfologia temporal.

### 3.5 Descarte do Transitório Inicial

Os primeiros **10 segundos** de cada gravação são descartados para eliminar o transitório do sistema e quaisquer artefatos de início de captura:

```python
cut = int(10 * fs_eff)
if X.shape[0] <= cut:
    continue
X = X[cut:]
```

Esse período de *warm-up* corresponde ao tempo necessário para a estabilização da posição do participante e do filtro digital (especialmente relevante para `filtfilt`).

### 3.6 Carregamento do Ground Truth

O *ground truth* é carregado exclusivamente do **smartwatch** (arquivo `.json`), que suporta múltiplos formatos de exportação:

```python
def load_smartwatch_gt(fp: Path) -> Optional[pd.DataFrame]:
    # Suporte a três esquemas JSON:
    # 1. {"Data": [{"HeartRate": ..., "StartTime": ...}]}
    # 2. {"heart_rate": [...], "start_time": [...]}
    # 3. Lista de objetos com "HeartRate"/"Value" e "StartTime"/"Time"
```

O pareamento com o arquivo CSI é feito por correspondência de nome de arquivo: o script remove o sufixo `_bw_*` do nome do NPZ e busca recursivamente o arquivo `<base>_HeartRateData.json` no diretório de GT. O tempo do GT é normalizado para zero (relativo ao primeiro registro).

### 3.7 Janelamento Deslizante com Associação ao GT

O núcleo do pré-processamento é a função `sliding_window_with_gt`, que segmenta o sinal CSI filtrado em **janelas deslizantes** e associa cada janela a um valor escalar de FC do smartwatch.

**Parâmetros de janelamento:**

| Parâmetro | Valor | Justificativa |
|---|---|---|
| `window_sec` | 20 segundos | Janela longa o suficiente para capturar múltiplos ciclos cardíacos (mínimo ~16 batidas @ 48 BPM) |
| `step_sec` | 0.5 segundos | Passo de 500 ms gera alta densidade de amostras com sobreposição de 97.5% |
| Dimensão da janela (amostras) | `int(20 × fs_eff)` | Adaptado à taxa real do arquivo |

**Dimensão final padronizada:**

Após a segmentação, as janelas são redimensionadas para `TARGET_WINDOW = int(window_sec × DEFAULT_FS) = int(20 × 8.33) = 166` amostras na dimensão temporal. Isso garante que todas as janelas tenham exatamente a mesma forma, independente de variações na taxa efetiva de captura entre arquivos.

**Processo de associação ao GT:**

Para cada janela centrada no instante $\bar{t}$:

1. Calcula-se $\bar{t} = \text{mean}(ts[\text{start:end}])$
2. Encontra-se o índice $i^* = \arg\min_i |t^{GT}_i - \bar{t}|$ (vizinho mais próximo no tempo)
3. Aceita-se a janela se $|t^{GT}_{i^*} - \bar{t}| \leq 0.75$ segundos (`max_gt_gap_s`)

**Normalização intra-janela (Z-score):**

Cada janela é normalizada independentemente:

$$\tilde{X}_{t,k} = \frac{X_{t,k} - \mu_{\text{janela}}}{\sigma_{\text{janela}} + \varepsilon}$$

com $\varepsilon = 10^{-8}$ para estabilidade numérica. Essa normalização local torna o modelo invariante a diferenças absolutas de potência de sinal entre posições, participantes e ambientes.

### 3.8 Filtros de Qualidade por Janela

Três mecanismos adicionais de controle de qualidade são aplicados para cada janela candidata:

**1. Exclusão por intervalos de falha:**

Intervalos de tempo marcados como defeituosos (arquivo `faltas.txt` no diretório GT) são excluídos:

```python
def is_valid_window(t_mean, fault_intervals):
    for start, end in fault_intervals:
        if start <= t_mean <= end:
            return False
    return True
```

**2. Rejeição por gap temporal ao GT:**

Janelas cujo instante central dista mais de 750 ms do GT mais próximo são descartadas:

```python
if abs_dt > max_gt_gap_s:  # max_gt_gap_s = 0.75 s
    stats["drop_gt_gap_too_large"] += 1
    continue
```

**3. Rejeição por instabilidade local da FC:**

Se dentro de uma janela de ±3 segundos em torno do instante central o GT apresentar variação de FC superior a 25 BPM, a janela é considerada instável e descartada:

```python
local_hr_window_s = 3.0
hr_jump_bpm = 25.0

mask = (gt_t >= t0) & (gt_t <= t1)
if mask.sum() >= 3:
    if np.nanmax(gt_hr[mask]) - np.nanmin(gt_hr[mask]) > hr_jump_bpm:
        stats["drop_hr_unstable_local"] += 1
        continue
```

Esse filtro elimina janelas que coincidem com transições abruptas de FC (início/fim de atividade), onde o rótulo escalar seria ambíguo.

### 3.9 Divisão por Sujeito (Subject-Wise Split)

A divisão em conjuntos de treino, validação e teste é feita **por sujeito** (não por amostra), garantindo que nenhum participante apareça em mais de um conjunto. Isso previne **data leakage** e garante que a avaliação reflita a generalização do modelo para novos indivíduos não vistos.

```python
TEST_SIZE  = 0.15  # 15% dos sujeitos
VAL_SIZE   = 0.15  # 15% dos sujeitos
# restante → treino (~70%)

rng = np.random.default_rng(RANDOM_STATE)  # RANDOM_STATE = 42
subjects = np.unique(s)
rng.shuffle(subjects)

n_test = max(1, round(n * TEST_SIZE))   # ~13 sujeitos
n_val  = max(1, round(n * VAL_SIZE))    # ~13 sujeitos
# train → ~59 sujeitos
```

Dado que há 85 participantes completos, a divisão resulta aproximadamente em:

| Conjunto | Sujeitos | Papel |
|---|---|---|
| Treino | ~59 | Otimização dos parâmetros do modelo |
| Validação | ~13 | Seleção de hiperparâmetros e early stopping |
| Teste | ~13 | Avaliação final sem viés de seleção |

Os arrays salvos preservam os índices originais (`idx_train.npy`, `idx_val.npy`, `idx_test.npy`) e os IDs de sujeitos por partição, permitindo auditoria completa e reprodutibilidade.

**Formato final dos tensores:**

| Arquivo | Shape | Descrição |
|---|---|---|
| `X_train.npz` | `(N_train, T, 256)` | Janelas CSI · treino |
| `X_val.npz` | `(N_val, T, 256)` | Janelas CSI · validação |
| `X_test.npz` | `(N_test, T, 256)` | Janelas CSI · teste |
| `y_*.npy` | `(N,)` | Rótulos de FC em BPM |
| `positions_*.npy` | `(N,)` | Posição experimental (1–17) |
| `subject_*.npy` | `(N,)` | ID do participante |

onde `T = 166` amostras (TARGET\_WINDOW = `int(20 × 500/60)`).

---

## 4. Arquitetura dos Modelos

Dois modelos são implementados com arquitetura espelhada, diferindo apenas na célula recorrente utilizada.

### 4.1 Modelo LSTM Bidirecional com Atenção

Arquivo: [`src/train_lstm.py`](src/train_lstm.py) · Classe: `PulseFiModel`

```
Entrada: (B, T, 256)   B=batch, T=166 timesteps, 256 features (subportadoras)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Bi-LSTM  (input=256, hidden=256, layers=2, dropout=0.3)│
│  → saída: (B, T, 512)   [256 forward + 256 backward]   │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  LayerNorm(512)                                         │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Atenção Softmax                                        │
│  w_t = softmax(Linear(512 → 1))   shape: (B, T, 1)     │
│  context = Σ_t w_t · h_t          shape: (B, 512)      │
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
Saída: (B,)   FC em BPM (valor escalar por janela)
```

**Total de parâmetros estimado:** ~2,9M

### 4.2 Modelo GRU Bidirecional com Projeção de Entrada e Atenção

Arquivo: [`src/train_gru.py`](src/train_gru.py) · Classe: `PulseFiModelGRU`

O GRU incorpora uma **camada de projeção de entrada** ausente no LSTM. Essa camada aprende uma representação densa das 256 subportadoras antes de alimentar o RNN, reduzindo ruído de entrada e melhorando a convergência:

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
│  Bi-GRU  (input=256, hidden=256, layers=2, dropout=0.3) │
│  → saída: (B, T, 512)   [256 forward + 256 backward]   │
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
│  w_t = softmax(Linear(512 → 1))   shape: (B, T, 1)     │
│  context = Σ_t w_t · h_t          shape: (B, 512)      │
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

**Motivação da projeção de entrada:** as 256 subportadoras CSI possuem alta correlação inter-banda e ruído variável por subportadora. A camada `Linear → LayerNorm → GELU` antes do GRU funciona como um encoder que projeta esse espaço de alta dimensão numa representação mais compacta e normalmente distribuída, reduzindo a variância dos gradientes nas camadas recorrentes e acelerando a convergência.

**Total de parâmetros estimado:** ~2,3M

### 4.3 Comparação Estrutural LSTM vs GRU

| Componente | LSTM | GRU |
|---|---|---|
| Gates | Input, Forget, Output | Reset, Update |
| Estado interno | Hidden state + Cell state | Apenas hidden state |
| Parâmetros/camada | ~4 × (input+hidden) × hidden | ~3 × (input+hidden) × hidden |
| Capacidade de memória longa | Alta (cell state explícito) | Moderada |
| Velocidade de treinamento | Mais lento | Mais rápido |
| Risco de overfitting | Maior | Menor |
| Projeção de entrada | Não | Sim (Linear→LN→GELU) |

O LSTM mantém estruturalmente vantagem para sinais cardíacos periódicos de longa duração (janelas de 20 s contendo 20–40 ciclos cardíacos), pois o *cell state* age como uma linha de memória dedicada ao longo dos 166 timesteps. A projeção de entrada no GRU é uma compensação parcial: ao reduzir a dimensionalidade de entrada de 256 para 256 com normalização, o GRU recebe um sinal de entrada de menor variância, o que facilita a propagação de gradientes ao longo da sequência.

**Mecanismo de Atenção:**

A atenção temporal implementada é uma **atenção aditiva escalar** sem projeções de query/key separadas — equivalente a uma *scaled dot-product attention* simplificada com uma única projeção linear. Para cada timestep $t$, o peso de atenção é calculado como:

$$\alpha_t = \text{softmax}\left( \mathbf{w}^\top \mathbf{h}_t \right)$$

e o vetor de contexto como a soma ponderada:

$$\mathbf{c} = \sum_{t=1}^{T} \alpha_t \cdot \mathbf{h}_t$$

Isso permite ao modelo aprender quais regiões temporais da janela de 20 segundos são mais relevantes para a estimativa de FC, sem a rigidez de considerar apenas o último estado oculto.

---

## 5. Procedimento de Treinamento

Os modelos LSTM e GRU compartilham a maior parte dos hiperparâmetros, com diferenças pontuais na arquitetura do GRU:

| Hiperparâmetro | LSTM (`train_lstm.py`) | GRU (`train_gru.py`) |
|---|---|---|
| `BATCH_SIZE` | 64 | 64 |
| `EPOCHS` (máximo) | 250 | 150 |
| `LEARNING_RATE` | 5 × 10⁻⁴ | 5 × 10⁻⁴ |
| `HIDDEN_SIZE` | 256 | 256 |
| `NUM_LAYERS` | 2 | 2 |
| `PATIENCE` (early stopping) | 30 épocas | 30 épocas |
| Dropout recorrente | 0.3 | 0.3 |
| Dropout regressor | 0.2 | 0.2 |
| Clipping de gradiente | max_norm = 1.0 | max_norm = 1.0 |
| Projeção de entrada | Não | Sim (Linear→LN→GELU) |
| `SEED` | 42 | 42 |

O GRU utiliza um número menor de épocas máximas (150 vs 250) porque a projeção de entrada acelera a convergência — o modelo tipicamente aciona o early stopping antes do limite.

### 5.1 Função de Perda: Huber Loss

A função de perda utilizada é a **Huber Loss** com parâmetro $\delta = 3.0$ BPM:

$$\mathcal{L}_{\delta}(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{se } |y - \hat{y}| \leq \delta \\
\delta \cdot \left(|y - \hat{y}| - \frac{\delta}{2}\right) & \text{caso contrário}
\end{cases}$$

A Huber Loss combina o comportamento quadrático do MSE (para erros pequenos, sensível a desvios) com o comportamento linear do MAE (para erros grandes, robustez a outliers). O limiar $\delta = 3$ BPM foi escolhido para que erros de até 3 BPM sejam tratados com sensibilidade quadrática, enquanto erros maiores (possivelmente ruído de anotação ou posições corpo-antena desfavoráveis) não dominam o gradiente.

Para a avaliação em validação, usa-se diretamente o **MAE** (L1Loss), que é a métrica clínica mais interpretável:

```python
criterion = nn.HuberLoss(delta=3.0)  # treinamento
v_loss += nn.L1Loss()(model(bx), by).item()  # validação / early stopping
```

### 5.2 Otimizador e Agendamento de LR

Ambos os modelos utilizam o otimizador **Adam** com `ReduceLROnPlateau`:

```python
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5
)
```

O scheduler reduz o LR pela metade sempre que o val MAE não melhora por 10 épocas consecutivas, com LR mínimo de $10^{-5}$. Esse comportamento de *annealing* permite ao modelo convergir rapidamente nas fases iniciais e refinar os pesos com gradientes menores nas fases finais.

O **gradient clipping** (`max_norm=1.0`) previne gradientes explosivos, frequentes em RNNs profundas quando as sequências de entrada contêm variações abruptas de amplitude.

**Suporte a acelerador Apple Silicon (MPS):** o GRU detecta automaticamente o backend disponível, priorizando MPS > CUDA > CPU:

```python
def _get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

### 5.3 Early Stopping e Checkpointing

O treinamento é monitorado pelo val MAE a cada época. Se não houver melhora por `PATIENCE=30` épocas consecutivas, o treinamento é interrompido e o melhor modelo salvo é carregado para avaliação final:

```python
if val_mae < best_val_mae:
    best_val_mae = val_mae
    patience_counter = 0
    torch.save(model.state_dict(), checkpoint_path)
else:
    patience_counter += 1
    if patience_counter >= PATIENCE:
        print(f"Early stopping na época {epoch + 1}")
        break

model.load_state_dict(torch.load(checkpoint_path))
```

O checkpoint (`best_model_gru.pt` / `best_model_lstm.pt`) preserva apenas os pesos do estado ótimo de validação, não o do último estado, garantindo que o modelo final seja o menos *overfitado*. O carregamento usa `weights_only=True` para compatibilidade com PyTorch ≥ 2.4 e segurança contra deserialização arbitrária:

```python
model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
```

### 5.4 Reprodutibilidade

Para garantir reprodutibilidade completa dos experimentos, todas as fontes de aleatoriedade são fixadas com semente `SEED=42`:

```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

O flag `deterministic=True` garante que operações CUDA produzam resultados determinísticos (ao custo de eventual redução de velocidade). O `benchmark=False` desativa a seleção automática de algoritmos otimizados pelo cuDNN, que introduziria não-determinismo.

---

## 6. Métricas de Avaliação

A avaliação ao final do treinamento produz um painel de 6 visualizações gerado automaticamente (`resultado_gru.png` / `resultado_lstm.png`):

### 6.1 Métricas Quantitativas

| Métrica | Fórmula | Interpretação |
|---|---|---|
| **MAE** | $\frac{1}{N}\sum|y_i - \hat{y}_i|$ | Erro médio absoluto em BPM |
| **RMSE** | $\sqrt{\frac{1}{N}\sum(y_i - \hat{y}_i)^2}$ | Penaliza erros grandes |
| **R²** | $1 - \frac{\text{SS}_\text{res}}{\text{SS}_\text{tot}}$ | Variância explicada pelo modelo |
| **Viés (μ)** | $\frac{1}{N}\sum(y_i - \hat{y}_i)$ | Tendência sistemática de sub/superestimar |
| **Desvio (σ)** | $\text{std}(y_i - \hat{y}_i)$ | Dispersão dos resíduos |
| **LoA ±1.96σ** | $\mu \pm 1.96\sigma$ | Limites de concordância de Bland-Altman |
| **% ≤ 5 BPM** | $P(|\text{erro}| \leq 5)$ | Fração de predições clinicamente aceitáveis |
| **% ≤ 10 BPM** | $P(|\text{erro}| \leq 10)$ | Fração de predições dentro de tolerância ampliada |

### 6.2 Visualizações do Painel de Avaliação

1. **Curva de Aprendizado** — Evolução do MAE de treino e validação por época, com marcação da melhor época
2. **Real vs Predito** — Scatter plot com linha ideal (y=x) e reta de regressão ajustada
3. **Histograma de Resíduos** — Distribuição dos erros com curva Normal sobreposta
4. **Bland-Altman** — Diferença entre medição de referência e predição vs. média das duas; viés e limites de concordância (LoA)
5. **CDF do Erro Absoluto** — Função de distribuição acumulada do erro absoluto com thresholds em 5, 10 e 15 BPM
6. **Tabela de Métricas** — Resumo numérico completo de todas as métricas

O **diagrama de Bland-Altman** é particularmente relevante neste contexto, pois é o método padrão para avaliação de concordância entre dois métodos de medição clínica. Ele revela não apenas o viés sistemático mas também se a dispersão dos erros é homogênea ao longo do range de FC (homocedasticidade) — um pressuposto importante para a validade do modelo em populações com FC em repouso muito baixa ou muito alta.

---

## 7. Estrutura de Arquivos

```
frequencia_csi/
├── src/
│   ├── preprocess-pulseFi.py        # Pipeline de pré-processamento completo
│   ├── train_lstm.py                 # LSTM bidirecional com atenção (referência)
│   ├── train_gru.py                  # GRU bidirecional com projeção de entrada + atenção
│   ├── train_gru_lstm_compare.py     # LSTM standalone com MPS e AdamW
│   ├── train_lstm_2.py               # LSTM simplificado (sem bidirecional/atenção)
│   └── train_gru_2.py                # GRU simplificado (sem bidirecional/atenção)
│
├── saida_full/                       # Tensores pré-processados (dataset completo)
│   ├── X_train.npz                   # (N_train, 166, 256) float32
│   ├── X_val.npz
│   ├── X_test.npz
│   ├── y_train.npy                   # (N_train,) float32 — FC em BPM
│   ├── y_val.npy
│   ├── y_test.npy
│   ├── positions_*.npy               # Posição experimental (1–17)
│   ├── subject_*.npy                 # ID do participante
│   ├── idx_*.npy                     # Índices originais no dataset global
│   └── subjects_*_ids.npy           # IDs dos sujeitos por partição
│
├── saida_smoke/                      # Tensores de smoke test (subconjunto rápido)
│   └── [mesma estrutura de saida_full]
│
├── reports_dataset/                  # Relatório exploratório do dataset
│   ├── general_stats.json
│   ├── relatorio_completo.txt
│   ├── participant_summary.csv
│   └── [gráficos .png e .csv de distribuição]
│
├── resultado_gru.png                 # Painel de avaliação — GRU com projeção de entrada
├── resultado_lstm.png                # Painel de avaliação — LSTM
├── resultado_lstm_compare.png        # Painel de avaliação — LSTM (train_gru_lstm_compare.py)
├── best_model_gru.pt                 # Checkpoint do melhor modelo GRU
├── best_model_lstm.pt                # Checkpoint do melhor modelo LSTM
├── RELATORIO_METODOLOGICO.md        # Este documento
├── index.html                        # Visualização web do relatório
└── participantes_incompletos.csv
```

---

## 8. Dependências e Execução

### 8.1 Ambiente

```bash
Python 3.10
torch >= 2.0
numpy
pandas
scipy
scikit-learn
matplotlib
```

### 8.2 Passo 1 — Pré-processamento

```bash
python src/preprocess-pulseFi.py \
    --dataset_path Data_DS2_raspberry_npz/ \
    --gt_dir       Data_DS2_smartwatch-main/Data_Heart/ \
    --out_dir      saida_full/ \
    --window_sec   20.0 \
    --step_sec     0.5
```

O script gera os tensores em `saida_full/` e imprime estatísticas de aceitação/rejeição de janelas por arquivo.

### 8.3 Passo 2 — Treinamento

```bash
# LSTM bidirecional com atenção (melhor resultado: ~1.07 BPM MAE)
python src/train_lstm.py

# GRU bidirecional com projeção de entrada + atenção (~1.2 BPM MAE)
python src/train_gru.py

# LSTM standalone com MPS/AdamW (train_gru_lstm_compare.py renomeado)
python src/train_gru_lstm_compare.py
```

Todos os scripts leem automaticamente de `saida_full/`, treinam o modelo e geram o painel de avaliação em PNG ao final.

| Script | Modelo | Épocas máx. | Resultado esperado |
|---|---|---|---|
| `train_lstm.py` | LSTM Bi + atenção | 250 | ~1.07 BPM MAE |
| `train_gru.py` | GRU Bi + proj. entrada + atenção | 150 | ~1.1–1.2 BPM MAE |
| `train_gru_lstm_compare.py` | LSTM Bi + atenção (MPS/AdamW) | 150 | ~1.07 BPM MAE |

---

## 9. Referências Técnicas

- **Butterworth filter / filtfilt:** Oppenheim & Schafer, *Discrete-Time Signal Processing*, 3ª ed. — Prentice Hall.
- **Savitzky-Golay:** Savitzky, A.; Golay, M.J.E. (1964). *Smoothing and Differentiation of Data by Simplified Least Squares Procedures*. Analytical Chemistry.
- **CSI para sinais vitais:** Wang, F. et al. (2017). *E-eyes: Device-free location-oriented activity identification using fine-grained WiFi signatures*. IEEE INFOCOM.
- **Bi-LSTM:** Schuster, M.; Paliwal, K. (1997). *Bidirectional recurrent neural networks*. IEEE Transactions on Signal Processing.
- **GRU:** Cho, K. et al. (2014). *Learning phrase representations using RNN encoder-decoder for statistical machine translation*. EMNLP.
- **Huber Loss:** Huber, P.J. (1964). *Robust Estimation of a Location Parameter*. Annals of Mathematical Statistics.
- **Bland-Altman:** Bland, J.M.; Altman, D.G. (1986). *Statistical methods for assessing agreement between two methods of clinical measurement*. The Lancet.
- **ReduceLROnPlateau / Adam:** Kingma, D.P.; Ba, J. (2014). *Adam: A method for stochastic optimization*. ICLR 2015.

---

*Relatório gerado automaticamente a partir da análise do código-fonte do projeto `frequencia_csi`.*
