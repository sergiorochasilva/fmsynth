# Resumo da conversa (fmsynth)
Data: 2026-04-11

## Objetivo geral
Melhorar a predição de parâmetros FM (dataset_big4) e a qualidade perceptiva da ressíntese, com foco em:
- modelo supervisionado multi-head,
- análise por head,
- redução de consumo de memória,
- experimento de pré-treino com autoencoder.

## Principais decisões tomadas
1. Avaliação por head é obrigatória (não só métrica agregada).
2. `frequencia_base` deve ser tratada separadamente (head dedicada + métricas próprias em Hz/MAPE/cents).
3. Saídas categóricas e numéricas devem ter tratamento e pesos de loss separados.
4. A conversão MIDI->Hz no resynth estava 1 oitava acima (bug corrigido).
5. Ressíntese com projeções fortes ajuda robustez, mas pode mascarar erro do modelo.
6. Autoencoder para pré-treino faz sentido, desde que o encoder seja reutilizado no downstream.

## Arquivos criados/atualizados
### Predição/Ressíntese (0_4)
- `pred_nsynth_big4_fmsynth3_0_4.py` (novo): adaptado para modelo multi-head 0_4.
- `resynth_nsynth_big4_fmsynth3_0_4.py` (novo): defaults para 0_4.

### Correção de bug MIDI->Hz (sem nova versão)
- `resynth_nsynth_big4_fmsynth3_0_2.py`: corrigido `midi_to_hz`.
- `resynth_nsynth_big4_fmsynth3_0_4.py`: corrigido `midi_to_hz`.

### Modelo supervisionado 0_6
- `model_training_big4_fmsynth3_0_6.py` (novo/atualizado):
  - heads numéricas separadas (`ratio/index/detune/env/phase/other`),
  - `ratio` em `log2`,
  - `algorithm` com merge de rótulo equivalente (`dual_chain -> series2x2_parallel1`),
  - pesos por head (numéricas/categóricas/frequência),
  - backbone CNN ajustado,
  - métricas e logs por head no `results.json`.

### Economia de memória no 0_6
No mesmo `model_training_big4_fmsynth3_0_6.py`:
- batch defaults reduzidos,
- mixed precision opcional,
- áudio em `float16` opcional,
- XLA JIT opcionalmente desativado,
- treino por `Sequence` (streaming por batch),
- runtime_config expandido no `results.json`.

### Autoencoder
- `autoencoder_training_big4_fmsynth3_0_1.py` (novo/atualizado):
  - encoder CNN + decoder MLP,
  - loss híbrida: `log-mel + STFT`,
  - export de `encoder` e `latent_*`,
  - depois ajustado para modo econômico de memória:
    - `Sequence` por índice,
    - áudio `float16` opcional,
    - mixed precision opcional,
    - XLA JIT opcionalmente desativado,
    - redução de cópias grandes em memória.

## Erros importantes encontrados e causa
1. **OOM no modelo 0_6**
   - sinais de alocação grande em GPU e falha durante `model.fit`.
   - mitigado com redução de batch/backbone + streaming + flags de memória.

2. **OOM no autoencoder**
   - erro no `_EagerConst` tentando copiar ~1.43GiB CPU->GPU.
   - causa: `fit` com arrays grandes em memória de uma vez.
   - mitigado com `Sequence` e remoção de cópias grandes (`x_fit/x_val/x_train/x_test`).

## Resultado do autoencoder (última avaliação)
Arquivo: `autoencoder_training_big4_fmsynth3_0_1/results.json`
- `test_loss_keras`: ~2.2925
- `test_waveform_mae`: ~0.2537
- `test_log_mel_mae`: ~2.7825
- `test_stft_mae`: ~1.1492

Leitura feita na conversa:
- convergência boa,
- gap treino/val controlado,
- teste muito próximo (até ligeiramente melhor que validação),
- sem sinal forte de overfitting.

## Estado atual (para continuar depois)
1. Treino do autoencoder já roda em modo mais econômico, mas ainda pode exigir batch muito baixo dependendo da GPU.
2. Próximo passo lógico sugerido:
   - usar o encoder pré-treinado no modelo supervisionado (freeze inicial + unfreeze parcial com LR baixo),
   - medir ganho real no downstream (`algorithm accuracy`, MAE de `ratio/index/detune`, e avaliação auditiva).

## Comandos úteis (último formato recomendado)
### Modelo 0_6 (baixo consumo)
```bash
TRAIN_BATCH_SIZE=1 PRED_BATCH_SIZE=2 AUDIO_DTYPE=float16 MIXED_PRECISION=1 DISABLE_XLA_JIT=1 ./.venv/bin/python model_training_big4_fmsynth3_0_6.py
```

### Autoencoder (baixo consumo)
```bash
TF_GPU_ALLOCATOR=cuda_malloc_async AE_TRAIN_BATCH_SIZE=1 AE_PRED_BATCH_SIZE=2 AE_AUDIO_DTYPE=float16 AE_MIXED_PRECISION=1 AE_DISABLE_XLA_JIT=1 ./.venv/bin/python autoencoder_training_big4_fmsynth3_0_1.py
```
