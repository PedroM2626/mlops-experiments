# Computer Vision — Experimentos e Aplicações

> **Área:** Computer Vision
> **Tarefa:** Classificação de imagens (multiclasse e multi-label), detecção e reconhecimento facial
> **Métrica principal:** Acurácia (CIFAR-10) / F1-macro (multi-label)
> **Status:** Concluído
> **Datasets:** CIFAR-10 (50.000 treino / 10.000 teste, 10 classes); dataset próprio de pets (44 imagens, 2 classes multi-label); dataset local de faces; COCO/custom para YOLO

## 1. Resumo

Esta pasta reúne quatro notebooks de visão computacional: um comparativo de três paradigmas (HOG+SVM, ResNet18 e ViT) no CIFAR-10 — vencido pelo **ViT com 0,9805 de acurácia** —, um estudo multi-label de classificação de pets com quatro abordagens (ResNet18, VGG16, CLIP zero-shot e EfficientNet) — vencido pelo **ResNet18 com F1-macro 1,000** —, um aplicativo de reconhecimento facial com modos LBPH/CNN/transferência (YuNet) e um notebook de detecção com YOLO (OpenCV DNN). Conclui-se que transformers pré-treinados e fine-tuning supervisionado são os caminhos de maior acurácia, enquanto features manuais (HOG) falham em imagens de baixa resolução.

## 2. Contexto e Objetivos

O grupo investiga o vetor de técnicas de visão computacional disponíveis para problemas de escala:
1. **CIFAR-10 (cv-methods-comparison.ipynb):** quantificar o salto de representações manuais (HOG) para redes profundas (ResNet18 CNN residual) e para transformers visuais (ViT pré-treinado no ImageNet-21k), além do custo computacional de cada um.
2. **Multi-label pets (animal-classifier.ipynb):** comparar 4 fluxos (PyTorch/Keras/CLIP) para o problema de ativar múltiplos rótulos numa mesma imagem — duas gatas (Dime e Frida) que aparecem juntas em algumas fotos.
3. **Face recognition e YOLO:** disponibilizar aplicações funcionais (app embutido no notebook e detecção por YOLO via OpenCV DNN) sem dependência de scripts externos.

Questões de pesquisa: *o salto arquitetural importa mais em visão do que em texto? O fine-tuning supervisionado supera backbones congelados e zero-shot em cenários de poucos dados?*

## 3. Fundamentação Teórica (curta)

- **HOG (Histogram of Oriented Gradients):** representação clássica baseada em gradientes locais por célula (cell/block); eficaz para detecção de pedestres em média resolução, mas com baixa capacidade de generalização para classe variada.
- **CNN residuais (ResNet18):** blocos com conexões residuais permitem treinamento profundo sem degradação de gradiente; fine-tune sobre pesos ImageNet transfere features genéricas de textura/forma.
- **Vision Transformer (ViT):** divide a imagem em patches linearizados e aplica self-attention global; pré-treinamento em corpora gigantes (ImageNet-21k, 14M imagens/21k classes) confere antecedentes qualitativos sobre CNNs pré-treinadas no ImageNet-1k.
- **Aprendizado multi-label:** BCEWithLogitsLoss + sigmoid por classe; métricas de Exact Match, Hamming Loss, F1-micro/macro, precisão/recall.
- **CLIP zero-shot:** alinha texto-imagem (ViT-B/32); classificação via protótipos de classes (embedding médio) e similaridade de cosseno com limiar (threshold).
- **EfficientNet-B0:** escalonamento compound (profundidade × largura × resolução).
- **Reconhecimento facial:** LBPH (histogramas LBP + distância), CNN treinada do zero e transfer learning com MobileNetV2 sobre faces detectadas por YuNet.
- **YOLO (OpenCV DNN):** detecção one-stage (YOLOv3-tiny COCO) para classificação de objetos em imagens de upload.

## 4. Metodologia

### 4.1 Dados

**Experimento CIFAR-10 (cv-methods-comparison.ipynb):**
- CIFAR-10: 50.000 imagens de treino, 10.000 de teste, 10 classes, 32×32, coloridas (RGB).
- HOG+SVM usou subamostra de **10k treino / 2k teste** por limitação computacional; ResNet18 e ViT usaram **50k treino / 10k teste**.
- Hardware: NVIDIA RTX 4070 Laptop GPU (8GB), Python 3.8, PyTorch 2.4.

**Experimento multi-label (animal-classifier.ipynb):**
- 44 imagens rotuladas (22 por classe: Dime e Frida); multi-label porque as duas gatas aparecem juntas em algumas fotos.
- Split: 60% treino (30), 15% validação (7), 25% teste (7), estratificado por classe dominante.
- Cautela: dataset muito pequeno impede generalização robusta (valores perfeitos devem ser interpretados com ressalvas).

**Face recog & YOLO:** dataset local `dataset/<nome>/` (coleta por upload); YOLO usa COCO (YOLOv3-tiny) ou modelo custom (car, motorbike, threewheel, van, bus, truck), baixado automaticamente via `.env`.

### 4.2 Pré-processamento

- CIFAR-10: redimensionamento para 224×224 com normalização ImageNet (ResNet18 e ViT); no ViT, normalização própria do modelo; HOG baseado em 9 orientações, cell 8×8, block 3×3, gerando **2.916 features**.
- Pets: Data Augmentation (flip horizontal, rotação ±15°, jitter de cor, affine) aplicada aos fluxos ResNet18 e EfficientNet; normalização ImageNet.
- Faces: recorte de faces detectadas, salvando em `dataset/<nome>/`.

### 4.3 Métodos comparados

| Notebook | Fluxos/Arquiteturas | Estratégia |
|---|---|---|
| cv-methods-comparison.ipynb | HOG+SVM; ResNet18 (fine-tune 5 épocas, Adam lr=1e-4, batch 128); ViT `google/vit-base-patch16-224-in21k` (fine-tune 2 épocas, Adam lr=2e-5, batch 32) | Paradigmática: manual → CNN → Transformer |
| animal-classifier.ipynb | ResNet18+Aug (fine layer4+FC, adam lr=1e-4, 10 épocas); VGG16 (frozen + head 128/Dropout0.2 + sigmoid, 6 épocas); CLIP zero-shot (prototypes, threshold 0.75); EfficientNet-B0+Aug (blocks 4-5+FC, 10 épocas) | Supervisionado vs zero-shot |
| face_recognition_app.ipynb | LBPH (baseline), CNN (do zero), transfer_yunet (MobileNetV2 + detecção YuNet) | Reconhecimento facial |
| yolo_notebook.ipynb | YOLOv3-tiny COCO (OpenCV DNN) | Detecção one-stage |

Configurações env para o face app: `FACE_DETECTOR=yunet\|haar`; `FACE_TL_EPOCHS`, `FACE_TL_BATCH`; `FACE_CNN_EPOCHS`, `FACE_CNN_BATCH`; `YUNET_SCORE_THRESHOLD`, `YUNET_NMS_THRESHOLD`, `YUNET_TOP_K`.

### 4.4 Avaliação

- CIFAR-10: **acurácia** no teste e tempo de treinamento; análise por classe (F1) para cada método.
- Multi-label: **Exact Match, Hamming Loss, F1-micro, F1-macro, precisão micro, recall micro** sobre o conjunto de teste (7 imagens).
- Face recog: predição por upload com visualização do resultado.
- Seeds determinísticas e mesma partição para todos os fluxos do multi-label.

### 4.5 Reprodução

- Notebooks com outputs embutidos; abrir em Jupyter (Jupyter Notebook / VS Code) e executar célula a célula.
- `cv-methods-comparison.ipynb` requer GPU NVIDIA (RTX 4070) e PyTorch 2.4.
- Scripts de validação estrutural: `python scripts/validate_notebooks.py`.
- Pacote de saída (atípico deste grupo modelo): `experiments/artifacts/<experimento>_<timestamp>_<sha>/`.

## 5. Resultados

### 5.1 CIFAR-10 — Comparativo de Paradigmas (cv-methods-comparison.ipynb)

| Método | Acurácia | Tempo | Paradigma | Dados |
|--------|----------|-------|-----------|-------|
| **ViT** | **0,9805** | ~17 min (1 época) | Transformer visual pré-treinado (ImageNet-21k) | 50k treino |
| **ResNet18** | **0,9362** | 12,5 min (5 épocas) | CNN residual pré-treinada (ImageNet) | 50k treino |
| HOG+SVM | 0,3970 | 27 min | Features manuais + SVM | 10k treino |

**Análise por classe (F1):** HOG+SVM melhor em `automobile` (0,54 F1, bordas retilíneas) e pior em `cat` (0,25 F1, forma não rígida). ResNet18: melhores em `ship` (0,99 prescisão), `bird` (0,97), `horse` (0,97); pior `cat` (0,84 prescisão, 0,87 F1). ViT domina todas as classes com margem.

Comportamento do treino ResNet18: saturação rápida (época 1 = 0,9323, oscila ~0,94). ViT alcançou 0,9805 em **1 época**.

### 5.2 Multi-label de Pets — 4 Abordagens (animal-classifier.ipynb)

| Métrica | ResNet18 + Aug | VGG16 | CLIP zero-shot | EfficientNet + Aug |
|---------|:--------------:|:-----:|:--------------:|:------------------:|
| **Exact Match** | **1.000** | 0.429 | 0.000 | 0.714 |
| **Hamming Loss** | 0.000 | 0.286 | 0.500 | 0.143 |
| **F1-micro** | **1.000** | 0.714 | 0.667 | 0.833 |
| **F1-macro** | **1.000** | 0.714 | 0.664 | 0.829 |
| Precisão micro | 1.000 | 0.714 | 0.500 | 1.000 |
| Recall micro | 1.000 | 0.714 | 1.000 | 0.714 |

**Detalhes por fluxo:**
- **ResNet18**: desempenho perfeito (F1-macro 1.000), resultado da baixa complexidade de generalização (7 imagens de teste); fine-tuning seletivo (layer4+FC) suficiente e BCEWithLogitsLoss adequada para multi-label.
- **VGG16**: vencedor por 0.714; superior para Frida (F1=0.86) vs Dime (0.57); backbone congelado limita adaptação ao domínio (−28.6 pp vs ResNet18).
- **CLIP**: F1-macro 0.664, recall 1.0, precisão 0.5, exact match 0.0; threshold 0.75 excessivas permissives (falsos positivos); protótipos capturam as classes, mas a calibração do limiar é crítica.
- **EfficientNet+Aug**: 2º lugar (0.829), +11.5 pp sobre VGG16, perfil conservador (precisão 1.0, recall 0.714) — omite 28.6% das previsões positivas; escalonamento compound precisa de mais fine-tuning para calibrar sigmoid.

### 5.3 Face Recognition App (face_recognition_app.ipynb)

- Fluxo embutido no notebook: coleta de faces por upload, treino (LBPH/CNN/YuNet) e predição por upload com visualização.
- **TBD**: não há métricas de acurácia embutidas no notebook (resultado qualitativo por visualização). Modos: `lbph` (baseline OpenCV), `cnn` (CNN pequena), `transfer_yunet` (MobileNetV2 + YuNet).

### 5.4 YOLO (yolo_notebook.ipynb)

- Classificação/detecção por upload usando YOLOv3-tiny COCO via OpenCV DNN; classes custom exibidas quando modelo treinado baixado.
- **TBD**: sem métricas de acurácia embutidas (resultado por inspeção das classes detectadas).

## 6. Discussão

- **Features manuais não escalam**: HOG+SVM obtém 0,3970 em CIFAR-10; a representação de gradientes é suficiente para formas rígidas (automobile) mas não para a variabilidade dos gatos — e o custo (27 min) nem compensa.
- **Transformers ≈ novo padrão**: ViT supera ResNet18 por 4,4 pp com apenas 1 época. No 16.º experimento (DistilBERT vs TF-IDF+SVC em NLP) o salto arquitetural foi menor (0,9 pp), sugerindo que em visão o pré-treinamento em 21k classes dá vantagem qualitativa maior sobre dados de porte médio (50k).
- **Fine-tuning supervisionado domina multi-label**, mas o dataset de 44 imagens impede conclusões fortes; os resultados perfeitos de ResNet18 devem ser lidos com cautela (overfitting benéfico).
- **Zero-shot é opção para zero-dados**, porém a calibração do threshold é o fator decisivo: com 0.75 o CLIP ganhou recall e perdeu precisão (exact match 0.000).
- **Limitações**: espaço para budgets de hardware (GPU obrigatória no comparativo CIFAR-10); face app e YOLO não possuem métricas objetivas nesta pasta (avaliação por inspeção).

## 7. Conclusões e Recomendações

- Para **classificação de imagens de médio porte**: use ViT para máxima acurácia (~0,985+ com 3 épocas, ~50 min) ou ResNet18 para prototipagem rápida (0,9362, 12,5 min); evite HOG (não recomendado).
- Para **multi-label em poucos dados**: preferir fine-tune supervisionado seletivo (ResNet18) que reduziu de fine com data augmentation; se não houver rótulos, CLIP zero-shot exige calibração cuidadosa do threshold (0.75 é permissivo demais).
- Para **aplicativos de produção**: o notebook face app funciona em CPU (LBPH roda em CPU; `transfer_yunet` acelera com GPU, mas também funciona em CPU) e o YOLO viabiliza detecção de objetos via OpenCV DNN sem treinamento.

## 8. Referências e Arquivos

- Notebooks (relativos a esta pasta):
  - `./cv-methods-comparison.ipynb` — comparativo CIFAR-10 (HOG+SVM / ResNet18 / ViT)
  - `./animal-classifier.ipynb` — multi-label pets (ResNet18, VGG16, CLIP, EfficientNet)
  - `./face_recognition_app.ipynb` — aplicativo de reconhecimento facial (LBPH / CNN / transfer_yunet)
  - `./yolo_notebook.ipynb` — detecção YOLO via OpenCV DNN
- Referências: Deng et al. (2009) CIFAR-10; He et al. (2016) *Deep Residual Learning*; Dosovitskiy et al. (2021) *An Image is Worth 16x16 Words*; Radford et al. (2021) *Learning Transferable Visual Models From Natural Language Supervision* (CLIP); Tan & Le (2019) *EfficientNet: Rethinking Model Scaling*; Sedhain et al. (2015) — ver também documentos em `docs/modelo-academico-readme.md`.