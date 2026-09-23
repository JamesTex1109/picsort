# PicSort: Automated Photo Organization with Transfer Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JamesTex1109/picsort/blob/main/PicSort_ProjectFinal.ipynb)

PicSort automatically sorts photos into scene categories so large photo collections don't have to be organized by hand. I built it after seeing how long it took to sort photos from my own photography work. It fine-tunes a pretrained ResNet18 and reaches **93.1% test accuracy** on 6 categories.

## Categories
`buildings` · `forest` · `glacier` · `mountain` · `sea` · `street`

It also labels each image's **orientation** (vertical, horizontal, or square) alongside its category.

## Results

| Model | Test Accuracy |
|---|---|
| Baseline (pretrained, no fine-tuning) | 18.3% |
| v1: final layer retrained | 90.8% |
| **v2: layer4 + final layer retrained** | **93.1%** |

Most of the remaining errors come from **glacier vs. mountain** (snowy peaks look alike) and **buildings vs. street**, which is what I expected going in.

<!-- Add your confusion matrix screenshot here -->
<!-- ![Confusion matrix](images/confusion_matrix.png) -->

## How It Works
1. **Dataset:** [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) (~25,000 labeled images, pre-split into train and test)
2. **Preprocessing:** resize to 224×224, normalize with ImageNet mean and std
3. **Stage 1:** freeze the pretrained ResNet18 backbone and replace the final layer with a 6-class head (Adam, LR 0.001, 3 epochs)
4. **Stage 2:** unfreeze `layer4` so the deepest features adapt to these scenes (Adam, LR 0.0001, 3 epochs)
5. **Evaluation:** test accuracy, confusion matrix, and per-class precision and recall

The idea behind transfer learning: ResNet18 already learned general visual features (edges, textures, shapes) from ImageNet, so I only have to teach it how to map those features onto my 6 categories. That's why it trains in a couple of minutes on a free Colab GPU instead of hours.

## Tech Stack
Python · PyTorch · torchvision · scikit-learn · matplotlib · Google Colab (T4 GPU)

## Running It
1. Click the **Open in Colab** badge above.
2. Add your Kaggle API token as a Colab secret named `KAGGLE_API_TOKEN` (🔑 icon in the sidebar).
3. Set the runtime to GPU and run all cells. The full pipeline takes about 2.5 minutes.

## Future Work
- Data augmentation to cut down glacier and mountain confusion
- A drag-and-drop desktop app for sorting a real photo folder
