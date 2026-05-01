# 🎨 Akina-Draw

> **Real-time hand drawing classification powered by deep learning.**  
> Draw on screen — watch the AI predict what you're drawing before you even finish.

Presented at **Robotech Fair** · Built by a team of 3  
**89 final classes** after iterative data refinement

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)](https://tensorflow.org)
[![Pygame](https://img.shields.io/badge/GUI-Pygame-green?style=flat-square)](https://pygame.org)
[![Dataset](https://img.shields.io/badge/Dataset-Google%20Quick%2C%20Draw!-yellow?style=flat-square)](https://quickdraw.withgoogle.com/data)

---

## 📌 Table of Contents

- [What is Akina-Draw?](#what-is-akina-draw)
- [Demo](#demo)
- [How it works](#how-it-works)
- [Project journey](#project-journey)
  - [Phase 1 — Data curation](#phase-1--data-curation)
  - [Phase 2 — Parallel model training](#phase-2--parallel-model-training)
  - [Phase 3 — Integration, the bug, and the pivot](#phase-3--integration-the-bug-and-the-pivot)
- [Final architecture](#final-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Supported classes](#supported-classes)
- [Tech stack](#tech-stack)
- [Team](#team)
- [Links](#links)

---

## What is Akina-Draw?

Akina-Draw is a desktop application where a user draws freely on a digital canvas and a trained convolutional neural network (CNN) classifies the drawing **in real time** — updating its prediction with every stroke, before the drawing is finished.

The system supports **89 object categories** including animals, fruits, vehicles, everyday objects, and landmarks, all drawn from the [Google Quick, Draw!](https://quickdraw.withgoogle.com/data) dataset.

Key features:
- 🖊️ Live prediction as you draw — no need to finish first
- 📊 Confidence score with colour-coded bar (green = high, amber = moderate)
- 📋 Side panel listing all 89 classes with verified reference images
- 🖱️ Adjustable brush size slider
- ✅ Works entirely offline after model is loaded

---

## Demo

📊 Full project slides:  
[View on Canva](https://www.canva.com/design/DAHHcMuVh0U/c1Y_KCjQVCFGVMFHEI-HPQ/edit)

---

## How it works

```
User draws on canvas
        │
        ▼
 OpenCV preprocessing
 (grayscale → resize to 128×128)
        │
        ▼
  Inception CNN
  (89-class softmax)
        │
        ▼
  Top prediction + confidence %
        │
        ▼
  Pygame GUI — live display
  (label · bar · class panel)
```

Every time the mouse button is released (or continuously in prediction mode), the canvas is sent through this pipeline. The result updates instantly on screen.

---

## Project journey

This project did not go in a straight line. Here is the full story.

### Phase 1 — Data curation

**Starting point:** The Google Quick, Draw! dataset contains 340 hand-drawn doodle classes with ~3,000 images each. That is far too many for one team to train and maintain properly.

**What we did:**
- Manually reviewed all 340 classes and selected **107** as a starting point — visually distinct categories that people would naturally want to draw: animals, fruits, vehicles, landmarks, everyday objects.
- Through iterative training and refinement across all three phases, this was further reduced to a **final set of 89 classes** — removing anything that consistently caused confusion, overlapped with another class, or couldn't reach acceptable accuracy.
- Considered adding a Roboflow dataset of simple geometric shapes (circle, square, triangle), but **rejected it**. A hand-drawn circle is nearly indistinguishable from a ball. Adding ambiguous classes would hurt accuracy across the board. Better to do fewer things well.
- Designed the original detection architecture: use **YOLO v8** to detect a whiteboard or white sheet in a live camera feed, crop the drawing area, and send it to a CNN classifier.
- Trained YOLO v8 on a whiteboard detection dataset on Roboflow — achieved **91% accuracy** on the first run.

---

### Phase 2 — Parallel model training

**The challenge:** 89 classes is a lot for one person to train and iterate on alone.

**What we did:**
- Split the 89 classes evenly across 3 team members — roughly **30 classes each**.
- Reduced to **1,000 images per class** to keep the dataset balanced and avoid bias from class-size differences.
- Each member uploaded their portion to Roboflow and trained independently in parallel.
- Tested two architectures: **MobileNetV2** and **MobileNetV3** — both lightweight CNNs suited for real-time on-device inference.

**The iteration process:**  
Training was never one-and-done. After each run, we inspected accuracy **per class** — printing test images alongside true and predicted labels to catch failures visually. For each problematic class:
- Low accuracy → try adding more images
- Still low → try removing noisy images
- Fundamentally ambiguous → drop the class entirely
- Suspiciously 100% accurate → check for data leakage

This loop repeated until each member's subset reached satisfactory per-class accuracy.

---

### Phase 3 — Integration, the bug, and the pivot

**Merging:** All three trained subsets were combined into a single Roboflow workspace. A final **MobileNetV2** model was trained on all 89 classes together — requiring another full round of per-class maintenance at scale.

**The integration attempt:**  
We connected the trained CNN to the YOLO whiteboard detector to build the full real-world pipeline. Results were immediately wrong:
- The model predicted **"bee"** for almost every input
- Confidence scores stayed below **10%** regardless of what was drawn

**Root cause — double normalisation:**  
After investigation, we found the bug: the image was being **normalised twice** — once inside the YOLO preprocessing step, and again before being passed to the CNN. The resulting pixel values were completely outside the distribution the model had trained on. That is why it collapsed to a single class.

**Fixing the bug wasn't enough:**  
Even with the normalisation fixed, a deeper problem remained. Our CNN was trained exclusively on **clean digital doodles** — black strokes on white, consistent line width, no noise. Real whiteboard photos have camera noise, lighting variation, shadow, perspective distortion, and surface texture. The domain gap between training data and real-world input was too large to bridge cleanly.

**The pivot:**  
Rather than fight the domain gap, we removed it entirely. We dropped the camera and built a **digital drawing canvas** into the application itself. The user draws directly on screen, so the input the model receives at inference time is generated in exactly the same conditions as the training data — a clean digital sketch. Suddenly everything worked.

---

## Final architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AKINA-DRAW GUI                           │
│                       (Pygame desktop app)                      │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │  Class list  │    │    Canvas    │    │   Controls panel  │  │
│  │  89 classes │    │  128×128 px  │    │  Start/Stop/Clear │  │
│  │  + previews  │    │  white bg    │    │  Brush size       │  │
│  └──────────────┘    └──────┬───────┘    └───────────────────┘  │
│                             │                                   │
└─────────────────────────────┼───────────────────────────────────┘
                              │ on mouse release
                              ▼
                   ┌──────────────────────┐
                   │  OpenCV preprocess   │
                   │  RGB → grayscale     │
                   │  resize → 128×128    │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │    MobileNetV2 CNN   │
                   │    89-class output  │
                   │    softmax layer     │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │   Prediction result  │
                   │   Class label        │
                   │   Confidence score   │
                   │   Colour-coded bar   │
                   └──────────────────────┘
```

**Deprecated path (Phase 1 plan — not in final product):**
```
Camera → YOLO v8 whiteboard detection → crop → CNN → bounding box label
```
Abandoned due to domain gap between real whiteboard photos and digital training data.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Hager-ali191/Akina-Draw.git
cd Akina-Draw

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
```
tensorflow>=2.8
opencv-python
pygame
numpy
```

Place the trained model file `hand_drawing_model_Inception.h5` in the root directory.

If you want the class reference images in the side panel, place them in an `examples/` folder:
```
examples/
  cat.png
  elephant.png
  apple.png
  ...
```

---

## Usage

```bash
python main.py
```

**Controls:**

| Action | How |
|---|---|
| Draw | Hold left mouse button on the canvas |
| Start predicting | Click **▶ Start** — model predicts on every stroke |
| Stop predicting | Click **■ Stop** |
| Clear canvas | Click **✕ Clear** |
| Change brush size | Drag the brush slider |
| Change eraser size | Drag the eraser slider |
| Erase | Press E to toggle eraser — press again to switch back to pen |
| Undo last strokePress | Ctrl + Z |
| Browse all classes | Click the **▶** tab on the left to open the class panel |
| Preview a class | Hover over any class name in the panel |
| Show class panelPress | → (right arrow) |
| Hide class panelPress | ← (left arrow) |
| Quit | Press `ESC` or `Q` |

**Prediction colours:**
- 🟢 **Green** — confidence above 50%
- 🟠 **Amber** — confidence between 25–50%
- ⚪ **Grey** — confidence below 25%

---

## Supported classes

89 classes across animals, objects, foods, vehicles, and landmarks. A selection:

`airplane` · `apple` · `banana` · `bee` · `bicycle` · `bird` · `brain` · `bus` · `butterfly` · `cactus` · `cake` · `camel` · `camera` · `car` · `castle` · `cat` · `cloud` · `cow` · `dolphin` · `elephant` · `fish` · `flower` · `giraffe` · `guitar` · `hammer` · `hat` · `helicopter` · `house` · `ice cream` · `laptop` · `lion` · `monkey` · `octopus` · `panda` · `penguin` · `pineapple` · `pizza` · `rabbit` · `rocket` · `sailboat` · `sea turtle` · `sheep` · `smiley face` · `snowman` · `spider` · `squirrel` · `star` · `strawberry` · `sun` · `train` · `tree` · `umbrella` · `vase` · `watermelon` · and more.

Full list available in the application's class panel.

---

## Tech stack

| Component | Technology |
|---|---|
| GUI | Pygame |
| Image processing | OpenCV |
| Deep learning framework | TensorFlow / Keras |
| CNN architecture | MobileNetV2 |
| Dataset | Google Quick, Draw! |
| Training platform | Roboflow |
| Language | Python 3.8+ |

---

## Team

Built by a team of 3 for the **Robotech Fair**.

- **[Hager Ali]** — [@GitHub](https://github.com/Hager-ali191)
- **[TMohammed Gamal]** — [@GitHub](#https://github.com/MohamadGemy04)
- **[Mohammed Ahmed Hassan]** — [@GitHub](#https://github.com/Mohamed-Ahmed-prog)

And Also not to forget our **supervisor**
- **[Malak Hisham]** — [@GitHub](#https://github.com/malakhishams)

---

## Links

- 🔗 **GitHub:** https://github.com/Hager-ali191/Akina-Draw
- 📊 **Project slides:** https://www.canva.com/design/DAHHcMuVh0U/c1Y_KCjQVCFGVMFHEI-HPQ/edit
- 📦 **Dataset:** https://quickdraw.withgoogle.com/data

---

> *"The model isn't wrong — the input is wrong."*  
> The lesson that changed our whole architecture.
