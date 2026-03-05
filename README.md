## RCMEL
RCMEL: An Image-Enhanced Refinement and Contrast Frameworkfor Multimodal Entity Linking

Zhi Zhang , Kehai Chen , Bing Xu , Muyun Yang , Hailong Cao , Conghui Zhu , Hongjiao Guan ,
Wenpeng Lu , and Tiejun Zhao

📖 Overview

RCMEL is a novel framework for Multimodal Entity Linking (MEL) that leverages image-enhanced refinement and contrastive learning strategies to achieve state-of-the-art performance on benchmark datasets.

Key Features:
• 🖼️ Multimodal Fusion: Seamlessly integrates textual and visual information

• 🔄 Image-enhanced Refinement: Enhances entity representations with visual cues

• ⚖️ Contrastive Learning: Improves discrimination through knowledge contrast

• 🚀 High Performance: Achieves SOTA results on WikiMEL and WikiDiverse datasets

🏗️ Architecture

The framework consists of four core components:

1. Information Refinement: Extracts and enhances visual features from multimodal contexts
2. Retrieval Augmentation: Selects candidate entities based on similarity matching
3. Knowledge Contrast: Captures dissimilarities between candidate entities
4. Ranking Decoding: Formulates MEL as a ranking task for optimal entity selection

📦 Installation

Prerequisites
  - torch>=1.9.0
  - transformers>=4.20.0
  - numpy>=1.21.0
  - pillow>=9.0.0
  - tqdm>=4.60.0
  - scikit-learn>=1.0.0


🚀 Quick Start

#### 1.Data Preparation

1. Download the WikiMEL and WikiDiverse datasets
2. Place the data in the data/ directory following the structure:

data/
├── wikimel/
│   ├── train.json
│   ├── valid.json
│   └── test.json
└── wikidiverse/
    ├── train.json
    ├── valid.json
    └── test.json



#### 2.Infer

bash src/wikidiverse.sh
