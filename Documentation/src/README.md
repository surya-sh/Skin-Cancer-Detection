---
updated: 2025-04-07T08:22
---
# An Interpretable Deep Learning Approach for Skin Cancer Categorization

This repository contains the code and datasets used in the paper titled **"An Interpretable Deep Learning Approach for Skin Cancer Categorization"** accepted and presented at the 26th International Conference on Computer and Information Technology (ICCIT) 2023.

**Paper Link:** [PDF](https://www.researchgate.net/publication/380199255_An_Interpretable_Deep_Learning_Approach_for_Skin_Cancer_Categorization)

## Table of Contents
  - [Dataset](#dataset)
  - [Result](#result)
  - [Citation](#citation)

## Dataset
We used in this paper publicly available [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) Dataset

## Result
#### Model-specific Classification Report of Weighted Average

| Models            |  Accuracy  | Precision |  Recall  | F1 Score |
| ----------------- | :--------: | :-------: | :------: | :------: |
| **XceptionNet**   | **88.72%** | **0.89**  | **0.89** | **0.89** |
| EfficientNetV2S   |   88.02%   |   0.88    |   0.88   |   0.88   |
| InceptionResNetV2 |   85.73%   |   0.86    |   0.86   |   0.85   |
| EfficientNetV2M   |   85.02%   |   0.89    |   0.89   |   0.89   |


## Citation
If you found this code helpful please consider citing,
```
@INPROCEEDINGS{10508527,
            author={Mahmud, Faysal and Mahfiz, Md. Mahin and Kabir, Md. Zobayer Ibna and Abdullah, Yusha},
            booktitle={2023 26th International Conference on Computer and Information Technology (ICCIT)}, 
            title={An Interpretable Deep Learning Approach for Skin Cancer Categorization}, 
            year={2023},
            volume={},
            number={},
            pages={1-6},
            keywords={Deep learning;Visualization;Explainable AI;Computational modeling;Medical services;Skin;Lesions;Skin Cancer Detection;Deep Learning;Pre-trained Models;Convolutional             Neural Networks (CNN);HAM10000;Medical Imaging;Explainable Artificial Intelligence (XAI)},
            doi={10.1109/ICCIT60459.2023.10508527}
}
```

# TODO 
Save the .h5 files of the four models
## License

This repository is licensed under the MIT License. See the [LICENSE](https://github.com/Faysal-MD/An-Interpretable-Deep-Learning-Approach-for-Skin-Cancer-Categorization-IEEE2023/blob/main/LICENSE) file for more information.
