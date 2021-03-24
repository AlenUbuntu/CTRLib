# CTRLib

## Table of Contents
* Supported Models
  * LR
  * FM
  * DeepCross
  * Wide&Deep
  * DeepFM
  * DCN
  * xDeepFM
  * AutoInt+
  * FiBiNet
  * AFN+

* Datasets
  * [Criteo 1TB Click Logs dataset](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/)
  * [Avazu: Click-Through Rate Prediction](https://www.kaggle.com/c/avazu-ctr-prediction/data)


## How to Run



## Experiment
### Dataset
| Dataset | Positive | Negative| Total | Imbalance Ratio|
| :------:| :------: | :------:| :------:| :------:|
| Criteo |  29,040 | 970,960 | 1,000,000| 33.44 |

### Criteo
|   Method  | AUC | Log Loss |
|:---------:|:---:|:--------:|
|     LR    |  0.74896 &pm; 0.00018 |     0.33974 &pm; 0.00114     |
|     FM    |  0.75236 &pm; 0.00058   |   0.27502 &pm; 0.00329       |
| DeepCross |     |          |
| Wide&Deep |     |          |
|   DeepFM  |     |          |
|    DCN    |     |          |
|  xDeepFM  |     |          |
|  AutoInt+ |     |          |
|  FiBiNet  |     |          |
|    AFN+   |     |          |

### Avazu


## Future Work

* Support imbalance learning strategies other than re-sampling
