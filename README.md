# CTRLib

## Supported Models

 |    Name   |                                                   Paper                                                  | Publisher | Year |
|:---------:|:--------------------------------------------------------------------------------------------------------:|:---------:|:----:|
|     LR    |                                            Logistic Regression                                           |     -     |   -  |
|     FM    |                                          Factorization Machines                                          |    ICDM   | 2010 |
|    DNN    |                             Deep Neural Networks for YouTube Recommendations                             |   RecSys  | 2016 |
| Wide&Deep |                               Wide & Deep Learning for Recommender Systems                               |    DLRS   | 2016 |
|    PNN    |                        Product-Based Neural Networks for User Response Prediction.                       |    ICDM   | 2016 |
| DeepCross |            Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features.            |    KDD    | 2016 |
|    NFM    |                       Neural Factorization Machines for Sparse Predictive Analytics                      |   SIGIR   | 2017 |
|    AFM    |  Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks  |   IJCAI   | 2017 |
|   DeepFM  |                  DeepFM: A Factorization-Machine based Neural Network for CTR Prediction                 |   IJCAI   | 2017 |
|    DCN    |                              Deep & Cross Network for Ad Click Predictions.                              |   ADKDD   | 2017 |
|  xDeepFM  |          xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems.          |    KDD    | 2018 |
|    DIN    |                          Deep Interest Network for Click-Through Rate Prediction                         |    KDD    | 2018 |
|  AutoInt+ |             AutoInt: Automatic Feature Interaction Learning via SelfAttentive Neural Networks            |    CIKM   | 2019 |
|  FiBiNet  | FiBiNET: combining feature importance and bilinear feature interaction for click-through rate prediction |   RecSys  | 2019 |
|    DIEN   |                     Deep Interest Evolution Network for Click-Through Rate Prediction                    |   AAAI    | 2019 |
|    AFN+   |               Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions.              |    AAAI   | 2020 | 

## How to Run



## Experiment
### Dataset
  * [Criteo 1TB Click Logs dataset](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/)
   * [Avazu: Click-Through Rate Prediction](https://www.kaggle.com/c/avazu-ctr-prediction/data)

| Dataset | Positive | Negative| Total | Imbalance Ratio|
| :------:| :------: | :------:| :------:| :------:|
| Criteo |  29,040 | 970,960 | 1,000,000| 33.44 |

### Criteo

|   Method  | AUC | Log Loss | #Params | Time x Epochs (Training) | FLOPS |
|:---------:|:---:|:--------:|:-------:|:------------------------:|:-----:|
|     LR    |  0.74896 &pm; 0.00018   |    0.33974 &pm; 0.00114      |         |                          |       |
|     FM    |  0.75236 &pm; 0.00058   |     0.27502 &pm; 0.00329     |         |                          |       |
| DeepCross |     |          |         |                          |       |
| Wide&Deep |     |          |         |                          |       |
|   DeepFM  |     |          |         |                          |       |
|    DCN    |     |          |         |                          |       |
|  xDeepFM  |     |          |         |                          |       |
|  AutoInt+ |     |          |         |                          |       |
|  FiBiNet  |     |          |         |                          |       |
|    AFN+   |     |          |         |                          |       |

### Avazu


## Future Work

* Support imbalance learning strategies other than re-sampling
