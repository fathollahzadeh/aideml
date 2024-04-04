# AIDE: Autonomous AI for Data Science
Welcome to the official repository for AIDE, an AI system that can automatically solve data science tasks at a human level, and with human input, it can perform even better. We believe giving developers and researchers direct access to AIDE locally, with local compute and choice to use their own LLM keys, is the most straightforward way to make it useful. That's why we'll open-source it, and the tentative timeline is it will arrive before the end of April. Currently, this repository serves as a gallery showcasing its solutions for 60+ Kaggle competitions we tested.

## About AIDE
AIDE is an AI-powered data science assistant that can autonomously understand task requirements, design, and implement solutions. By leveraging large language models and innovative agent architectures, such as the Solution Space Tree Search algorithm, AIDE has achieved human-level performance on a wide range of data science tasks, outperforming over 50% of human data scientists on Kaggle competitions.

## Gallery
| Domain                           | Task                                                                    | Top%        | Solution Link                                                     | Competition Link                                                                                   |
|:---------------------------------|:------------------------------------------------------------------------|:------------|:------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------|
| Urban Planning                   | Forecast city bikeshare system usage                                    | 0.05        | [link](examples/bike-sharing-demand.py)                           | [link](https://www.kaggle.com/competitions/bike-sharing-demand/overview)                           |
| Physics                          | Predicting Critical Heat Flux                                           | 0.56        | [link](examples/playground-series-s3e15.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e15/overview)                       |
| Genomics                         | Classify bacteria species from genomic data                             | 0.0         | [link](examples/tabular-playground-series-feb-2022.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-feb-2022/overview)            |
| Agriculture                      | Predict blueberry yield                                                 | 0.58        | [link](examples/playground-series-s3e14.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e14/overview)                       |
| Healthcare                       | Predict disease prognosis                                               | 0.0         | [link](examples/playground-series-s3e13.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e13/overview)                       |
| Economics                        | Predict monthly microbusiness density in a given area                   | 0.35        | [link](examples/godaddy-microbusiness-density-forecasting.py)     | [link](https://www.kaggle.com/competitions/godaddy-microbusiness-density-forecasting/overview)     |
| Cryptography                     | Decrypt shakespearean text                                              | 0.91        | [link](examples/ciphertext-challenge-iii.py)                      | [link](https://www.kaggle.com/competitions/ciphertext-challenge-iii/overview)                      |
| Data Science Education           | Predict passenger survival on Titanic                                   | 0.78        | [link](examples/tabular-playground-series-apr-2021.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-apr-2021/overview)            |
| Software Engineering             | Predict defects in c programs given various attributes about the code   | 0.0         | [link](examples/playground-series-s3e23.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e23/overview)                       |
| Real Estate                      | Predict the final price of homes                                        | 0.05        | [link](examples/home-data-for-ml-course.py)                       | [link](https://www.kaggle.com/competitions/home-data-for-ml-course/overview)                       |
| Real Estate                      | Predict house sale price                                                | 0.36        | [link](examples/house-prices-advanced-regression-techniques.py)   | [link](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)   |
| Entertainment Analytics          | Predict movie worldwide box office revenue                              | 0.62        | [link](examples/tmdb-box-office-prediction.py)                    | [link](https://www.kaggle.com/competitions/tmdb-box-office-prediction/overview)                    |
| Entertainment Analytics          | Predict scoring probability in next 10 seconds of a rocket league match | 0.21        | [link](examples/tabular-playground-series-oct-2022.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-oct-2022/overview)            |
| Environmental Science            | Predict air pollution levels                                            | 0.12        | [link](examples/tabular-playground-series-jul-2021.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-jul-2021/overview)            |
| Environmental Science            | Classify forest categories using cartographic variables                 | 0.55        | [link](examples/forest-cover-type-prediction.py)                  | [link](https://www.kaggle.com/competitions/forest-cover-type-prediction/overview)                  |
| Computer Vision                  | Predict the probability of machine failure                              | 0.32        | [link](examples/playground-series-s3e17.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e17/overview)                       |
| Computer Vision                  | Identify handwritten digits                                             | 0.14        | [link](examples/digit-recognizer.py)                              | [link](https://www.kaggle.com/competitions/digit-recognizer/overview)                              |
| Manufacturing                    | Predict missing values in dataset                                       | 0.7         | [link](examples/tabular-playground-series-jun-2022.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-jun-2022/overview)            |
| Manufacturing                    | Predict product failures                                                | 0.48        | [link](examples/tabular-playground-series-aug-2022.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/overview)            |
| Manufacturing                    | Cluster control data into different control states                      | 0.96        | [link](examples/tabular-playground-series-jul-2022.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-jul-2022/overview)            |
| Natural Language Processing      | Classify toxic online comments                                          | 0.78        | [link](examples/jigsaw-toxic-comment-classification-challenge.py) | [link](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/overview) |
| Natural Language Processing      | Predict passenger transport to an alternate dimension                   | 0.59        | [link](examples/spaceship-titanic.py)                             | [link](https://www.kaggle.com/competitions/spaceship-titanic/overview)                             |
| Natural Language Processing      | Classify sentence sentiment                                             | 0.42        | [link](examples/sentiment-analysis-on-movie-reviews.py)           | [link](https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/overview)           |
| Natural Language Processing      | Predict whether a tweet is about a real disaster                        | 0.48        | [link](examples/nlp-getting-started.py)                           | [link](https://www.kaggle.com/competitions/nlp-getting-started/overview)                           |
| Business Analytics               | Predict total sales for each product and store in the next month        | 0.87        | [link](examples/competitive-data-science-predict-future-sales.py) | [link](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/overview) |
| Business Analytics               | Predict book sales for 2021                                             | 0.66        | [link](examples/tabular-playground-series-sep-2022.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-sep-2022/overview)            |
| Business Analytics               | Predict insurance claim amount                                          | 0.8         | [link](examples/tabular-playground-series-feb-2021.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-feb-2021/overview)            |
| Business Analytics               | Minimize penalty cost in scheduling families to santa's workshop        | 1.0         | [link](examples/santa-2019-revenge-of-the-accountants.py)         | [link](https://www.kaggle.com/competitions/santa-2019-revenge-of-the-accountants/overview)         |
| Business Analytics               | Predict yearly sales for learning modules                               | 0.26        | [link](examples/playground-series-s3e19.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e19/overview)                       |
| Business Analytics               | Binary classification of manufacturing machine state                    | 0.6         | [link](examples/tabular-playground-series-may-2022.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-may-2022/overview)            |
| Business Analytics               | Forecast retail store sales                                             | 0.36        | [link](examples/tabular-playground-series-jan-2022.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-jan-2022/overview)            |
| Business Analytics               | Predict reservation cancellation                                        | 0.54        | [link](examples/playground-series-s3e7.py)                        | [link](https://www.kaggle.com/competitions/playground-series-s3e7/overview)                        |
| Finance                          | Predict the probability of an insurance claim                           | 0.13        | [link](examples/tabular-playground-series-mar-2021.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-mar-2021/overview)            |
| Finance                          | Predict loan loss                                                       | 0.0         | [link](examples/tabular-playground-series-aug-2021.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-aug-2021/overview)            |
| Finance                          | Predict a continuous target                                             | 0.42        | [link](examples/tabular-playground-series-jan-2021.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-jan-2021/overview)            |
| Finance                          | Predict customer churn                                                  | 0.24        | [link](examples/playground-series-s4e1.py)                        | [link](https://www.kaggle.com/competitions/playground-series-s4e1/overview)                        |
| Finance                          | Predict median house value                                              | 0.58        | [link](examples/playground-series-s3e1.py)                        | [link](https://www.kaggle.com/competitions/playground-series-s3e1/overview)                        |
| Finance                          | Predict closing price movements for nasdaq listed stocks                | 0.99        | [link](examples/optiver-trading-at-the-close.py)                  | [link](https://www.kaggle.com/competitions/optiver-trading-at-the-close/overview)                  |
| Finance                          | Predict taxi fare                                                       | 1.0         | [link](examples/new-york-city-taxi-fare-prediction.py)            | [link](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/overview)            |
| Finance                          | Predict insurance claim probability                                     | 0.62        | [link](examples/tabular-playground-series-sep-2021.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-sep-2021/overview)            |
| Biotech                          | Predict cat in dat                                                      | 0.66        | [link](examples/cat-in-the-dat-ii.py)                             | [link](https://www.kaggle.com/competitions/cat-in-the-dat-ii/overview)                             |
| Biotech                          | Predict the biological response of molecules                            | 0.62        | [link](examples/tabular-playground-series-oct-2021.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-oct-2021/overview)            |
| Biotech                          | Predict medical conditions                                              | 0.92        | [link](examples/icr-identify-age-related-conditions.py)           | [link](https://www.kaggle.com/competitions/icr-identify-age-related-conditions/overview)           |
| Biotech                          | Predict wine quality                                                    | 0.61        | [link](examples/playground-series-s3e5.py)                        | [link](https://www.kaggle.com/competitions/playground-series-s3e5/overview)                        |
| Biotech                          | Predict binary target without overfitting                               | 0.98        | [link](examples/dont-overfit-ii.py)                               | [link](https://www.kaggle.com/competitions/dont-overfit-ii/overview)                               |
| Biotech                          | Predict concrete strength                                               | 0.86        | [link](examples/playground-series-s3e9.py)                        | [link](https://www.kaggle.com/competitions/playground-series-s3e9/overview)                        |
| Biotech                          | Predict crab age                                                        | 0.46        | [link](examples/playground-series-s3e16.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e16/overview)                       |
| Biotech                          | Predict enzyme characteristics                                          | 0.1         | [link](examples/playground-series-s3e18.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e18/overview)                       |
| Biotech                          | Classify activity state from sensor data                                | 0.51        | [link](examples/tabular-playground-series-apr-2022.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-apr-2022/overview)            |
| Biotech                          | Predict horse health outcomes                                           | 0.86        | [link](examples/playground-series-s3e22.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e22/overview)                       |
| Biotech                          | Predict the mohs hardness of a mineral                                  | 0.64        | [link](examples/playground-series-s3e25.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e25/overview)                       |
| Biotech                          | Predict cirrhosis patient outcomes                                      | 0.51        | [link](examples/playground-series-s3e26.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e26/overview)                       |
| Biotech                          | Predict obesity risk                                                    | 0.62        | [link](examples/playground-series-s4e2.py)                        | [link](https://www.kaggle.com/competitions/playground-series-s4e2/overview)                        |
| Biotech                          | Classify presence of feature in data                                    | 0.66        | [link](examples/cat-in-the-dat.py)                                | [link](https://www.kaggle.com/competitions/cat-in-the-dat/overview)                                |
| Biotech                          | Predict patient's smoking status                                        | 0.4         | [link](examples/playground-series-s3e24.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e24/overview)                       |