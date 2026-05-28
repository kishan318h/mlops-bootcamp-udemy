# What needs to be monitored in production?
## Functional level: the input, model, output
- Pydantic to ensure data integrity and types, etc
- Data drift: 2 sample t-test to check if train and test data are from same population or not, KS-statstic
- Model drift: decline in the model's predictive accuracy caused by changes in the real world environment. Monitored for predictive performance over time and statistical tests
- Tracking model version: ensure correct version is in production, manage version history and prediction records
- Monitoring output using ground truth labels - for example, model predicted the number of people likely to click on an impression. After serving the impressions check, how many actually clicked on the impressions. It may not be available in many use cases
- Prediction drift: check the change in distribution of prediction. For example, ratio of true vs false in a classification model

### Drift in Machine learning:
#### Data drift: 
- Also known as feature drift, population drift or covariate shift
- occurs when distribution or characteristic of input data changes
- drift happens because model is prepared for the new distribution

#### Prediction shift:
- data drift can lead to changes in prediction variable over time, resulting in prediction drift
- Also called prior probability shift, label drift or unconditional class shift
- it can occur due to addition or removal of classes in the data
- retrain the model to mitigate the impact

#### Concept Shift:
- also called posterior class shift
- it occurs when relationship between independent variable and dependent variable changes
- significant concept shift can lead to unreliable model predictions

#### Measure drift:
For continuous data:
1. Statistical distance metrics: 
    - wasserstein distance
    - Population Stability Index (PSI)
    - Characterstic Stability Index (CSI)
    - Kullback-leibler divergence (KL-divergence)
    - Jensen-Shannon divergence (JS-divergence)
    - Kolmogorov-Smirnov (KS) statistic
2. PCA: reconstruct low dimensional components back to features (training and production data). Check the mean and standard deviation. **Limitations** captures linear relationships only. Captures data drift and not concept drift
3. Mean, median, correlation, min, max

For categorical data:
1. cardinality test, chi-squared test, entropy
2. histograms and control charts
3. platforms like: whylabs, and libraries like alibi-detect and deep checks

### Strategies to tackle drift:
Retrain the model after identifying concept drift or data drift. When production data is insufficient, combine historical data and production data. Four strategies for model retraining:
1. periodic retraining: at scheduled times
2. event driven: when new data is available
3. model or metric-driven: based on evaluation metric or SLA threshold
4. online learning for continuous real-time

## Operational level: System performance, pipeline and costs
1. Memory usage
2. Latency
3. CPU/GPU utilization
4. Cost: compute heavy metrics require cost monitoring important. It can increase the cost by using more compute on AWS. etc.
5. Data pipeline

### Tools for monitoring
1. For infrastructure monitoring we can use Prometheus and Grafana. Prometheus can do application level monitoring.
We can use Prometheus to scrape the logs which can be loaded using Grafana. Whylabs (whylog) to track drift and data/model level tracking