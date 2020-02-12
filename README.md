# Suicidal Ideation Detection in Online User Contents

Project archived. No updates, no dataset licensing. Please consider using UMD Suicidality dataset instead.

# Getting Started
Due to the anonymity of online media and social networks, people tend to express their feelings and sufferings in online communities.
In order to prevent suicide, it is necessary to detect suicide-related posts and users' suicide ideation in cyberspace by natural language processing methods.
We focus on the online community called Reddit and the social networking website Twitter, and classify users' posts with potential suicide and without suicidal risk through texts features processing and machine learning based methods.


# Datasets
We collect two sets of data from Reddit and Twitter.
The Reddit data set includes 5,326 suicidal ideation samples and a number of non-suicide texts (20k). The Twitter dataset has totally 10k tweets with 594 tweets (around 6\%) with suicidal ideation.

The Reddit word cloud (left) and Twitter word cloud (right) are shown as follow:

<img width="400" alt="Reddit word cloud" src="https://github.com/shaoxiongji/sw-detection/blob/master/output/reddit.jpg"><img width="400" alt="Twitter word cloud" src="https://github.com/shaoxiongji/sw-detection/blob/master/output/twitter.jpg">

The original text data can not be provided publicly for the consideration of users privacy. It will be provided by request, see the data availability in our paper.

**Notice** Only Reddit dataset is available for sharing. Please contact me using your _institutional email_ to identify yourself when request for it. Due to a large number of messages, I may miss your message or it may been misclassified as spam. Sorry for that. 

If you'd like to collect your own data, please refer this repository: [web spider](https://github.com/shaoxiongji/webspider-eda).

These two xlsx files in this project contain some sample data composed by the author.
*Notice: when running the scripts, please replace them with requested data or your own data.*

# Features Precessing
We extracted six sets of features, i.e., statistical features, POS counts, TF-IDF, Topics probability, [LIWC](http://liwc.wpengine.com), and pre-trained [word2vec](https://radimrehurek.com/gensim/models/word2vec.html) word embedding.

The csv files are the processed features using [LIWC](http://liwc.wpengine.com).
*Notice: when running the scripts, please replace them with your own data.*

All these features are visualized in the following 6 pictures, using PCA as dimensionality reduction.

<img width="500" alt="features" src="https://github.com/shaoxiongji/sw-detection/blob/master/output/all.png">

# Running the scripts
Six models were implemented. They are logistic regression, random forest, gradient boosting decision tree, xgboost, support vector machine, and LSTM networks.

Former five models for Reddit and Twitter were implemented by
`python clf.py`
and
`python clf_reddit.py`.
The LSTM model for Reddit and Twitter by
`python lstm.py` and `python lstm_reddit.py`.
`python lstm_word2vec.py` and `python lstm_word2vec_reddit.py`.

These scripts were written in Python 3.6. Please check the requirements before running.

# Results
Part of experimental results as below on Reddit SuicideWatch vs. all dataset with 5,326 posts containing suicidal ideation.
 
| Model	| Acc.	    | Pre.	    | Rec.      |	F1	    | AUC      |
|------ | ------    | ------    | ------    | ------    | ------   |
|RF	    | 0.941440  | 0.958286	| 0.906931	| 0.931861	| 0.986029 |
|GBDT	| 0.961845	| 0.964161	| 0.948894	| 0.956437	| 0.991860 |
|XGB	| 0.965660  | 0.969280	| 0.952525	| 0.960796	| 0.993403 |
|LSTM	| 0.961098	| 0.959305	| 0.952117	| 0.955449	| 0.992637 |

# References
Please cite our paper if you use this repo.
```
@article{ji2018supervised,
  title={Supervised Learning for Suicidal Ideation Detection in Online User Content},
  author={Ji, Shaoxiong and Yu, Celina Ping and Fung, Sai-fu and Pan, Shirui and Long, Guodong},
  journal={Complexity},
  volume={2018},
  year={2018},
  publisher={Hindawi}
}

@article{ji2020suicidal,
  title={Suicidal Ideation Detection: A Review of Machine Learning Methods and Applications},
  author={Ji, Shaoxiong and Pan, Shirui and Li, Xue and Cambria, Erik and Long, Guodong and Huang, Zi},
  journal={arXiv preprint arXiv:1910.12611},
  year={2020}
}
```

There is also a remarkable work from University of Maryland which was finished almost at the same period of our work.

> Han-Chin Shing, Suraj Nair, Ayah Zirikly, Meir Friedenberg, Hal Daumé III, and Philip Resnik, "[Expert, Crowdsourced, and Machine Assessment of Suicide Risk via Online Postings](http://aclweb.org/anthology/W18-0603)", Proceedings of the Fifth Workshop on Computational Linguistics and Clinical Psychology: From Keyboard to Clinic, pages 25–36, New Orleans, Louisiana, June 5, 2018.

The [UMD Reddit Suicidality Dataset](http://users.umiacs.umd.edu/~resnik/umd_reddit_suicidality_dataset.html) published in June 5, 2018 is highly recommended for research on suicidality and suicide prevention, which was constructed from the [2015 Full Reddit Submission Corpus](https://www.reddit.com/r/datasets/comments/3mg812/full_reddit_submission_corpus_now_available_2006/)

The following publication(s) use this dataset. 
> Tadesse, M. M., Lin, H., Xu, B., & Yang, L. (2020). Detection of Suicide Ideation in Social Media Forums Using Deep Learning. Algorithms, 13(1), 7.
