## Synthesizing Entity Resolution Datasets
**This is a project that is aiming to synthesize entity resolution datasets.**

## Description
Entity resolution (ER) is a core problem in data integration. Many companies have lots of datasets where ER needs to be conducted to integrate the data. On the one hand, it is nontrivial for non-ER experts within companies to design ER solutions. On the other hand, most companies are reluctant to release their real datasets for multiple reasons (e.g., privacy issues). 
A typical solution from the machine learning (ML) and the statistical community is to create surrogate (\aka analogous) datasets based on the real dataset, release these surrogate datasets to the public to train ML models, such that these models trained on surrogate datasets can be either directly used or be adapted for the real dataset by the companies.
We study the problem of synthesizing surrogate ER datasets using transformer models, with the goal that the ER model trained on the synthesized dataset can be used directly on the real dataset.
The synthesized ER datasets have the 3 following properties:
1. **Indistinguishable Entities**: one cannot tell one entity is real or synthesized.
2. **Performance Preservation**: the ER models trained by the real and synthesized ER datasets have similar performance on a same test set.
3. **Privacy Preserving**: the synthesized ER datasets will not leak privacy of entites in the real ER dataset.


## Datasets
|#|Dataset|Domain|URL|
|---|----|-----|-----|
|1|DBLP-ACM|scholar|https://dbs.uni-leipzig.de/file/DBLP-ACM.zip
|2|Restaurant|restaurant|http://www.cs.utexas.edu/users/ml/riddle/data/restaurant.tar.gz
|3|Walmart-Amazon|electronics|http://pages.cs.wisc.edu/∼anhai/data/corleone_data/products/walmart.csv
|4|iTunes-Amazon|music|https://pages.cs.wisc.edu/∼anhai/data1/deepmatcher_data/Structured/iTunes-Amazon/


## Platforms
Code has been tested on **OS X**, **CentOS** and **Linux**.


## Usage
### Dependencies 
- [x] Python 3.6+
- [x] [sklearn](https://scikit-learn.org/stable/)
- [x] [numpy](https://github.com/numpy/numpy)
- [x] [pandas](https://pandas.pydata.org/)
- [x] [py_entitymatching](https://github.com/anhaidgroup/py_entitymatching)
- [x] [The Daisy repository](https://github.com/ruclty/Daisy)


### How to use
If you want to get the latest source code, please clone it from Github repo with the following command (take the dataset DBLP-ACM as an example):
```
https://github.com/Qinxuedi/ersyn
cd DBLP-ACM
python3 generate_paper.py
```


## Contributors
|#|Contributor|Affiliation|Contact|
|---|----|-----|-----|
|1|Xuedi Qin| PhD Candidate, Tsinghua University| qxd17@mails.tsinghua.edu.cn
|2|Chengliang Chai| Postdoc Researcher, Tsinghua University | ccl@mail.tsinghua.edu.cn
|3|[Yuyu Luo](https://luoyuyu.vip)| PhD Candidate, Tsinghua University| luoyy18@mails.tsinghua.edu.cn
|4|[Nan Tang](http://da.qcri.org/ntang/index.html)|Senior Scientist, Qatar Computing Research Institute|ntang@hbku.edu.qa
|5|[Guoliang Li](http://dbgroup.cs.tsinghua.edu.cn/ligl/)|Professor, Tsinghua University| LastName+FirstName@tsinghua.edu.cn
##### If you have any questions or feedbacks about this project, please feel free to contact Xuedi Qin (qxd17@mails.tsinghua.edu.cn).
