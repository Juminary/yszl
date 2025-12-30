---
license: apache-2.0
task_categories:
- text-classification
- question-answering
- conversational
- text-generation
language:
- zh
tags:
- medical
pretty_name: Huatuo26M_v2
size_categories:
- 100K<n<1M
---







# Huatuo26M-Lite 📚

- ## Table of Contents 🗂

  - [Dataset Description](#dataset-description) 📝
  - [Dataset Information](#dataset-information) ℹ️
  - [Data Distribution](#data-distribution) 📊
  - [Usage](#usage) 🔧
  - [Citation](#citation) 📖

## Dataset Description 📝

Huatuo26M-Lite is a refined and optimized dataset based on the Huatuo26M dataset, which has undergone multiple purification processes and rewrites. It has more data dimensions and higher data quality. We welcome you to try using it.

## Dataset Information ℹ️

- **Dataset Name:** Huatuo26M-Lite
- **Version:** _[0.0.1]_
- **Size:** _[178k]_
- **Language:** _[Chinese]_

### Abstract 📄

We collected 26 million pieces of original QA data in the medical field, but it was not easy to use and had some risks because it was obtained from Common Crawl. Therefore, we took the following steps based on the original 26 million data: deduplication, cleaning, extraction of high-frequency questions, scoring of high-frequency questions using ChatGPT, and filtering only high-scoring questions. We then used ChatGPT to rewrite the answers to the high-scoring questions, resulting in a completely refined dataset. Please refer to our paper for the specific processing methods.

### Data Collection 🕵️‍♂️

ur question data was collected from the internet, and we extracted the high-frequency portion. The answers were rewritten by ChatGPT based on the original answers as a reference, and their quality was judged to be better than the original answers through manual evaluation. Therefore, please feel free to use our dataset with confidence.

### Preprocessing/Cleaning 🧹

The dataset has been processed to remove duplicates and cleaned to ensure high-quality data. It was then refined using OpenAI's ChatGPT, which helped in enhancing the overall quality of the dataset.

## Data Distribution 📊

This section provides a visual overview of the distribution of data in the Huatuo26M-Lite dataset.

**Data Categories Bar Chart:** ![label](http://file.huatuogpt.cn/files/models_ref/huatuo26m/high_quality_huatuoshine.png)

This chart represents the distribution of data categories in the dataset.





**Top 20 Associated Diseases Table:**

| topn | disease    | nums | ratio   |
| ---- | ---------- | ---- | ------- |
| 1    | 白癜风     | 3308 | 1.8615% |
| 2    | 人流       | 2686 | 1.5115% |
| 3    | 感冒       | 2371 | 1.3342% |
| 4    | 癫痫       | 2217 | 1.2476% |
| 5    | 痔疮       | 2134 | 1.2009% |
| 6    | 疼痛       | 1842 | 1.0366% |
| 7    | 咳嗽       | 1799 | 1.0124% |
| 8    | 前列腺炎   | 1564 | 0.8801% |
| 9    | 尖锐湿疣   | 1516 | 0.8531% |
| 10   | 肺癌       | 1408 | 0.7923% |
| 11   | 出血       | 1400 | 0.7878% |
| 12   | 鼻炎       | 1370 | 0.7709% |
| 13   | 肝癌       | 1354 | 0.7619% |
| 14   | 糖尿病     | 1348 | 0.7586% |
| 15   | 过敏性鼻炎 | 1295 | 0.7287% |
| 16   | 发烧       | 1265 | 0.7119% |
| 17   | 乙肝       | 1232 | 0.6933% |
| 18   | 便秘       | 1214 | 0.6832% |
| 19   | 甲亢       | 1178 | 0.6629% |
| 20   | 脱发       | 1173 | 0.6601% |







This table shows the top 20 diseases associated with the data entries in the dataset, along with their respective data entry counts and proportions.



## Usage 🔧

```python

from datasets import load_dataset

dataset = load_dataset("FreedomIntelligence/Huatuo26M-Lite")

```

## Citation 📖

```
@misc{li2023huatuo26m,
      title={Huatuo-26M, a Large-scale Chinese Medical QA Dataset}, 
      author={Jianquan Li and Xidong Wang and Xiangbo Wu and Zhiyi Zhang and Xiaolong Xu and Jie Fu and Prayag Tiwari and Xiang Wan and Benyou Wang},
      year={2023},
      eprint={2305.01526},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

---

Please note that this dataset is distributed "AS IS" without any warranty, express or implied, from the provider. Users should cite the dataset appropriately and respect any licensing or usage restrictions.