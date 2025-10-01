# [WWW'25] CROWN: A Novel Approach to Comprehending Users' Preferences for Accurate Personalized News Recommendation
This repository provides an implementation of *CROWN* as described in the paper: [CROWN: A Novel Approach to Comprehending Users' Preferences for Accurate Personalized News Recommendation](https://dl.acm.org/doi/10.1145/3696410.3714752) by Yunyong Ko, Seongeun Ryu and Sang-Wook Kim, In Proceedings of the ACM Web Conference (WWW) 2025

## The overview of CROWN
![The overview of CROWN](./assets/CROWN_overview.PNG)

## Available dataset
1. [MIND Dataset](https://msnews.github.io/)
2. [Adressa Dataset](https://reclab.idi.ntnu.no/dataset/)

## Datasets
|Datasets|# of Users|# of News|Avg. title len|Avg. body len|
|:---:|:---:|:---:|:---:|:---:|
|MIND|94,057|65,238|11.67|41.01|
|Adressa|601,215|73,844|6.63|552.15|

## Dependencies
Our code runs on the Intel i7-9700k CPU with 64GB memory and NVIDIA RTX 2080 Ti GPU with 12GB, with the following packages installed:
```
python 3.10.16
torch 2.0.1
torchtext 0.15.2
pandas
numpy
argparse
sklearn
```
## How to run
```
python main.py --news_encoder=CROWN --user_encoder=CROWN
```

## Citation
Please cite our paper if you have used the code in your work. You can use the following BibTex citation:
```
@inproceedings{ko2025crown,
  title={CROWN: A Novel Approach to Comprehending Users' Preferences for Accurate Personalized News Recommendation},
  author={Ko, Yunyong and Ryu, Seongeun and Kim, Sang-Wook},
  booktitle={Proceedings of the ACM Web Conference (WWW) 2025},
  pages={xxxx--xxxx},
  year={2025}
}
```
