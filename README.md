# GenEx: A Commonsense-aware Unified Generative Framework for Explainable Cyberbullying Detection (EMNLP 2023)

This is the official repository accompanying the EMNLP 2023 full paper [GenEx: A Commonsense-aware Unified Generative Framework for Explainable Cyberbullying Detection](https://aclanthology.org/2023.emnlp-main.1035.pdf). This repository contains codebase and  dataset.

# Authors
Krishanu Maity, Prince Jha, Raghav Jain, Sriparna Saha, Pushpak Bhattacharyya
# Dataset Description
We created an explainable cyberbullying dataset called BullyExplain, addressing four tasks simultaneously: Cyberbullying Detection (CD), Sentiment Analysis (SA), Target Identification (TI), and Detection of Rationales (RD). Each tweet in this dataset is annotated with four classes: Bully (Yes/No), Sentiment (Positive/Neutral/Negative), Target (Religion/Sexual-Orientation/Attacking-Relatives-and-Friends/Organization/Community/Profession/Miscellaneous), and Rationales (highlighted parts of the text justifying the classification decision). If the post is non-bullying, the rationales are not marked, and the target class is selected as NA (Not Applicable). The BullyExplain dataset comprises a total of 6,084 samples, with 3,034 samples belonging to the non-bully class and the remaining 3,050 samples marked as bully. The number of tweets with positive and neutral sentiments is 1,536 and 1,327, respectively, while the remaining tweets express negative sentiments.


# Citation
If you find this repository to be helpful please cite us.


@inproceedings{DBLP:conf/emnlp/MaityJJ0B23,
  author       = {Krishanu Maity and
                  Raghav Jain and
                  Prince Jha and
                  Sriparna Saha and
                  Pushpak Bhattacharyya},
  editor       = {Houda Bouamor and
                  Juan Pino and
                  Kalika Bali},
  title        = {GenEx: {A} Commonsense-aware Unified Generative Framework for Explainable
                  Cyberbullying Detection},
  booktitle    = {Proceedings of the 2023 Conference on Empirical Methods in Natural
                  Language Processing, {EMNLP} 2023, Singapore, December 6-10, 2023},
  pages        = {16632--16645},
  publisher    = {Association for Computational Linguistics},
  year         = {2023},
  url          = {https://aclanthology.org/2023.emnlp-main.1035},
  timestamp    = {Wed, 13 Dec 2023 17:20:20 +0100},
  biburl       = {https://dblp.org/rec/conf/emnlp/MaityJJ0B23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
