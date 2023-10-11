# InfoCL

Code for Findings of EMNLP 2023: *[InfoCL: Alleviating Catastrophic Forgetting in Continual Text Classification from An Information Theoretic Perspective](https://arxiv.org/abs/2310.06362)*
## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# FewRel
python main.py +task_args=FewRel

# TACRED
python main.py +task_args=TACRED

# MAVEN
python main.py +task_args=MAVEN

# HWU64
python main.py +task_args=HWU64
```

## Citation

If you find this repo useful, please cite us.

```bibtex
@misc{song2023infocl,
      title={InfoCL: Alleviating Catastrophic Forgetting in Continual Text Classification from An Information Theoretic Perspective}, 
      author={Yifan Song and Peiyi Wang and Weimin Xiong and Dawei Zhu and Tianyu Liu and Zhifang Sui and Sujian Li},
      year={2023},
      eprint={2310.06362},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
