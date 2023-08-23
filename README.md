# InfoCL

Code for "InfoCL: Alleviating Catastrophic Forgetting in Continual Text Classification from An Information Theoretic Perspective"

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

