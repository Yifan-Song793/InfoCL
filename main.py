import os
import sys
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from types import SimpleNamespace
from functools import partial

import numpy as np
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    set_seed,
)

from train import default_hyper_train

logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = "false"


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    args = OmegaConf.create()   # cfg seems to be read-only
    args = OmegaConf.merge(args, cfg.task_args, cfg.model_args, cfg.training_args)
    args = SimpleNamespace(**args)
    
    logging.basicConfig(
        format="%(asctime)s - %(le5velname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.setLevel(logging.INFO)

    set_seed(args.seed)

    default_hyper_train(args)


if __name__ == "__main__":
    main()
