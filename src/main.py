import logging
from omegaconf import DictConfig, OmegaConf
import hydra

# A logger for this file
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=".", config_name="../configs/config")
def my_app(cfg):
    print(OmegaConf.to_yaml(cfg))
    log.info("info log test")

if __name__ == "__main__":
    my_app()