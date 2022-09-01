import hydra
from omegaconf import DictConfig
import mlflow
import os
import logging


logger = logging.getLogger()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("START")
    mlflow.set_experiment(cfg["mlflow.experiment"]["name"])
    
    with mlflow.start_run(**cfg["mlflow.run"]):
        mlflow.log_artifact(
            os.path.join(os.getcwd(), __file__)
        )
        mlflow.log_artifacts(
            os.path.join(os.getcwd(), "cb_pred"), "cb_pred"
        )
    
    logger.info("FINISH")


if __name__ == "__main__":
    main()
