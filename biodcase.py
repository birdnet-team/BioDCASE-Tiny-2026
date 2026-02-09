from embedded_code_generation import run_embedded_code_generation
from feature_extraction import run_feature_extraction
from data_preprocessing import run_preprocessing
from config import load_config
from model_training import run_model_training


if __name__ == '__main__':
    config = load_config()
    run_preprocessing(config)
    run_feature_extraction(config)
    run_model_training(config)
    run_embedded_code_generation(config)
