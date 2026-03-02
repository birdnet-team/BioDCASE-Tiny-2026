<div align="center">
  <picture>
    <img src="../docs/under_construction.png" alt="Under construction" width="200">
  </picture>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../docs/BioDCASE_header_light.svg">
    <source media="(prefers-color-scheme: light)" srcset="../docs/BioDCASE_header_dark.svg">
    <img src="../docs/BioDCASE_header_dark.svg" alt="BioDCASE Logo" width="600">
  </picture>
  <picture>
    <img src="../docs/under_construction.png" alt="Under construction" width="200">
  </picture>
  <br><br>
</div>


This folder contains the pipeline for a **Tensorflow** machine learning framework based on previous year's challenge repo: [BioDCASE-Tiny-2025 Baseline](https://github.com/birdnet-team/BioDCASE-Tiny-2025).



## Setup and Installation

### Prerequisites

1. Python >=3.11 and <=3.13 with pip and venv
2. ESP32-S3-Korvo-2 development board and USB cable

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/birdnet-team/BioDCASE-Tiny-2026.git
cd BioDCASE-Tiny-2026/tensorflow_framework
```

2. Install your favourite python version, we used 3.11 for testing

3. Create a virtual environment (recommended)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

4. Install Python dependencies:
```bash
pip install -r requirements_tensorflow.txt
```

5. Set your serial device port in the pipeline_config.yaml

```yaml
embedded_code_generation:
  serial_device: <YOUR_DEVICE> 
```

### Running on Windows

As the required tflite-micro package is not easily available for Windows we recommend using WSL to run this project.

To make your device accessible for WSL you can use this guide: https://learn.microsoft.com/en-us/windows/wsl/connect-usb 

To determine your serial device port you can use the following command:

```bash
dmesg | grep tty
```

You might also need to grant some rights to run the deployment:

```bash
sudo adduser $USER dialout
sudo chmod a+rw $SERIAL_PORT
```

## Usage

- Modify model.py with your architecture (make sure to compile with optimizer and loss)
- Modify the training loop in the same file, if you need to
- Modify pipeline_config.yaml parameters of feature extraction
- run biodcase2026_tiny_ml_tensorflow.py

> [!IMPORTANT]
> Writing custom features rather than using what is implemented here, requires implementing a numerically equivalent version on the embedded target too!
> This is a necessary condition for the evaluated model to behave identically on the host and on the embedded target likewise.
> Note, that this is a non-trivial undertaking and we generally advice to stick to what is already implemented in this repository!


<!-- > [!IMPORTANT]
> Crucial information necessary for users to succeed. -->

## Development

### Quickstart

To run the complete pipeline execute (make sure your installed virtual python environment is activated):
   ```bash
   python biodcase2026_tiny_ml_tensorflow.py
   ```

This will execute the data preprocessing, extract the features, train the model and deploy it to your board.

Once deployed, benchmarking code on the ESP32-S3 will display info about the runtime performance of the preprocessing steps and actual model via serial monitor.

#### Step-by-Step Deployment Instructions

The steps of the pipeline can be executed individually

1. Data Preprocessing
   ```bash
   python data_preprocessing.py
   ```

2. Feature Extraction
   ```bash
   python feature_extraction.py
   ```

3. Model Training
   ```bash
   python model_training.py
   ```

4. Deployment
   ```bash
   python embedded_code_generation.py
   ```