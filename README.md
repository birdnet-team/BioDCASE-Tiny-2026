<div align="center">
  <picture>
    <img src="docs/under_construction.png" alt="Under construction" width="200">
  </picture>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/BioDCASE_header_light.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/BioDCASE_header_dark.svg">
    <img src="docs/BioDCASE_header_dark.svg" alt="BioDCASE Logo" width="600">
  </picture>
  <picture>
    <img src="docs/under_construction.png" alt="Under construction" width="200">
  </picture>
  <br><br>
</div>

**BioDCASE-Tiny 2026 competition (Task 3)** - A machine learning challenge to bird sound recognition on tiny hardware, also visit the [official BioDCASE 2026 Task 3 website](https://biodcase.github.io/challenge2026/task3) for more information.


## Background

BioDCASE-Tiny is a competition for developing efficient machine learning models for bird audio recognition that can run on resource-constrained embedded devices. The project uses the ESP32-S3-Korvo-2 development board, which offers audio processing capabilities in a small form factor suitable for field deployment.
This year we changed the main framework to work on pytorch and transfered the updated tensorflow baseline to the `tensorflow_framework` folder.


## Table of Contents
- [Dataset](#dataset)
- [Setup and Installation](#setup-and-installation)
- [Development](#development)
- [ESP32-S3-Korvo-2 Development Board](#esp32-s3-korvo-2-development-board)
- [Code Structure](#code-structure)
- [Development Tips](#development-tips)
- [Evaluation Metrics](#evaluation-metrics)
- [License](#license)
- [Citation](#citation)
- [Funding](#funding)
- [Partners](#partners)


## Dataset

The dataset for this year's task consists of field recordings from Germany of 10 bird species and an additional set of urban backround sounds:

- 11 class labels categorized in folders
- 3,300 recordings of 3 seconds each...
- audio is sampled at 24 kHz, mono, 16-bit PCM wav files

The dataset is organized as follows:

```
Development_Set/
├── Train/
│   ├── species_1/
|       ├── recording_1.wav
|       ├── recording_2.wav
|       ├── ...
│   ├── species_2/
│   ├── ...
├── Validation/
│   ├── species_1/
│   ├── species_2/
```

Download the dataset from: [BioDCASE-Tiny 2026 Dataset]()


## Setup and Installation

### Prerequisites

1. Python >=3.11 and <=3.13 with pip and venv
2. [Docker](https://www.docker.com/get-started/) (runs ESP-IDF in a container) or locally installed [ESP-IDF](https://github.com/espressif/esp-idf)  
3. (optional) ESP32-S3-Korvo-2 development board and two USB cables (power and serial connection)

> [!IMPORTANT]
> You can also participate in the challenge if you do not want to buy a Korvo-2 dev board, but consider that you will not be able to check if your model is actually deployable on the korvo system.
> Note, that we will not accept models that do not run on our system.

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/birdnet-team/BioDCASE-Tiny-2026.git
cd BioDCASE-Tiny-2026
```

2. Install your favorite python version, we used 3.13 for testing

3. Choose if you want to develop in pytorch or tensorflow. If you choose pytorch, you can continue here. If you choose tensorflow go to the `tensorflow_framework` folder in this repository.

3. Create a virtual environment (recommended)

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

4. Install Python dependencies:
```bash
pip install -r requirements_<pytorch|tensorflow>.txt
```

5. Install Docker on your system ([link](https://www.docker.com/get-started/)). Afterwards you have to activate the docker deamon and add it to your user group. On linux it looks something like:
```bash
systemctl status docker.socket
systemctl enable docker.socket
sudo gpasswd -a <your_username> docker
```
then run a test with:
```bash
docker run hello-world
```

6. (alternatively) install the [ESP-IDF](https://github.com/espressif/esp-idf) framework on your local pc and compile the code by hand.

### Configuration

1. Make sure you add some rights to your usb device to connect to the korvo board. This depends a bit on the system, on linux, you have to add udev rules in `/etc/udev/rules.d/`, e.g. add a file  there `/etc/udev/rules.d/10-custom-usb.rules` with content:
```
# ubuntu vs. arch
# GROUP="dialout" vs. GROUP="uucp"
# rule for esp32 (devkit-c or korvo)
KERNEL=="ttyUSB0", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", GROUP="uucp", MODE="0666"
```
and add to your group ("uucp" for arch, "dialout" for ubuntu), then reload and restart your pc:
```
sudo gpasswd -a <your_username> uucp
sudo udevadm control --reload-rules
```

2. Set your serial device port in the `config.yaml` for linux it is usually `/dev/ttyUSB0`

```yaml
generate_embedded_code:
  serial_device: <YOUR_DEVICE> 
```

3. Move the downloaded dataset to any location on your pc, for instance, `./output/00_raw/` and edit `config.yaml` file at following location:
```yaml
datamodule:
  dataset:
    root_path: ./output/00_raw/
```
you can also change the intermediate and cache (feature) paths at:
```yaml
datamodule:
  intermediate:
    root_path: ./output/01_intermediate/
  caching:
    root_path: ./output/02_features/
```

### Installation and running on Windows

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

## Development

- Modify `config.yaml` to change feature extraction or model parameters
- Modify `model_tiny_ml.py` or create your own model that is based on the `ModelBase` class implemented

> [!IMPORTANT]
> Writing custom features rather than using what is implemented here, requires implementing a numerically equivalent version on the embedded target too!
> This is a necessary condition for the evaluated model to behave identically on the host and on the embedded target likewise.
> Note, that this is a non-trivial undertaking and we generally advice to stick to what is already implemented in this repository!

### Quickstart

To run the complete pipeline execute:
```bash
python biodcase2026_tiny_ml.py
```

This will execute the data preprocessing, extract the features, train the model, and deploy it to your board.

Once deployed, benchmarking code on the ESP32-S3 will display info about the runtime performance of the preprocessing steps and actual model via serial monitor (over USB cable).

Data Preprocessing and feature extraction can also be run separately to visualize some data examples:
```bash
python datamodule_tiny_ml.py
```

Compilation of the created src code (only works if model is trained and src is created by running `biodcase2026_tiny_ml.py` first):
```bash
python compile_embedded_src_code.py
```

Deployment of compiled code on the actual device (only works if code was compiled before):
```bash
python deploy_embedded_compiled_code.py
```

### Data Processing and Feature Extraction

The data processing pipeline follows these steps:
1. Raw audio files are read, preprocessed, and checked
2. Features are extracted according to configuration in `config.yaml`
```yaml
datamodule:
  feature_extraction:
    window_len: 4096
    window_stride: 512
    ...
```

### Model Training

The model training process is managed in `model_base.py` by the `ModelBase` class.
You can customize the model architecture in `model_tiny_ml.py` and overwrite any function you wish to change.
See also how the training is done in `biodcase2026_tiny_ml.py`
You can also simply create a new model file, but make sure that your model class inherits `ModelBase`.

### ESP32-S3 Deployment

To deploy your model to the ESP32-S3-Korvo-2 board, you'll use the built-in deployment tools that handle model conversion, code generation, and flashing. The deployment process:

1. Converts your trained pytorch model to TensorFlow Lite format optimized for the ESP32-S3
2. Packages your feature extraction configuration for embedded use
3. Generates C++ code that integrates with the ESP-IDF framework
4. Compiles the firmware using Docker-based ESP-IDF toolchain
5. Flashes the compiled firmware to your connected ESP32-S3-Korvo-2 board


The [ESP32-S3-Korvo-2](https://docs.espressif.com/projects/esp-adf/en/latest/design-guide/dev-boards/user-guide-esp32-s3-korvo-2.html) board features:
- ESP32-S3 dual-core processor
- Built-in microphone array
- Audio codec for high-quality audio processing
- Wi-Fi and Bluetooth connectivity
- [Software Support](https://components.espressif.com/components/espressif/esp32_s3_korvo_2/versions/4.1.2/readme)

and can be bought for instance [here](https://www.digikey.de/de/products/detail/espressif-systems/ESP32-S3-KORVO-2/15822448).

<div align="left">
  <picture>
    <img src="docs/BioDCASE_Tiny_Dev_Board.jpg" alt="ESP32-S3-Korvo-2 development board" width="400">
  </picture>
  <br><br>
</div>

### Code Structure

- `biodcase2026_tiny_ml.py` - Main execution pipeline
- `datamodule_tiny_ml.py` - Datamodule for preprocessing, feature extraction, and data loading
- `model_base.py` - Base model class: inherit this class for your own model and define its architecture, training, and more.
- `model_tiny_ml.py` - Define your model
- `biodcase_tiny/embedded/esp_target.py` - ESP target definition and configuration
- `biodcase_tiny/embedded/esp_toolchain.py` - ESP toolchain and for docker IDF
- `biodcase_tiny/embedded/firmware/main` - Firmware source code that will be copied and modified for the ESP target


### Development Tips

1. **Feature Extraction Parameters**: Carefully tune the feature extraction parameters in `config.yaml`.

2. **Model Size**: Keep your model compact. The ESP32-S3 has limited memory, so optimize your architecture accordingly.

3. **Profiling**: Use the profiling tools to identify bottlenecks in your implementation.

4. **Memory Management**: Be mindful of memory allocation on the ESP32. Monitor the allocations reported by the firmware.

5. **Docker Environment**: The toolchain uses Docker to provide a consistent ESP-IDF environment, making it easier to build on any host system.

## Evaluation Metrics

The BioDCASE-Tiny competition evaluates models based on multiple criteria:

### Classification Performance
- **Average precision**: the average value of precision across all recall levels from 0 to 1. 

### Resource Efficiency
- **Model Size**: Tflite model file size (KB)
- **Inference Time**: Average time required for single audio classification, including feature extraction (ms)
- **Peak Memory Usage**: Maximum RAM usage during inference (KB)

### Ranking
Participants will be ranked separately for each one of the evaluation criteria.


## Submission
t.b.d


## Limitations
This framework is still not perfect as we do not use real microphone data from the korvo-2 and merely run a profiler to check upon the model and feature extraction.
Therefore, we are always looking for interested collaborators to improve upon this project and create an even better challenge starting point for BioDCASE.

If you find issues in code or have problems in getting started, please create a github issue.


## License

This project is licensed under the Apache License 2.0 - see the license headers in individual files for details.


## Citation
t.b.d
<!-- If you use the BioDCASE-Tiny framework or dataset in your research, please cite the following:

### Framework Citation

```bibtex
@misc{biodcase_tiny_2026_repo,
  author = {tba},
  title = {tba},
  year = {2026},
  institution = {tba},
  type = {Software},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/birdnet-team/BioDCASE-Tiny-2026}}
}
```

### Dataset Citation

```bibtex
@dataset{biodcase_tiny_2026_dataset,
  author = {tba},
  title = {BioDCASE 2026 Task 3: Bioacoustics for Tiny Hardware Development Set},
  year = {2026},
  publisher = {Zenodo},
  doi = {tba},
  url = {tba}
}
``` -->


## Funding

<!-- Our work in the K. Lisa Yang Center for Conservation Bioacoustics is made possible by the generosity of K. Lisa Yang to advance innovative conservation technologies to inspire and inform the conservation of wildlife and habitats.

The development of BirdNET is supported by the German Federal Ministry of Research, Technology and Space (FKZ 01|S22072), the German Federal Ministry for the Environment, Climate Action, Nature Conservation and Nuclear Safety (FKZ 67KI31040E), the German Federal Ministry of Economic Affairs and Energy (FKZ 16KN095550), the Deutsche Bundesstiftung Umwelt (project 39263/01) and the European Social Fund. -->
t.b.d


## Partners

BirdNET is a joint effort of partners from academia and industry.
Without these partnerships, this project would not have been possible.
Thank you!

![Logos of all partners](https://tuc.cloud/index.php/s/KSdWfX5CnSRpRgQ/download/box_logos.png)
