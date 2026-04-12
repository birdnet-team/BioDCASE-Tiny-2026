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

**BioDCASE-Tiny 2026 competition (Task 3)** - A machine learning competition for bird sound recognition on tiny hardware, also visit the [official BioDCASE 2026 Task 3 website](https://biodcase.github.io/challenge2026/task3) for additional information.


## Background

BioDCASE-Tiny is a competition for developing efficient machine learning models for bird audio recognition that can run on resource-constrained embedded devices.
The project uses the ESP32-S3-Korvo-2 development board, which offers audio processing capabilities in a small form factor suitable for field deployment.
This year we added a pytorch framework along the tensorflow framework from last year offering participants to choose between one of the two machine learning frameworks. 


## Table of Contents
- [Dataset](#dataset)
- [Setup and Installation](#setup-and-installation)
- [Development](#development)
- [Rules and Submissions](#rules-and-submissions)
- [Limitations](#limitations)
- [Support](#support)
- [License](#license)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)


## Dataset

The dataset consists of 2750 audio field recordings of 3 s length from 10 bird species plus 1 background class from urban environments resulting in **11 classes** in total.
The dataset is available for download on [Zenodo](https://zenodo.org/records/19453065) where additional details are provided as well.


## Setup and Installation

### Prerequisites

1. Python >=3.11 and <=3.13 with pip and venv
2. [Docker](https://www.docker.com/get-started/) (runs ESP-IDF in a container) **OR** locally installed [ESP-IDF](https://github.com/espressif/esp-idf)  
3. (optional) ESP32-S3-Korvo-2 development board and two USB cables (power and serial connection)

> [!IMPORTANT]
> You can also participate in the challenge if you do not want to buy a Korvo-2 dev board, but consider that you will not be able to check if your model is actually deployable on the korvo system.
> If the deployable model (`.tflite`) does not exist or isn't able to run on our system, you will be ranked lower in the competition.
> We still recommend to buy the development board to get the full embedded feeling of this task.

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/birdnet-team/BioDCASE-Tiny-2026.git
cd BioDCASE-Tiny-2026
```

2. Install your favorite python version (as long as it is able to build the requirements, see prerequisities for working version).

3. Create a virtual environment (recommended)

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

4. Install Python dependencies depending on the framework you want to use (pytorch or tensorflow):
```bash
pip install -r requirements_<pytorch|tensorflow>.txt
```

5. Install [Docker](https://www.docker.com/get-started/) on your system. Afterwards you have to activate the docker deamon and add it to your user group. On linux (depends on your distribution) this could be achieved with:
```bash
systemctl status docker.socket
systemctl enable docker.socket
sudo gpasswd -a <your_username> docker
```
To test if your Docker runs you can do:
```bash
docker run hello-world
```

6. (alternatively) install the [ESP-IDF](https://github.com/espressif/esp-idf) framework on your local PC and compile the code using `idf.py` commands. You have to have a look into the code of the repository and figure out which docker commands are used for building and deploying the code.

### Configuration

1. Move the downloaded dataset to any location on your PC, for instance, `/path/to/your/downloaded/dataset/` and edit `config.yaml` file at following location (the path can also be relative to the path you start the script):
```yaml
datamodule:
  dataset:
    root_path: /path/to/your/downloaded/dataset/
```
you can also change the intermediate and cache (feature) paths at:
```yaml
datamodule:
  intermediate:
    root_path: ./output/01_intermediate/
  caching:
    root_path: ./output/02_features/
```

2. (optional) Make sure to add rights to your usb device connecting to the korvo board. This depends a bit on the system, on linux, you have to add udev rules in `/etc/udev/rules.d/`, e.g. add a file  there `/etc/udev/rules.d/10-custom-usb.rules` with content:
```
# ubuntu vs. arch: GROUP="dialout" vs. GROUP="uucp"
# rule for esp32 (devkit-c or korvo)
KERNEL=="ttyUSB0", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", GROUP="uucp", MODE="0666"
```
and add to your group ("uucp" for arch, "dialout" for ubuntu), then reload and restart your PC:
```
sudo gpasswd -a <your_username> uucp
sudo udevadm control --reload-rules
```

3. (optional) Set your serial device port in the `config.yaml` for linux it is usually `/dev/ttyUSB0`

```yaml
generate_embedded_code:
  serial_device: <YOUR_DEVICE> 
```

### Running on Windows

We recommend using Windows Subsystem for Linux (WSL) to run this project.
To make your device accessible for WSL you can use this [Guide](https://learn.microsoft.com/en-us/windows/wsl/connect-usb).
To determine your serial device port you can use the following command:

```bash
dmesg | grep tty
```

You might also need to grant rights to run the deployment:

```bash
sudo adduser $USER dialout
sudo chmod a+rw $SERIAL_PORT
```


## Development

- Modify `config.yaml` to change feature extraction or framework model parameters
- Modify corresponding model files in either `pipeline_pytorch/` or `pipeline_tensorflow/` framework folders.

> [!IMPORTANT]
> Writing custom features rather than using the implemented ones, requires implementing a numerically equivalent version on the embedded target too!
> This is a necessary condition for the evaluated model to behave identically on the host and on the embedded target likewise.
> Note, that this is a non-trivial undertaking and we generally advice to stick to already implemented feature extraction in this repository!

### Quickstart

To run the complete pipeline on the pytorch framework execute:
```bash
python biodcase2026_tiny_ml_pytorch.py
```
and for the tensorflow framework:
```bash
python biodcase2026_tiny_ml_tensorflow.py
```

This will run the data preprocessing, extraction of features, training of the model, and deployment on your development board.
Once deployed, the benchmarking code on the ESP32-S3 will display information about the runtime performance of the preprocessing steps and the deployed model via serial monitor (over USB cable).

### Code Structure

- `biodcase2026_tiny_ml__<pytorch|tensorflow>.py` - Main execution pipeline
- `datamodule.py` - Datamodule for preprocessing, feature extraction, and data storing
- `pipeline_pytorch/` - pytorch framework folder
- `pipeline_tensorflow/` - tensorflow framework folder
- `biodcase_tiny/embedded/firmware/main` - Firmware source code that will be copied and modified for the ESP target
- `biodcase_tiny/embedded/esp_target.py` - ESP build target creation 
- `biodcase_tiny/embedded/esp_toolchain.py` - ESP toolchain with Docker IDF to build, flash, and monitor

### Data Processing and Feature Extraction

Both data processing and feature extraction is handled by `datamodule.py` and follows following steps:
1. Raw audio files are read, checked, and stored.
2. Features are extracted by `feature_handler.py` according to the configuration set in `config.yaml` where both `feature_extraction` and `feature_handler_add_kwargs` are passed to the `FeatureHandler` class. The config to adapt is:
```yaml
datamodule:
  feature_extraction:
    window_len: 4096
    window_stride: 512
    ...
  feature_handler_add_kwargs:
    target_sample_rate: 24000
    transpose_features_extracted: True
    ...

```
3. The preprocessed data (intermediate) and features (caching) are stored and reloaded once they exist. 
Therefore, if you change the feature extraction parameters you either have to delete the cached folders or set the `redo` flags to `True` in
```yaml
datamodule:
  redo_all: False
  redo_intermediate: False
  redo_cache: False
```
or you change the ids or root path in
```yaml
datamodule:
    intermediate:
      root_path: './output/01_intermediate'
      intermediate_id: 'intermediate0'
      ...
    caching:
      root_path: './output/02_features'
      cache_id: 'cache0'
      ...
```

If you want to run only the data preprocessing and feature extraction with a visualization of data samples (and audio playback) execute:
```bash
python datamodule.py
```

### Model Training

The model training is started either by `biodcase2026_tiny_ml_pytorch.py` or `biodcase2026_tiny_ml_tensorflow.py` depending on your framework of choice.

#### Pytorch Framework
When choosing the pytorch framework, the model depending files are located in `./pipeline_pytorch`.
The model training process is managed in `model_training.py` where the training/validation steps are defined in `model_base.py` by the `ModelBase` class.
You can customize the model architecture in `model_tiny_ml.py` and overwrite any function you wish to change.
You can also create a new model file, but make sure that your model class inherits `ModelBase` which is required for the submission process.

#### Tensorflow Framework
When choosing the pytorch framework, the model depending files are located in `./pipeline_tensorflow`.
The model training process is managed in `model_training.py` where [Keras](https://keras.io/) model creation and training step is defined in `model.py`.
This framework was adapted from previous year's [BioDCASE 2025 competition](https://github.com/birdnet-team/BioDCASE-Tiny-2025).

### ESP32-S3 Build and Deployment

<!-- To deploy your model to the ESP32-S3-Korvo-2 board, you'll use the built-in deployment tools that handle model conversion, code generation, and flashing. 
The deployment process: -->

Our model build and deployment procedure to the ESP32-S3-Korvo-2 board follows following three main steps:
1. Build target creation
2. Compilation (Docker or ESP-IDF)
3. Deployment and Monitoring (Docker or ESP-IDF)

In the build target creation step, the firmware in `./biodcase_tiny/embedded/firmware/main/` is copied and processed with [Jinja](https://jinja.palletsprojects.com/en/stable/) where the serialized `.tflite` model (stored by the pipeline in e.g. `./output/03_models/pytorch`) and feature extraction parameters are inserted.
The build target is stored in `./output/05_generated_embedded_code/src/`.

Once the build target is ready you can run
```bash
python compile_embedded_src_code.py
```
to manually compile the target.
Afterwards you can deploy the compiled code on the actual device with:
```bash
python deploy_embedded_compiled_code.py
```

Note: If you do not want to use docker, have a look into `embedded_code_generation.py` and  `biodcase_tiny/embedded/esp_toolchain.py` where you find the corresponding `idf.py` commands for building (build), deploying (flash), and monitoring (monitor). We still recommend to use docker, because we also store the monitor output to e.g. `./output/04_reports/pytorch/monitor_report.yaml`.

#### Technical Details

The [ESP32-S3-Korvo-2](https://docs.espressif.com/projects/esp-adf/en/latest/design-guide/dev-boards/user-guide-esp32-s3-korvo-2.html) board features:
- ESP32-S3 dual-core processor
- Built-in microphone array
- Audio codec for high-quality audio processing
- Wi-Fi and Bluetooth connectivity
- [Software Support](https://components.espressif.com/components/espressif/esp32_s3_korvo_2/versions/4.1.2/readme)

and can be bought, for instance, [here](https://www.digikey.de/de/products/detail/espressif-systems/ESP32-S3-KORVO-2/15822448).

<div align="left">
  <picture>
    <img src="docs/BioDCASE_Tiny_Dev_Board.jpg" alt="ESP32-S3-Korvo-2 development board" width="400">
  </picture>
  <br><br>
</div>


### Development Tips

1. **Feature Extraction Parameters**: Carefully tune the feature extraction parameters in `config.yaml`.

2. **Model Size**: Keep your model compact. The ESP32-S3 has limited memory, so optimize your architecture accordingly.

3. **Profiling**: Use the profiling tools to identify bottlenecks in your implementation.

4. **Memory Management**: Be mindful of memory allocation on the ESP32. Monitor the allocations reported by the firmware.

5. **Docker Environment**: The toolchain uses Docker to provide a consistent ESP-IDF environment, making it easier to build on any host system.


## Rules and Submissions
> <font color="red">**Note:**</font> <br>
> We are still updating the submission rules to make it easy for you to submit your work and easy for us to evaluate your submission!
> The final submission rules are to be expected by end of April, so please stay up to date on this repository!

Generally, you will have to submit a .zip file that contains:
1. Inference model
2. (optional) .tflite model
3. (alternative) Feature extraction algorithm that does not follow the baseline algorithm
4. A YAML metadata file describing details of your submission
5. Technical report

Your solution will be evaluated on a hidden test set and the scores will be presented in the upcoming results section of the BioDCASE website.

### Evaluation Metrics

The BioDCASE-Tiny competition evaluates models based on multiple criteria on **classification performance** and **resource efficiency**:

- **Average precision**: the average value of precision across all recall levels from 0 to 1. 
- **Model Size**: `.tflite` model file size (KB)
- **Inference Time**: Average time required for single audio classification, including feature extraction (ms)
- **Peak Memory Usage**: Maximum RAM usage during inference (KB)

### Ranking
Participants will be ranked separately for each one of the evaluation metrics.


## Limitations
This framework does not yet use real microphone data from the korvo-2!
Instead, it runs a profiler to evaluate the model and feature extraction in size and time consumption.
Therefore, we are looking for interested collaborators to improve this project (especially on the embedded side) and create an even better challenge starting point for future editions of BioDCASE.


## Support

If you find errors in code or if you have problems in getting started, please create a **GitHub issue** within this repository.
We are happy to get any feedback to improve this repository!


## License

This project is licensed under the Apache License 2.0 - see the license headers in individual files for details.


## Citation

If you use the BioDCASE-Tiny framework or dataset in your research, please cite the following:

### Framework Citation

```bibtex
@misc{biodcase_tiny_2026_repo,
  author = {Walter, Christian and Benhamadi, Yasmine and Seidel, Tom and Carmantini, Giovanni and Kahl, Stefan},
  title = {BioDCASE-Tiny 2026: A Framework for Bird Species Recognition on Resource-Constrained Hardware},
  year = {2026},
  type = {Software},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/birdnet-team/BioDCASE-Tiny-2026}},
}
```

### Dataset Citation

```bibtex
@dataset{biodcase_tiny_2026_dataset,
  author = {Kahl, Stefan, and Martin, Ralph},
  title = {BioDCASE 2026 Task 3: Bioacoustics for Tiny Hardware Development Set},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.19453065},
  url = {https://doi.org/10.5281/zenodo.19453065}
}
```


## Acknowledgement
- C.W. was supported by the University of Veterinary Medicine, Vienna.
- Y.B. was supported by the EU MSCA Doctoral Network Bioacoustic AI (BioacAI, 101071532).
- T.S. and S.K. were supported by Chemnitz University of Technology

