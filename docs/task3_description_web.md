Title: Bioacoustics for Tiny Hardware
Date: 2025-02-11 14:51
Category: challenge
Slug: challenge2026/task3
Author: DCASE 
HeaderTextSecondary: Task 3 Description
HeaderImage: images/header/veriko-dundua-BPKTTohN8wU-unsplash.jpg
HeaderImagecc: Background photo by Veriko Dundua
HeaderIcon:  
bpersonnel: True
bpersonnel_source: content/data/challenge2026/personnel.yaml
bpersonnel_set: task3
bpersonnel_header: Coordinators
bpersonnel_fields: affiliation.title, affiliation.url, affiliation_title, affiliation_url, email, photo
bpersonnel_panel_color: panel-default
btoc: True
btoc_panel_color: panel-default
brepository_source: content/data/challenge2026/repository.yaml
bsponsors_source:

# About
The goal of the **Bioacoustics on Tiny Hardware** task is to develop an automatic classifier of birdsong that complies with the resource constraints of low-cost and battery-powered autonomous recording units.

> <font color="red">**!!!Attention!!!**</font> <br>
> We are still updating this content, so please stay up to date!

# Description 

The next generation of autonomous recording units contains programmable chips, thus offering the opportunity the opportunity to perform BioDCASE tasks. On-device processing has multiple advantages, such as high durability, low latency, and privacy preservation. However, such “tiny hardware” is limited in terms of memory and compute, which calls for the development of original methods in audio content analysis.

In this context, task participants will revisit the well-known problem of automatic detection of birdsong while adjusting their systems to meet the specifications of a commercially available microcontroller. The challenge focuses on detecting the vocalizations of 10 different bird species using the ESP32-S3-Korvo-2 development board.

<figure>
    <div class="row row-centered">
        <div class="col-xs-10 col-md-8 col-centered">
            <img src="{static}/images/tasks/challenge2025/task3_dev-board.jpg" class="img img-responsive">
            <figcaption>Photograph of the ESP32-S3-Korvo-2, the "tiny hardware" of BioDCASE task 3.</figcaption>
        </div>
    </div>
</figure>
<br>



The primary challenge is striking the optimal balance between classification accuracy and resource usage. While conventional deep learning approaches might achieve high accuracy, they may not be feasible on embedded hardware. Participants must explore techniques such efficient architecture design, and optimized feature extraction to create a solution that performs well within the hardware constraints.

A baseline implementation is provided as a starting point, which participants can modify and improve upon. Solutions will be evaluated based on classification performance, model size, inference time, and memory usage.

**Note:** We encourage participants to buy the ESP32-S3-Korvo-2 development board to test their solutions. The board is available for purchase on various online platforms (e.g, at [DigiKey](https://www.digikey.de/de/products/detail/espressif-systems/ESP32-S3-KORVO-2/15822448)). However, the competition can be completed without the board, as the evaluation will be performed on a hidden test set using the baseline system by the organizers.

# Dataset

The dataset for this year's task uses field recordings from 10 bird species and an additional set of urban backround sounds from Germany:

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

The dataset is available for download on [Zenodo](https://zenodo.org/records/19453065)

<div class="brepository-item" data-item="biodcase2026-task3"></div>

 
# Evaluation and Baseline System

We provide a baseline system that includes a complete pipeline for audio processing, feature extraction, model training, and deployment to the ESP32-S3-Korvo-2 development board. The baseline system is designed to be easily adaptable for participants to build upon.

You can find the baseline system in the GitHub repository: [**BioDCASE-Tiny 2026 Baseline System**](https://github.com/birdnet-team/BioDCASE-Tiny-2026)

Submissions will be evaluated based on both classification performance and resource efficiency:

### Classification Performance
- **Average precision**: The average value of precision across all recall levels from 0 to 1.
 
### Resource Efficiency (measured on the ESP32-S3-Korvo-2 development board)
- **Model Size**: TFLite model file size (KB)
- **Inference Time**: Average time required for single audio classification, including feature extraction (ms)
- **Peak Memory Usage**: Maximum RAM usage during inference (KB)

### Ranking
Participants will be ranked separately for each one of the evaluation criteria.

A baseline system is provided in the GitHub repository, including a complete pipeline for audio processing, feature extraction, model training, and deployment to the ESP32-S3-Korvo-2 development board.

# External Data Resources

The competition focuses on the provided 10-bird species dataset. External data use may be regulated according to the [official competition rules](https://biodcase.github.io/challenge2026/rules)

# Rules and Submissions
Please follow the "Rules and Submission" guidlines described in the [baseline repository](https://github.com/birdnet-team/BioDCASE-Tiny-2026).
Generally, you will have to submit a .zip file that contains:
1. Inference model
2. (optional) .tflite model
3. (alternative) Feature extraction algorithm that does not follow the baseline algorithm
4. A YAML metadata file describing details of your submission
5. Technical report

Your solution will be evaluated on a hidden test set and the scores will be presented in the upcoming results section of the biodcase website.

<!-- > <font color="red">**Note:**</font> <br>
> The submission rules are still in development but will be available in time, so stay up to date! -->

<!-- 1. Solutions must be deployable on the ESP32-S3-Korvo-2 development board
2. Models must process audio from the onboard microphone array in real-time
3. Submissions must include .h5 model files, pipeline_config.yaml and monitor_report.yaml in a zip file
4. Solutions will be evaluated on a hidden test set using the baseline system
5. Participants must cite the competition framework and dataset in any publications

A complete YAML metadata file with submission details should be provided. Here is an example <code>Lastname_task3_1.meta.yaml</code>:

<div class="panel-group" id="metadata-accordion" role="tablist" aria-multiselectable="true">
    <div class="panel panel-default">
        <div class="panel-heading" role="tab" id="task3-example-header">
            <h4 class="panel-title">
                <a class="collapsed accordion-toggle" role="button" data-toggle="collapse"
                   data-parent="#metadata-accordion" href="#task3-example-collapse"
                   aria-expanded="true" aria-controls="collapseOne">               
                   Metadata
                </a>
            </h4>
        </div>
        <div id="task3-example-collapse" class="panel-collapse collapse" role="tabpanel" aria-labelledby="task3-example-header">
            <div class="panel-body" style="padding: 0px">
                <pre class="font110" style="padding:0;border:0;border-radius:0;"><code class="yaml">{include::content/data/challenge2026/Lastname_task3_1.meta.yaml}</code></pre>
            </div>
        </div>
    </div>
</div> -->

# Citation

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

# Support

If you have questions, please use the [BioDCASE Google Groups community forum](https://groups.google.com/g/biodcase-community) or create an issue in the Github baseline repo: [BioDCASE-Tiny 2026 Baseline System](https://github.com/birdnet-team/BioDCASE-Tiny-2026).