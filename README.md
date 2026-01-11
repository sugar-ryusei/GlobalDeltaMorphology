# GlobalDeltaMorphology

*GlobalDeltaMorphology* introduces a novel method to quantify global delta morphologies using a convolutional autoencoder (CAE) model. This method is based on <a href="https://doi.org/10.1038/s43247-025-03144-w" target="_blank">Sato and Naruse (2025)</a>.

<div align="center">
<img src="https://raw.githubusercontent.com/sugar-ryusei/GlobalDeltaMorphology/main/figure/clustering.png" width="600">
</div>

This approach utilizes delta images sampled from the surface water distribution dataset. The CAE model effectively captures the morphological features in the delta images and compresses them into 70-dimensional feature vectors. The X-means clustering method can be applied to the feature vectors to assign morphotypes to the deltas.

Developed by <a href="https://orcid.org/0009-0008-3182-0980" target="_blank">Ryusei Sato</a> and <a href="https://orcid.org/0000-0003-3863-3404" target="_blank">Hajime Naruse</a> from Kyoto University, Japan.

## Usage

*GlobalDeltaMorphology* includes the standardized delta images as the training dataset for the CAE model.

If needed, use conda environment `delta_cae.yml` files are stored in `conda_environment` folder.

To analyse the global delta morphologies, run the Python files in `working_directory` folder in this order.

### 1. Acquire Morphometrics

<div align="center">
<img src="https://raw.githubusercontent.com/sugar-ryusei/GlobalDeltaMorphology/main/figure/cae_model.png" width="600">
</div>

In the architecture of the CAE model, the encoder reduces the data size and extracts the features through convolutional processing, and the decoder expands the data size and reproduces the similar image as the input delta image. The latent codes at the center of the network  are obtained as the compressed representation of the delta morphology.

The model is implemented using Python version 3.9 with TensorFlow version 2.8.2 and Keras version 2.8.0.

Before executing the following commands, extend `delta_image` in `working_directory`.

To build and train the CAE model, execute:

    python train.py

To acquire feature vectors for the delta images using the trained CNN model:

    python predict.py

### 2. Classify Delta Morphology
Using the extracted feature vectors, the X-means clustering method can categorize delta morphologies into the optimal number of classes based on Bayesian Information Criterion. The clustering result is visualized through the t-SNE method.

To classify and visualize the result:

    python analysis.py

## Citation
If *GlobalDeltaMorphology* contributes to your project, please cite:

    @article{Sato2025,
       author = {Ryusei Sato and Hajime Naruse},
       doi = {10.1038/s43247-025-03144-w},
       issn = {2662-4435},
       journal = {Communications Earth \& Environment},
       title = {Identifying controlling factors of delta morphology using a convolutional autoencoder},
       url = {https://doi.org/10.1038/s43247-025-03144-w},
       year = {2025}
    }
