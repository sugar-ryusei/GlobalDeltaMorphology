# GlobalDeltaMorphology

*GlobalDeltaMorphology* introduces a novel method to quantify global delta morphologies using a convolutional autoencoder (CAE) model.

<img src="https://raw.githubusercontent.com/sugar-ryusei/GlobalDeltaMorphology/main/figure/clustering.png" width="600">

This approach utilizes delta images sampled from the surface water distribution dataset. The CAE model effectively captures the morphological features in the delta images and compresses them into 70-dimensional feature vectors. The X-means clustering method can be applied to the feature vectors to assign morphotypes to the deltas.

Developed by <a href="https://orcid.org/0009-0008-3182-0980" target="_blank">Ryusei Sato</a> and <a href="https://orcid.org/0000-0003-3863-3404" target="_blank">Hajime Naruse</a> from Kyoto University, Japan.

## Usage

*GlobalDeltaMorphology* includes the standardized delta images as the training dataset for the CAE model.

If needed, use conda environment `delta_cae.yml` files are stored in `conda_environment` folder.

To analyse the global delta morphologies, run the Python files in `working_directory` folder in this order.

### 1. Acquire Feature Vectors

![](https://github.com/sugar-ryusei/GlobalDeltaMorphology/raw/main/figure/cae_model.png)

In the architecture of the CAE model, the encoder reduces the data size and extracts the features through convolutional processing, and the decoder expands the data size and reproduces the similar image as the input delta image. The latent codes at the center of the network  are obtained as the compressed representation of the delta morhpology.

The model is implemented using Python version 3.9 with TensorFlow version 2.8.2 and Keras version 2.8.0.

To build and train the CAE model, execute:

    python train.py

To acquire feature vectors for the delta images using the trained CNN model:

    python predict.py

### 2. Classify Delta Morphology
Using the extracted feature vectors, the X-means clustering method can categorize delta morphologies into the optimal number of classes based on Bayesian optimization. The clustering result is visualized through the t-SNE method.

To classify and visualize the result:

    python analyse.py

## Citation
If *GlobalDeltaMorphology* contributes to your project, please cite:

    @article{
    }
