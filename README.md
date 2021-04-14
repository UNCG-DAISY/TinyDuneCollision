# TinyDuneCollision

This repository outlines the development of a CNN model and the deployment of the (quantized) model on a microcontroller (with a camera) to detect when waves are hitting the dunes.

The key elements of this repository are the 2 jupyter notebooks describing the development of the CNN model, the tensorflow lite model (`.tflite` file), and 1 micropython script to load onto the microcontroller.

The data used to develop the model can be found on figshare ([Goldstein & Beuzen 2020](https://doi.org/10.6084/m9.figshare.12765494.v1)). The images were labeled with the [Coastal Image Labeler](https://github.com/UNCG-DAISY/Coastal-Image-Labeler).

## Explanation Notebooks and Code in the `/src` directory:

### TinyML_End_to_End-OpenMV.ipynb

This notebook builds the CNN model, trains the model, prunes the model, quantizes the final pruned model to 8 bits (i.e., post-training quantization) for use on a microcontroller (via TF Lite Micro)

### TinyML_Model_Explanation.ipynb

In this notebook we compare the custom CNN architecture to Mobilenet v1. We look at both models performance, number of ops, and the model size.

For the custom CNN model, we examine gradcam outputs from teh last layer.

We continue by pruning the custom CNN and quantizing the model to 8 bits, and then inspect the quantized outputs of the model. Finally, we look at the perfomance of the quantized model for this binary task (using a confusion amtrix) and also inspect the distribution of model outputs for the whole dataset.

### CollisionCam.py

This is the micropython code to be loaded onto the OpneMV H7+ board. Note that the `.tflite` model must also loaded onto the camera (you can find that here: `TinyDuneCollision/my-log-dir/saved_model/post_fullint_quantized.tflite`).

Use the OpenMV IDE to `Save script to OpenMV Cam (as main.py)` and this model will run when the OpenMV cam is powered.

