# SMILER Docker Model Skeleton

This directory is a template to aid in creation of new SMILER models. Copy these files over to `SMILER/models/docker/YOUR_MODEL_NAME`, and fill them in with details specific to your model.

The directory structure for a new model should look like this:

```
    SMILER
    ├── models
         ├── docker
            ├── YOUR_MODEL_NAME
                ├── model
                │   ├── run_model.py
                │   ├── *.prototxt
                │   ├── *.caffemodel
                │   ├── *.h5
                │   ├── etc.
                ├── smiler.json
                ├── other_libraries (e.g. custom caffe build)     
```

SMILER's Docker models rely on the following being present:

## 1. `smiler.json`

Holds metadata about the model.
Fill in the following information in `smiler.json`:

1. `name` - short name of the model (e.g. DGII)
2. `long name` - full name of the model  (e.g. DeepGaze II)
3. `version` - version within SMILER (e.g. 1.0.0)
4. `citation` - paper citation information (e.g. M. K{\"u}mmerer et al., "DeepGaze II: Reading fixations from deep features trained on object recognition", arXiv preprint arXiv:1610.01563, 2016.)
5. `model_type` - docker (for matlab models see `model_skeletons/MATLAB/`)
6. `model_files` - files required by the model (e.g. pretrained SVM, caffe weights, tensorflow checkpoint, etc.). These files should be placed in `docker/models/YOUR_MODEL_NAME/model/` directory as shown in the diagram above.
7. `docker_image` - name of the docker container (e.g. dgii).
8. `run_command` - replace `python` with `python3` if needed.
9. `shell_command` - replace `python` with `python3` if needed.
10. `parameters` - any extra parameters that the model requires.

To make sure that your ```smiler.json``` file is correct run ```./smiler info``` and see if your model appears in the list of models and no errors are generated.

## 2. `model/run_model.py`

A file that serves as the entry point for SMILER. It should perform model initialization, and call `smiler_tools.runner.run_model` with a single argument: a function that takes in a path to a single image (not directory), and returns a saliency map as a numpy array. SMILER will handle pre- and post-processing of the image. See examples in ```models/docker/```.

NOTE: if your saliency model returns saliency maps normalized between 0 and 1, multiply the final result by 255 since the image is converted to uint8 internally for post-processing.

If you wish to bypass SMILER's `smiler_tools` functions, you can modify the `run_command` parameter in `smiler.json` to be something else, but this is not recommended.

## 3. A Docker Image

In `smiler.json`, you will specify a Docker image this model will be run within. Some great choices are:

- SMILER's pre-built containers for other models: https://hub.docker.com/u/tsotsoslab
- Base CUDA containers: https://hub.docker.com/r/nvidia/cuda
- The official TensorFlow container: https://hub.docker.com/r/tensorflow/tensorflow
- The official caffe container: https://hub.docker.com/r/bvlc/caffe

### 3.1. Writing your own Dockerfile
If you write your own Dockerfile you should build an image locally.

```model_skeletons/docker/model/Dockerfile.example``` explains the steps that most saliency models would require with minor modifications. Also see dockerfiles for other models in ```SMILER/dockerfiles```.

First select the image for the model (typically tensorflow, caffe or cuda).

When building custom libraries (e.g. modified caffe or crf library for post-processing) add or remove dependencies from the ```Apt and pip dependencies``` step.

Comment out the step ```Build custom caffe model``` or replace it with any other custom library you need to build.

Do not modify the rest of the steps.

Once finished, name your dockerfile as ```Dockerfile.YOUR_MODEL_NAME``` and place it in ```SMILER/dockerfiles```.

To build the model run (from the SMILER root directory):

```
dockerfiles/build_model.sh YOUR_MODEL_NAME tag_name
```

where ```YOUR_MODEL_NAME``` is the same as `name` and `tag_name` is the same as `version` in `smiler.json`.


## 4. Running your model

If previous steps were successfull you may now use your model in the same way as the predefined SMILER models. See the main README for the available options.