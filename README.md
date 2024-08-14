## Project Description
Flower-Image-Classifier is a deep learning project built with TensorFlow that classifies flowers into one of 102 categories. This project includes a command-line interface (CLI) written in Python that allows users to interact with the trained model for inference.

## Features
- Classifies flowers into 102 distinct categories.
- Displays the probability of the top predicted classes.
- Provides an option to map class labels to flower names using a JSON file.
- Command-line interface for easy interaction with the model.

## Installation Instructions
To get started with the Flower-Image-Classifier, you'll need to have Python installed along with several dependencies:

### Dependencies
- Python
- Jupyter Notebook (optional, for viewing the notebook locally)
- TensorFlow (version 2.14.0)
- Matplotlib
- PIL (Python Imaging Library)
You can install the required Python packages using the following command:

`pip install tensorflow==2.14.0 matplotlib pillow`

### Setting Up the Project
1. Clone this repository to your local machine:

`git clone https://github.com/mawada-sweis/Flower-Image-Classifier.git`

2. Navigate to the project directory:
   
`cd flower-image-classifier`

3. If you want to view the Jupyter Notebook locally, you can install Jupyter and open the notebook:

`pip install notebook`

`jupyter notebook`

Alternatively, you can view the notebook through the provided HTML file without needing Jupyter.

## Usage Instructions
To use the Flower-Image-Classifier, you need to have a trained Keras model and an image you want to classify.

### Basic Usage
`python predict.py /path/to/image saved_model`

### Options
- Return the top K most likely classes:

`python predict.py /path/to/image saved_model --top_k K`

- Map labels to flower names using a JSON file:

`python predict.py /path/to/image saved_model --category_names map.json`

### Example
Assuming you have an image `orchid.jpg` in a folder named `test_images/` and a Keras model saved as `my_model.h5`, here are some example commands:

- Basic Usage:

`python predict.py ./test_images/orchid.jpg my_model.h5`

- Return the top 3 most likely classes:

`python predict.py ./test_images/orchid.jpg my_model.h5 --top_k 3`

- Use a label_map.json file to map labels to flower names:

`python predict.py ./test_images/orchid.jpg my_model.h5 --category_names label_map.json`
