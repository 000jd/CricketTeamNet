# CricketTeamNet# Cricket Team Selection Project

This project focuses on using machine learning to assist in the selection of a cricket team based on various features and attributes of players.

## Project Structure

The project is structured as follows:

- **data**: Contains the dataset used for training and testing the model.
  - *dataset.csv*: The dataset in CSV format.

- **models**: Contains the model classes used in this project.
  - *cricket_team_net.py*: Defines the neural network architecture.

- **notebooks**: Contains Jupyter notebooks used for data analysis and exploration.
  - *data_analysis.ipynb*: Notebook for analyzing the dataset.

- **scripts**: Contains scripts for data preprocessing, model training, and team generation.
  - *data_preprocessing.py*: Script for preprocessing the dataset.
  - *model_training.py*: Script for training the model.
  - *team_generation.py*: Script for generating the cricket team.

- **utils**: Contains utility functions used throughout the project.

- **main.py**: The main script to run the entire project.

- **README.md**: The README file providing an overview of the project.

## Usage

1. **Data Preprocessing**: Run `scripts/data_preprocessing.py` to preprocess the dataset and save the preprocessed data.

2. **Model Training**: Run `scripts/model_training.py` to train the neural network model.

3. **Team Generation**: Run `scripts/team_generation.py` to generate a cricket team using the trained model.

## Running the Code

Ensure you have all the necessary dependencies installed by running:

```bash
pip install -r requirements.txt
