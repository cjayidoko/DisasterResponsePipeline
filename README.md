# DisasterResponsePipeline

## Table of Contents
1. Installation
2. Project Motivation
3. File Descriptions
4. Results
5. Licensing, Authors, and Acknowledgements

## Installation: 
The codes .py and .ipynb files in this repository run on an anaconda installed Python environment. The .py files can run on terminal

## Project Motivation:
The motivation behind this project is develop a supervised predictive model that can classify text messages recievdd during a disaster into the different categories of disasters following the different categories seen in the dataset. Further, the project also include bulding an application that will apply the machine-learning model to any new text data and bring out clean visualization of the data. Natural Language Took-kits are used in the developemnt of the machine-learning model

## File Descriptions:
The dataset utilized for this study are obtained from Figure 8, and provided to Udacity, and thus not publicly available. The datasets include the messages.csv data nd the categries.csv. The message.csv data contains columns: message (raw message recieved during disaster), the message id, the genre (whether it is social media, news, or direct text messages), and the categories. The categories.csv contain the id and the different categories the message belongs to. Some of the categories include: medical related, search and rescue, request help, millitary, etc. 
 - The 'ETL Pipleline Preparation.ipynb' file is the python notebook used in cleaning the dataset.
- The 'ML Pipeline preparation.ipynb' is the python notebook used in building, training, testing, evaluating, and applying the machine learning models.
- The 'process_data2.py' file is the python script that applies all the works done on 'ETL Pipeline' and can take in the raw dataset and return a cleaned SQLite table.
- The 'train_model1.py' file will take in an SQLite table of the cleaned dataset and apply all the works done on ML Pipeline preparation while creating a machine learning model saved as a pickle file.
- The 'run.py' file will take the results of 'process_data2.py', and 'train_model1.py' and produce a web app that one can type in any message during disaster and get a category of the message. 
- These can run on the terminal. 

## Results: 
The findings of this study are in the repository and published on my medium [here](https://medium.com/@idokochijioke)
## Licensing, Authors, Acknowledgements
Many thanks to Figure-8 for making this available to Udacity for training purposes. Special thanks to udacity for the training. Feel free to utilize the contents of this while citing me, udacity, and/or figure-8 accordingly.
