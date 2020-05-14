![Image description](https://github.com/kjanjua26/Octofying-COVID19-Literature/raw/master/video/logo.png)

# Octofying-COVID19-Literature

This repository contains the code for analyzing the semantics in published literature on COVID-19 in hopes of finding some relation.

## Working
![Output sample](https://github.com/kjanjua26/Octofying-COVID19-Literature/raw/master/video/out.gif)

## To Run
To run the code, clone this repository and install the following dependencies:
<ol>
  <li>sentence_transformers => ```pip install -U sentence-transformers``` </li>
</ol>
The dataset can be downloaded from Kaggle, the link is: <a href="https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge">DOWNLOAD</a>

Once the dataset and dependencies are installed, you need the pre-trained weights of BERT as well.
Download those from here <a href="https://drive.google.com/uc?id=1-PYF5y1hIpzwoXosNIpBP2L_Gyt5oUbv&export=download">WEIGHTS FILE</a>

Finally, type the following comand to run: ```python3 GUI_for_results.py``` file.

## Note
This repo builds on the initial work done by: <a href="https://www.kaggle.com/theamrzaki/covid-19-bert-researchpapers-semantic-search">CLICK HERE</a>
