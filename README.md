### Music Genre Classification Using Deep Learning
## Member: Nguyen Tran
## Description: 
Implement a deep learning model to classify music based on genre. Currently we have 10 genres: Disco, Blues, Metal, Pop, Country, Hip-Hop, Classical, Rock, Reggae-Dub, Jazz. My ultimate goals is to have a music generating app. 

## Data Source
- gtzan dataset [https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification] exclude the file 'jazz.00054.wav'
- Free Music Archive large [https://github.com/mdeff/fma?tab=readme-ov-file]
I do not recommend running through the label file since I have included a copy of the data in the repository. 

### Prerequisites
I used Ubuntu 18.04 and Python 3.9.19 for this project, lower version might not be capable. 

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/ntran4869/audio-classification-sp24.git
   ```
2. Set up conda environment
   ```sh
   conda create -f environmeny.yml
   ```
2. Or necessary library
   ```sh
   pip install -r requirement.txt
   ```

## Usage

- run dataload.ipynb to generate pickle files
- run model.ipynb to generate the model (mymodel.keras)
- run eval.ipynb to see accuracy score and obtain the confusion matrix
- if you run label.ipynb, make sure to download the fma_large data (90+GB)
- run demo to test your own stuffs!

## UPCOMING
- Music Generation App
