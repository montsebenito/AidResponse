# Disaster Response Pipeline Project

![homepage](/pics/homepage_top.jpg?raw=true)

### Project Overview:

Created a NPL-ML pipeline to categorize real messages that were sent during disaster events, so that can be routed to an appropriate disaster relief agency. This project includes a web app where an emergency worker can input a new message and get classification results in several categories. This project app also displays visualizations of the training data. 

##### Business Understanding
Following a disaster, when disaster response organizations have the least capacity to filter and then pull out the messages that are more important (often it really is only one of thousand messages that may be relevant to the disaster response professionals), it's very common to get millions and millions of communications, either direct or via social media.
Moreover, different organizations tipically take care of different parts of the problem: One organization will take care of water, another will take care of blocked roads, another will take care  of medical supplies.

##### Data Understanding
Data is based on real messages sent during disaster events. Each message is categorized into 36 categories, most of them binary with the exception of multiclass 'related' category.

![trainingdata](/pics/TrainingDataOverview_01.jpg?raw=true)

##### Data Preparation: ETL Pipeline
Built a data cleaning pipeline, process_data.py, that loads the messages and categories datasets, merges the two datasets, cleans the data and stores it in a SQLite database.

##### Data Modeling: ML Pipeline
Built a ML pipeline, train_classifier.py, that loads data from the SQLite database, splits the dataset into training and test sets, builds a text processing (custom tokenize function using nltk to case normalize, lemmatize, and tokenize text) as well as machine learning pipeline, trains and tunes a model using GridSearchCV, outputs results on the test set and exports the final model as a pickle file.

##### Results Evaluation: Flask Web App
Created Web app using Flask, html, css and Javascript, where an emergency worker can input a new message and get classification results in several categories and also displays visualizations of the data using Potly. 
![classifier](/pics/Classifier01.jpg?raw=true)

### Next Steps:
- Use an API to directly read the messages from social media.
- Most of the Disaster Categories are highly imbalance. For example, category Child_Alone does not have a single observation in the training dataset to learn from.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



### Code and Resources Used:
Python Version: 3.8.5
Packages: pandas, numpy, sklearn, plotly, nlkt , sqlalchemy


### Acknowledgements:
Thanks to Udacity for providing the data, guidelines as well as reviewing the project.


