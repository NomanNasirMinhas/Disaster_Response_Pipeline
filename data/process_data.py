# import libraries
import pandas as pd
from sqlalchemy import create_engine
import numpy as np


def load_csv(messages, categories):
	'''
	Loads csv files into dataframes

	INPUT:
	messages = Name of the messages csv file
	categories = Name of the categories csv file

	OUTPUT:
	messages = Dataframe containing messages
	categories = Dataframe containing categories
	'''

	# load messages dataset
	messages = pd.read_csv(messages)
	messages.head()
	# load categories dataset
	categories = pd.read_csv(categories)
	categories.head()

	return messages, categories

messages , categories = load_csv('messages.csv', 'categories.csv')


# merge datasets
df = pd.merge(messages, categories, on='id')
df.head()


def transform_data():
	'''
	Performs cleaning and transformation of the data
	'''
	# create a dataframe of the 36 individual category columns
	categories =df['categories'].str.split(';',expand=True)
	categories.head()


	# select the first row of the categories dataframe
	row = categories.iloc[[0]]


	# use this row to extract a list of new column names for categories.
	# one way is to apply a lambda function that takes everything 
	# up to the second to last character of each string with slicing
	category_colnames =[]
	for col in row:
	    col_name=row[col][0]
	    category_colnames.append(col_name[:-2])
	print(category_colnames)
	#print(row)



	# rename the columns of `categories`
	categories.columns = category_colnames
	categories.head()


	#Convert category values to just numbers 0 or 1.
	cols = len(category_colnames)
	rows = len(categories.index)
	for i in range(rows):
	    for j in range(cols):
	        current = categories.iloc[i][j]
	        current = int(current[-1])
	        categories.iloc[i][j] = current
	    
	categories.head()


	#Replace `categories` column in `df` with new category columns.
	# drop the original categories column from `df`
	df.drop(columns=['categories'],inplace=True)
	df.head()


	# concatenate the original dataframe with the new `categories` dataframe
	df = pd.concat([df, categories],axis=1)
	df.head()


	#Remove duplicates.

	# check number of duplicates
	df_msg = df['message'].unique()
	duplicates = rows - len(df_msg)
	#print(len(df_msg))
	print(f"There are {duplicates} duplicates on MESSAGE column")

	# drop duplicates
	msg_df = df.drop_duplicates(subset=['message','genre'], keep='first')
	msg_df.head()
	#print(len(msg_df.index))


	# check number of duplicates
	unique_original = msg_df['original'].unique()
	duplicates = len(msg_df.index) - len(unique_original)
	print(f"There are {duplicates} duplicates on ORIGINAL column")
	#print(len(unique_original))

	#drop duplicates
	original_df = msg_df.drop_duplicates(subset=['original','genre'], keep='first')
	original_df.head()
	#print(len(original_df.index))
	unique_dataset = original_df.drop_duplicates(subset=['id'], keep='first')
	unique_dataset.head()
	#print(len(unique_dataset.index))


	print(f"There are {len(unique_dataset.index)} unique records")


#Save the clean dataset into an sqlite database.
def save_data(db_name):
	'''
	saves cleaned and tranformed data in a database

	INPUT: 
	db_name = Name of the database file to be saved as
	'''
	engine = create_engine(f'sqlite:///{db_name}')
	df.to_sql('messages', engine, index=False)