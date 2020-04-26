#!/usr/bin/env python
# coding: utf-8

# # ETL Pipeline Preparation
# Follow the instructions below to help you create your ETL pipeline.
# ### 1. Import libraries and load datasets.
# - Import Python libraries
# - Load `messages.csv` into a dataframe and inspect the first few lines.
# - Load `categories.csv` into a dataframe and inspect the first few lines.

# In[105]:


# import libraries
import pandas as pd
from sqlalchemy import create_engine
import numpy as np


# In[106]:


# load messages dataset
messages = pd.read_csv('messages.csv')
messages.head()


# In[107]:


# load categories dataset
categories = pd.read_csv('categories.csv')
categories.head()


# ### 2. Merge datasets.
# - Merge the messages and categories datasets using the common id
# - Assign this combined dataset to `df`, which will be cleaned in the following steps

# In[108]:


# merge datasets
df = pd.merge(messages, categories, on='id')
df.head()


# ### 3. Split `categories` into separate category columns.
# - Split the values in the `categories` column on the `;` character so that each value becomes a separate column. You'll find [this method](https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.Series.str.split.html) very helpful! Make sure to set `expand=True`.
# - Use the first row of categories dataframe to create column names for the categories data.
# - Rename columns of `categories` with new column names.

# In[109]:


# create a dataframe of the 36 individual category columns
categories =df['categories'].str.split(';',expand=True)
categories.head()


# In[110]:


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


# In[111]:


# rename the columns of `categories`
categories.columns = category_colnames
categories.head()


# ### 4. Convert category values to just numbers 0 or 1.
# - Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). For example, `related-0` becomes `0`, `related-1` becomes `1`. Convert the string to a numeric value.
# - You can perform [normal string actions on Pandas Series](https://pandas.pydata.org/pandas-docs/stable/text.html#indexing-with-str), like indexing, by including `.str` after the Series. You may need to first convert the Series to be of type string, which you can do with `astype(str)`.

# In[112]:


cols = len(category_colnames)
rows = len(categories.index)
for i in range(rows):
    for j in range(cols):
        current = categories.iloc[i][j]
        current = int(current[-1])
        categories.iloc[i][j] = current
    
categories.head()


# ### 5. Replace `categories` column in `df` with new category columns.
# - Drop the categories column from the df dataframe since it is no longer needed.
# - Concatenate df and categories data frames.

# In[113]:


# drop the original categories column from `df`
df.drop(columns=['categories'],inplace=True)
df.head()


# In[114]:


# concatenate the original dataframe with the new `categories` dataframe
df = pd.concat([df, categories],axis=1)
df.head()


# ### 6. Remove duplicates.
# - Check how many duplicates are in this dataset.
# - Drop the duplicates.
# - Confirm duplicates were removed.

# In[119]:


# check number of duplicates
df_msg = df['message'].unique()
duplicates = rows - len(df_msg)
#print(len(df_msg))
print(f"There are {duplicates} duplicates on MESSAGE column")


# In[120]:


# drop duplicates
msg_df = df.drop_duplicates(subset=['message','genre'], keep='first')
msg_df.head()
#print(len(msg_df.index))


# In[121]:


# check number of duplicates
unique_original = msg_df['original'].unique()
duplicates = len(msg_df.index) - len(unique_original)
print(f"There are {duplicates} duplicates on ORIGINAL column")
#print(len(unique_original))


# In[122]:


original_df = msg_df.drop_duplicates(subset=['original','genre'], keep='first')
original_df.head()
#print(len(original_df.index))


# In[123]:


unique_dataset = original_df.drop_duplicates(subset=['id'], keep='first')
unique_dataset.head()
#print(len(unique_dataset.index))


# In[124]:


print(f"There are {len(unique_dataset.index)} unique records")


# ### 7. Save the clean dataset into an sqlite database.
# You can do this with pandas [`to_sql` method](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_sql.html) combined with the SQLAlchemy library. Remember to import SQLAlchemy's `create_engine` in the first cell of this notebook to use it below.

# In[100]:





# In[125]:


engine = create_engine('sqlite:///disaster_response.db')
df.to_sql('messages', engine, index=False)


# ### 8. Use this notebook to complete `etl_pipeline.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database based on new datasets specified by the user. Alternatively, you can complete `etl_pipeline.py` in the classroom on the `Project Workspace IDE` coming later.

# In[ ]:




