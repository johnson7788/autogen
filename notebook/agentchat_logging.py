#!/usr/bin/env python
# coding: utf-8

# # Runtime Logging with AutoGen 
# 
# AutoGen offers utilities to log data for debugging and performance analysis. This notebook demonstrates how to use them. 
# 
# we log data in different modes:
# - SQlite Database
# - File 
# 
# In general, users can initiate logging by calling `autogen.runtime_logging.start()` and stop logging by calling `autogen.runtime_logging.stop()`

# In[9]:


import json

import pandas as pd

import autogen
from autogen import AssistantAgent, UserProxyAgent

# Setup API key. Add your own API key to config file or environment variable
llm_config = {
    "config_list": autogen.config_list_from_json(
        env_or_file="OAI_CONFIG_LIST",
    ),
    "temperature": 0.9,
}

# Start logging
logging_session_id = autogen.runtime_logging.start(config={"dbname": "logs.db"})
print("Logging session ID: " + str(logging_session_id))

# Create an agent workflow and run it
assistant = AssistantAgent(name="assistant", llm_config=llm_config)
user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config=False,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
)

user_proxy.initiate_chat(
    assistant, message="What is the height of the Eiffel Tower? Only respond with the answer and terminate"
)
autogen.runtime_logging.stop()


# ## Getting Data from the SQLite Database 
# 
# `logs.db` should be generated, by default it's using SQLite database. You can view the data with GUI tool like `sqlitebrowser`, using SQLite command line shell or using python script:
# 
# 

# In[10]:


def get_log(dbname="logs.db", table="chat_completions"):
    import sqlite3

    con = sqlite3.connect(dbname)
    query = f"SELECT * from {table}"
    cursor = con.execute(query)
    rows = cursor.fetchall()
    column_names = [description[0] for description in cursor.description]
    data = [dict(zip(column_names, row)) for row in rows]
    con.close()
    return data


# In[11]:


def str_to_dict(s):
    return json.loads(s)


log_data = get_log()
log_data_df = pd.DataFrame(log_data)

log_data_df["total_tokens"] = log_data_df.apply(
    lambda row: str_to_dict(row["response"])["usage"]["total_tokens"], axis=1
)

log_data_df["request"] = log_data_df.apply(lambda row: str_to_dict(row["request"])["messages"][0]["content"], axis=1)

log_data_df["response"] = log_data_df.apply(
    lambda row: str_to_dict(row["response"])["choices"][0]["message"]["content"], axis=1
)

log_data_df


# ## Computing Cost 
# 
# One use case of logging data is to compute the cost of a session.

# In[12]:


# Sum totoal tokens for all sessions
total_tokens = log_data_df["total_tokens"].sum()

# Sum total cost for all sessions
total_cost = log_data_df["cost"].sum()

# Total tokens for specific session
session_tokens = log_data_df[log_data_df["session_id"] == logging_session_id]["total_tokens"].sum()
session_cost = log_data_df[log_data_df["session_id"] == logging_session_id]["cost"].sum()

print("Total tokens for all sessions: " + str(total_tokens) + ", total cost: " + str(round(total_cost, 4)))
print(
    "Total tokens for session "
    + str(logging_session_id)
    + ": "
    + str(session_tokens)
    + ", cost: "
    + str(round(session_cost, 4))
)


# ## Log data in File mode
# 
# By default, the log type is set to `sqlite` as shown above, but we introduced a new parameter for the `autogen.runtime_logging.start()`
# 
# the `logger_type = "file"` will start to log data in the File mode.

# In[4]:



import pandas as pd

import autogen
from autogen import AssistantAgent, UserProxyAgent

# Setup API key. Add your own API key to config file or environment variable
llm_config = {
    "config_list": autogen.config_list_from_json(
        env_or_file="OAI_CONFIG_LIST",
    ),
    "temperature": 0.9,
}

# Start logging with logger_type and the filename to log to
logging_session_id = autogen.runtime_logging.start(logger_type="file", config={"filename": "runtime.log"})
print("Logging session ID: " + str(logging_session_id))

# Create an agent workflow and run it
assistant = AssistantAgent(name="assistant", llm_config=llm_config)
user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config=False,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
)

user_proxy.initiate_chat(
    assistant, message="What is the height of the Eiffel Tower? Only respond with the answer and terminate"
)
autogen.runtime_logging.stop()


# This should create a `runtime.log` file in your current directory. 
