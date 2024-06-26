#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/microsoft/autogen/blob/main/notebook/agentchat_langchain.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Auto Generated Agent Chat: Task Solving with Langchain Provided Tools as Functions
# 
# AutoGen offers conversable agents powered by LLM, tool, or human, which can be used to perform tasks collectively via automated chat. This framework allows tool use and human participants through multi-agent conversation. Please find documentation about this feature [here](https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat).
# 
# In this notebook, we demonstrate how to use `AssistantAgent` and `UserProxyAgent` to make function calls with the new feature of OpenAI models (in model version 0613) with a set of Langchain-provided tools and toolkits, to demonstrate how to leverage the 35+ tools available. 
# A specified prompt and function configs must be passed to `AssistantAgent` to initialize the agent. The corresponding functions must be passed to `UserProxyAgent`, which will execute any function calls made by `AssistantAgent`. Besides this requirement of matching descriptions with functions, we recommend checking the system message in the `AssistantAgent` to ensure the instructions align with the function call descriptions.
# 
# ## Requirements
# 
# AutoGen requires `Python>=3.8`. To run this notebook example, please install `pyautogen` and `Langchain`:
# ```bash
# pip install pyautogen Langchain
# ```

# In[ ]:


get_ipython().run_line_magic('pip', 'install "pyautogen>=0.2.3" Langchain')


# ## Set your API Endpoint
# 
# The [`config_list_from_models`](https://microsoft.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_models) function tries to create a list of configurations using Azure OpenAI endpoints and OpenAI endpoints for the provided list of models. It assumes the api keys and api bases are stored in the corresponding environment variables or local txt files:
# 
# - OpenAI API key: os.environ["OPENAI_API_KEY"] or `openai_api_key_file="key_openai.txt"`.
# - Azure OpenAI API key: os.environ["AZURE_OPENAI_API_KEY"] or `aoai_api_key_file="key_aoai.txt"`. Multiple keys can be stored, one per line.
# - Azure OpenAI API base: os.environ["AZURE_OPENAI_API_BASE"] or `aoai_api_base_file="base_aoai.txt"`. Multiple bases can be stored, one per line.
# 
# It's OK to have only the OpenAI API key, or only the Azure OpenAI API key + base.
# If you open this notebook in google colab, you can upload your files by clicking the file icon on the left panel and then choosing "upload file" icon.
# 
# The following code excludes Azure OpenAI endpoints from the config list because some endpoints don't support functions yet. Remove the `exclude` argument if they do.

# In[ ]:


import math
import os
from typing import Optional, Type

# Starndard Langchain example
from langchain.agents import create_spark_sql_agent
from langchain.agents.agent_toolkits import SparkSQLToolkit
from langchain.chat_models import ChatOpenAI

# Import things that are needed generically
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain.tools.file_management.read import ReadFileTool
from langchain.utilities.spark_sql import SparkSQL
from pyspark.sql import SparkSession

import autogen

config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
    },
)


# It first looks for environment variable "OAI_CONFIG_LIST" which needs to be a valid json string. If that variable is not found, it then looks for a json file named "OAI_CONFIG_LIST". It filters the configs by models (you can filter by other keys as well). Only the models with matching names are kept in the list based on the filter condition.
# 
# The config list looks like the following:
# ```python
# config_list = [
#     {
#         'model': 'gpt-4',
#         'api_key': '<your OpenAI API key here>',
#     },
#     {
#         'model': 'gpt-3.5-turbo',
#         'api_key': '<your Azure OpenAI API key here>',
#         'base_url': '<your Azure OpenAI API base here>',
#         'api_type': 'azure',
#         'api_version': '2024-02-15-preview',
#     },
#     {
#         'model': 'gpt-3.5-turbo-16k',
#         'api_key': '<your Azure OpenAI API key here>',
#         'base_url': '<your Azure OpenAI API base here>',
#         'api_type': 'azure',
#         'api_version': '2024-02-15-preview',
#     },
# ]
# ```
# 
# You can set the value of config_list in any way you prefer. Please refer to this [notebook](https://github.com/microsoft/autogen/blob/main/website/docs/topics/llm_configuration.ipynb) for full code examples of the different methods.

# ## Making Function Calls
# 
# In this example, we demonstrate function call execution with `AssistantAgent` and `UserProxyAgent`. With the default system prompt of `AssistantAgent`, we allow the LLM assistant to perform tasks with code, and the `UserProxyAgent` would extract code blocks from the LLM response and execute them. With the new "function_call" feature, we define functions and specify the description of the function in the OpenAI config for the `AssistantAgent`. Then we register the functions in `UserProxyAgent`.
# 

# In[ ]:


class CircumferenceToolInput(BaseModel):
    radius: float = Field()


class CircumferenceTool(BaseTool):
    name = "circumference_calculator"
    description = "Use this tool when you need to calculate a circumference using the radius of a circle"
    args_schema: Type[BaseModel] = CircumferenceToolInput

    def _run(self, radius: float):
        return float(radius) * 2.0 * math.pi


def get_file_path_of_example():
    # Get the current working directory
    current_dir = os.getcwd()

    # Go one directory up
    parent_dir = os.path.dirname(current_dir)

    # Move to the target directory
    target_folder = os.path.join(parent_dir, "test")

    # Construct the path to your target file
    file_path = os.path.join(target_folder, "test_files/radius.txt")

    return file_path


# In[ ]:


# Define a function to generate llm_config from a LangChain tool


def generate_llm_config(tool):
    # Define the function schema based on the tool's args_schema
    function_schema = {
        "name": tool.name.lower().replace(" ", "_"),
        "description": tool.description,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }

    if tool.args is not None:
        function_schema["parameters"]["properties"] = tool.args

    return function_schema


# Instantiate the ReadFileTool
read_file_tool = ReadFileTool()
custom_tool = CircumferenceTool()

# Construct the llm_config
llm_config = {
    # Generate functions config for the Tool
    "functions": [
        generate_llm_config(custom_tool),
        generate_llm_config(read_file_tool),
    ],
    "config_list": config_list,  # Assuming you have this defined elsewhere
    "timeout": 120,
}

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)

# Register the tool and start the conversation
user_proxy.register_function(
    function_map={
        custom_tool.name: custom_tool._run,
        read_file_tool.name: read_file_tool._run,
    }
)

chatbot = autogen.AssistantAgent(
    name="chatbot",
    system_message="For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
    llm_config=llm_config,
)

user_proxy.initiate_chat(
    chatbot,
    message=f"Read the file with the path {get_file_path_of_example()}, then calculate the circumference of a circle that has a radius of that files contents.",  # 7.81mm in the file
    llm_config=llm_config,
)


# # A PySpark Example

# In[ ]:


get_ipython().run_line_magic('pip', 'install pyspark')


# In[ ]:


spark = SparkSession.builder.getOrCreate()
schema = "langchain_example"
spark.sql(f"CREATE DATABASE IF NOT EXISTS {schema}")
spark.sql(f"USE {schema}")
csv_file_path = "./sample_data/california_housing_train.csv"
table = "california_housing_train"
spark.read.csv(csv_file_path, header=True, inferSchema=True).write.option(
    "path", "file:/content/spark-warehouse/langchain_example.db/california_housing_train"
).mode("overwrite").saveAsTable(table)
spark.table(table).show()


# In[ ]:


# Note, you can also connect to Spark via Spark connect. For example:
# db = SparkSQL.from_uri("sc://localhost:15002", schema=schema)
spark_sql = SparkSQL(schema=schema)
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
toolkit = SparkSQLToolkit(db=spark_sql, llm=llm)
agent_executor = create_spark_sql_agent(llm=llm, toolkit=toolkit, verbose=True)


# In[ ]:


# Starndard Langchain example
agent_executor.run("Describe the california_housing_train table")


# In[ ]:


# LangChain direct tool usage instead of toolkit example
# from langchain.tools.spark_sql.tool import (
#     InfoSparkSQLTool,
#     ListSparkSQLTool,
#     QueryCheckerTool,
#     QuerySparkSQLTool,
# )
# debug_toolkit = [
#   QuerySparkSQLTool(db=spark_sql),
#   InfoSparkSQLTool(db=spark_sql),
#   ListSparkSQLTool(db=spark_sql),
#   QueryCheckerTool(db=spark_sql, llm=llm),
# ]


# In[ ]:


# Now use AutoGen with Langchain Tool Bridgre
tools = []
function_map = {}

for tool in toolkit.get_tools():  # debug_toolkit if you want to use tools directly
    tool_schema = generate_llm_config(tool)
    print(tool_schema)
    tools.append(tool_schema)
    function_map[tool.name] = tool._run

# Construct the llm_config
llm_config = {
    "functions": tools,
    "config_list": config_list,  # Assuming you have this defined elsewhere
    "timeout": 120,
}

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)

print(function_map)

# Register the tool and start the conversation
user_proxy.register_function(function_map=function_map)

chatbot = autogen.AssistantAgent(
    name="chatbot",
    system_message="For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
    llm_config=llm_config,
)

user_proxy.initiate_chat(
    chatbot,
    message="Describe the table names california_housing_train",
    llm_config=llm_config,
)

