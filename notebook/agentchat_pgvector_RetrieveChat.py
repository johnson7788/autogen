#!/usr/bin/env python
# coding: utf-8

# # Using RetrieveChat Powered by PGVector for Retrieve Augmented Code Generation and Question Answering
# 
# AutoGen offers conversable agents powered by LLM, tool or human, which can be used to perform tasks collectively via automated chat. This framework allows tool use and human participation through multi-agent conversation.
# Please find documentation about this feature [here](https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat).
# 
# RetrieveChat is a conversational system for retrieval-augmented code generation and question answering. In this notebook, we demonstrate how to utilize RetrieveChat to generate code and answer questions based on customized documentations that are not present in the LLM's training dataset. RetrieveChat uses the `RetrieveAssistantAgent` and `RetrieveUserProxyAgent`, which is similar to the usage of `AssistantAgent` and `UserProxyAgent` in other notebooks (e.g., [Automated Task Solving with Code Generation, Execution & Debugging](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_auto_feedback_from_code_execution.ipynb)). Essentially, `RetrieveAssistantAgent` and  `RetrieveUserProxyAgent` implement a different auto-reply mechanism corresponding to the RetrieveChat prompts.
# 
# ## Table of Contents
# We'll demonstrate six examples of using RetrieveChat for code generation and question answering:
# 
# - [Example 1: Generate code based off docstrings w/o human feedback](#example-1)
# - [Example 2: Answer a question based off docstrings w/o human feedback](#example-2)
# 
# 
# ````{=mdx}
# :::info Requirements
# Some extra dependencies are needed for this notebook, which can be installed via pip:
# 
# ```bash
# pip install pyautogen[retrievechat-pgvector] flaml[automl]
# ```
# 
# For more information, please refer to the [installation guide](/docs/installation/).
# :::
# ````
# 
# Ensure you have a PGVector instance. 
# 
# If not, a test version can quickly be deployed using Docker.
# 
# `docker-compose.yml`
# ```yml
# version: '3.9'
# 
# services:
#   db:
#     hostname: db
#     image: ankane/pgvector
#     ports:
#       - 5432:5432
#     restart: always
#     environment:
#       - POSTGRES_DB=postgres
#       - POSTGRES_USER=postgres
#       - POSTGRES_PASSWORD=postgres
#       - POSTGRES_HOST_AUTH_METHOD=trust
#     volumes:
#       - ./init.sql:/docker-entrypoint-initdb.d/init.sql
# ```
# 
# Create `init.sql` file
# ```SQL
# CREATE EXTENSION IF NOT EXISTS vector;
# ```
# 

# ## Set your API Endpoint
# 
# The [`config_list_from_json`](https://microsoft.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_json) function loads a list of configurations from an environment variable or a json file.
# 

# In[9]:


import json
import os

import chromadb

import autogen
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

# Accepted file formats for that can be stored in
# a vector database instance
from autogen.retrieve_utils import TEXT_FORMATS

config_list = [
    {
        "model": "Meta-Llama-3-8B-Instruct-imatrix",
        "api_key": "YOUR_API_KEY",
        "base_url": "http://localhost:8080/v1",
        "api_type": "openai",
    },
    {"model": "gpt-3.5-turbo-0125", "api_key": "YOUR_API_KEY", "api_type": "openai"},
    {
        "model": "gpt-35-turbo",
        "base_url": "...",
        "api_type": "azure",
        "api_version": "2023-07-01-preview",
        "api_key": "...",
    },
]

assert len(config_list) > 0
print("models to use: ", [config_list[i]["model"] for i in range(len(config_list))])


# ````{=mdx}
# :::tip
# Learn more about configuring LLMs for agents [here](/docs/topics/llm_configuration).
# :::
# ````
# 
# ## Construct agents for RetrieveChat
# 
# We start by initializing the `RetrieveAssistantAgent` and `RetrieveUserProxyAgent`. The system message needs to be set to "You are a helpful assistant." for RetrieveAssistantAgent. The detailed instructions are given in the user message. Later we will use the `RetrieveUserProxyAgent.message_generator` to combine the instructions and a retrieval augmented generation task for an initial prompt to be sent to the LLM assistant.

# In[5]:


print("Accepted file formats for `docs_path`:")
print(TEXT_FORMATS)


# In[6]:


# 1. create an RetrieveAssistantAgent instance named "assistant"
assistant = RetrieveAssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
    },
)

# 2. create the RetrieveUserProxyAgent instance named "ragproxyagent"
# By default, the human_input_mode is "ALWAYS", which means the agent will ask for human input at every step. We set it to "NEVER" here.
# `docs_path` is the path to the docs directory. It can also be the path to a single file, or the url to a single file. By default,
# it is set to None, which works only if the collection is already created.
# `task` indicates the kind of task we're working on. In this example, it's a `code` task.
# `chunk_token_size` is the chunk token size for the retrieve chat. By default, it is set to `max_tokens * 0.6`, here we set it to 2000.
# `custom_text_types` is a list of file types to be processed. Default is `autogen.retrieve_utils.TEXT_FORMATS`.
# This only applies to files under the directories in `docs_path`. Explicitly included files and urls will be chunked regardless of their types.
# In this example, we set it to ["non-existent-type"] to only process markdown files. Since no "non-existent-type" files are included in the `websit/docs`,
# no files there will be processed. However, the explicitly included urls will still be processed.
ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
    retrieve_config={
        "task": "code",
        "docs_path": [
            "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Examples/Integrate%20-%20Spark.md",
            "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Research.md",
            os.path.join(os.path.abspath(""), "..", "website", "docs"),
        ],
        "custom_text_types": ["non-existent-type"],
        "chunk_token_size": 2000,
        "model": config_list[0]["model"],
        "vector_db": "pgvector",  # PGVector database
        "collection_name": "flaml_collection",
        "db_config": {
            "connection_string": "postgresql://postgres:postgres@localhost:5432/postgres",  # Optional - connect to an external vector database
            # "host": postgres, # Optional vector database host
            # "port": 5432, # Optional vector database port
            # "database": postgres, # Optional vector database name
            # "username": postgres, # Optional vector database username
            # "password": postgres, # Optional vector database password
            "model_name": "all-MiniLM-L6-v2",  # Sentence embedding model from https://huggingface.co/models?library=sentence-transformers or https://www.sbert.net/docs/pretrained_models.html
        },
        "get_or_create": True,  # set to False if you don't want to reuse an existing collection
        "overwrite": False,  # set to True if you want to overwrite an existing collection
    },
    code_execution_config=False,  # set to False if you don't want to execute the code
)


# ### Example 1
# 
# [Back to top](#table-of-contents)
# 
# Use RetrieveChat to help generate sample code and automatically run the code and fix errors if there is any.
# 
# Problem: Which API should I use if I want to use FLAML for a classification task and I want to train the model in 30 seconds. Use spark to parallel the training. Force cancel jobs if time limit is reached.

# In[7]:


# reset the assistant. Always reset the assistant before starting a new conversation.
assistant.reset()

# given a problem, we use the ragproxyagent to generate a prompt to be sent to the assistant as the initial message.
# the assistant receives the message and generates a response. The response will be sent back to the ragproxyagent for processing.
# The conversation continues until the termination condition is met, in RetrieveChat, the termination condition when no human-in-loop is no code block detected.
# With human-in-loop, the conversation will continue until the user says "exit".
code_problem = "How can I use FLAML to perform a classification task and use spark to do parallel training. Train for 30 seconds and force cancel jobs if time limit is reached."
chat_result = ragproxyagent.initiate_chat(
    assistant, message=ragproxyagent.message_generator, problem=code_problem, search_string="spark"
)


# ### Example 2
# 
# [Back to top](#table-of-contents)
# 
# Use RetrieveChat to answer a question that is not related to code generation.
# 
# Problem: Who is the author of FLAML?

# In[8]:


# reset the assistant. Always reset the assistant before starting a new conversation.
assistant.reset()

qa_problem = "Who is the author of FLAML?"
chat_result = ragproxyagent.initiate_chat(assistant, message=ragproxyagent.message_generator, problem=qa_problem)

