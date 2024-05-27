#!/usr/bin/env python
# coding: utf-8

# # Using RetrieveChat with Qdrant for Retrieve Augmented Code Generation and Question Answering
# 
# [Qdrant](https://qdrant.tech/) is a high-performance vector search engine/database.
# 
# This notebook demonstrates the usage of `QdrantRetrieveUserProxyAgent` for RAG, based on [agentchat_RetrieveChat.ipynb](https://colab.research.google.com/github/microsoft/autogen/blob/main/notebook/agentchat_RetrieveChat.ipynb).
# 
# 
# RetrieveChat is a conversational system for retrieve augmented code generation and question answering. In this notebook, we demonstrate how to utilize RetrieveChat to generate code and answer questions based on customized documentations that are not present in the LLM's training dataset. RetrieveChat uses the `RetrieveAssistantAgent` and `QdrantRetrieveUserProxyAgent`, which is similar to the usage of `AssistantAgent` and `UserProxyAgent` in other notebooks (e.g., [Automated Task Solving with Code Generation, Execution & Debugging](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_auto_feedback_from_code_execution.ipynb)).
# 
# We'll demonstrate usage of RetrieveChat with Qdrant for code generation and question answering w/ human feedback.
# 
# ````{=mdx}
# :::info Requirements
# Some extra dependencies are needed for this notebook, which can be installed via pip:
# 
# ```bash
# pip install "pyautogen[retrievechat-qdrant]" "flaml[automl]"
# ```
# 
# For more information, please refer to the [installation guide](/docs/installation/).
# :::
# ````

# In[7]:


get_ipython().run_line_magic('pip', 'install "pyautogen[retrievechat-qdrant]" "flaml[automl]"')


# ## Set your API Endpoint
# 
# The [`config_list_from_json`](https://microsoft.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_json) function loads a list of configurations from an environment variable or a json file.
# 

# In[8]:


from qdrant_client import QdrantClient

import autogen
from autogen.agentchat.contrib.qdrant_retrieve_user_proxy_agent import QdrantRetrieveUserProxyAgent
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent

# Accepted file formats for that can be stored in
# a vector database instance
from autogen.retrieve_utils import TEXT_FORMATS

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")

assert len(config_list) > 0
print("models to use: ", [config_list[i]["model"] for i in range(len(config_list))])


# ````{=mdx}
# :::tip
# Learn more about configuring LLMs for agents [here](/docs/topics/llm_configuration).
# :::
# ````

# In[9]:


print("Accepted file formats for `docs_path`:")
print(TEXT_FORMATS)


# ## Construct agents for RetrieveChat
# 
# We start by initializing the `RetrieveAssistantAgent` and `QdrantRetrieveUserProxyAgent`. The system message needs to be set to "You are a helpful assistant." for RetrieveAssistantAgent. The detailed instructions are given in the user message. Later we will use the `QdrantRetrieveUserProxyAgent.generate_init_prompt` to combine the instructions and a retrieval augmented generation task for an initial prompt to be sent to the LLM assistant.
# 
# ### You can find the list of all the embedding models supported by Qdrant [here](https://qdrant.github.io/fastembed/examples/Supported_Models/).

# In[10]:


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

# 2. create the QdrantRetrieveUserProxyAgent instance named "ragproxyagent"
# By default, the human_input_mode is "ALWAYS", which means the agent will ask for human input at every step. We set it to "NEVER" here.
# `docs_path` is the path to the docs directory. It can also be the path to a single file, or the url to a single file. By default,
# it is set to None, which works only if the collection is already created.
#
# Here we generated the documentations from FLAML's docstrings. Not needed if you just want to try this notebook but not to reproduce the
# outputs. Clone the FLAML (https://github.com/microsoft/FLAML) repo and navigate to its website folder. Pip install and run `pydoc-markdown`
# and it will generate folder `reference` under `website/docs`.
#
# `task` indicates the kind of task we're working on. In this example, it's a `code` task.
# `chunk_token_size` is the chunk token size for the retrieve chat. By default, it is set to `max_tokens * 0.6`, here we set it to 2000.
# We use an in-memory QdrantClient instance here. Not recommended for production.
# Get the installation instructions here: https://qdrant.tech/documentation/guides/installation/
ragproxyagent = QdrantRetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    retrieve_config={
        "task": "code",
        "docs_path": [
            "https://raw.githubusercontent.com/microsoft/flaml/main/README.md",
            "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Research.md",
        ],  # change this to your own path, such as https://raw.githubusercontent.com/microsoft/autogen/main/README.md
        "chunk_token_size": 2000,
        "model": config_list[0]["model"],
        "client": QdrantClient(":memory:"),
        "embedding_model": "BAAI/bge-small-en-v1.5",
    },
    code_execution_config=False,
)


# <a id="example-1"></a>
# ### Example 1
# 
# [back to top](#toc)
# 
# Use RetrieveChat to answer a question and ask for human-in-loop feedbacks.
# 
# Problem: Is there a function named `tune_automl` in FLAML?

# In[11]:


# reset the assistant. Always reset the assistant before starting a new conversation.
assistant.reset()

qa_problem = "Is there a function called tune_automl?"
ragproxyagent.initiate_chat(assistant, message=ragproxyagent.message_generator, problem=qa_problem)


# <a id="example-2"></a>
# ### Example 2
# 
# [back to top](#toc)
# 
# Use RetrieveChat to answer a question that is not related to code generation.
# 
# Problem: Who is the author of FLAML?

# In[12]:


# reset the assistant. Always reset the assistant before starting a new conversation.
assistant.reset()

qa_problem = "Who is the author of FLAML?"
ragproxyagent.initiate_chat(assistant, message=ragproxyagent.message_generator, problem=qa_problem)

