#!/usr/bin/env python
# coding: utf-8

# # Assistants with Azure Cognitive Search and Azure Identity
# 
# This notebook demonstrates the use of Assistant Agents in conjunction with Azure Cognitive Search and Azure Identity. Assistant Agents use tools that interact with Azure Cognitive Search to extract pertinent data.
# 

# ## Prerequisites
# 
# Before running this notebook, please ensure the following prerequisites are met:
#  
# 
# ### Dependencies
# 1. **Autogen**
# 2. **Azure SDK**
# 3. **Cognitive Search**/**AI Search**
# 
# If you have AI search enabled in your Azure Portal, you can use the following code to create an assistant agent that can search Azure Cognitive Search.
# 
# **AI search setup details:**
# - Documentation:   
#     - Create search service: https://learn.microsoft.com/en-us/azure/search/search-create-service-portal 
#     - Search index:  https://learn.microsoft.com/en-us/azure/search/search-how-to-create-search-index?tabs=portal 
#     hybrid search: https://learn.microsoft.com/en-us/azure/search/hybrid-search-how-to-query
# 
# - Youtube walkthrough: https://www.youtube.com/watch?v=6Zfuw-UJZ7k
# 
# 
# ### Install Azure CLI
# This notebook requires the Azure CLI for authentication purposes. Follow these steps to install and configure it:
# 
# 1. **Download and Install Azure CLI**:
#    - Visit the [Azure CLI installation page](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli) and follow the instructions for your operating system.
#    - Mac users can install Azure CLI using Homebrew with the command `brew install azure-cli`   
# 
# 2. **Verify Installation**:
#    - In the below cell execute `az --version` to check if Azure CLI is installed correctly.
# 
# 4. **Login to Azure**:
#    - In the below cell execute `az login` to log into your Azure account. This step is necessary as the notebook uses `AzureCliCredential` which retrieves the token based on the Azure account currently logged in.
# 
# ### Check Azure CLI Installation
# Run the cell below to check if Azure CLI is installed and properly configured on your system.

# ## Check Azure CLI Installation and Login Status

# In[ ]:


# Check Azure CLI installation and login status
# !az --version
# !az login


# ## Install required packages
# Run the cell below to install the required packages for this notebook.
# 
# 

# In[ ]:


get_ipython().system('pip3 install pyautogen==0.2.16')
get_ipython().system('pip3 install python-dotenv==1.0.1')
get_ipython().system('pip3 install pyautogen[graph]>=0.2.11')
get_ipython().system('pip3 install azure-search-documents==11.4.0b8')
get_ipython().system('pip3 install azure-identity==1.12.0')


# Next you will import the required packages for this notebook.
# 
# 

# In[ ]:


import json
import os

import requests
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv

import autogen
from autogen import AssistantAgent, UserProxyAgent, register_function
from autogen.cache import Cache

load_dotenv()

# Import Cognitive Search index ENV
AZURE_SEARCH_SERVICE = os.getenv("AZURE_SEARCH_SERVICE")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_API_VERSION = os.getenv("AZURE_SEARCH_API_VERSION")
AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG = os.getenv("AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG")
AZURE_SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")


# Next, you need to authenticate and create a `SearchClient` instance.

# In[ ]:


credential = DefaultAzureCredential()
endpoint = AZURE_SEARCH_SERVICE_ENDPOINT

from azure.identity import AzureCliCredential

credential = AzureCliCredential()
token = credential.get_token("https://cognitiveservices.azure.com/.default")

print("TOKEN", token.token)

client = SearchClient(endpoint=endpoint, index_name="test-index", credential=credential)


# 
# Then, load the configuration list and define the configuration for the `AssistantAgent`.

# In[13]:


config_list = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST",
)

gpt4_config = {
    "cache_seed": 42,
    "temperature": 0,
    "config_list": config_list,
    "timeout": 120,
}


# 
# Define your tool function `search` that will interact with the Azure Cognitive Search service.

# In[14]:


def search(query: str):
    payload = json.dumps(
        {
            "search": query,
            "vectorQueries": [{"kind": "text", "text": query, "k": 5, "fields": "vector"}],
            "queryType": "semantic",
            "semanticConfiguration": AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG,
            "captions": "extractive",
            "answers": "extractive|count-3",
            "queryLanguage": "en-US",
        }
    )

    response = list(client.search(payload))

    output = []
    for result in response:
        result.pop("titleVector")
        result.pop("contentVector")
        output.append(result)

    return output


# 
# Define the `AssistantAgent` and `UserProxyAgent` instances, and register the `search` function to them.

# In[ ]:


cog_search = AssistantAgent(
    name="COGSearch",
    system_message="You are a helpful AI assistant. "
    "You can help with Azure Cognitive Search."
    "Return 'TERMINATE' when the task is done.",
    llm_config=gpt4_config,
)

user_proxy = UserProxyAgent(
    name="User",
    llm_config=False,
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
)

register_function(
    search,
    caller=cog_search,
    executor=user_proxy,
    name="search",
    description="A tool for searching the Cognitive Search index",
)


# Finally, initiate a chat.

# In[16]:


if __name__ == "__main__":
    import asyncio

    async def main():
        with Cache.disk() as cache:
            await user_proxy.a_initiate_chat(
                cog_search,
                message="Search for 'What is Azure?' in the 'test-index' index",
                cache=cache,
            )

    await main()


# In[ ]:




