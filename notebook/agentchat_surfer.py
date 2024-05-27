#!/usr/bin/env python
# coding: utf-8

# # WebSurferAgent
# 
# AutoGen provides a proof-of-concept WebSurferAgent that can command a simple text-based browser (similar to [Lynx](https://en.wikipedia.org/wiki/Lynx_(web_browser))) to search the web, visit pages, navigate within pages, download files, etc. The browsing is stateful, meaning that browsing history, viewport state, and other details are maintained throughout the conversation. 
# 
# This work was largely inspired by OpenAI's [WebGPT](https://openai.com/research/webgpt) project from December 2021. 
# 
# ## Requirements
# 
# AutoGen requires `Python>=3.8`. To run this notebook example, please install AutoGen with the optional `websurfer` dependencies:
# ```bash
# pip install "pyautogen[websurfer]"
# ```

# In[1]:


# %pip install --quiet "pyautogen[websurfer]"


# ## Set your API Endpoint
# 
# The [`config_list_from_json`](https://microsoft.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_json) function loads a list of configurations from an environment variable or a json file.
# 
# It first looks for environment variable "OAI_CONFIG_LIST" which needs to be a valid json string. If that variable is not found, it then looks for a json file named "OAI_CONFIG_LIST". It filters the configs by models (you can filter by other keys as well).
# 
# The WebSurferAgent uses a combination of models. GPT-4 and GPT-3.5-turbo-16 are recommended.
# 
# Your json config should look something like the following:
# ```json
# [
#     {
#         "model": "gpt-4",
#         "api_key": "<your OpenAI API key here>"
#     },
#     {
#         "model": "gpt-3.5-turbo-16k",
#         "api_key": "<your OpenAI API key here>"
#     }
# ]
# ```
# 
# If you open this notebook in colab, you can upload your files by clicking the file icon on the left panel and then choose "upload file" icon.
# 

# In[2]:


import autogen  # noqa: E402

llm_config = {
    "timeout": 600,
    "cache_seed": 44,  # change the seed for different trials
    "config_list": autogen.config_list_from_json(
        "OAI_CONFIG_LIST",
        filter_dict={"model": ["gpt-4", "gpt-4-0613", "gpt-4-32k", "gpt-4-32k-0613", "gpt-4-1106-preview"]},
    ),
    "temperature": 0,
}

summarizer_llm_config = {
    "timeout": 600,
    "cache_seed": 44,  # change the seed for different trials
    "config_list": autogen.config_list_from_json(
        "OAI_CONFIG_LIST",
        filter_dict={"model": ["gpt-3.5-turbo-1106", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-16k"]},
    ),
    "temperature": 0,
}


# ## Configure Bing
# 
# For WebSurferAgent to be reasonably useful, it needs to be able to search the web -- and that means it needs a Bing API key. 
# You can read more about how to get an API on the [Bing Web Search API](https://www.microsoft.com/en-us/bing/apis/bing-web-search-api) page.
# 
# Once you have your key, either set it as the `BING_API_KEY` system environment variable, or simply input your key below.
# 

# In[3]:


import os  # noqa: E402

bing_api_key = os.environ["BING_API_KEY"]


# ### Construct Agents
# 
# We now create out WebSurferAgent, and a UserProxyAgent to surf the web. 

# In[4]:


from autogen.agentchat.contrib.web_surfer import WebSurferAgent  # noqa: E402

web_surfer = WebSurferAgent(
    "web_surfer",
    llm_config=llm_config,
    summarizer_llm_config=summarizer_llm_config,
    browser_config={"viewport_size": 4096, "bing_api_key": bing_api_key},
)

user_proxy = autogen.UserProxyAgent(
    "user_proxy",
    human_input_mode="NEVER",
    code_execution_config=False,
    default_auto_reply="",
    is_termination_msg=lambda x: True,
)


# ### Example 1: Search, summarize
# - Search for information about Microsoft AutoGen
# - Summarize the results
# - Visit the Getting Started Docs page

# In[5]:


task1 = """
Search the web for information about Microsoft AutoGen
"""

user_proxy.initiate_chat(web_surfer, message=task1)


# In[6]:


task2 = "Summarize these results"
user_proxy.initiate_chat(web_surfer, message=task2, clear_history=False)


# In[7]:


task3 = "Click the 'Getting Started' result"
user_proxy.initiate_chat(web_surfer, message=task3, clear_history=False)


# ### Example 2: Navigational search, scroll, answer questions
# - Search for Microsoft's wikipedia page, then naviagate to it
# - Scroll down
# - Answer questions about the content

# In[8]:


task4 = """Find Microsoft's Wikipedia page."""
user_proxy.initiate_chat(web_surfer, message=task4, clear_history=False)


# In[9]:


task5 = """Scroll down."""
user_proxy.initiate_chat(web_surfer, message=task5, clear_history=False)


# In[10]:


task6 = """Where was the first office location, and when did they move to Redmond?"""
user_proxy.initiate_chat(web_surfer, message=task6, clear_history=False)

