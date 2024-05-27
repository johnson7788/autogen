#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/microsoft/autogen/blob/main/notebook/oai_client_cost.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Copyright (c) Microsoft Corporation. All rights reserved. 
# 
# Licensed under the MIT License.
# 
# # Usage tracking with AutoGen
# ## 1. Use AutoGen's OpenAIWrapper for cost estimation
# The `OpenAIWrapper` from `autogen` tracks token counts and costs of your API calls. Use the `create()` method to initiate requests and `print_usage_summary()` to retrieve a detailed usage report, including total cost and token usage for both cached and actual requests.
# 
# - `mode=["actual", "total"]` (default): print usage summary for non-caching completions and all completions (including cache).
# - `mode='actual'`: only print non-cached usage.
# - `mode='total'`: only print all usage (including cache).
# 
# Reset your session's usage data with `clear_usage_summary()` when needed.
# 
# ## 2. Track cost and token count for agents
# We also support cost estimation for agents. Use `Agent.print_usage_summary()` to print the cost summary for the agent.
# You can retrieve usage summary in a dict using `Agent.get_actual_usage()` and `Agent.get_total_usage()`. Note that `Agent.reset()` will also reset the usage summary.
# 
# To gather usage data for a list of agents, we provide an utility function `autogen.gather_usage_summary(agents)` where you pass in a list of agents and gather the usage summary.
# 
# ## Caution when using Azure OpenAI!
# If you are using azure OpenAI, the model returned from completion doesn't have the version information. The returned model is either 'gpt-35-turbo' or 'gpt-4'. From there, we are calculating the cost based on gpt-3.5-turbo-0125: (0.0005, 0.0015) per 1k prompt and completion tokens and gpt-4-0613: (0.03, 0.06). This means the cost can be wrong if you are using a different version from azure OpenAI.
# 
# This will be improved in the future. However, the token count summary is accurate. You can use the token count to calculate the cost yourself.
# 
# ## Requirements
# 
# AutoGen requires `Python>=3.8`:
# ```bash
# pip install "pyautogen"
# ```

# ## Set your API Endpoint
# 
# The [`config_list_from_json`](https://microsoft.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_json) function loads a list of configurations from an environment variable or a json file.
# 

# In[1]:


import autogen
from autogen import AssistantAgent, OpenAIWrapper, UserProxyAgent, gather_usage_summary

config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "tags": ["gpt-3.5-turbo", "gpt-3.5-turbo-16k"],  # comment out to get all
    },
)


# It first looks for environment variable "OAI_CONFIG_LIST" which needs to be a valid json string. If that variable is not found, it then looks for a json file named "OAI_CONFIG_LIST". It filters the configs by tags (you can filter by other keys as well).
# 
# The config list looks like the following:
# ```python
# config_list = [
#     {
#         "model": "gpt-3.5-turbo",
#         "api_key": "<your OpenAI API key>",
#         "tags": ["gpt-3.5-turbo"],
#     },  # OpenAI API endpoint for gpt-3.5-turbo
#     {
#         "model": "gpt-35-turbo-0613",  # 0613 or newer is needed to use functions
#         "base_url": "<your Azure OpenAI API base>", 
#         "api_type": "azure", 
#         "api_version": "2024-02-15-preview", # 2023-07-01-preview or newer is needed to use functions
#         "api_key": "<your Azure OpenAI API key>",
#         "tags": ["gpt-3.5-turbo", "0613"],
#     }
# ]
# ```
# 
# You can set the value of config_list in any way you prefer. Please refer to this [notebook](https://github.com/microsoft/autogen/blob/main/website/docs/topics/llm_configuration.ipynb) for full code examples of the different methods.

# ## OpenAIWrapper with cost estimation

# In[2]:


client = OpenAIWrapper(config_list=config_list)
messages = [
    {"role": "user", "content": "Can you give me 3 useful tips on learning Python? Keep it simple and short."},
]
response = client.create(messages=messages, cache_seed=None)
print(response.cost)


# ## Usage Summary for OpenAIWrapper
# 
# When creating a instance of OpenAIWrapper, cost of all completions from the same instance is recorded. You can call `print_usage_summary()` to checkout your usage summary. To clear up, use `clear_usage_summary()`.
# 

# In[3]:


client = OpenAIWrapper(config_list=config_list)
messages = [
    {"role": "user", "content": "Can you give me 3 useful tips on learning Python? Keep it simple and short."},
]
client.print_usage_summary()  # print usage summary


# In[4]:


# The first creation
# By default, cache_seed is set to 41 and enabled. If you don't want to use cache, set cache_seed to None.
response = client.create(messages=messages, cache_seed=41)
client.print_usage_summary()  # default to ["actual", "total"]
client.print_usage_summary(mode="actual")  # print actual usage summary
client.print_usage_summary(mode="total")  # print total usage summary


# In[5]:


# take out cost
print(client.actual_usage_summary)
print(client.total_usage_summary)


# In[6]:


# Since cache is enabled, the same completion will be returned from cache, which will not incur any actual cost.
# So actual cost doesn't change but total cost doubles.
response = client.create(messages=messages, cache_seed=41)
client.print_usage_summary()


# In[7]:


# clear usage summary
client.clear_usage_summary()
client.print_usage_summary()


# In[8]:


# all completions are returned from cache, so no actual cost incurred.
response = client.create(messages=messages, cache_seed=41)
client.print_usage_summary()


# ## Usage Summary for Agents
# 
# - `Agent.print_usage_summary()` will print the cost summary for the agent.
# - `Agent.get_actual_usage()` and `Agent.get_total_usage()` will return the usage summary in a dict. When an agent doesn't use LLM, they will return None.
# - `Agent.reset()` will reset the usage summary.
# - `autogen.gather_usage_summary` will gather the usage summary for a list of agents.

# In[9]:


assistant = AssistantAgent(
    "assistant",
    system_message="You are a helpful assistant.",
    llm_config={
        "timeout": 600,
        "cache_seed": None,
        "config_list": config_list,
    },
)

ai_user_proxy = UserProxyAgent(
    name="ai_user",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
    code_execution_config=False,
    llm_config={
        "config_list": config_list,
    },
    # In the system message the "user" always refers to the other agent.
    system_message="You ask a user for help. You check the answer from the user and provide feedback.",
)
assistant.reset()

math_problem = "$x^3=125$. What is x?"
ai_user_proxy.initiate_chat(
    assistant,
    message=math_problem,
)


# In[10]:


ai_user_proxy.print_usage_summary()
print()
assistant.print_usage_summary()


# In[11]:


user_proxy = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=2,
    code_execution_config=False,
    default_auto_reply="That's all. Thank you.",
)
user_proxy.print_usage_summary()


# In[12]:


print("Actual usage summary for assistant (excluding completion from cache):", assistant.get_actual_usage())
print("Total usage summary for assistant (including completion from cache):", assistant.get_total_usage())

print("Actual usage summary for ai_user_proxy:", ai_user_proxy.get_actual_usage())
print("Total usage summary for ai_user_proxy:", ai_user_proxy.get_total_usage())

print("Actual usage summary for user_proxy:", user_proxy.get_actual_usage())
print("Total usage summary for user_proxy:", user_proxy.get_total_usage())


# In[13]:


usage_summary = gather_usage_summary([assistant, ai_user_proxy, user_proxy])
usage_summary["usage_including_cached_inference"]

