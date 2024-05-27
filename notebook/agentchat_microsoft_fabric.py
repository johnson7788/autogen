#!/usr/bin/env python
# coding: utf-8

# # Use AutoGen in Microsoft Fabric
# 
# AutoGen offers conversable LLM agents, which can be used to solve various tasks with human or automatic feedback, including tasks that require using tools via code.
# Please find documentation about this feature [here](https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat).
# 
# [Microsoft Fabric](https://learn.microsoft.com/en-us/fabric/get-started/microsoft-fabric-overview) is an all-in-one analytics solution for enterprises that covers everything from data movement to data science, Real-Time Analytics, and business intelligence. It offers a comprehensive suite of services, including data lake, data engineering, and data integration, all in one place. Its pre-built AI models include GPT-x models such as `gpt-4-turbo`, `gpt-4`, `gpt-4-8k`, `gpt-4-32k`, `gpt-35-turbo`, `gpt-35-turbo-16k` and `gpt-35-turbo-instruct`, etc. It's important to note that the Azure Open AI service is not supported on trial SKUs and only paid SKUs (F64 or higher, or P1 or higher) are supported. Azure Open AI is being enabled in stages, with access for all users expected by March 2024.
# 
# In this notebook, we demonstrate how to use `AssistantAgent` and `UserProxyAgent` to write code and execute the code. Here `AssistantAgent` is an LLM-based agent that can write Python code (in a Python coding block) for a user to execute for a given task. `UserProxyAgent` is an agent which serves as a proxy for the human user to execute the code written by `AssistantAgent`, or automatically execute the code. Depending on the setting of `human_input_mode` and `max_consecutive_auto_reply`, the `UserProxyAgent` either solicits feedback from the human user or returns auto-feedback based on the result of code execution (success or failure and corresponding outputs) to `AssistantAgent`. `AssistantAgent` will debug the code and suggest new code if the result contains error. The two agents keep communicating to each other until the task is done.
# 
# ## Requirements
# 
# AutoGen requires `Python>=3.8`. To run this notebook example, please install:
# ```bash
# pip install "pyautogen"
# ```
# 
# Also, this notebook depends on Microsoft Fabric pre-built LLM endpoints. Running it elsewhere may encounter errors.

# ## AutoGen version < 0.2.0
# 
# For AutoGen version < 0.2.0, the Azure OpenAI endpoint is pre-configured.

# In[ ]:


get_ipython().run_line_magic('pip', 'install "pyautogen<0.2.0"')


# In[ ]:


from synapse.ml.mlflow import get_mlflow_env_config

import autogen

# Choose different models
config_list = [
    {
        "model": "gpt-4-turbo",
    },
]

# Set temperature, timeout and other LLM configurations
llm_config = {
    "config_list": config_list,
    "temperature": 0,
}


# In[ ]:


# create an AssistantAgent instance named "assistant"
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
)

# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",  # input() doesn't work, so needs to be "NEVER" here
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    },
    llm_config=llm_config,
    system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet.""",
)

# the assistant receives a message from the user, which contains the task description
user_proxy.initiate_chat(
    assistant,
    message="""
Who should read this paper: https://arxiv.org/abs/2308.08155
""",
)


# ## AutoGen version >= 0.2.0
# 
# For AutoGen version >= 0.2.0, we need to set up an API endpoint because the version of the openai-python package is different from the pre-configured version.

# In[ ]:


get_ipython().run_line_magic('pip', 'install "pyautogen>=0.2.0"')


# ## Set your API endpoint

# In[ ]:


mlflow_env_configs = get_mlflow_env_config()
access_token = mlflow_env_configs.driver_aad_token
prebuilt_AI_base_url = mlflow_env_configs.workload_endpoint + "cognitive/openai/"


# In[ ]:


config_list = [
    {
        "model": "gpt-4-turbo",
        "api_key": access_token,
        "base_url": prebuilt_AI_base_url,
        "api_type": "azure",
        "api_version": "2024-02-15-preview",
    },
]


# In[ ]:


# create an AssistantAgent named "assistant"
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={
        # "cache_seed": 42,  # seed for caching and reproducibility
        "config_list": config_list,  # a list of OpenAI API configurations
        # "temperature": 0,  # temperature for sampling
    },  # configuration for autogen's enhanced inference API which is compatible with OpenAI API
)
# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    },
)
# the assistant receives a message from the user_proxy, which contains the task description
user_proxy.initiate_chat(
    assistant,
    message="""What date is today? Compare the year-to-date gain for META and TESLA.""",
)


# In[ ]:




