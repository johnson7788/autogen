#!/usr/bin/env python
# coding: utf-8

# ## OpenAI Assistants in AutoGen
# 
# This notebook shows a very basic example of the [`GPTAssistantAgent`](https://github.com/microsoft/autogen/blob/main/autogen/agentchat/contrib/gpt_assistant_agent.py#L16C43-L16C43), which is an experimental AutoGen agent class that leverages the [OpenAI Assistant API](https://platform.openai.com/docs/assistants/overview) for conversational capabilities,  working with
# `UserProxyAgent` in AutoGen.

# In[1]:


import logging
import os

from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

assistant_id = os.environ.get("ASSISTANT_ID", None)

config_list = config_list_from_json("OAI_CONFIG_LIST")
llm_config = {"config_list": config_list}

assistant_config = {"assistant_id": assistant_id}

gpt_assistant = GPTAssistantAgent(
    name="assistant",
    instructions=AssistantAgent.DEFAULT_SYSTEM_MESSAGE,
    llm_config=llm_config,
    assistant_config=assistant_config,
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
)
user_proxy.initiate_chat(gpt_assistant, message="Print hello world")


# In[2]:


user_proxy.initiate_chat(gpt_assistant, message="Write py code to eval 2 + 2", clear_history=True)


# In[3]:


gpt_assistant.delete_assistant()

