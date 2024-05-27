#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install "pyautogen>=0.2.3"')
get_ipython().system('pip install chromadb')
get_ipython().system('pip install sentence_transformers')
get_ipython().system('pip install tiktoken')
get_ipython().system('pip install pypdf')


# In[ ]:


import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import nest_asyncio

from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.agentchat.user_proxy_agent import UserProxyAgent


# In[ ]:


# Define an asynchronous function that simulates some asynchronous task (e.g., I/O operation)


async def my_asynchronous_function():
    print("Start asynchronous function")
    await asyncio.sleep(2)  # Simulate some asynchronous task (e.g., I/O operation)
    print("End asynchronous function")
    return "input"


# Define a custom class `CustomisedUserProxyAgent` that extends `UserProxyAgent`


class CustomisedUserProxyAgent(UserProxyAgent):
    # Asynchronous function to get human input
    async def a_get_human_input(self, prompt: str) -> str:
        # Call the asynchronous function to get user input asynchronously
        user_input = await my_asynchronous_function()

        return user_input

    # Asynchronous function to receive a message

    async def a_receive(
        self,
        message: Union[Dict, str],
        sender,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        # Call the superclass method to handle message reception asynchronously
        await super().a_receive(message, sender, request_reply, silent)


class CustomisedAssistantAgent(AssistantAgent):
    # Asynchronous function to get human input
    async def a_get_human_input(self, prompt: str) -> str:
        # Call the asynchronous function to get user input asynchronously
        user_input = await my_asynchronous_function()

        return user_input

    # Asynchronous function to receive a message
    async def a_receive(
        self,
        message: Union[Dict, str],
        sender,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        # Call the superclass method to handle message reception asynchronously
        await super().a_receive(message, sender, request_reply, silent)


# In[ ]:


def create_llm_config(model, temperature, seed):
    config_list = [
        {
            "model": "<model_name>",
            "api_key": "<api_key>",
        },
    ]

    llm_config = {
        "seed": int(seed),
        "config_list": config_list,
        "temperature": float(temperature),
    }

    return llm_config


# In[ ]:


get_ipython().system('pip install nest-asyncio')


# In[25]:


nest_asyncio.apply()


async def main():
    boss = CustomisedUserProxyAgent(
        name="boss",
        human_input_mode="ALWAYS",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    assistant = CustomisedAssistantAgent(
        name="assistant",
        system_message="You will provide some agenda, and I will create questions for an interview meeting. Every time when you generate question then you have to ask user for feedback and if user provides the feedback then you have to incorporate that feedback and generate new set of questions and if user don't want to update then terminate the process and exit",
        llm_config=create_llm_config("gpt-4", "0.4", "23"),
    )

    await boss.a_initiate_chat(
        assistant,
        message="Resume Review, Technical Skills Assessment, Project Discussion, Job Role Expectations, Closing Remarks.",
        n_results=3,
    )


await main()  # noqa: F704

