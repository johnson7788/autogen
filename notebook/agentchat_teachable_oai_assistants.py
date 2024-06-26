#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/microsoft/autogen/blob/main/notebook/agentchat_teachability.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Making OpenAI Assistants Teachable
# 
# Conversational assistants based on LLMs can remember the current chat with the user, and can even demonstrate in-context learning of things that the user teaches the assistant during the chat. But these memories and learnings are lost once the chat is over, or when a single chat grows too long for the LLM to handle effectively. In subsequent chats, the user is forced to repeat any necessary instructions over and over.
# 
# The optional agent capability called `Teachability` addresses these limitations by persisting user teachings across chat boundaries in long-term memory (a vector database). Memories (called memos) are created and saved to disk throughout a conversation, then loaded from disk later. Instead of copying all the memos into the context window, which would eat up valuable space, individual memos are retrieved into context only as needed. This allows the user to teach many facts, preferences and skills to the teachable agent just once, and have it remember them in later chats.
# 
# In making decisions about memo storage and retrieval, `Teachability` calls an instance of `TextAnalyzerAgent` to analyze pieces of text in several different ways. This adds extra LLM calls involving a relatively small number of tokens. These calls can add a few seconds to the time a user waits for a response.
# 
# This notebook demonstrates how `Teachability` can be added to instances of `GPTAssistantAgent`
# so that they can learn facts, preferences, and skills from users. As explained [here](https://microsoft.github.io/autogen/blog/2023/11/13/OAI-assistants), each instance of `GPTAssistantAgent` wraps an OpenAI Assistant that can be given a set of tools including functions, code interpreter, and retrieval. Assistants with these tools are demonstrated in separate standalone sections below, which can be run independently.
# 
# ## Requirements
# 
# AutoGen requires `Python>=3.8`. To run this notebook example, please install the [teachable] option.
# ```bash
# pip install "pyautogen[teachable]"
# ```

# In[ ]:


get_ipython().run_cell_magic('capture', '--no-stderr', '# %pip install "pyautogen[teachable]"')


# ## Set your API Endpoint
# 
# The [`config_list_from_json`](https://microsoft.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_json) function loads a list of configurations from an environment variable or a json file.

# In[1]:


import logging
import os

import requests

import autogen
from autogen import UserProxyAgent, config_list_from_json

# from autogen.agentchat import UserProxyAgent
from autogen.agentchat.contrib.capabilities.teachability import Teachability
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent

config_list = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST",
    file_location=".",
    filter_dict={
        "model": ["gpt-4", "gpt-4-1106-preview", "gpt4", "gpt-4-32k"],
    },
)

print(config_list[0]["model"])


# It first looks for environment variable "OAI_CONFIG_LIST" which needs to be a valid json string. If that variable is not found, it then looks for a json file named "OAI_CONFIG_LIST". It filters the configs by models (you can filter by other keys as well). After application of the filter shown above, only the gpt-4 models are considered.
# 
# The config list may look like the following:
# ```python
# config_list = [
#     {
#         'model': 'gpt-4-1106-preview',
#         'api_key': '<your OpenAI API key here>',
#     },
#     {
#         'model': 'gpt-4',
#         'api_key': '<your OpenAI API key here>',
#     },
#     {
#         'model': 'gpt-4',
#         'api_key': '<your Azure OpenAI API key here>',
#         'base_url': '<your Azure OpenAI API base here>',
#         'api_type': 'azure',
#         'api_version': '2024-02-15-preview',
#     },
#     {
#         'model': 'gpt-4-32k',
#         'api_key': '<your Azure OpenAI API key here>',
#         'base_url': '<your Azure OpenAI API base here>',
#         'api_type': 'azure',
#         'api_version': '2024-02-15-preview',
#     },
# ]
# ```
# 
# If you open this notebook in colab, you can upload your files by clicking the file icon on the left panel and then choose "upload file" icon.
# 
# You can set the value of config_list in other ways if you prefer, e.g., loading from a YAML file.

# ***

# ## Teachable OpenAI Assistant with Functions
# This section is based on [agentchat_oai_assistant_function_call.ipynb](https://github.com/microsoft/autogen/blob/teach_cap/notebook/agentchat_oai_assistant_function_call.ipynb), which demonstrates an assistant accomplishing a task through **function calling**.

# ### Function schema and implementation
# This example leverages OSS Insight (Open Source Software Insight) for advanced GitHub data analysis.

# In[2]:


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

ossinsight_api_schema = {
    "name": "ossinsight_data_api",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "Enter your GitHub data question in the form of a clear and specific question to ensure the returned data is accurate and valuable. "
                    "For optimal results, specify the desired format for the data table in your request."
                ),
            }
        },
        "required": ["question"],
    },
    "description": "This is an API endpoint allowing users (analysts) to input question about GitHub in text format to retrieve the realted and structured data.",
}


def get_ossinsight(question):
    """
    Retrieve the top 10 developers with the most followers on GitHub.
    """
    url = "https://api.ossinsight.io/explorer/answer"
    headers = {"Content-Type": "application/json"}
    data = {"question": question, "ignoreCache": True}

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        answer = response.json()
    else:
        return f"Request to {url} failed with status code: {response.status_code}"

    report_components = []
    report_components.append(f"Question: {answer['question']['title']}")
    if answer["query"]["sql"] != "":
        report_components.append(f"querySQL: {answer['query']['sql']}")

    if answer.get("result", None) is None or len(answer["result"]["rows"]) == 0:
        result = "Result: N/A"
    else:
        result = "Result:\n  " + "\n  ".join([str(row) for row in answer["result"]["rows"]])
    report_components.append(result)

    if answer.get("error", None) is not None:
        report_components.append(f"Error: {answer['error']}")
    return "\n\n".join(report_components) + "\n\n"


# ### Create the OpenAI Assistant with function calling as a tool

# In[3]:


assistant_id = os.environ.get("ASSISTANT_ID", None)
config_list = config_list_from_json("OAI_CONFIG_LIST")
llm_config = {
    "config_list": config_list,
    "assistant_id": assistant_id,
    "tools": [
        {
            "type": "function",
            "function": ossinsight_api_schema,
        }
    ],
}

oss_analyst = GPTAssistantAgent(
    name="OSS_Analyst",
    instructions=(
        "Hello, Open Source Project Analyst. You'll conduct comprehensive evaluations of open source projects or organizations on the GitHub platform, "
        "analyzing project trajectories, contributor engagements, open source trends, and other vital parameters. "
        "Please carefully read the context of the conversation to identify the current analysis question or problem that needs addressing."
    ),
    llm_config=llm_config,
    verbose=True,
)
oss_analyst.register_function(
    function_map={
        "ossinsight_data_api": get_ossinsight,
    }
)


# ### Give the assistant a task involving function calls

# In[4]:


user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
)

user_proxy.initiate_chat(oss_analyst, message="Top 10 developers with the most followers")


# ### Make the assistant teachable
# We can make any `ConversableAgent` teachable by adding a `Teachability` object to it.

# In[5]:


teachability = Teachability(reset_db=True, llm_config={"config_list": config_list})
teachability.add_to_agent(oss_analyst)


# ### Test the teachable assistant
# This time let's teach the assistant a more specific way of formatting lists.

# In[6]:


user_proxy.initiate_chat(
    oss_analyst,
    message="List the top 10 developers with the most followers. When listing things, please put them in a table.",
)


# Finally, let's clear the chat history to see whether the assistant can remember to use our preferred formatting in a new chat without having to be told again.

# In[7]:


user_proxy.initiate_chat(oss_analyst, message="List the top 10 developers with the most followers.", clear_history=True)


# ### Delete the teachable assistant
# All OpenAI Assistants can be created, viewed and deleted manually through OpenAI's [website](https://platform.openai.com/assistants). They can also be deleted through the `GPTAssistantAgent` instance.

# In[8]:


oss_analyst.delete_assistant()


# ***

# ## Teachable OpenAI Assistant with Code Interpreter
# This section is based on [agentchat_oai_code_interpreter.ipynb](https://github.com/microsoft/autogen/blob/teach_cap/notebook/agentchat_oai_code_interpreter.ipynb), which demonstrates an assistant accomplishing a task by **writing code and executing it** in a sandbox.

# ### Create the OpenAI Assistant with code interpreter as a tool

# In[9]:


# Initiate an agent equipped with code interpreter
coder_assistant = GPTAssistantAgent(
    name="Coder_Assistant",
    llm_config={
        "tools": [{"type": "code_interpreter"}],
        "config_list": config_list,
    },
    instructions="You are an expert at solving math questions. Write code and run it to solve math problems. Reply TERMINATE when the task is solved and there is no problem.",
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    },
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
)


# ### Give the assistant a task involving code execution

# In[10]:


user_proxy.initiate_chat(
    coder_assistant, message="If $725x + 727y = 1500$ and $729x+ 731y = 1508$, what is the value of $x - y$ ?"
)


# ### Make the assistant teachable
# We can make any `ConversableAgent` teachable by adding a `Teachability` object to it.

# In[11]:


teachability = Teachability(reset_db=True, llm_config={"config_list": config_list})
teachability.add_to_agent(coder_assistant)


# ### Test the teachable assistant
# This time let's teach the assistant to show its work.

# In[12]:


user_proxy.initiate_chat(
    coder_assistant,
    message="If $725x + 727y = 1500$ and $729x+ 731y = 1508$, what is the value of $x - y$ ? After finding the values of variables, always explain how to find the solution.",
)


# Finally, let's clear the chat history to see whether the assistant can remember to show its work in a new chat without having to be told again.

# In[13]:


user_proxy.initiate_chat(
    coder_assistant,
    message="If $725x + 727y = 1500$ and $729x+ 731y = 1508$, what is the value of $x - y$ ?",
    clear_history=True,
)


# ### Delete the teachable assistant
# All OpenAI Assistants can be created, viewed and deleted manually through OpenAI's [website](https://platform.openai.com/assistants). They can also be deleted through the `GPTAssistantAgent` instance.

# In[14]:


coder_assistant.delete_assistant()


# ***

# ## Teachable OpenAI Assistant with Retrieval
# This section is based on [agentchat_oai_assistant_retrieval.ipynb](https://github.com/microsoft/autogen/blob/teach_cap/notebook/agentchat_oai_assistant_retrieval.ipynb), which demonstrates an assistant accomplishing a task through **retrieval augmented generation (RAG)**.

# ### Create the OpenAI Assistant with retrieval as a tool
# For this example, first upload the [conversable_agent.py](https://github.com/microsoft/autogen/blob/main/autogen/agentchat/conversable_agent.py) file to your OpenAI API account. This can be done manually through the [website](https://platform.openai.com/assistants). Then find the uploaded File ID on the [Files page](https://platform.openai.com/files), and paste that ID into the `file_ids` list in the code below.

# In[15]:


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

assistant_id = os.environ.get("ASSISTANT_ID", None)

config_list = config_list_from_json("OAI_CONFIG_LIST")
llm_config = {
    "config_list": config_list,
    "assistant_id": assistant_id,
    "tools": [{"type": "retrieval"}],
    "file_ids": ["file-HPDOsp8k4dz95QRx9bnfNJHp"],
}

rag_assistant = GPTAssistantAgent(
    name="RAG_Assistant", instructions="You are adapt at question answering", llm_config=llm_config
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config=False,
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
    human_input_mode="ALWAYS",
)


# ### Give the assistant a task involving file retrieval
# When prompted, type "Does the file contain any bugs?". On the next prompt, type "exit" to end the chat.

# In[16]:


user_proxy.initiate_chat(rag_assistant, message="What is the name of the class of agents in the file I gave you?")


# ### Make the assistant teachable
# We can make any `ConversableAgent` teachable by adding a `Teachability` object to it.

# In[17]:


teachability = Teachability(reset_db=True, llm_config={"config_list": config_list})
teachability.add_to_agent(rag_assistant)


# ### Test the teachable assistant
# This time let's teach the assistant to report its confidence.

# In[18]:


user_proxy.initiate_chat(
    rag_assistant,
    message="What is the name of the class of agents in the file I gave you? When you answer a question based on a file, always report your confidence in the answer as a percentage.",
)


# Finally, let's clear the chat history to see whether the assistant can remember to report its confidence in a new chat without having to be told again.

# In[19]:


user_proxy.initiate_chat(
    rag_assistant, message="What is the name of the class of agents in the file I gave you?", clear_history=True
)


# ### Delete the teachable assistant
# All OpenAI Assistants can be created, viewed and deleted manually through OpenAI's [website](https://platform.openai.com/assistants). They can also be deleted through the `GPTAssistantAgent` instance.

# In[20]:


rag_assistant.delete_assistant()


# ***
