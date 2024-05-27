#!/usr/bin/env python
# coding: utf-8

# # Handling A Long Context via `TransformChatHistory`
# 
# <div class="alert alert-danger" role="alert">
#     <strong>Deprecation Notice:</strong> <code>TransformChatHistory</code> is no longer supported and will be removed in version <code>0.2.30</code>. Please transition to using <code>TransformMessages</code> as the new standard method. For a detailed introduction to this method, including how to limit the number of tokens in message context history to replace <code>TransformChatHistory</code>, visit our guide <a href="https://microsoft.github.io/autogen/docs/topics/handling_long_contexts/intro_to_transform_messages" target="_blank">Introduction to Transform Messages</a>.
# </div>
# 
# This notebook illustrates how you can use the `TransformChatHistory` capability to give any `Conversable` agent an ability to handle a long context. 
# 
# ````{=mdx}
# :::info Requirements
# Install `pyautogen`:
# ```bash
# pip install pyautogen
# ```
# 
# For more information, please refer to the [installation guide](/docs/installation/).
# :::
# ````

# In[2]:


import os

import autogen
from autogen.agentchat.contrib.capabilities import context_handling


# In[3]:


llm_config = {
    "config_list": [{"model": "gpt-3.5-turbo", "api_key": os.environ.get("OPENAI_API_KEY")}],
}


# ````{=mdx}
# :::tip
# Learn more about configuring LLMs for agents [here](/docs/topics/llm_configuration).
# :::
# ````
# 
# To add this ability to any agent, define the capability and then use `add_to_agent`.

# In[4]:


assistant = autogen.AssistantAgent(
    "assistant",
    llm_config=llm_config,
)


# Instantiate the capability to manage chat history
manage_chat_history = context_handling.TransformChatHistory(max_tokens_per_message=50, max_messages=10, max_tokens=1000)
# Add the capability to the assistant
manage_chat_history.add_to_agent(assistant)

user_proxy = autogen.UserProxyAgent(
    "user_proxy",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },
    max_consecutive_auto_reply=10,
)

user_proxy.initiate_chat(assistant, message="plot and save a graph of x^2 from -10 to 10")


# ## Why is this important?
# This capability is especially useful if you expect the agent histories to become exceptionally large and exceed the context length offered by your LLM.
# For example, in the example below, we will define two agents -- one without this ability and one with this ability.
# 
# The agent with this ability will be able to handle longer chat history without crashing.

# In[5]:


assistant_base = autogen.AssistantAgent(
    "assistant",
    llm_config=llm_config,
)

assistant_with_context_handling = autogen.AssistantAgent(
    "assistant",
    llm_config=llm_config,
)
# suppose this capability is not available
manage_chat_history = context_handling.TransformChatHistory(max_tokens_per_message=50, max_messages=10, max_tokens=1000)
manage_chat_history.add_to_agent(assistant_with_context_handling)

user_proxy = autogen.UserProxyAgent(
    "user_proxy",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },
    max_consecutive_auto_reply=2,
)

# suppose the chat history is large
# Create a very long chat history that is bound to cause a crash
# for gpt 3.5
long_history = []
for i in range(1000):
    # define a fake, very long message
    assitant_msg = {"role": "assistant", "content": "test " * 1000}
    user_msg = {"role": "user", "content": ""}

    assistant_base.send(assitant_msg, user_proxy, request_reply=False, silent=True)
    assistant_with_context_handling.send(assitant_msg, user_proxy, request_reply=False, silent=True)
    user_proxy.send(user_msg, assistant_base, request_reply=False, silent=True)
    user_proxy.send(user_msg, assistant_with_context_handling, request_reply=False, silent=True)

try:
    user_proxy.initiate_chat(assistant_base, message="plot and save a graph of x^2 from -10 to 10", clear_history=False)
except Exception as e:
    print("Encountered an error with the base assistant")
    print(e)
    print("\n\n")

try:
    user_proxy.initiate_chat(
        assistant_with_context_handling, message="plot and save a graph of x^2 from -10 to 10", clear_history=False
    )
except Exception as e:
    print(e)

