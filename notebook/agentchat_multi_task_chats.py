#!/usr/bin/env python
# coding: utf-8

# # Solving Multiple Tasks in a Sequence of Chats
# 
# This notebook showcases how to use the new chat interface of conversational agents in AutoGen: initiate_chats, to conduct a series of tasks. This new interface allows one to pass multiple tasks and their corresponding dedicated agents. Once initiate_chats is invoked, the tasks will be solved sequentially, with the summaries from previous tasks provided to subsequent tasks as context, if the `summary_method` argument is specified.
# 
# \:\:\:info Requirements
# 
# Install `pyautogen`:
# ```bash
# pip install pyautogen
# ```
# 
# For more information, please refer to the [installation guide](/docs/installation/).
# 
# \:\:\:

# In[ ]:


import autogen

config_list = autogen.config_list_from_json(env_or_file="OAI_CONFIG_LIST")
llm_config = {"config_list": config_list}


# \:\:\:tip
# 
# Learn more about the various ways to configure LLM endpoints [here](/docs/topics/llm_configuration).
# 
# \:\:\:

# ### Example Tasks
# Below are three example tasks, with each task being a string of text describing the request. The completion of later tasks requires or benefits from the results of previous tasks.
# 

# In[1]:


financial_tasks = [
    """What are the current stock prices of NVDA and TESLA, and how is the performance over the past month in terms of percentage change?""",
    """Investigate possible reasons of the stock performance.""",
]

writing_tasks = ["""Develop an engaging blog post using any information provided."""]


# ### Scenario 1: Solve the tasks with a series of chats
# 
# The `initiate_chats` interface can take a list of dictionaries as inputs. Each dictionary preserves the following fields: 
# - `message`: is a string of text (typically a message containing the task);
# - `recipient`: a conversable agent dedicated for the task;
# - `summary_method`: A string specifying the method to get a summary from the chat. Currently supported choices include `last_msg`, which takes the last message from the chat history as the summary, and `reflection_with_llm`, which uses an LLM call to reflect on the chat history and summarize a takeaway;
# - `summary_prompt`: A string specifying how to instruct an LLM-backed agent (either the recipient or the sender in the chat) to reflect on the chat history and derive a summary. If not otherwise specified, a default prompt will be used when `summary_method` is `reflection_with_llm`.
# "Summarize the takeaway from the conversation. Do not add any introductory phrases. If the intended request is NOT properly addressed, please point it out."
# - `carryover`: A string or a list of string to specify additional context to be used in the chat. With `initiate_chats`, summary from previous chats will be added as carryover. They will be appended after the carryover provided by the user.

# In[2]:


financial_assistant = autogen.AssistantAgent(
    name="Financial_assistant",
    llm_config=llm_config,
)
research_assistant = autogen.AssistantAgent(
    name="Researcher",
    llm_config=llm_config,
)
writer = autogen.AssistantAgent(
    name="writer",
    llm_config=llm_config,
    system_message="""
        You are a professional writer, known for
        your insightful and engaging articles.
        You transform complex concepts into compelling narratives.
        Reply "TERMINATE" in the end when everything is done.
        """,
)

user = autogen.UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "tasks",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)

chat_results = user.initiate_chats(
    [
        {
            "recipient": financial_assistant,
            "message": financial_tasks[0],
            "clear_history": True,
            "silent": False,
            "summary_method": "last_msg",
        },
        {
            "recipient": research_assistant,
            "message": financial_tasks[1],
            "summary_method": "reflection_with_llm",
        },
        {
            "recipient": writer,
            "message": writing_tasks[0],
            "carryover": "I want to include a figure or a table of data in the blogpost.",
        },
    ]
)


# #### Check chat results
# The `initiate_chat` method returns a `ChatResult` object, which is a dataclass object storing information about the chat. Currently, it includes the following attributes:
# 
# - `chat_history`: a list of chat history.
# - `summary`: a string of chat summary. A summary is only available if a summary_method is provided when initiating the chat.
# - `cost`: a tuple of (total_cost, total_actual_cost), where total_cost is a dictionary of cost information, and total_actual_cost is a dictionary of information on the actual incurred cost with cache.
# - `human_input`: a list of strings of human inputs solicited during the chat. (Note that since we are setting `human_input_mode` to `NEVER` in this notebook, this list is always empty.)

# In[6]:


for i, chat_res in enumerate(chat_results):
    print(f"*****{i}th chat*******:")
    print(chat_res.summary)
    print("Human input in the middle:", chat_res.human_input)
    print("Conversation cost: ", chat_res.cost)
    print("\n\n")


# ### Scenario 2: With human inputs revising tasks in the middle
# 
# Since AutoGen agents support soliciting human inputs during a chat if `human_input_mode` is specificed properly, the actual task might be revised in the middle of a chat. 
# 
# The example below showcases that even if a task is revised in the middle (for the first task, the human user requests to get Microsoft's stock price information as well, in addition to NVDA and TSLA), the `reflection_with_llm`` summary method can still capture it, as it reflects on the whole conversation instead of just the original request.

# In[7]:


user = autogen.UserProxyAgent(
    name="User",
    human_input_mode="ALWAYS",  # ask human for input at each step
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "tasks",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)

chat_results = user.initiate_chats(
    [
        {
            "recipient": financial_assistant,
            "message": financial_tasks[0],
            "clear_history": True,
            "silent": False,
            "summary_method": "reflection_with_llm",
        },
        {
            "recipient": research_assistant,
            "message": financial_tasks[1],
            "summary_method": "reflection_with_llm",
        },
        {
            "recipient": writer,
            "message": writing_tasks[0],
        },
    ]
)


# #### Check chat results

# In[8]:


for i, chat_res in enumerate(chat_results):
    print(f"*****{i}th chat*******:")
    print(chat_res.summary)
    print("Human input in the middle:", chat_res.human_input)
    print("Conversation cost: ", chat_res.cost)
    print("\n\n")


# ### Scenario 3: Solve the tasks with a series of chats involving group chat

# In[9]:


user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "groupchat",
        "use_docker": False,
    },
)

research_assistant = autogen.AssistantAgent(
    name="Researcher",
    llm_config={"config_list": config_list},
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
)

writer = autogen.AssistantAgent(
    name="Writer",
    llm_config={"config_list": config_list},
    system_message="""
    You are a professional writer, known for
    your insightful and engaging articles.
    You transform complex concepts into compelling narratives.
    Reply "TERMINATE" in the end when everything is done.
    """,
)

critic = autogen.AssistantAgent(
    name="Critic",
    system_message="""Critic. Double check plan, claims, code from other agents and provide feedback. Check whether the plan includes adding verifiable info such as source URL.
    Reply "TERMINATE" in the end when everything is done.
    """,
    llm_config={"config_list": config_list},
)

groupchat_1 = autogen.GroupChat(agents=[user_proxy, research_assistant, critic], messages=[], max_round=50)

groupchat_2 = autogen.GroupChat(agents=[user_proxy, writer, critic], messages=[], max_round=50)

manager_1 = autogen.GroupChatManager(
    groupchat=groupchat_1,
    name="Research_manager",
    llm_config={"config_list": config_list},
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "groupchat",
        "use_docker": False,
    },
)
manager_2 = autogen.GroupChatManager(
    groupchat=groupchat_2,
    name="Writing_manager",
    llm_config={"config_list": config_list},
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "groupchat",
        "use_docker": False,
    },
)

user = autogen.UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "tasks",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)
user.initiate_chats(
    [
        {"recipient": research_assistant, "message": financial_tasks[0], "summary_method": "last_msg"},
        {"recipient": manager_1, "message": financial_tasks[1], "summary_method": "reflection_with_llm"},
        {"recipient": manager_2, "message": writing_tasks[0]},
    ]
)

