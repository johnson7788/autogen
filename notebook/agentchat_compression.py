#!/usr/bin/env python
# coding: utf-8

# # Conversations with Chat History Compression Enabled
# 
# <div class="alert alert-danger" role="alert">
#   <strong>Deprecation Notice:</strong> <code>CompressibleAgent</code> has been deprecated and will no longer be available as of version <code>0.2.30</code>. Please transition to using <code>TransformMessages</code>, which is now the recommended approach. For a detailed guide on implementing this new standard, refer to our user guide on <a href="https://microsoft.github.io/autogen/docs/topics/handling_long_contexts/compressing_text_w_llmligua" target="_blank">Compressing Text with LLMLingua</a>. This guide provides examples for effectively utilizing LLMLingua transform as a replacement for <code>CompressibleAgent</code>.
# </div>
# 
# AutoGen offers conversable agents powered by LLM, tools, or humans, which can be used to perform tasks collectively via automated chat. This framework allows tool use and human participance through multi-agent conversation. Please find documentation about this feature [here](https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat).
# 
# In this notebook, we demonstrate how to enable compression of history messages using the `CompressibleAgent`. While this agent retains all the default functionalities of the `AssistantAgent`, it also provides the added feature of compression when activated through the `compress_config` setting.
# 
# Different compression modes are supported:
# 
# 1. `compress_config=False` (Default): `CompressibleAgent` is equivalent to `AssistantAgent`.
# 2. `compress_config=True` or `compress_config={"mode": "TERMINATE"}`: no compression will be performed. However, we will count token usage before sending requests to the OpenAI model. The conversation will be terminated directly if the total token usage exceeds the maximum token usage allowed by the model (to avoid the token limit error from OpenAI API).
# 3. `compress_config={"mode": "COMPRESS", "trigger_count": <your pre-set number>, "leave_last_n": <your pre-set number>}`: compression is enabled.
# 
#     ```python
#     # default compress_config
#     compress_config = {
#         "mode": "COMPRESS",
#         "compress_function": None,
#         "trigger_count": 0.7, # default to 0.7, or your pre-set number
#         "broadcast": True, # the compressed with be broadcast to sender. This will not be used in groupchat.
# 
#         # the following settings are for this mode only
#         "leave_last_n": 2, # leave the last n messages in the history to avoid compression
#         "verbose": False, # if True, print out the content to be compressed and the compressed content
#     }
#     ```
# 
#     Currently, our compression logic is as follows:
#         1. We will always leave the first user message (as well as system prompts) and compress the rest of the history messages.
#         2. You can choose to not compress the last n messages in the history with "leave_last_n".
#         2. The summary is performed on a per-message basis, with the role of the messages (See compressed content in the example below).
# 
# 4. `compress_config={"mode": "CUSTOMIZED", "compress_function": <A customized function for compression>}t`: the `compress_function` function will be called on trigger count. The function should accept a list of messages as input and return a tuple of (is_success: bool, compressed_messages: List[Dict]). The whole message history (except system prompt) will be passed.
# 
# 
# By adjusting `trigger_count`, you can decide when to compress the history messages based on existing tokens. If this is a float number between 0 and 1, it is interpreted as a ratio of max tokens allowed by the model. For example, the AssistantAgent uses gpt-4 with max tokens 8192, the trigger_count = 0.7 * 8192 = 5734.4 -> 5734. Do not set `trigger_count` to the max tokens allowed by the model, since the same LLM is employed for compression and it needs tokens to generate the compressed content. 
# 
# 
# 
# ## Limitations
# - For now, the compression feature **is not well-supported for groupchat**. If you initialize a `CompressibleAgent` in a groupchat with compression, the compressed cannot be broadcast to all other agents in the groupchat. If you use this feature in groupchat, extra cost will be incurred since compression will be performed on at per-agent basis.
# - We do not support async compression for now.
# 
# ## Requirements
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

# ## Set your API Endpoint
# 
# The [`config_list_from_json`](https://microsoft.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_json) function loads a list of configurations from an environment variable or a json file.
# 

# In[2]:


# define functions according to the function description
from IPython import get_ipython

import autogen
from autogen.agentchat.contrib.compressible_agent import CompressibleAgent
from autogen.agentchat.contrib.math_user_proxy_agent import MathUserProxyAgent

config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4-1106-preview"],
    },
)


# ````{=mdx}
# :::tip
# Learn more about configuring LLMs for agents [here](/docs/topics/llm_configuration).
# :::
# ````

# ## Example 1
# This example is from [agentchat_MathChat.ipynb](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_MathChat.ipynb). Compression with code execution.
# 
# You must set the `model` field in `llm_config`, as it will be used to calculate the token usage.
# 
# Note: we set `trigger_count=600`, and `leave_last_n=2`. In this example, we set a low trigger_count to demonstrate the compression feature. 
# The token count after compression is still bigger than trigger count, mainly because the trigger count is low an the first and last 2 messages are not compressed. Thus, the compression is performed at each turn. In practice, you want to adjust the trigger_count to a bigger number and properly set the `leave_last_n` to avoid compression at each turn. 
# 

# In[3]:


# 1. replace AssistantAgent with CompressibleAgent
assistant = CompressibleAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
        "model": "gpt-4-1106-preview",  # you must set the model field in llm_config, as it will be used to calculate the token usage.
    },
    compress_config={
        "mode": "COMPRESS",
        "trigger_count": 600,  # set this to a large number for less frequent compression
        "verbose": True,  # to allow printing of compression information: context before and after compression
        "leave_last_n": 2,
    },
)

# 2. create the MathUserProxyAgent instance named "mathproxyagent"
mathproxyagent = MathUserProxyAgent(
    name="mathproxyagent",
    human_input_mode="NEVER",
    code_execution_config={
        "use_docker": False
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    max_consecutive_auto_reply=5,
)
math_problem = (
    "Find all $x$ that satisfy the inequality $(2x+10)(x+3)<(3x+9)(x+8)$. Express your answer in interval notation."
)
mathproxyagent.initiate_chat(assistant, message=mathproxyagent.message_generator, problem=math_problem)


# ## Example 2
# This example is from [agentchat_function_call.ipynb](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_function_call.ipynb). Compression with function calls. 

# In[4]:


llm_config = {
    "model": "gpt-4-1106-preview",
    "functions": [
        {
            "name": "python",
            "description": "run cell in ipython and return the execution result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cell": {
                        "type": "string",
                        "description": "Valid Python cell to execute.",
                    }
                },
                "required": ["cell"],
            },
        },
        {
            "name": "sh",
            "description": "run a shell script and return the execution result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "Valid shell script to execute.",
                    }
                },
                "required": ["script"],
            },
        },
    ],
    "config_list": config_list,
    "timeout": 120,
}

chatbot = CompressibleAgent(
    name="chatbot",
    system_message="For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
    llm_config=llm_config,
    compress_config={
        "mode": "COMPRESS",
        "trigger_count": 600,  # set this to a large number for less frequent compression
        "verbose": True,  # set this to False to suppress the compression log
        "leave_last_n": 2,
    },
)

# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)


def exec_python(cell):
    ipython = get_ipython()
    result = ipython.run_cell(cell)
    log = str(result.result)
    if result.error_before_exec is not None:
        log += f"\n{result.error_before_exec}"
    if result.error_in_exec is not None:
        log += f"\n{result.error_in_exec}"
    return log


def exec_sh(script):
    return user_proxy.execute_code_blocks([("sh", script)])


# register the functions
user_proxy.register_function(
    function_map={
        "python": exec_python,
        "sh": exec_sh,
    }
)

# start the conversation
user_proxy.initiate_chat(
    chatbot,
    message="Draw two agents chatting with each other with an example dialog. Don't add plt.show().",
)


# ## Example 3
# This example is from [agent_chat_web_info.ipynb](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_web_info.ipynb). 
# We use this example to demonstrate how to pass in a customized compression function. We pass in an compression function `constrain_num_messages`, which constrains the number of messages to be 3 or less. 
# The customized function should accept a list of messages as input and return a tuple of `(is_success: bool, compressed_messages: List[Dict])`.

# In[5]:


def constrain_num_messages(messages):
    """Constrain the number of messages to 3.

    This is an example of a customized compression function.

    Returns:
        bool: whether the compression is successful.
        list: the compressed messages.
    """
    if len(messages) <= 3:
        # do nothing
        return False, None

    # save the first and last two messages
    return True, messages[:1] + messages[-2:]


# create a CompressibleAgent instance named "assistant"
assistant = CompressibleAgent(
    name="assistant",
    llm_config={
        "timeout": 600,
        "cache_seed": 43,
        "config_list": config_list,
        "model": "gpt-4-1106-preview",
    },
    compress_config={
        "mode": "CUSTOMIZED",
        "compress_function": constrain_num_messages,  # this is required for customized compression
        "trigger_count": 1600,
    },
)

# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE")
    or x.get("content", "").rstrip().endswith("TERMINATE."),
    code_execution_config={
        "work_dir": "web",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet.""",
)

user_proxy.initiate_chat(
    assistant,
    message="""Show me the YTD gain of 10 largest technology companies as of today.""",
)

