#!/usr/bin/env python
# coding: utf-8

# # AgentOptimizer: An Agentic Way to Train Your LLM Agent
# 
# AutoGen offers conversable agents powered by LLM, tool, or human, which can be used to perform tasks collectively via automated chat. This framework allows tool use and human participation through multi-agent conversation.
# Please find documentation about this feature [here](https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat).
# 
# In traditional ML pipeline, we train a model by updating its parameter according to the loss on the training set, while in the era of LLM agents, how should we train an agent? Here, we take an initial step towards the agent training. Inspired by the [function calling](https://platform.openai.com/docs/guides/function-calling) capabilities provided by OpenAI, we draw an analogy between model parameters and agent functions/skills, and update agent’s functions/skills based on its historical performance on the training set. As an agentic way of training an agent, our approach help enhance the agents’ abilities without requiring access to the LLMs parameters.
# 
# In this notebook, we introduce a new class, ‘AgentOptimizer’, which is able to improve the function list of one Assistant-UserProxy pair according to the historical conversation histories.
# This feature would support agents in improving their ability to solve problems of the same type as previous tasks.
# Specifically, given a set of training data, AgentOptimizer would iteratively prompt the LLM to optimize the existing function list of the AssistantAgent and UserProxyAgent with code implementation if necessary. It also includes two strategies, roll-back, and early-stop, to streamline the training process.
# In the example scenario, we test the proposed AgentOptimizer in solving problems from the [MATH dataset](https://github.com/hendrycks/math). 
# 
# ![AgentOptimizer](../website/blog/2023-12-23-AgentOptimizer/img/agentoptimizer.png)
# 
# More information could be found in the [paper](https://arxiv.org/abs/2402.11359).
# 
# Authors:
# - [Shaokun Zhang](https://github.com/skzhang1), Ph.D. student at the The Pennsylvania State University
# - [Jieyu Zhang](https://jieyuz2.github.io), Ph.D. student at the University of Washington

# In[17]:


import copy
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from openai import BadRequestError

import autogen
from autogen import config_list_from_json
from autogen.agentchat import Agent
from autogen.agentchat.contrib.agent_optimizer import AgentOptimizer
from autogen.agentchat.contrib.math_user_proxy_agent import MathUserProxyAgent
from autogen.code_utils import extract_code
from autogen.math_utils import get_answer


# # MathUserProxy with function_call
# 
# This agent is a customized MathUserProxy inherits from its [parent class](https://github.com/microsoft/autogen/blob/main/autogen/agentchat/contrib/math_user_proxy_agent.py).
# 
# It supports using both function_call and python to solve math problems.
# 

# In[18]:


def is_termination_msg_mathchat(message):
    """Check if a message is a termination message."""
    if isinstance(message, dict):
        message = message.get("content")
        if message is None:
            return False
    cb = extract_code(message)
    contain_code = False
    for c in cb:
        if c[0] == "python":
            contain_code = True
            break
    if message.rstrip().find("TERMINATE") >= 0:
        return True
    return not contain_code and get_answer(message) is not None and get_answer(message) != ""


class MathUserProxyAgent(MathUserProxyAgent):
    MAX_CONSECUTIVE_AUTO_REPLY = 15
    DEFAULT_REPLY = "Continue. Please keep solving the problem until you need to query. (If you get to the answer, put it in \\boxed{}.)"
    PROMPTS = """Let's solve a math problem.
Query requirements:
You should always use the 'print' function for the output and use fractions/radical forms instead of decimals.
You can use packages like sympy to help you.
You must follow the formats below to write your code:
```python
# your code
```
If some packages are missing, you could also suggest a code to install the corresponding package.

Please follow this process:
1. Solve the problem step by step (do not over-divide the steps).
2. Take out any queries that can be asked through Python code (for example, any calculations or equations that can be calculated) and functions you know in the context of this conversation.

Please
(1) do not mix suggested Python codes and function calls in one step.
(2) You MUST remember that you don’t have a function named "python" available.

You must follow the formats below to write your Python code:
```python
# your code
```

3. Wait for me to give the results or wait for the executed results of the function call.
4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.

After all the queries are run and you get the answer, put the answer in \\boxed{}.

Problem:
"""

    def __init__(
        self,
        name: Optional[str] = "MathChatAgent",
        is_termination_msg: Optional[Callable[[Dict], bool]] = is_termination_msg_mathchat,
        human_input_mode: Optional[str] = "NEVER",
        default_auto_reply: Optional[Union[str, Dict, None]] = DEFAULT_REPLY,
        max_invalid_q_per_step=3,
        **kwargs,
    ):
        super().__init__(
            name=name,
            is_termination_msg=is_termination_msg,
            human_input_mode=human_input_mode,
            default_auto_reply=default_auto_reply,
            max_invalid_q_per_step=max_invalid_q_per_step,
            **kwargs,
        )
        del self._reply_func_list[2]
        self.register_reply([Agent, None], MathUserProxyAgent._generate_math_reply, position=4)
        del self._reply_func_list[3]
        self.register_reply(
            trigger=autogen.ConversableAgent, reply_func=MathUserProxyAgent.generate_function_call_reply, position=3
        )
        self.register_reply(
            trigger=autogen.ConversableAgent, reply_func=MathUserProxyAgent._check_final_result, position=0
        )

        self.max_function_call_trial = 3
        self.query = None
        self.answer = None
        self.is_correct = None

    def generate_function_call_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[autogen.ConversableAgent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[Dict, None]]:
        """Generate a reply using function call."""
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        if "function_call" in message:
            is_exec_success, func_return = self.execute_function(message["function_call"])
            if is_exec_success:
                self.max_function_call_trial = 3
                return True, func_return
            else:
                if self.max_function_call_trial == 0:
                    error_message = func_return["content"]
                    self.max_function_call_trial = 3
                    return (
                        True,
                        "The func is executed failed many times. "
                        + error_message
                        + ". Please directly reply me with TERMINATE. We need to terminate the conversation.",
                    )
                else:
                    revise_prompt = "You may make a wrong function call (It may due the arguments you provided doesn't fit the function arguments like missing required positional argument).                     If you think this error occurs due to you make a wrong function arguments input and you could make it success, please try to call this function again using the correct arguments.                     Otherwise, the error may be caused by the function itself. Please directly reply me with TERMINATE. We need to terminate the conversation. "
                    error_message = func_return["content"]
                    return True, "The func is executed failed." + error_message + revise_prompt
        return False, None

    def initiate_chat(
        self,
        recipient,
        answer: None,
        silent: Optional[bool] = False,
        **context,
    ):
        self.query = context["problem"]
        if not isinstance(answer, str):
            answer = str(answer)
            if answer.endswith(".0"):
                answer = answer[:-2]
            self._answer = answer
        else:
            self._answer = answer

        self.is_correct = None

        self._prepare_chat(recipient, True)
        error_message = None
        try:
            prompt = self.PROMPTS + context["problem"]
            self.send(prompt, recipient, silent=silent)
        except BadRequestError as e:
            error_message = str(e)
            self.is_correct = 0
            print("error information: {}".format(error_message))

        recipient.reset()
        is_correct = copy.deepcopy(self.is_correct)
        self._reset()
        return is_correct

    def _check_final_result(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[autogen.Agent] = None,
        config: Optional[Any] = None,
    ):

        messages = messages[-1]
        if isinstance(messages, dict):
            messages = messages.get("content")
            if messages is None:
                return False, None

        cb = extract_code(messages)
        contain_code = False
        for c in cb:
            if c[0] == "python":
                contain_code = True
                break
        if not contain_code and get_answer(messages) is not None and get_answer(messages) != "":
            if get_answer(messages) == self._answer:
                self.is_correct = 1
                return True, "The result is Correct. Please reply me with TERMINATE."
            else:
                self.is_correct = 0
                return False, None
        else:
            return False, None

    def _reset(self):
        super()._reset()
        self.max_function_call_trial = 3
        self.is_correct = None
        self.query = None
        self.answer = None


# # Load dataset
# 
# MATAH dataset contains 12,500 challenging competition mathematics problems. Each problem in MATH has a full step-by-step solution which can be used to teach models to generate answer derivations and explanations. 
# 
# We strictly follow the [train](https://github.com/lifan-yuan/CRAFT/blob/main/tab_and_math/MATH/dataset/train/algebra.jsonl)/[test](https://github.com/lifan-yuan/CRAFT/blob/main/tab_and_math/MATH/dataset/algebra.jsonl) splits of [Craft](https://github.com/lifan-yuan/CRAFT). Please specific your own path to the dataset. Here we sample the first 10 algebra problems as examples. 

# In[25]:


test_data, train_data = [], []
with open("MATH/dataset/algebra.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        test_data.append(json.loads(line))
with open("MATH/dataset/train/algebra.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        train_data.append(json.loads(line))
test_data, train_data = test_data[0:10], train_data[0:10]


# # Agents construction
# 
# Constructing MathUserProxyAgent and AssistantAgent used in solving these problems. Here, we use gpt-4-1106-preview to construct the AssistantAgent. 

# In[26]:


llm_config = {
    "config_list": [
        {
            "model": "gpt-4-1106-preview",
            "api_type": "azure",
            "api_key": os.environ["AZURE_OPENAI_API_KEY"],
            "base_url": "https://ENDPOINT.openai.azure.com/",
            "api_version": "2023-07-01-preview",
        }
    ]
}

assistant = autogen.AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config=llm_config,
)
user_proxy = MathUserProxyAgent(
    name="mathproxyagent",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "_output", "use_docker": False},
)


# # Test without agent optimizations 
# 
# Below is the code to get the performance without the agents optimization process. 
# 
# In this case, the AssistantAgent and MathUserProxyAgent don't have any function calls but solely solve problems with Python.

# In[ ]:


sum = 0
for index, query in enumerate(test_data):
    is_correct = user_proxy.initiate_chat(recipient=assistant, answer=query["answer"], problem=query["question"])
    print(is_correct)
    sum += is_correct
success_rate_without_agent_training = sum / 10


# # Agent Training 
# 
# Then, we use the AgentOptimizer to iteratively optimize the agents by optimizing the function calls according to the historical conversations and performance.
# The AgentOptimizer yields register_for_llm and register_for_executor at each iteration, which are subsequently utilized to update the assistant and user_proxy agents, respectively. 
# Here we optimize these two agents for ten epochs. 

# In[ ]:


EPOCH = 10
optimizer_model = "gpt-4-1106-preview"
optimizer = AgentOptimizer(max_actions_per_step=3, llm_config=llm_config, optimizer_model=optimizer_model)
for i in range(EPOCH):
    for index, query in enumerate(train_data):
        is_correct = user_proxy.initiate_chat(assistant, answer=query["answer"], problem=query["question"])
        history = assistant.chat_messages_for_summary(user_proxy)
        optimizer.record_one_conversation(history, is_satisfied=is_correct)
    register_for_llm, register_for_exector = optimizer.step()
    for item in register_for_llm:
        assistant.update_function_signature(**item)
    if len(register_for_exector.keys()) > 0:
        user_proxy.register_function(function_map=register_for_exector)


# # Test with agent optimizations 
# 
# After agent optimization, the agents obtained a list of functions from the AgentOptimizers after 10 optimization iterations as shown below.
# 
# We then show the final performances with/without the agent optimization process. We observe the agents after optimization are obviously better.
# 
# 

# In[ ]:


sum = 0
for index, query in enumerate(test_data):
    is_correct = user_proxy.initiate_chat(recipient=assistant, answer=query["answer"], problem=query["question"])
    sum += is_correct
success_rate_with_agent_training = sum / 10


# In[2]:


print(
    "------------------------------------------------Functions learned------------------------------------------------"
)
for func in assistant.llm_config["functions"]:
    print(func["name"] + ": " + func["description"] + "\n")
print("------------------------------------------------Summary------------------------------------------------\n")
print("success_rate_without_agent_training: {average}%\n".format(average=success_rate_without_agent_training * 100))
print("success_rate_with_agent_training: {average}%\n".format(average=success_rate_with_agent_training * 100))

