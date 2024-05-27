#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2024/5/24 15:18
# @File  : simple_question.py
# @Author: 
# @Desc  : 简单问答


import os
from autogen import ConversableAgent, config_list_from_json
config_list = config_list_from_json(
    env_or_file="chatconfig.json",
)

agent = ConversableAgent(
    "chatbot",
    llm_config={"config_list": config_list},
    code_execution_config=False,  # Turn off code execution, by default it is off.
    function_map=None,  # No registered functions, by default it is None.
    human_input_mode="NEVER",  # Never ask for human input.
)

reply = agent.generate_reply(messages=[{"content": "Tell me a joke, haha", "role": "user"}])
print(reply)