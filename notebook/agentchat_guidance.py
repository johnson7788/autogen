#!/usr/bin/env python
# coding: utf-8

# # Using Guidance with AutoGen
# 
# This notebook shows how Guidance can be used to enable structured responses from AutoGen agents. In particular, this notebook focuses on creating agents that always output a valid code block or valid json object.
# 

# In[14]:


import re

from guidance import assistant, gen, models, system, user
from pydantic import BaseModel

from autogen import Agent, AssistantAgent, UserProxyAgent, config_list_from_json

llm_config = config_list_from_json("OAI_CONFIG_LIST")[0]  # use the first config
gpt = models.OpenAI("gpt-4", api_key=llm_config.get("api_key"))


# The example below uses guidance to create a `guidance_coder` agent that only responds with valid code blocks.

# In[20]:


def is_valid_code_block(code):
    pattern = r"```[\w\s]*\n([\s\S]*?)\n```"
    match = re.search(pattern, code)
    if match:
        return True
    else:
        return False


def generate_structured_response(recipient, messages, sender, config):
    gpt = models.OpenAI("gpt-4", api_key=llm_config.get("api_key"), echo=False)

    # populate the recipient with the messages from the history
    with system():
        lm = gpt + recipient.system_message

    for message in messages:
        if message.get("role") == "user":
            with user():
                lm += message.get("content")
        else:
            with assistant():
                lm += message.get("content")

    # generate a new response and store it
    with assistant():
        lm += gen(name="initial_response")
    # ask the agent to reflect on the nature of the response and store it
    with user():
        lm += "Does the very last response from you contain code? Respond with yes or no."
    with assistant():
        lm += gen(name="contains_code")
    # if the response contains code, ask the agent to generate a proper code block
    if "yes" in lm["contains_code"].lower():
        with user():
            lm += "Respond with a single blocks containing the valid code. Valid code blocks start with ```"
        with assistant():
            lm += "```" + gen(name="code")
            response = "```" + lm["code"]

            is_valid = is_valid_code_block(response)
            if not is_valid:
                raise ValueError(f"Failed to generate a valid code block\n {response}")

    # otherwise, just use the initial response
    else:
        response = lm["initial_response"]

    return True, response


guidance_agent = AssistantAgent("guidance_coder", llm_config=llm_config)
guidance_agent.register_reply(Agent, generate_structured_response, 1)
user_proxy = UserProxyAgent(
    "user",
    human_input_mode="TERMINATE",
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content"),
)
user_proxy.initiate_chat(guidance_agent, message="Plot and save a chart of nvidia and tsla stock price change YTD.")


# The example below uses Guidance to enable a `guidance_labeler` agent that only responds with a valid JSON that labels a given comment/joke.

# In[16]:


class Response(BaseModel):
    label: str
    explanation: str


response_prompt_instructions = """The label must be a JSON of the format:
{
    "label": str,
    "explanation": str
}"""


def generate_structured_response(recipient, messages, sender, config):
    gpt = models.OpenAI("gpt-4", api_key=llm_config.get("api_key"), echo=False)

    # populate the recipient with the messages from the history
    with system():
        lm = gpt + recipient.system_message

    for message in messages:
        if message.get("role") == "user":
            with user():
                lm += message.get("content")
        else:
            with assistant():
                lm += message.get("content")

    # generate a new response and store it
    with assistant():
        lm += gen(name="initial_response")
    # ask the agent to reflect on the nature of the response and store it
    with user():
        lm += "Does the very last response from you contain JSON object? Respond with yes or no."
    with assistant():
        lm += gen(name="contains_json")
    # if the response contains code, ask the agent to generate a proper code block
    if "yes" in lm["contains_json"].lower():
        with user():
            lm += (
                "What was that JSON object? Only respond with that valid JSON string. A valid JSON string starts with {"
            )
        with assistant():
            lm += "{" + gen(name="json")
            response = "{" + lm["json"]
            # verify that the response is valid json
            try:
                response_obj = Response.model_validate_json(response)
                response = response_obj.model_dump_json()
            except Exception as e:
                response = str(e)
    # otherwise, just use the initial response
    else:
        response = lm["initial_response"]

    return True, response


guidance_agent = AssistantAgent("guidance_labeler", llm_config=llm_config, system_message="You are a helpful assistant")
guidance_agent.register_reply(Agent, generate_structured_response, 1)
user_proxy = UserProxyAgent("user", human_input_mode="ALWAYS", code_execution_config=False)
user_proxy.initiate_chat(
    guidance_agent,
    message=f"""
Label the TEXT via the following instructions:

{response_prompt_instructions}

TEXT: what did the fish say when it bumped into a wall? Dam!

""",
)

