#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2024/5/27 17:15
# @File  : questionaire.py
# @Author: 
# @Desc  : GPT4写的代码，不对

import os
from autogen import AssistantAgent, UserProxyAgent

# Configuration for the LLM
llm_config = {"model": "gpt-3.5-turbo", "api_key": os.environ.get("ANY_WHERE_KEY"),
                               'base_url': 'https://api.chatanywhere.tech/v1'}

# Initialize agents
assistant = AssistantAgent("assistant", llm_config=llm_config)
user_proxy = UserProxyAgent("user_proxy", code_execution_config=False)

# Define the questionnaire
questions = [
    "What is your name?",
    "How old are you?",
    "What is your gender?",
    "What is your occupation?",
    "What is your favorite hobby?",
    "What is your highest level of education?",
    "What is your favorite food?",
    "What is your favorite book?",
    "What is your favorite movie?",
    "What is your favorite travel destination?"
]


# Define validation logic for each question (this can be customized)
def validate_response(question, response):
    if not response or len(response.strip()) == 0:
        return False, "Response cannot be empty. Please provide a valid answer."
    if question == "How old are you?" and not response.isdigit():
        return False, "Please provide a numerical value for your age."
    return True, ""


def conduct_survey(assistant, user_proxy, questions):
    responses = {}
    for question in questions:
        valid = False
        while not valid:
            user_proxy.initiate_chat(
                assistant,
                message=question
            )
            user_response = input(f"{question} ")
            valid, error_message = validate_response(question, user_response)
            if not valid:
                print(error_message)
        responses[question] = user_response
    return responses


def summarize_responses(assistant, user_proxy, responses):
    summary_prompt = "Please summarize the following responses:\n"
    for question, response in responses.items():
        summary_prompt += f"{question}: {response}\n"

    user_proxy.initiate_chat(
        assistant,
        message=summary_prompt
    )
    summary_response = assistant.generate_text(summary_prompt)
    return summary_response


# Conduct the survey
survey_responses = conduct_survey(assistant, user_proxy, questions)

# Summarize the responses
summary = summarize_responses(assistant, user_proxy, survey_responses)
print("Survey Summary:")
print(summary)
