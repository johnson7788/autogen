#!/usr/bin/env python
# coding: utf-8

# # Config loader utility functions
# 
# For an introduction to configuring LLMs, refer to the [main configuration docs](https://microsoft.github.io/autogen/docs/topics/llm_configuration). This guide will run through examples of the more advanced utility functions for managing API configurations.
# 
# Managing API configurations can be tricky, especially when dealing with multiple models and API versions. The provided utility functions assist users in managing these configurations effectively. Ensure your API keys and other sensitive data are stored securely. You might store keys in `.txt` or `.env` files or environment variables for local development. Never expose your API keys publicly. If you insist on storing your key files locally on your repo (you shouldn't), ensure the key file path is added to the `.gitignore` file.
# 
# ## Storing API keys
# 
# 1. Obtain API keys from OpenAI and optionally from Azure OpenAI (or other provider).
# 2. Store them securely using either:
#     - Environment Variables: `export OPENAI_API_KEY='your-key'` in your shell.
#     - Text File: Save the key in a `key_openai.txt` file.
#     - Env File: Save the key to a `.env` file eg: `OPENAI_API_KEY=sk-********************`
# 
# ## Utility functions
# 
# There are several utility functions for loading LLM config lists that may be useful depending on the situation.
# 
# - [`get_config_list`](#get_config_list): Generates configurations for API calls, primarily from provided API keys.
# - [`config_list_openai_aoai`](#config_list_openai_aoai): Constructs a list of configurations using both Azure OpenAI and OpenAI endpoints, sourcing API keys from environment variables or local files.
# - [`config_list_from_json`](#config_list_from_json): Loads configurations from a JSON structure, either from an environment variable or a local JSON file, with the flexibility of filtering configurations based on given criteria.
# - [`config_list_from_models`](#config_list_from_models): Creates configurations based on a provided list of models, useful when targeting specific models without manually specifying each configuration.
# - [`config_list_from_dotenv`](#config_list_from_dotenv): Constructs a configuration list from a `.env` file, offering a consolidated way to manage multiple API configurations and keys from a single file.

# ### get_config_list
# 
# Used to generate configurations for API calls.

# In[4]:


import autogen

api_keys = ["YOUR_OPENAI_API_KEY"]
base_urls = None  # You can specify API base URLs if needed. eg: localhost:8000
api_type = "openai"  # Type of API, e.g., "openai" or "aoai".
api_version = None  # Specify API version if needed.

config_list = autogen.get_config_list(api_keys, base_urls=base_urls, api_type=api_type, api_version=api_version)

print(config_list)


# ### config_list_openai_aoai
# 
# This method creates a list of configurations using Azure OpenAI endpoints and OpenAI endpoints. It tries to extract API keys and bases from environment variables or local text files.
# 
# Steps:
# - Store OpenAI API key in:
#     - Environment variable: `OPENAI_API_KEY`
#     - or Local file: `key_openai.txt`
# - Store Azure OpenAI API key in:
#     - Environment variable: `AZURE_OPENAI_API_KEY`
#     - or Local file: `key_aoai.txt` (Supports multiple keys, one per line)
# - Store Azure OpenAI API base in:
#     - Environment variable: `AZURE_OPENAI_API_BASE`
#     - or Local file: `base_aoai.txt` (Supports multiple bases, one per line)

# In[ ]:


config_list = autogen.config_list_openai_aoai(
    key_file_path=".",
    openai_api_key_file="key_openai.txt",
    aoai_api_key_file="key_aoai.txt",
    aoai_api_base_file="base_aoai.txt",
    exclude=None,  # The API type to exclude, eg: "openai" or "aoai".
)


# ### config_list_from_json
# 
# This method loads configurations from an environment variable or a JSON file. It provides flexibility by allowing users to filter configurations based on certain criteria.
# 
# Steps:
# - Setup the JSON Configuration:
#     1. Store configurations in an environment variable named `OAI_CONFIG_LIST` as a valid JSON string.
#     2. Alternatively, save configurations in a local JSON file named `OAI_CONFIG_LIST.json`
#     3. Add `OAI_CONFIG_LIST` to your `.gitignore` file on your local repository.
# 
# Your JSON structure should look something like this:
# 
# ```json
# # OAI_CONFIG_LIST file example
# [
#     {
#         "model": "gpt-4",
#         "api_key": "YOUR_OPENAI_API_KEY"
#     },
#     {
#         "model": "gpt-3.5-turbo",
#         "api_key": "YOUR_OPENAI_API_KEY",
#         "api_version": "2023-03-01-preview"
#     }
# ]
# 
# ```
# 

# In[ ]:


config_list = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST",  # or OAI_CONFIG_LIST.json if file extension is added
    filter_dict={
        "model": {
            "gpt-4",
            "gpt-3.5-turbo",
        }
    },
)


# ### config_list_from_models
# 
# This method creates configurations based on a provided list of models. It's useful when you have specific models in mind and don't want to manually specify each configuration. The [`config_list_from_models`](/docs/reference/oai/openai_utils#config_list_from_models) function tries to create a list of configurations using Azure OpenAI endpoints and OpenAI endpoints for the provided list of models. It assumes the api keys and api bases are stored in the corresponding environment variables or local txt files. It's okay to only have the OpenAI API key, OR only the Azure OpenAI API key + base. For Azure the model name refers to the OpenAI Studio deployment name.
# 
# Steps:
# - Similar to method 1, store API keys and bases either in environment variables or `.txt` files.

# In[6]:


config_list = autogen.config_list_from_models(
    key_file_path=".",
    openai_api_key_file="key_openai.txt",
    aoai_api_key_file="key_aoai.txt",
    aoai_api_base_file="base_aoai.txt",
    exclude="aoai",
    model_list=["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
)


# ### config_list_from_dotenv
# 
# If you are interested in keeping all of your keys in a single location like a `.env` file rather than using a configuration specifically for OpenAI, you can use `config_list_from_dotenv`. This allows you to conveniently create a config list without creating a complex `OAI_CONFIG_LIST` file.
# 
# The `model_api_key_map` parameter is a dictionary that maps model names to the environment variable names in the `.env` file where their respective API keys are stored. It lets the code know which API key to use for each model. 
# 
# If not provided, it defaults to using `OPENAI_API_KEY` for `gpt-4` and `OPENAI_API_KEY` for `gpt-3.5-turbo`.
# 
# ```python
#     # default key map
#     model_api_key_map = {
#         "gpt-4": "OPENAI_API_KEY",
#         "gpt-3.5-turbo": "OPENAI_API_KEY",
#     }
# ```
# 
# Here is an example `.env` file:
# 
# ```bash
# OPENAI_API_KEY=sk-*********************
# HUGGING_FACE_API_KEY=**************************
# ANOTHER_API_KEY=1234567890234567890
# ```

# In[1]:


config_list = autogen.config_list_from_dotenv(
    dotenv_file_path=".env",  # If None the function will try to find in the working directory
    filter_dict={
        "model": {
            "gpt-4",
            "gpt-3.5-turbo",
        }
    },
)

config_list


# In[2]:


# gpt-3.5-turbo will default to OPENAI_API_KEY
config_list = autogen.config_list_from_dotenv(
    dotenv_file_path=".env",  # If None the function will try to find in the working directory
    model_api_key_map={
        "gpt-4": "ANOTHER_API_KEY",  # String or dict accepted
    },
    filter_dict={
        "model": {
            "gpt-4",
            "gpt-3.5-turbo",
        }
    },
)

config_list


# In[3]:


# example using different environment variable names
config_list = autogen.config_list_from_dotenv(
    dotenv_file_path=".env",
    model_api_key_map={
        "gpt-4": "OPENAI_API_KEY",
        "vicuna": "HUGGING_FACE_API_KEY",
    },
    filter_dict={
        "model": {
            "gpt-4",
            "vicuna",
        }
    },
)

config_list


# You can also provide additional configurations for APIs, simply by replacing the string value with a dictionary expanding on the configurations. See the example below showing the example of using `gpt-4` on `openai` by default, and using `gpt-3.5-turbo` with additional configurations for `aoai`.

# In[4]:


config_list = autogen.config_list_from_dotenv(
    dotenv_file_path=".env",
    model_api_key_map={
        "gpt-4": "OPENAI_API_KEY",
        "gpt-3.5-turbo": {
            "api_key_env_var": "ANOTHER_API_KEY",
            "api_type": "aoai",
            "api_version": "v2",
            "base_url": "https://api.someotherapi.com",
        },
    },
    filter_dict={
        "model": {
            "gpt-4",
            "gpt-3.5-turbo",
        }
    },
)

config_list

