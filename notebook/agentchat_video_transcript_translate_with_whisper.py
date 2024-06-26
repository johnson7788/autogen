#!/usr/bin/env python
# coding: utf-8

# # Translating Video audio using Whisper and GPT-3.5-turbo
# 
# In this notebook, we demonstrate how to use whisper and GPT-3.5-turbo with `AssistantAgent` and `UserProxyAgent` to recognize and translate
# the speech sound from a video file and add the timestamp like a subtitle file based on [agentchat_function_call.ipynb](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_function_call.ipynb)
# 

# ## Requirements
# 
# ````{=mdx}
# :::info Requirements
# Some extra dependencies are needed for this notebook, which can be installed via pip:
# 
# ```bash
# pip install pyautogen openai openai-whisper
# ```
# 
# For more information, please refer to the [installation guide](/docs/installation/).
# :::
# ````

# ## Set your API Endpoint
# It is recommended to store your OpenAI API key in the environment variable. For example, store it in `OPENAI_API_KEY`.

# In[2]:


import os

config_list = [
    {
        "model": "gpt-4",
        "api_key": os.getenv("OPENAI_API_KEY"),
    }
]


# ````{=mdx}
# :::tip
# Learn more about configuring LLMs for agents [here](/docs/topics/llm_configuration).
# :::
# ````
# 
# ## Example and Output
# Below is an example of speech recognition from a [Peppa Pig cartoon video clip](https://drive.google.com/file/d/1QY0naa2acHw2FuH7sY3c-g2sBLtC2Sv4/view?usp=drive_link) originally in English and translated into Chinese.
# 'FFmpeg' does not support online files. To run the code on the example video, you need to download the example video locally. You can change `your_file_path` to your local video file path.

# In[ ]:


from typing import Annotated, List

import whisper
from openai import OpenAI

import autogen

source_language = "English"
target_language = "Chinese"
key = os.getenv("OPENAI_API_KEY")
target_video = "your_file_path"

assistant = autogen.AssistantAgent(
    name="assistant",
    system_message="For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
    llm_config={"config_list": config_list, "timeout": 120},
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={},
)


def translate_text(input_text, source_language, target_language):
    client = OpenAI(api_key=key)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Directly translate the following {source_language} text to a pure {target_language} "
                f"video subtitle text without additional explanation.: '{input_text}'",
            },
        ],
        max_tokens=1500,
    )

    # Correctly accessing the response content
    translated_text = response.choices[0].message.content if response.choices else None
    return translated_text


@user_proxy.register_for_execution()
@assistant.register_for_llm(description="using translate_text function to translate the script")
def translate_transcript(
    source_language: Annotated[str, "Source language"], target_language: Annotated[str, "Target language"]
) -> str:
    with open("transcription.txt", "r") as f:
        lines = f.readlines()

    translated_transcript = []

    for line in lines:
        # Split each line into timestamp and text parts
        parts = line.strip().split(": ")
        if len(parts) == 2:
            timestamp, text = parts[0], parts[1]
            # Translate only the text part
            translated_text = translate_text(text, source_language, target_language)
            # Reconstruct the line with the translated text and the preserved timestamp
            translated_line = f"{timestamp}: {translated_text}"
            translated_transcript.append(translated_line)
        else:
            # If the line doesn't contain a timestamp, add it as is
            translated_transcript.append(line.strip())

    return "\n".join(translated_transcript)


@user_proxy.register_for_execution()
@assistant.register_for_llm(description="recognize the speech from video and transfer into a txt file")
def recognize_transcript_from_video(filepath: Annotated[str, "path of the video file"]) -> List[dict]:
    try:
        # Load model
        model = whisper.load_model("small")

        # Transcribe audio with detailed timestamps
        result = model.transcribe(filepath, verbose=True)

        # Initialize variables for transcript
        transcript = []
        sentence = ""
        start_time = 0

        # Iterate through the segments in the result
        for segment in result["segments"]:
            # If new sentence starts, save the previous one and reset variables
            if segment["start"] != start_time and sentence:
                transcript.append(
                    {
                        "sentence": sentence.strip() + ".",
                        "timestamp_start": start_time,
                        "timestamp_end": segment["start"],
                    }
                )
                sentence = ""
                start_time = segment["start"]

            # Add the word to the current sentence
            sentence += segment["text"] + " "

        # Add the final sentence
        if sentence:
            transcript.append(
                {
                    "sentence": sentence.strip() + ".",
                    "timestamp_start": start_time,
                    "timestamp_end": result["segments"][-1]["end"],
                }
            )

        # Save the transcript to a file
        with open("transcription.txt", "w") as file:
            for item in transcript:
                sentence = item["sentence"]
                start_time, end_time = item["timestamp_start"], item["timestamp_end"]
                file.write(f"{start_time}s to {end_time}s: {sentence}\n")

        return transcript

    except FileNotFoundError:
        return "The specified audio file could not be found."
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


# Now, start the chat:

# In[5]:


user_proxy.initiate_chat(
    assistant,
    message=f"For the video located in {target_video}, recognize the speech and transfer it into a script file, "
    f"then translate from {source_language} text to a {target_language} video subtitle text. ",
)


# In[ ]:




