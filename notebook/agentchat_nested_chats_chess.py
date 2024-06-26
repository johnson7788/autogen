#!/usr/bin/env python
# coding: utf-8

# # Nested Chats for Tool Use in Conversational Chess
# 
# This notebook demonstrates how to create agents that can play chess with each other
# while communicating in natural language.
# The key concept covered in this notebook is the use of nested chats
# to enable tool use and packaging an LLM-based agent with a tool executor agent
# into a single agent.
# 
# Related tutorials:
# - [Tool Use](/docs/tutorial/tool-use)
# - [Nested Chats](/docs/tutorial/conversation-patterns#nested-chats)
# 
# In this setting, each player is an agent backed by an LLM equipped two tools:
# - `get_legal_moves` to get a list of current legal moves.
# - `make_move` to make a move.
# 
# A board proxy agent is set up to execute the tools and manage the game.
# It is important to use a board proxy as a non-LLM "guard rail" to ensure the game
# is played correctly and to prevent agents from making illegal moves.
# 
# Each time a player agent receives a message from the other player agent, 
# it instantiates a nested chat with the board proxy agent to get the legal moves
# and make a move using the tools given. 
# The nested chat between the player agent and the board agent
# continues until the a legal move is made by the tool.
# Once the nested chat concludes, the player agent sends a message to the
# other player agent about the move made.

# ## Installation
# 
# First you need to install the `pyautogen` and `chess` packages to use AutoGen.

# In[2]:


get_ipython().system(' pip install -qqq pyautogen chess')


# ## Setting up LLMs
# 
# Now you can set up the models you want to use.

# In[4]:


import os

player_white_config_list = [
    {
        "model": "gpt-4-turbo-preview",
        "api_key": os.environ.get("OPENAI_API_KEY"),
    },
]

player_black_config_list = [
    {
        "model": "gpt-4-turbo-preview",
        "api_key": os.environ.get("OPENAI_API_KEY"),
    },
]


# ## Creating tools
# 
# Write functions for getting legal moves and making a move.

# In[5]:


from typing import List

import chess
import chess.svg
from IPython.display import display
from typing_extensions import Annotated

# Initialize the board.
board = chess.Board()

# Keep track of whether a move has been made.
made_move = False


def get_legal_moves() -> Annotated[str, "A list of legal moves in UCI format"]:
    return "Possible moves are: " + ",".join([str(move) for move in board.legal_moves])


def make_move(move: Annotated[str, "A move in UCI format."]) -> Annotated[str, "Result of the move."]:
    move = chess.Move.from_uci(move)
    board.push_uci(str(move))
    global made_move
    made_move = True
    # Display the board.
    display(
        chess.svg.board(board, arrows=[(move.from_square, move.to_square)], fill={move.from_square: "gray"}, size=200)
    )
    # Get the piece name.
    piece = board.piece_at(move.to_square)
    piece_symbol = piece.unicode_symbol()
    piece_name = (
        chess.piece_name(piece.piece_type).capitalize()
        if piece_symbol.isupper()
        else chess.piece_name(piece.piece_type)
    )
    return f"Moved {piece_name} ({piece_symbol}) from {chess.SQUARE_NAMES[move.from_square]} to {chess.SQUARE_NAMES[move.to_square]}."


# ## Creating agents
# 
# Let's create the agents. We have three different agents:
# - `player_white` is the agent that plays white.
# - `player_black` is the agent that plays black.
# - `board_proxy` is the agent that moves the pieces on the board.

# In[6]:


from autogen import ConversableAgent, register_function

player_white = ConversableAgent(
    name="Player White",
    system_message="You are a chess player and you play as white. "
    "First call get_legal_moves() first, to get list of legal moves. "
    "Then call make_move(move) to make a move.",
    llm_config={"config_list": player_white_config_list, "cache_seed": None},
)

player_black = ConversableAgent(
    name="Player Black",
    system_message="You are a chess player and you play as black. "
    "First call get_legal_moves() first, to get list of legal moves. "
    "Then call make_move(move) to make a move.",
    llm_config={"config_list": player_black_config_list, "cache_seed": None},
)

# Check if the player has made a move, and reset the flag if move is made.


def check_made_move(msg):
    global made_move
    if made_move:
        made_move = False
        return True
    else:
        return False


board_proxy = ConversableAgent(
    name="Board Proxy",
    llm_config=False,
    # The board proxy will only terminate the conversation if the player has made a move.
    is_termination_msg=check_made_move,
    # The auto reply message is set to keep the player agent retrying until a move is made.
    default_auto_reply="Please make a move.",
    human_input_mode="NEVER",
)


# Register tools for the agents. See [tutorial chapter on tool use](/docs/tutorial/tool-use) 
# for more information.

# In[7]:


register_function(
    make_move,
    caller=player_white,
    executor=board_proxy,
    name="make_move",
    description="Call this tool to make a move.",
)

register_function(
    get_legal_moves,
    caller=player_white,
    executor=board_proxy,
    name="get_legal_moves",
    description="Get legal moves.",
)

register_function(
    make_move,
    caller=player_black,
    executor=board_proxy,
    name="make_move",
    description="Call this tool to make a move.",
)

register_function(
    get_legal_moves,
    caller=player_black,
    executor=board_proxy,
    name="get_legal_moves",
    description="Get legal moves.",
)


# Now the agents have their tools ready. You can inspect the auto-generated
# tool schema for each agent.

# In[8]:


player_black.llm_config["tools"]


# Register nested chats for the player agents.
# Nested chats allows each player agent to chat with the board proxy agent
# to make a move, before communicating with the other player agent.
# 
# In the code below, in each nested chat, the board proxy agent starts
# a conversation with the player agent using the message recieved from the other
# player agent (e.g., "Your move"). The two agents continue the conversation
# until a legal move is made using the `make_move` tool.
# The last message in the nested chat is a message from the player agent about
# the move made,
# and this message is then sent to the other player agent.
# 
# The following diagram illustrates the nested chat between the player agent and the board agent.
# 
# ![Conversational Chess](nested-chats-chess.png)
# 
# See [nested chats tutorial chapter](/docs/tutorial/conversation-patterns#nested-chats)
# for more information.

# In[9]:


player_white.register_nested_chats(
    trigger=player_black,
    chat_queue=[
        {
            # The initial message is the one received by the player agent from
            # the other player agent.
            "sender": board_proxy,
            "recipient": player_white,
            # The final message is sent to the player agent.
            "summary_method": "last_msg",
        }
    ],
)

player_black.register_nested_chats(
    trigger=player_white,
    chat_queue=[
        {
            # The initial message is the one received by the player agent from
            # the other player agent.
            "sender": board_proxy,
            "recipient": player_black,
            # The final message is sent to the player agent.
            "summary_method": "last_msg",
        }
    ],
)


# ## Playing the game
# 
# Start the chess game.

# In[11]:


# Clear the board.
board = chess.Board()

chat_result = player_black.initiate_chat(
    player_white,
    message="Let's play chess! Your move.",
    max_turns=4,
)


# In the output above, you can see "Start a new chat" is displayed
# whenever a new nested chat is started between the board proxy agent and a player agent.
# The "carryover" is empty as it is a new chat in the sequence.
