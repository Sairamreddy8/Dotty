"""Dotty: a personal AI assistant with long-term memory.

Run `python dotty.py` to chat. Dotty keeps three kinds of long-term memory
(your profile, your ToDo list, and your preferences for how ToDos are written)
in a local SQLite file, so it remembers you between runs.
"""

import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from trustcall import create_extractor

from langchain_core.messages import HumanMessage, SystemMessage, merge_message_runs
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq

from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore
from langgraph.store.sqlite import SqliteStore

load_dotenv()

MODEL = "openai/gpt-oss-120b"
DEFAULT_USER_ID = "default-user"
DB_PATH = Path(__file__).with_name("dotty.db")
TRUSTCALL_STEP_LIMIT = 10

def get_user_id(config: RunnableConfig) -> str:
    return config.get("configurable", {}).get("user_id") or DEFAULT_USER_ID

## Schema definitions

# User profile schema
class Profile(BaseModel):
    """This is the profile of the user you are chatting with"""
    name: Optional[str] = Field(description="The user's name", default=None)
    location: Optional[str] = Field(description="The user's location", default=None)
    job: Optional[str] = Field(description="The user's job", default=None)
    connections: list[str] = Field(
        description="Personal connection of the user, such as family members, friends, or coworkers",
        default_factory=list
    )
    interests: list[str] = Field(
        description="Interests that the user has",
        default_factory=list
    )

# ToDo schema
# Every field has a default and none uses a JSON-schema `format`: Groq validates tool
# arguments against the schema and rejects the whole call if the model omits a required
# field or writes a date-only value like "2026-12-01" into a `format: date-time` field.
class ToDo(BaseModel):
    """This is a task that the user wants to complete, along with any relevant information about it."""
    task: str = Field(description="The task to be completed.")
    time_to_complete: Optional[int] = Field(description="Estimated time to complete the task (minutes).", default=None)
    deadline: Optional[str] = Field(
        description="When the task needs to be completed by (if applicable), as an ISO 8601 date, e.g. 2026-12-01",
        default=None
    )
    solutions: list[str] = Field(
        description="List of specific, actionable solutions (e.g., specific ideas, service providers, or concrete options relevant to completing the task)",
        default_factory=list
    )
    status: Literal["not started", "in progress", "done", "archived"] = Field(
        description="Current status of the task",
        default="not started"
    )

# ponytail: the whole list is ONE document, loaded into the prompt every turn and
# re-validated on every update. Fine for a personal list; move to one document per
# task (Trustcall enable_inserts) plus semantic search once it reaches hundreds of
# tasks AND you are on a model that emits parallel tool calls (see README).
class ToDoList(BaseModel):
    """The user's complete ToDo list."""
    todos: list[ToDo] = Field(description="Every task on the user's ToDo list", default_factory=list)

## Initialize the model and tools

# Update memory tool
class UpdateMemory(TypedDict):
    """ Decision on what memory type to update """
    update_type: Literal['user', 'todo', 'instructions']

# Initialize the model (timeout so a stuck request can't hang the chat loop)
model = ChatGroq(model=MODEL, temperature=0, timeout=60)

# Trustcall extractors for updating the user profile and the ToDo list
profile_extractor = create_extractor(model, tools=[Profile], tool_choice="Profile")
todo_extractor = create_extractor(model, tools=[ToDoList], tool_choice="ToDoList")

## Prompts

# Chatbot instruction for choosing what to update and what tools to call
MODEL_SYSTEM_MESSAGE = """You are Dotty, a helpful personal AI assistant.

You are designed to be a companion to a user, helping them keep track of their ToDo list.

You have a long term memory which keeps track of three things:
1. The user's profile (general information about them)
2. The user's ToDo list
3. General instructions for updating the ToDo list

Here is the current User Profile (may be empty if no information has been collected yet):
<user_profile>
{user_profile}
</user_profile>

Here is the current ToDo List (may be empty if no tasks have been added yet):
<todo>
{todo}
</todo>

Here are the current user-specified preferences for updating the ToDo list (may be empty if no preferences have been specified yet):
<instructions>
{instructions}
</instructions>

Here are your instructions for reasoning about the user's messages:

1. Reason carefully about the user's messages as presented below.

2. Decide whether any of the your long-term memory should be updated:
- If personal information was provided about the user, update the user's profile by calling UpdateMemory tool with type `user`
- If tasks are mentioned, update the ToDo list by calling UpdateMemory tool with type `todo`
- If the user has specified preferences for how to update the ToDo list, update the instructions by calling UpdateMemory tool with type `instructions`

3. Tell the user that you have updated your memory, if appropriate:
- Do not tell the user you have updated the user's profile
- Tell the user them when you update the todo list
- Do not tell the user that you have updated instructions

4. Err on the side of updating the todo list. No need to ask for explicit permission.

5. The result of an UpdateMemory call shows exactly what is now saved. Only tell the user something was saved or changed if that result confirms it, and quote the saved values rather than your own guess. If the result says nothing was saved, say so.

6. Respond naturally to user user after a tool call was made to save memories, or if no tool call was made.

Today is {today}."""

# Appended to the system message when Dotty must answer without another memory update
ALREADY_UPDATED = """

Your memory has already been updated for this message. Do not call any tools: reply to the user now, based on the tool results above."""

# Trustcall instruction
TRUSTCALL_INSTRUCTION = """Reflect on following interaction.

Use the provided tools to retain any necessary memories about the user.

Make every change the interaction calls for in a single tool call: if you are patching an existing document, include one JSON patch per change.

{instructions}

System Time: {time}"""

# Instructions for updating the ToDo list
CREATE_INSTRUCTIONS = """Reflect on the following interaction.

Based on this interaction, update your instructions for how to update ToDo list items. Use any feedback from the user to update how they like to have items added, etc.

Your current instructions are:

<current_instructions>
{current_instructions}
</current_instructions>"""

def now() -> str:
    # Weekday included: models are bad at working out "next Friday" from a bare date
    return datetime.now().strftime("%A %Y-%m-%d %H:%M")

def updates_this_turn(messages) -> set:
    """The memory types already updated since the user's last message."""
    done = set()
    for m in reversed(messages):
        if m.type == "human":
            break
        for call in getattr(m, "tool_calls", None) or []:
            done.add(call["args"].get("update_type"))
    return done

## Node definitions

def dotty(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Load memories from the store and use them to personalize the chatbot's response."""

    user_id = get_user_id(config)

    # Retrieve profile memory from the store
    memory = store.get(("profile", user_id), "Profile")
    user_profile = memory.value if memory else None

    # Retrieve task memory from the store
    memory = store.get(("todo", user_id), "ToDoList")
    todo = "\n".join(f"{t}" for t in memory.value["todos"]) if memory else ""

    # Retrieve custom instructions
    memory = store.get(("instructions", user_id), "user_instructions")
    instructions = memory.value["memory"] if memory else ""

    system_msg = MODEL_SYSTEM_MESSAGE.format(user_profile=user_profile, todo=todo, instructions=instructions, today=now())

    # Respond using memory as well as the chat history
    messages = [SystemMessage(content=system_msg)]+state["messages"]
    response = model.bind_tools([UpdateMemory], parallel_tool_calls=False).invoke(messages)

    # Each memory type is updated at most once per user message. If the model asks to
    # repeat an update it already made this turn (which would loop, and duplicate
    # ToDos), make it answer in text instead.
    if response.tool_calls and response.tool_calls[0]["args"].get("update_type") in updates_this_turn(state["messages"]):
        messages[0] = SystemMessage(content=system_msg + ALREADY_UPDATED)
        response = model.invoke(messages)

    return {"messages": [response]}

def run_trustcall(state: MessagesState, store: BaseStore, namespace: tuple, schema: type[BaseModel], extractor, instructions: str = ""):

    """Reflect on the chat history with a Trustcall extractor, save the resulting document
    to the store, and return a summary of what was actually saved."""

    # The document Trustcall should patch, keyed by its schema name. When nothing is saved
    # yet we hand it an empty document rather than None, so the model always answers with
    # JSON patches: Trustcall validates and repairs those itself, whereas a whole new
    # document goes through Groq's stricter server-side validation, which fails the request.
    tool_name = schema.__name__
    existing_item = store.get(namespace, tool_name)
    before = existing_item.value if existing_item else schema().model_dump(mode="json")

    # Merge the chat history and the instruction
    instruction = TRUSTCALL_INSTRUCTION.format(instructions=instructions, time=now())
    updated_messages = list(merge_message_runs(messages=[SystemMessage(content=instruction)] + state["messages"][:-1]))

    # Invoke the extractor. Trustcall swallows model API errors (rate limits, rejected
    # tool calls) and retries with no cap of its own, so cap its internal graph: a clean
    # update takes 4 steps and its full validate-and-repair cycle 10.
    failed = f"Nothing was saved: the {tool_name} update failed. Tell the user and offer to retry."
    try:
        result = extractor.invoke({"messages": updated_messages,
                                   "existing": {tool_name: before}},
                                  {"recursion_limit": TRUSTCALL_STEP_LIMIT})
    except GraphRecursionError:
        return failed

    # Report what was actually saved (not what the model attempted), so Dotty
    # never claims an update that failed
    if not result["responses"]:
        return failed
    after = result["responses"][0].model_dump(mode="json")
    if after == before:
        return f"No changes were made to {tool_name}."
    store.put(namespace, tool_name, after)
    return f"Saved. {tool_name} is now: {after}"

def update_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the chat history and update the user profile."""

    saved = run_trustcall(state, store, ("profile", get_user_id(config)), Profile, profile_extractor)

    # Respond to the tool call made in dotty, confirming the update
    tool_calls = state['messages'][-1].tool_calls
    return {"messages": [{"role": "tool", "content": saved, "tool_call_id":tool_calls[0]['id']}]}

def update_todos(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the chat history and update the ToDo list."""

    user_id = get_user_id(config)

    # The user's preferences for how ToDo items should be written (procedural memory)
    memory = store.get(("instructions", user_id), "user_instructions")
    instructions = f"Follow the user's preferences for ToDo items:\n{memory.value['memory']}" if memory else ""

    saved = run_trustcall(state, store, ("todo", user_id), ToDoList, todo_extractor, instructions)

    # Respond to the tool call made in dotty, confirming the update
    tool_calls = state['messages'][-1].tool_calls
    return {"messages": [{"role": "tool", "content": saved, "tool_call_id":tool_calls[0]['id']}]}

def update_instructions(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the chat history and rewrite the ToDo instructions."""

    namespace = ("instructions", get_user_id(config))
    key = "user_instructions"

    existing_memory = store.get(namespace, key)

    # Format the memory in the system prompt
    system_msg = CREATE_INSTRUCTIONS.format(current_instructions=existing_memory.value if existing_memory else None)
    new_memory = model.invoke([SystemMessage(content=system_msg)]+state['messages'][:-1] + [HumanMessage(content="Please update the instructions based on the conversation")])

    # Overwrite the existing memory in the store
    store.put(namespace, key, {"memory": new_memory.content})
    tool_calls = state['messages'][-1].tool_calls
    return {"messages": [{"role": "tool", "content": "updated instructions", "tool_call_id":tool_calls[0]['id']}]}

# Conditional edge
def route_message(state: MessagesState, config: RunnableConfig, store: BaseStore) -> Literal[END, "update_todos", "update_instructions", "update_profile"]:

    """Route to the memory updater Dotty asked for, or end the turn if it just replied."""
    message = state['messages'][-1]
    if len(message.tool_calls) ==0:
        return END
    else:
        tool_call = message.tool_calls[0]
        if tool_call['args']['update_type'] == "user":
            return "update_profile"
        elif tool_call['args']['update_type'] == "todo":
            return "update_todos"
        elif tool_call['args']['update_type'] == "instructions":
            return "update_instructions"
        else:
            raise ValueError

# Create the graph + all nodes
builder = StateGraph(MessagesState)

builder.add_node(dotty)
builder.add_node(update_todos)
builder.add_node(update_profile)
builder.add_node(update_instructions)

# Define the flow
builder.add_edge(START, "dotty")
builder.add_conditional_edges("dotty", route_message)
builder.add_edge("update_todos", "dotty")
builder.add_edge("update_profile", "dotty")
builder.add_edge("update_instructions", "dotty")

# Compiled without a store/checkpointer for LangGraph Studio, which provides its own
graph = builder.compile()

## Terminal chat

def main():
    sys.stdout.reconfigure(encoding="utf-8")

    # Long-term (across-run) memory lives in SQLite; short-term (this chat) memory in RAM
    with SqliteStore.from_conn_string(str(DB_PATH)) as store:
        store.setup()
        app = builder.compile(checkpointer=MemorySaver(), store=store)
        config = {"configurable": {"thread_id": str(uuid.uuid4()), "user_id": DEFAULT_USER_ID}}

        print("Dotty is ready. /memory shows what I remember, /quit exits.")
        while True:
            try:
                text = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            if text == "/quit":
                break
            if text == "/memory":
                for kind in ("profile", "todo", "instructions"):
                    print(f"[{kind}]")
                    for item in store.search((kind, DEFAULT_USER_ID)):
                        print(f"  {item.value}")
                continue

            try:
                result = app.invoke({"messages": [HumanMessage(content=text)]}, config)
            except Exception as e:
                # A failed turn can leave an unanswered tool call in the thread, which
                # the model API rejects from then on. Start a fresh thread; long-term
                # memory is untouched.
                config["configurable"]["thread_id"] = str(uuid.uuid4())
                print(f"dotty> Something went wrong ({e}). I've reset this conversation; your saved memory is safe.")
                continue
            print(f"dotty> {result['messages'][-1].content}")

if __name__ == "__main__":
    main()
