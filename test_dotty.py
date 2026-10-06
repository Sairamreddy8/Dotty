"""Offline self-check: a scripted fake model stands in for Groq, so this makes no API calls.

Run with `python test_dotty.py` or pytest.
"""

import os

os.environ.setdefault("GROQ_API_KEY", "not-a-real-key")

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph import END
from langgraph.store.memory import InMemoryStore
from trustcall import create_extractor

import dotty


class Scripted(BaseChatModel):
    """Replays `script` one item per call (repeating the last); an Exception item is raised."""
    script: list
    calls: int = 0

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        step = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(step, Exception):
            raise step
        return ChatResult(generations=[ChatGeneration(message=step.model_copy(deep=True))])


def update_memory(update_type):
    return AIMessage(content="", tool_calls=[{"name": "UpdateMemory", "args": {"update_type": update_type}, "id": "u1"}])


def patch_doc(patches):
    args = {"json_doc_id": "ToDoList", "planned_edits": "edit the list", "patches": patches}
    return AIMessage(content="", tool_calls=[{"name": "PatchDoc", "args": args, "id": "p1"}])


def use_model(script):
    """Point Dotty and its ToDo extractor at a scripted model."""
    fake = Scripted(script=script)
    dotty.model = fake
    dotty.todo_extractor = create_extractor(fake, tools=[dotty.ToDoList], tool_choice="ToDoList")
    return fake


MILK = {"task": "buy milk", "time_to_complete": 5, "deadline": None, "solutions": ["corner shop"], "status": "not started"}
ASK_TODO = {"messages": [HumanMessage(content="add milk"), update_memory("todo")]}


def test_route_message():
    assert dotty.route_message({"messages": [AIMessage(content="hi")]}, {}, None) == END
    assert dotty.route_message({"messages": [update_memory("user")]}, {}, None) == "update_profile"
    assert dotty.route_message({"messages": [update_memory("todo")]}, {}, None) == "update_todos"
    assert dotty.route_message({"messages": [update_memory("instructions")]}, {}, None) == "update_instructions"


def test_updates_this_turn():
    # Only tool calls made since the user's last message count
    history = [HumanMessage(content="a"), update_memory("user"), HumanMessage(content="b"), update_memory("todo")]
    assert dotty.updates_this_turn(history) == {"todo"}
    assert dotty.updates_this_turn(history[:3]) == set()


def test_turn_end_to_end():
    # Dotty asks for a todo update, Trustcall patches the (empty) list, then Dotty asks
    # for the same update AGAIN: the guard must make it answer instead of looping.
    fake = use_model([
        update_memory("todo"),
        patch_doc([{"op": "add", "path": "/todos/-", "value": MILK}]),
        update_memory("todo"),
        AIMessage(content="Added buy milk."),
    ])
    store = InMemoryStore()
    out = dotty.builder.compile(store=store).invoke({"messages": [HumanMessage(content="add milk")]})

    assert out["messages"][-1].content == "Added buy milk."
    assert fake.calls == 4
    assert store.get(("todo", dotty.DEFAULT_USER_ID), "ToDoList").value == {"todos": [MILK]}
    # The tool message Dotty saw reports the saved list
    assert "Saved" in out["messages"][2].content and "buy milk" in out["messages"][2].content


def test_several_changes_in_one_update():
    use_model([patch_doc([
        {"op": "replace", "path": "/todos/0/status", "value": "done"},
        {"op": "add", "path": "/todos/-", "value": dict(MILK, task="call mom")},
    ])])
    store, ns = InMemoryStore(), ("todo", "u")
    store.put(ns, "ToDoList", {"todos": [MILK]})

    msg = dotty.run_trustcall(ASK_TODO, store, ns, dotty.ToDoList, dotty.todo_extractor)

    assert msg.startswith("Saved")
    assert [(t["task"], t["status"]) for t in store.get(ns, "ToDoList").value["todos"]] == [
        ("buy milk", "done"), ("call mom", "not started")]


def test_failed_update_is_reported_not_claimed():
    store, ns = InMemoryStore(), ("todo", "u")
    store.put(ns, "ToDoList", {"todos": [MILK]})

    # A patch that changes nothing
    use_model([patch_doc([{"op": "replace", "path": "/todos/0/status", "value": "not started"}])])
    assert "No changes" in dotty.run_trustcall(ASK_TODO, store, ns, dotty.ToDoList, dotty.todo_extractor)

    # A patch that cannot be applied
    use_model([patch_doc([{"op": "replace", "path": "/todos/7/status", "value": "done"}])])
    assert "Nothing was saved" in dotty.run_trustcall(ASK_TODO, store, ns, dotty.ToDoList, dotty.todo_extractor)

    # The model API keeps failing (e.g. rate limit): Trustcall retries, so the retries must be capped
    fake = use_model([RuntimeError("429 rate limit")])
    assert "Nothing was saved" in dotty.run_trustcall(ASK_TODO, store, ns, dotty.ToDoList, dotty.todo_extractor)
    assert fake.calls <= dotty.TRUSTCALL_STEP_LIMIT

    assert store.get(ns, "ToDoList").value == {"todos": [MILK]}


if __name__ == "__main__":
    test_route_message()
    test_updates_this_turn()
    test_turn_end_to_end()
    test_several_changes_in_one_update()
    test_failed_update_is_reported_not_claimed()
    print("ok")
