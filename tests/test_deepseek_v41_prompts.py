"""DeepSeek-V4.1-Flash prompts through DeepSeek's V4.1 encoder, and its spaced DSML tool calls through the server's parser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorfold.engine.call_gate import call_format
from tensorfold.families.deepseek_v41.prompts import DeepSeekV41Tokenizer, effort_budget, render
from tensorfold.families.deepseek_v41.vendor.encoding_dsv41 import parse_message_from_completion_text
from tensorfold.server.request_options import coerce_effort, effort_levels
from tensorfold.server.text import _CALLS, hide_tool_calls, split_thinking
from tensorfold.server.tools import parse_tool_calls_from_content

CASES = Path(__file__).parent / "fixtures" / "deepseek_v41" / "encoding" / "tests"
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}},
         {"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}]
ONE = ('<｜DSML｜ calls>\n<｜DSML｜ invoke name="bash">\n'
       '<｜DSML｜ parameter name="command" string="true">ls -la /tmp</｜DSML｜ parameter>\n'
       '</｜DSML｜ invoke>\n</｜DSML｜ calls>')
TWO = ('Checking both.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n'
       '<｜DSML｜ parameter name="location" string="true">Beijing</｜DSML｜ parameter>\n'
       '<｜DSML｜ parameter name="days" string="false">3</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n'
       '<｜DSML｜ invoke name="bash">\n<｜DSML｜ parameter name="command" string="true">date</｜DSML｜ parameter>\n'
       '</｜DSML｜ invoke>\n</｜DSML｜ calls>')


@pytest.mark.parametrize("case_id", [1, 2, 3, 4, 5])
def test_render_matches_deepseeks_golden_prompts(case_id):
    """The server's render of each official case (OpenAI shape, tools passed apart) is DeepSeek's own prompt."""

    case = json.loads((CASES / f"test_input_{case_id}.json").read_text())
    case = case if isinstance(case, dict) else {"messages": case}
    got = render(case["messages"], tools=case.get("tools"), thinking=case.get("thinking_mode") == "thinking",
                 reasoning_effort=case.get("reasoning_effort"))
    assert got == (CASES / f"test_output_{case_id}.txt").read_text()


def test_effort_is_the_encoders_budget():
    talk = [{"role": "user", "content": "Hi"}]
    for effort, budget in ((None, 75), ("low", 50), ("medium", 75), ("high", 75), ("xhigh", 100), ("max", 100),
                           ("37", 37)):
        assert f"Reasoning Effort: {budget} (range 1-100" in render(talk, thinking=True, reasoning_effort=effort)
    assert "Reasoning Effort" not in render(talk, reasoning_effort="max")              # chat mode: no budget
    assert effort_budget("none") is None
    for bad in ("0", "101", "huge"):
        with pytest.raises(ValueError):
            effort_budget(bad)
    levels = effort_levels(DeepSeekV41Tokenizer.chat_template)                       # what the server hears
    assert levels == {"low", "high", "max"}
    assert [coerce_effort(e, levels) for e in ("minimal", "medium", "high", "xhigh")] == ["low", "high", "high", "xhigh"]


def test_single_and_parallel_tool_calls_parse():
    content, calls = parse_tool_calls_from_content(ONE, TOOLS)
    assert content == "" and [c["function"]["name"] for c in calls] == ["bash"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "ls -la /tmp"}
    content, calls = parse_tool_calls_from_content(TWO, TOOLS)
    assert content == "Checking both."
    assert [c["function"]["name"] for c in calls] == ["get_weather", "bash"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"location": "Beijing", "days": 3}


def test_malformed_or_mixed_blocks_stay_text():
    for bad in (TWO.replace('"bash"', '"other"'),                                    # a tool not offered
                ONE.replace("</｜DSML｜ invoke>", ""),                                 # an unclosed invoke
                ONE.replace("<｜DSML｜ parameter", "<｜DSML｜parameter"),               # V4 tags inside a V4.1 block
                ONE.replace('string="true"', 'string="maybe"'),
                TWO.replace(">3<", ">three<")):                                       # string="false" that is not JSON
        assert parse_tool_calls_from_content(bad, TOOLS) == (bad.strip(), None)
    v4 = ONE.replace("｜DSML｜ calls", "｜DSML｜tool_calls").replace("｜DSML｜ ", "｜DSML｜")   # V4's dialect still parses
    assert [c["function"]["name"] for c in parse_tool_calls_from_content(v4, TOOLS)[1]] == ["bash"]


def test_streaming_hides_the_block_and_its_partial_opener():
    assert hide_tool_calls(TWO, finished=True).strip() == "Checking both."
    for k in range(len("Checking both.\n\n"), len(TWO)):
        shown = hide_tool_calls(TWO[:k], finished=False)
        assert "DSML" not in shown and "<｜" not in shown and "Checking both.".startswith(shown.strip()[:14])
    reasoning, answer = split_thinking("why</think>" + ONE, finished=True)
    assert reasoning == "why" and answer == ONE


def test_tool_call_history_round_trips():
    """Calls parsed from a reply, sent back as history with their results, render as DeepSeek writes them and parse again."""

    _, calls = parse_tool_calls_from_content(TWO, TOOLS)
    talk = [{"role": "user", "content": "Weather and date?"},
            {"role": "assistant", "content": "Checking both.", "tool_calls": calls},
            {"role": "tool", "tool_call_id": calls[0]["id"], "content": "sunny"},
            {"role": "tool", "tool_call_id": calls[1]["id"], "content": "Sat Oct 11"}]
    text = render(talk, tools=TOOLS, thinking=True)
    assert TWO.split("\n\n", 1)[1] in text                                            # the block exactly as generated
    assert "<tool_result>sunny</tool_result>\n\n<tool_result>Sat Oct 11</tool_result><｜Assistant｜><think>" in text
    head = "<｜Assistant｜><think>"
    reply = text[text.index(head) + len(head):text.index("<｜end▁of▁sentence｜>")]
    parsed = parse_message_from_completion_text(reply + "<｜end▁of▁sentence｜>", thinking_mode="thinking")
    assert [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in parsed["tool_calls"]] == \
        [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls]


def test_tokenizer_switches_and_call_probe():
    class Chars:
        def encode(self, text, add_special_tokens=True):
            return [ord(c) for c in text]

    tok = DeepSeekV41Tokenizer(Chars())
    talk = [{"role": "user", "content": "Hi"}]
    assert tok.apply_chat_template(talk, tokenize=False, enable_thinking=True).endswith("<｜Assistant｜><think>")
    assert tok.apply_chat_template(talk, tokenize=False).endswith("<｜Assistant｜></think>")
    assert not tok.apply_chat_template(talk, tokenize=False, add_generation_prompt=False).endswith("</think>")
    probe = render([{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_0", "type": "function", "function": {"name": "tfprobe_fn", "arguments": {}}}]}])
    assert call_format(probe, "tfprobe_fn", [o for o, _ in _CALLS])[0] == "<｜DSML｜ calls>"
