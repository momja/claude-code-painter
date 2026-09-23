"""
A small watercolor painting problem wired as a three-node graph (toolkit, agent, critic).

It runs on numpy with no API keys so the observability layer has something real to watch. The agent is a
scripted painter standing in for the LLM agent: its "prompt" is a set of strategy parameters rendered as
text. Swap `paint()` for an LLM tool-use loop and the rest of the graph stays the same.
"""
