"""
A shared, unbounded canvas that agents paint on together, each through a viewport of its own.

  tiles.py    the canvas in the database: tiles, versions, the op log, and a transaction per call
  server.py   one agent's MCP server: the instrument's tools on its viewport, plus look, move_viewport,
              write_message and paint_batch, counted against the agent's tool-call budget
  agent.py    spawning an agent from the catalog, and the process that runs it
  catalog.py  every instrument and painter prompt saved in any run, deduplicated, plus the seeds
  views.py    what the canvas page reads: canvases, agents, tile heads, the replay history
  prompts.py  what an agent reads
"""
