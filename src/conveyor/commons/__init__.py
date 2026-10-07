"""
A shared, unbounded canvas that agents paint on together, each through a viewport of its own.

  tiles.py    the canvas in the database: tiles, versions, the op log, and a transaction per call
  server.py   one agent's MCP server: the instrument's tools on its viewport, plus look, overview,
              move_viewport, write_message, paint_batch and spawn_successor, counted against the agent's budget
  overview.py the whole canvas shrunk to one picture, for the overview tool
  agent.py    spawning an agent from the catalog, and the process that runs it
  catalog.py  painters: each run's 5 best instrument and prompt pairs, or the seeds on an empty database
  views.py    what the canvas page reads: canvases, agents, tile heads, the replay history
  prompts.py  what an agent reads
"""
