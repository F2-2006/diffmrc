"""Flow control: what to run, in what order, and when to refuse.

The driver owns the state machine -- start cheap, escalate when a
self-check fails, refuse rather than emit an untrustworthy number. It
calls into policy for decisions and into core for numerics, and it is the
only layer that knows about the sequence.
"""
