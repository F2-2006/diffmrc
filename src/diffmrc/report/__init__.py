"""Presentation of a finished run, and of the reasons behind it.

This layer takes plain data classes and turns them into something a
person can read and a machine can parse. It knows nothing about how the
numbers were produced: the module may not import ``core``, ``policy`` or
``driver``, which is enforced by an import contract.

That restriction is what makes the report testable on hand-built input.
A report whose correctness could only be checked by running a whole
electronic-structure calculation would not be checked often.
"""
