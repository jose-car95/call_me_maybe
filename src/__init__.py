"""Application package for the call_me_maybe function-calling tool.

The package intentionally exposes no public shortcuts from this module.
Import concrete components from their own modules to keep startup predictable
and avoid loading model-related dependencies unless the CLI requests them.
"""
