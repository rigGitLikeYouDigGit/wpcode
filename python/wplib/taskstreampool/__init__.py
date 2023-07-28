
"""
Bespoke thread pool-like object, designed for many small, frequently-updated
tasks, such as updating UI elements, propagating user actions in 3d scenes, etc.

Imagine incoming streams of tasks from different sources,
instead of a single queue of tasks to be processed.
"""