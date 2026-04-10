"""Task registry for benchmarked trading scenarios."""

from .task1_easy import TASK as TASK_EASY
from .task2_medium import TASK as TASK_MEDIUM
from .task3_hard import TASK as TASK_HARD

TASK_REGISTRY = {
    TASK_EASY["id"]: TASK_EASY,
    TASK_MEDIUM["id"]: TASK_MEDIUM,
    TASK_HARD["id"]: TASK_HARD,
}

TASK_CONFIGS = [
    {
        "task_id": task["id"],
        "difficulty": task["difficulty"],
        "target_steps": task["target_steps"],
        "name": task["name"],
        "description": task["description"],
    }
    for task in (TASK_EASY, TASK_MEDIUM, TASK_HARD)
]
