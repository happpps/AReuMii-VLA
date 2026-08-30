"""Language-command helpers for color cube pick-and-place."""

COLOR_NAMES = {0: "red", 1: "yellow", 2: "green"}

TASK_INSTRUCTIONS = {
    0: "Pick up the red cube and put it in the blue basket.",
    1: "Pick up the yellow cube and put it in the blue basket.",
    2: "Pick up the green cube and put it in the blue basket.",
}

def target_color_to_instruction(color_id: int) -> str:
    return TASK_INSTRUCTIONS[int(color_id)]
