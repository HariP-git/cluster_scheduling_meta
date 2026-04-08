# Updated reward clamping

reward = round(max(0.01, min(0.99, demand / MAX_TASK_DEMAND)), 4)