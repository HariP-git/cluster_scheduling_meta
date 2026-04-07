---
description: How to use the Swagger UI to manually schedule tasks locally
---

# 1. Start the Environment server
Execute the following to bring up the FastAPI server natively:
```bash
uv run --project d:\meta\scheduler server
```
Or you can use `uv run uvicorn d:\meta\scheduler.server.app:app`

# 2. Open the Swagger UI
Visit: [http://localhost:8000/docs](http://localhost:8000/docs) in your local web browser.

# 3. Reset the Environment
- Find the `POST /reset` endpoint.
- Click "Try it out" and hit "Execute".
- Note the initialized cluster: It will create exactly 60 machines with roughly 70% capacity pre-utilized.

# 4. Process a Single Task (6 Steps)
To completely parse a task, you must sequence through 6 endpoints sequentially using the `POST /step` API:

**Step 1: Intake (Inject Task)**
Submit the following JSON:
```json
{
  "stage_id": 1,
  "task_difficulty": "medium"
}
```

**Step 2: Profiling**
```json
{
  "stage_id": 2
}
```

**Step 3: Matching**
```json
{
  "stage_id": 3
}
```

**Step 4: Assignment**
```json
{
  "stage_id": 4
}
```

**Step 5: Balancing**
```json
{
  "stage_id": 5
}
```

**Step 6: Monitoring**
```json
{
  "stage_id": 6
}
```

By Step 6, you will be able to see the reward generated and your cluster metrics inside the returned `observation` object. You can now cycle back to Step 1 and provide a "hard" or "easy" `task_difficulty` to repeat the process.
