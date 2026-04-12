# OpenEnv Reference Environment Analysis
## Comprehensive Summary of 5 Reference Environments

Generated from examination of: calendar_env, reasoning_gym_env, tbench2_env, repl_env, carla_env

---

## 1. STANDARD DIRECTORY STRUCTURE

Every compliant OpenEnv environment follows this layout:

```
envs/<name>_env/
├── __init__.py              # Package init, re-exports key classes
├── client.py                # EnvClient subclass (client-side)
├── models.py                # Pydantic Action, Observation, State models
├── openenv.yaml             # Environment descriptor
├── pyproject.toml           # Python project metadata + dependencies
├── server/                  # Server-side code (runs in Docker)
│   ├── __init__.py
│   ├── Dockerfile           # Container build instructions
│   ├── app.py               # FastAPI app entry point using create_app()
│   └── <name>_environment.py  # Environment class implementing core logic
├── README.md
└── uv.lock (optional)
```

Key insight: The Dockerfile lives INSIDE server/, not at the env root.

---

## 2. openenv.yaml FORMAT

All environments use identical structure:
```yaml
spec_version: 1
name: <env_name>
type: space
runtime: fastapi
app: server.app:app
port: 8000          # (calendar_env uses 8004)
```

Fields:
- spec_version: Always 1
- name: Environment identifier
- type: Always "space"
- runtime: Always "fastapi"
- app: Always "server.app:app" (module path to FastAPI app)
- port: Default 8000

---

## 3. MODELS (models.py) PATTERN

Each env defines Pydantic models inheriting from OpenEnv base types:

```python
from openenv.core.env_server.types import Action, Observation, State

class MyAction(Action):
    """Fields specific to this env's actions."""
    ...

class MyObservation(Observation):
    """Fields specific to this env's observations."""
    ...
    # Inherited: done: bool, reward: float|None, metadata: dict

class MyState(State):  # Optional - only if needed
    """Server-side state beyond base episode_id + step_count."""
    ...
```

### Comparison of Action/Observation designs:

| Environment | Action Fields | Observation Fields | State |
|------------|---------------|-------------------|-------|
| reasoning_gym | answer: str | question, score, correct_answer, dataset_metadata | Base State |
| tbench2 | action_type, command, session_id, block, wait_seconds, file_path, content | instruction, output, success, error, task_id, session_id, info | task_id, task_path, terminal_ready |
| repl | code: str, is_final: bool, final_answer | result (CodeBlockResult), context_preview, available_variables, iteration, max_iterations | context, task_prompt, iteration, namespace_keys, final_answer |
| carla | action_type (Literal), throttle, steer, brake, lane_direction, target_speed, navigation params | scene_description, speed_kmh, location, rotation, goal_distance, nearby_actors, collision_detected, rubric_reward, camera_image | scenario_name, town, weather, total_distance, collisions, movement metrics |
| calendar | action_type (ListToolsAction/ToolCallAction), tool_name, arguments | success, tools_list, tool_result, done, reward | Base State |

Key observations:
- Action inherits from `Action` which has extra="forbid" (rejects unknown fields) and a `metadata: Dict` field
- Observation inherits from `Observation` which provides `done: bool`, `reward`, and `metadata: Dict`
- State inherits from `State` which provides `episode_id` and `step_count`
- All use pydantic Field() with descriptions for auto-documentation

---

## 4. SERVER APP (server/app.py) PATTERN

The critical pattern is using `create_app()` from OpenEnv:

```python
from openenv.core.env_server.http_server import create_app
from .my_environment import MyEnvironment
from ..models import MyAction, MyObservation

app = create_app(
    MyEnvironment,              # Environment class OR factory function
    MyAction,                   # Action model class
    MyObservation,              # Observation model class
    env_name="my_env",          # Name string
    max_concurrent_envs=1,      # Optional: concurrent session limit
)
```

create_app() automatically generates:
- POST /reset - Reset environment
- POST /step - Execute action
- GET /state - Get current state
- GET /schema - Get action/observation JSON schemas
- GET /health - Health check
- WS /ws - WebSocket for persistent sessions

### Variations:
- **Simple**: Pass class directly (reasoning_gym, echo_env)
- **Factory function**: For env needing config (carla, repl - `def create_environment(): return MyEnv(config...)`)
- **Calendar**: Custom approach - builds its own FastAPI app with lifespan and manually registers OpenEnv routes
- **Tbench2**: Selects environment class based on TB2_MODE env var

---

## 5. ENVIRONMENT CLASS (server/<name>_environment.py) PATTERN

Must extend `Environment` ABC and implement 3 abstract methods:

```python
from openenv.core.env_server.interfaces import Environment

class MyEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True  # or False

    def __init__(self):
        # Initialize state, no super().__init__() needed unless using rubric

    def reset(self, seed=None, episode_id=None, **kwargs) -> MyObservation:
        # Reset env state, return initial observation
        # kwargs receives any extra params from client reset()

    def step(self, action: MyAction) -> MyObservation:
        # Execute action, return observation with done/reward

    @property
    def state(self) -> State:
        # Return current state
```

### Comparison of Environment implementations:

| Env | Concurrency | reset() params | step() logic | Reward mechanism |
|-----|------------|----------------|-------------|-----------------|
| reasoning_gym | True | dataset_name, dataset_config, seed, size | Single-step: score answer, always done=True | score from dataset.score_answer() |
| tbench2 | True | task_id, seed, episode_id | Multi-step: exec commands in terminal, check task completion | Binary success/failure from task checking |
| repl | True | context, task_prompt, expected_answer, hf_token, max_iterations | Multi-step: execute Python code, detect FINAL() | Rubric-based (REPLRubric) + expected_answer matching |
| carla | False(default) | scenario_name, scenario_config | Multi-step: vehicle control in CARLA sim | Rubric-based (CarlaNavigationRubric, CarlaTrolleyRubric) |
| calendar | N/A | database_id, auth_token | MCP tool calls (list_tools, call_tool) | Action success-based |

---

## 6. CLIENT (client.py) PATTERN

Extends `EnvClient[ActionT, ObservationT, StateT]`:

```python
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

class MyEnv(EnvClient[MyAction, MyObservation, MyState]):
    def _step_payload(self, action: MyAction) -> dict:
        # Serialize action to JSON dict
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult[MyObservation]:
        # Deserialize server response to StepResult
        obs = MyObservation(**payload["observation"])
        return StepResult(observation=obs, reward=payload.get("reward"), done=payload.get("done"))

    def _parse_state(self, payload: dict) -> MyState:
        # Deserialize state response
        return MyState(**payload)
```

Usage pattern (from inference scripts):
```python
env = MyEnv(base_url="http://localhost:8000")
# OR
env = MyEnv.from_docker_image("my-env:latest")

result = env.reset(...)    # Returns StepResult with .observation
result = env.step(MyAction(...))  # Returns StepResult
env.close()
```

---

## 7. DOCKERFILE PATTERN

Two patterns observed:

### Standard Pattern (reasoning_gym, tbench2, repl, echo):
Multi-stage build using openenv-base image:
```dockerfile
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder
WORKDIR /app
COPY . /app/env
WORKDIR /app/env
# Install uv, then uv sync dependencies
RUN uv sync --frozen --no-editable
# Final stage
FROM ${BASE_IMAGE}
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"
HEALTHCHECK ...
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
```

### Custom Pattern (calendar_env):
Simple single-stage with pip:
```dockerfile
FROM python:3.11-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8004"]
```

### Heavy Pattern (carla_env):
Full GPU image with CARLA server:
```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
# Install Python, CARLA, system deps
# Complex startup script launching CARLA + FastAPI
```

---

## 8. INFERENCE / AGENT PATTERN

There is NO `inference.py` file within environment packages. Instead, inference scripts live in `examples/`:

Pattern: Agent loop connecting LLM to environment:
```python
# examples/coding_env_inference.py pattern:
from openai import OpenAI
env = CodingEnv.from_docker_image("coding-env:latest")
client = OpenAI(base_url=..., api_key=...)

obs = env.reset().observation
for step in range(MAX_STEPS):
    response = client.chat.completions.create(model=MODEL, messages=history)
    code = extract_code(response)
    result = env.step(CodeAction(code=code))
    if result.observation.done:
        break
    history.append(feedback_message(result))
env.close()
```

---

## 9. WHAT MAKES GOOD RL ENVIRONMENTS

### reasoning_gym:
- Single-step episodes (ask question -> answer -> score)
- Large variety via reasoning_gym library datasets
- Clear reward signal (0.0-1.0 score)
- Good for: supervised fine-tuning, reward model training, basic RL

### tbench2 (Terminal Bench):
- Multi-step terminal interaction
- Real system administration tasks
- Binary success/failure reward
- Downloads real task repo, runs real commands
- Good for: tool use, multi-step reasoning, code generation

### repl_env:
- Multi-step Python code execution
- Supports the Recursive Language Model (RLM) paradigm
- FINAL() pattern for answer extraction
- Rubric-based reward computation
- LLM-in-the-loop capability (llm_query in namespace)
- Good for: code generation RL, recursive reasoning, tool use

### carla_env:
- Continuous control (throttle, steer, brake)
- Rich observation space (scene description, sensors, collisions)
- Multiple scenario types (navigation, trolley problems, free roam)
- Rubric-based rewards (navigation, safety)
- Mock + real CARLA modes
- Good for: embodied AI, continuous control, safety research

### calendar_env:
- MCP (Model Context Protocol) based tool use
- Simulates Google Calendar API
- Rich tool ecosystem (events, calendars, settings)
- Good for: function calling, agentic AI, API interaction training

---

## 10. KEY PATTERNS SUMMARY

1. **Separation of concerns**: client.py (client-side) vs server/ (server-side)
2. **create_app()** is the central factory - handles all HTTP/WS plumbing
3. **Environment ABC** requires: reset(), step(), state property
4. **Pydantic models** for type-safe Action/Observation/State
5. **openenv.yaml** as the manifest file
6. **Docker-first**: Dockerfile in server/ for containerized deployment
7. **WebSocket sessions**: Each client gets isolated env instance
8. **Dual import support**: try/except for in-repo vs standalone pip imports
9. **Rubric system**: Optional reward computation via rubric classes
10. **SUPPORTS_CONCURRENT_SESSIONS**: Class-level flag for multi-session support
