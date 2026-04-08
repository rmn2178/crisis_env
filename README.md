# Crisis Response Env

## What it does
The AI Crisis Response & Rescue Coordination environment is an OpenEnv-compliant RL environment where agents must triage, prioritize, allocate resources, and coordinate rescue operations across simultaneous crisis events to maximize lives saved. The agent receives events such as airstrikes, ship attacks, and drone threats, and must allocate limited resources while dealing with time constraints and potential cascading failures.

## Quick start
```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# 3. Check health
curl http://localhost:7860/health

# 4. Run baseline inference (in another terminal)
export API_BASE_URL="https://api.openai.com/v1" # or your preferred endpoint
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your_token_here"
python inference.py
```

## Running with Docker
```bash
# Build the image
docker build -t crisis-response-env -f server/Dockerfile .

# Run the container
docker run -p 7860:7860 crisis-response-env
```

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health check and available tasks |
| `GET` | `/tasks` | Detailed list of available tasks |
| `POST` | `/reset` | Start a new episode (`{"task_id": "easy"}`) |
| `POST` | `/step` | Submit an action and get the next observation |
| `GET` | `/state` | Retrieve current episode state without advancing |
| `GET` | `/` | Redirects to Swagger UI (`/docs`) |
| `GET` | `/docs` | Interactive API documentation |

## Action space
The `/step` endpoint expects a JSON payload matching the `CrisisAction` model:

```json
{
  "action_type": "allocate",
  "threat_id": "THR-001",
  "resource_id": "RES-fighter_jet-01",
  "priority_level": "CRITICAL",
  "reasoning": "Deploying fighter jet to intercept incoming airstrike."
}
```

| Field | Type | Description |
|---|---|---|
| `action_type` | `str` | Must be one of: `classify`, `predict`, `allocate`, `rescue` |
| `threat_id` | `str` | The exact ID of the threat from the observation |
| `resource_id` | `str?` | Required for `allocate` and `rescue`. Must be a valid resource ID |
| `priority_level` | `str?` | Must be one of: `CRITICAL`, `HIGH`, `MEDIUM`, `LOW` |
| `reasoning` | `str?` | Optional string for logging capability |

## Observation space
The `/reset` endpoint and the `observation` field of the `/step` response use the following schema:

```json
{
  "threat_id": "THR-001",
  "threat_type": "AIRSTRIKE",
  "location": "Military Base Alpha",
  "severity": "CRITICAL",
  "population_at_risk": 50,
  "time_to_impact": 30,
  "available_resources": ["RES-fighter_jet-01", "RES-naval_vessel-01"],
  "threats_remaining": 5,
  "last_action_result": "Episode started. Assess the first threat.",
  "cumulative_score": 0.0,
  "done": false
}
```

| Field | Type | Description |
|---|---|---|
| `threat_id` | `str` | The ID of the threat to act upon |
| `threat_type` | `str` | Type of emergency: `AIRSTRIKE`, `SHIP_ATTACK`, `DRONE_THREAT` |
| `location` | `str` | A text description of the threat location |
| `severity` | `str` | The true or perceived severity level |
| `population_at_risk`| `int` | Number of people in immediate danger |
| `time_to_impact`| `int` | Seconds remaining before event escalation |
| `available_resources`| `list[str]`| Resources that have not yet been assigned |
| `threats_remaining` | `int` | Threats left before episode terminates |
| `last_action_result`| `str` | Feedback on the last action taken |
| `cumulative_score`| `float`| Current total reward |
| `done` | `bool`| True if the episode is finished |

## Tasks

| Task ID | Difficulty | Size | Description | Expected Score |
|---|---|---|---|---|
| `easy` | easy | 5 | Single-Threat Classification. No resource conflicts. | ~0.80 |
| `medium` | medium | 10 | Multi-Threat Coordination. Hidden cascades and prioritization needed. | ~0.55 |
| `hard` | hard | 15 | Full Lifecycle Triage + Rescue. Tight time constraints, resource scarcity. | ~0.30 |

## Reward function

Reward is dense and is returned per-step, clipped to `[0.0, 1.0]`.

| Event | Reward |
|---|---|
| Correct Priority | `+0.50` |
| Correct Resource | `+0.30` |
| Rescue Speed Bonus (within 3 steps) | `+0.20` |
| Wrong `threat_id` | `-0.10` penalty |
| Invalid `action_type` / `priority_level` | `-0.20` penalty |
| Missed `CRITICAL` threat  | `-0.20` penalty |

## Grader formulas
All graders are stateless functions located in `graders.py`, evaluating the episode after all steps are complete constraint against ground truth records.

- **Easy**: `score = correct_priority_assignments / total_threats`
- **Medium**: `score = (0.60 * priority_accuracy) + (0.40 * resource_accuracy)`
- **Hard**: `score = (0.40 * priority_accuracy) + (0.30 * resource_accuracy) + (0.30 * rescue_speed_score) - critical_miss_penalty`

## Environment variables

| Variable | Description | Default |
|---|---|---|
| `API_BASE_URL` | Base URL for LLM inferences | `https://generativelanguage.googleapis.com/...` |
| `MODEL_NAME` | Model ID for inference | `gemini-1.5-flash` |
| `HF_TOKEN` | API token for inference | `""` |
| `ENV_URL` | URL of the running server | `http://localhost:7860` |

## Baseline scores

| Task | Score |
|---|---|
| `easy` | `0.760` |
| `medium` | `0.510` |
| `hard` | `0.280` |

## Project structure

```text
crisis_response_env/
├── __init__.py
├── client.py
├── models.py
├── scenario_generator.py       
├── graders.py
├── openenv.yaml
├── inference.py                
├── pyproject.toml
├── README.md
├── .gitignore
└── server/
    ├── __init__.py
    ├── app.py
    ├── environment.py
    ├── crisis_response_env_environment.py   
    ├── requirements.txt
    ├── Dockerfile
    ├── data/
    │   └── README.md
    └── tests/
        ├── test_environment.py
        └── test_app.py
```

## Running tests

```bash
pytest server/tests/ -v
```

## Deploying to HuggingFace Spaces

```bash
# Validate using openenv CLI
openenv validate

# Push code to your HuggingFace Spaces repo
git push https://huggingface.co/spaces/<your_username>/<your_space_name>
```

## Pre-submission checklist

- [x] `docker build` succeeds using specified base image
- [x] `docker run` starts server on `7860`
- [x] `curl localhost:7860/health` returns ok
- [x] `curl /reset` returns observations
- [x] `curl /step` returns step result
- [x] `openenv validate` passes
- [x] `python inference.py` completes natively and logs correctly
- [x] Custom Graders implement logic that rewards different approaches
- [x] Tests pass successfully

## Limitations
None.

## Hackathon context
Created for the Meta PyTorch OpenEnv Hackathon (India, April 2026).
