# LLM Integration for Quadruped Trajectory Optimization

This package implements the iterative refinement pipeline for LLM-based constraint generation as described in the research paper.

## Overview

The system translates natural language commands like "do a backflip" into precise mathematical constraints for trajectory optimization, using an iterative feedback loop with Large Language Models.

## Architecture

```
User Command → LLM Agent → Code Parser → Trajectory Opt → Results Analysis → Feedback Loop
     ↑                                                                            ↓
     └─────────────────── Context Update ←──────────────────────────────────────┘
```

## Components

### 1. **System Prompts** (`prompts.py`)
- Expert roboticist persona for LLM
- CasADi constraint generation templates
- Physical feasibility guidelines
- Feedback prompt templates

### 2. **Safe Code Parser** (`code_parser.py`)
- Secure execution of LLM-generated code
- AST validation for dangerous operations
- CasADi function signature validation
- Constraint feasibility checks

### 3. **Feedback Mechanism** (`feedback.py`)
- Trajectory analysis and metrics extraction
- Success criteria evaluation
- Issue identification and suggestions
- Structured feedback generation for LLM

### 4. **Iterative Refinement Engine** (`iterative_refinement.py`)
- Main orchestration of the refinement loop
- Integration with trajectory optimization
- Performance scoring and best result tracking
- Convergence criteria and termination logic

### 5. **LLM Client** (`llm_client.py`)
- Universal interface for multiple LLM providers
- Support for OpenAI, Anthropic, and local endpoints
- Automatic fallback and error handling
- Mock client for testing

### 6. **Configuration Management** (`config_loader.py`)
- Environment variable and .env file handling
- Multiple LLM provider auto-detection
- Parameter validation and defaults

## Quick Start

### 1. Setup API Keys

Copy the environment template:
```bash
cp .env.template .env
```

Add your API key to `.env`:
```bash
# For OpenAI
OPENAI_API_KEY=your_key_here

# For Anthropic (alternative)
ANTHROPIC_API_KEY=your_key_here

# For local LLM (alternative)
LLM_BASE_URL=http://localhost:8000
```

### 2. Run LLM-Integrated Optimization

```bash
# Basic usage
python llm_main.py "do a backflip"

# With mock LLM for testing
python llm_main.py --mock "jump as high as possible"

# Custom iteration limit
python llm_main.py --max-iterations 15 "do a front flip"
```

### 3. Check Configuration

```bash
python llm_main.py --config-status
```

## Usage Examples

### Basic Commands
```bash
python llm_main.py "do a backflip"
python llm_main.py "jump as high as possible"
python llm_main.py "do a front flip"
python llm_main.py "land on three legs"
```

### Advanced Usage
```python
from llm_integration.iterative_refinement import run_llm_constraint_generation

success, constraint_function, summary = run_llm_constraint_generation(
    command="do a backflip",
    kinodynamic_model=model,
    config=config,
    initial_state=initial_state,
    contact_sequence=contact_sequence,
    reference_trajectory=ref,
    max_iterations=10
)
```

## Configuration Options

Environment variables in `.env`:

```bash
# LLM Provider
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4                    # gpt-4, gpt-3.5-turbo
ANTHROPIC_API_KEY=your_key
ANTHROPIC_MODEL=claude-3-sonnet-20240229

# Generation Parameters
LLM_MAX_TOKENS=2000
LLM_TEMPERATURE=0.1
LLM_TIMEOUT=60

# Iteration Control
MAX_LLM_ITERATIONS=10
TRAJECTORY_OPTIMIZATION_TIMEOUT=120

# Testing
USE_MOCK_LLM=false                   # true for testing without API
```

## Output Structure

```
results/
├── llm_iterations/                   # LLM iteration logs
│   ├── iteration_001.json
│   ├── iteration_002.json
│   └── ...
├── state_traj_llm.npy               # Generated trajectory
├── grf_traj_llm.npy                 # Ground reaction forces
├── joint_torques_traj_llm.npy       # Joint torques
├── planned_traj_llm.mp4             # Planned trajectory video
├── trajectory_llm.mp4               # Simulated trajectory video
└── trajectory_comparison_llm.png    # Comparison plot
```

## Algorithm Flow

The system implements **Algorithm 1** from the research paper:

1. **Initialize**: System prompt + user command
2. **Generate**: LLM produces constraint function
3. **Parse**: Safe execution and validation
4. **Solve**: Trajectory optimization with constraints
5. **Analyze**: Extract metrics and success criteria
6. **Feedback**: Generate structured feedback for LLM
7. **Iterate**: Update context and repeat until success

## Constraint Function Format

LLM generates functions with this signature:

```python
def generated_constraints(x_k, u_k, kinodynamic_model, config, contact_k, k, horizon):
    """
    Args:
        x_k: State at time k (24x1 CasADi SX)
        u_k: Input at time k (24x1 CasADi SX)
        kinodynamic_model: Robot dynamics model
        config: Configuration parameters
        contact_k: Contact flags [FL, FR, RL, RR] (4x1 CasADi SX)
        k: Current time step
        horizon: Total horizon length

    Returns:
        tuple: (constraint_expression, lower_bound, upper_bound)
               or None if no constraint at this time step
    """

    # Example: Terminal rotation constraint for backflip
    if k == horizon:
        target_rotation = 2 * np.pi  # Full backward rotation
        pitch_constraint = x_k[MP_X_BASE_EUL][1] - target_rotation
        return pitch_constraint, -0.1, 0.1  # ±0.1 rad tolerance

    return None
```

## Success Metrics

The system evaluates trajectories on:

- **Optimization Convergence** (30%): Solver successfully finds solution
- **Constraint Satisfaction** (20%): Violations below tolerance
- **Height Clearance** (15%): Sufficient clearance for maneuver
- **Rotation Target** (20%): Achieves desired rotation
- **Landing Stability** (15%): Stable final velocities

## Safety Features

- **Code Sandboxing**: Restricted execution environment
- **AST Validation**: Blocks dangerous imports/operations
- **Input Validation**: Function signature verification
- **Timeout Protection**: Prevents infinite loops
- **Error Recovery**: Graceful handling of failures

## Debugging

### Check Configuration
```bash
python llm_integration/config_loader.py
```

### Test LLM Client
```python
from llm_integration.llm_client import get_llm_client

client = get_llm_client(use_mock=True)
response = client.generate_response("Test prompt")
print(response)
```

### View Iteration Logs
```bash
ls results/llm_iterations/
cat results/llm_iterations/iteration_001.json
```

## Extension Points

- **New Constraints**: Add constraint templates to prompts
- **LLM Providers**: Implement new providers in `llm_client.py`
- **Success Metrics**: Extend evaluation criteria in `feedback.py`
- **Commands**: Add command examples to `prompts.py`

## Troubleshooting

### No API Key Error
```
⚠️ No API key found! Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, or LLM_API_KEY
```
**Solution**: Copy `.env.template` to `.env` and add your API key

### Import Errors
```
ImportError: No module named 'casadi'
```
**Solution**: Install dependencies using the Docker environment

### Solver Failures
```
❌ Trajectory optimization failed
```
**Solution**: Check constraint bounds, increase iteration limit, or simplify command

### LLM Generation Errors
```
❌ Failed to generate constraints
```
**Solution**: Check API key, network connection, or use `--mock` for testing
