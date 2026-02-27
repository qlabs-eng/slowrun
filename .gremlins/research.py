import asyncio
import json


async def program(ctx, spec="", **kwargs):
    """Parallel hypothesis exploration for slowrun.

    Spawns a surveyor agent to read the slowrun codebase, analyze the
    training pipeline, and propose concrete experiments. Then spawns
    multiple experimenter agents in parallel, each implementing and
    testing one proposed modification. Collects all results at the end.
    """

    num_experiments = int(kwargs.get("num_experiments", 3))

    SURVEYOR_PROMPT = f"""\
You are a research surveyor for the NanoGPT Slowrun benchmark. Your job is to deeply \
read the training code, understand every design choice, and propose {num_experiments} concrete \
experiments that could lower validation loss.

## Background

NanoGPT Slowrun trains a 2.7B parameter transformer on 100M tokens from FineWeb with \
unlimited compute. The goal is the lowest possible validation loss. The baseline achieves \
3.402 val loss in ~47 minutes on 8xH100. Key features of the current baseline:

- Muon optimizer for matrix params, AdamW for embeddings/scalars
- Heavy weight decay (1.6) and dropout (0.1) for regularization
- Flash Attention 3 with SSSL windowed attention pattern
- Value embeddings (ResFormer) on alternating layers
- Softcap logits (15 * tanh(logits/15))
- Squared ReLU activation in MLP
- RoPE positional embeddings
- 12 epochs, 50% warmdown, no warmup
- Cautious weight decay (only when gradient and param have same sign)

## Your task

1. Read every file in the repo. Understand the model architecture, optimizer, data pipeline, \
and training loop in detail. Pay special attention to:
   - `train.py` (main training script, ~846 lines)
   - `prepare_data.py` (data preprocessing)
   - `ensemble/train.py` (ensemble training, may have ideas worth borrowing)
   - `limited/train.py` (limited compute track)
   - `README.md` (benchmark rules and baseline description)

2. Identify {num_experiments} promising research directions. Each must be:
   - A concrete, implementable modification to `train.py`
   - Motivated by a clear hypothesis about why it would help
   - Different from the others (explore diverse axes: architecture, optimization, \
regularization, data handling, training schedule, etc.)
   - Testable within a single training run

3. For each experiment, think about what the right evaluation strategy is. Can we run \
a quick ablation (fewer epochs, smaller model) to get a signal, or does the modification \
only make sense at full scale?

## Output format

Call submit with a JSON summary in exactly this format:

```json
{{
  "experiments": [
    {{
      "name": "short-kebab-case-name",
      "hypothesis": "One sentence: what you expect and why.",
      "modification": "Detailed description of what to change in train.py.",
      "eval_strategy": "How to test this: full run, reduced epochs, smaller model, etc.",
      "risk": "What could go wrong or why this might not help."
    }}
  ]
}}
```

Do not include any text outside the JSON block. The JSON must parse cleanly.

## Research spec

{spec if spec else "Open-ended: propose the most promising experiments you can find for lowering val loss."}
"""

    EXPERIMENTER_PROMPT_TEMPLATE = """\
You are a research experimenter for the NanoGPT Slowrun benchmark. You have been assigned \
one specific experiment to implement and test.

## Background

NanoGPT Slowrun trains a 2.7B parameter transformer on 100M tokens from FineWeb. The goal \
is the lowest possible validation loss. The baseline achieves 3.402 val loss. The repo is \
already cloned at /home/agent/repo.

## Your experiment

Name: {name}
Hypothesis: {hypothesis}
Modification: {modification}
Eval strategy: {eval_strategy}
Risk: {risk}

## Your task

1. Read `train.py` carefully. Understand the section you are modifying.

2. Create a new branch: `research/{name}`.

3. Implement the modification. Be surgical -- change only what is necessary for this experiment. \
Do not refactor unrelated code. Add a comment at each change site: `# EXPERIMENT: {name}`.

4. Verify the code compiles and runs. At minimum:
   - `python -c "import train"` should not crash (or handle the distributed init gracefully)
   - If GPUs are available, run a quick training with reduced settings:
     `torchrun --standalone --nproc_per_node=1 train.py --num-epochs 1 --device-batch-size 1 --n_layer 4 --n_head 4 --n_embd 256`
   - If no GPUs, verify the code parses and the logic is correct by reading through it.

5. If you can run training, record the val loss and compare against baseline at the same \
reduced scale. Run the baseline config too if time allows.

6. Commit your changes with a message like: `research: {name} - [brief description]`.

7. Push the branch and open a PR with your findings.

8. Call submit with a structured summary:

```
## Experiment: {name}

### Hypothesis
{hypothesis}

### Changes made
[List of specific code changes with file paths and line numbers]

### Results
[Val loss numbers if you ran training, or "code-only -- no GPU available"]
[Baseline comparison if available]

### Analysis
[Did the results support the hypothesis? What did you learn?]
[Any unexpected behavior or observations?]

### Next steps
[What would you try next based on these results?]
```

Do not give up if something breaks. Fix it and try again. The experiment is only done when \
you have either produced results or conclusively determined the modification cannot work.

## Research spec

{spec}
"""

    # -- Phase 1: Survey --

    surveyor = await ctx.agent("surveyor", model="claude", prompt=SURVEYOR_PROMPT, git="read")

    @surveyor.on("submit")
    async def on_surveyor_submit(summary=""):
        # Parse the surveyor's proposed experiments
        experiments = []
        try:
            # Handle both raw JSON and JSON wrapped in markdown code blocks
            clean = summary.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(clean)
            experiments = parsed.get("experiments", [])[:num_experiments]
        except (json.JSONDecodeError, KeyError, AttributeError):
            # If JSON parsing fails, fall back: give each experimenter the raw summary
            # and let them pick their own direction
            experiments = [
                {
                    "name": f"experiment-{i+1}",
                    "hypothesis": "See surveyor notes below.",
                    "modification": f"Pick experiment #{i+1} from the surveyor's proposals and implement it.",
                    "eval_strategy": "Run at reduced scale if GPUs available.",
                    "risk": "Unknown.",
                }
                for i in range(num_experiments)
            ]

        # -- Phase 2: Experiment (parallel) --

        results = {}
        total = len(experiments)

        async def spawn_experimenter(exp):
            prompt = EXPERIMENTER_PROMPT_TEMPLATE.format(
                name=exp["name"],
                hypothesis=exp["hypothesis"],
                modification=exp["modification"],
                eval_strategy=exp["eval_strategy"],
                risk=exp["risk"],
                spec=spec or "(open-ended research)",
            )
            agent_name = f"exp-{exp['name']}"
            agent = await ctx.agent(agent_name, model="claude", prompt=prompt, git="write")

            @agent.on("submit")
            async def on_exp_submit(summary=""):
                results[exp["name"]] = {
                    "hypothesis": exp["hypothesis"],
                    "summary": summary,
                }
                if len(results) == total:
                    ctx.done({
                        "surveyor_proposals": experiments,
                        "experiment_results": results,
                    })

        await asyncio.gather(*(spawn_experimenter(exp) for exp in experiments))
