# How to Add a New Dataset

This guide explains how to register new datasets for evaluation with the Q language harness.

## Dataset Format Requirements

All datasets must follow this exact schema with the following required columns:

```json
{
    "task_id": 0,
    "prompt": "// @overview\n// Your Q function description here\n//\n// @param x {type} Parameter description\n// @return {type} Return description\nfunction_name:{[x]\n    / function body\n    }",
    "tests": "def check(candidate):\n    assert candidate(input) == expected_output\n    assert candidate(input2) == expected_output2",
    "q_tests": [],
    "entry_point": "function_name",
    "test_setup_code": "",
    "canonical_solution": ""
}
```

### Required Fields:
- **`task_id`**: Unique integer identifier for each problem
- **`prompt`**: Q code template with function signature and documentation
- **`tests`**: Python test assertions using `def check(candidate):` format
- **`entry_point`**: Function name to be tested (must match prompt)
- **`q_tests`**: Array for future Q-native tests (keep empty for now)
- **`test_setup_code`**: Setup code for tests (can be empty string)
- **`canonical_solution`**: Reference solution (can be empty string)

## Supported File Formats

- **JSONL** (recommended): One JSON object per line
- **JSON**: Single JSON array (auto-detected)

## Registration Methods

### Method 1: Automatic Detection (Recommended)
Just use your dataset file path directly with the CLI:

```bash
qeval execute --dataset ./path/to/your_dataset.jsonl solutions.jsonl
```

The system automatically detects:
- File format (based on extension and content sampling)
- Schema type (based on column names)
- Language settings (defaults to Q→Python testing)

### Method 2: Manual Registration
For permanent registration, add to `src/datasets/registry.py`:

```python
DATASET_CONFIGS["your-dataset-name"] = {
    "path": "./datasets/your_dataset.jsonl",
    "format": "jsonl",
    "schema": "q_humaneval",  # or "q_mbpp", "generic"
    "language": "q",
    "test_language": "python",
    "prompt_template": "q_humaneval"
}
```

Then use with:
```bash
qeval execute --dataset your-dataset-name solutions.jsonl
```

## Testing Your Dataset

1. Place your dataset file in the `./datasets/` directory
2. Test with a small sample first:
   ```bash
   qeval execute --dataset ./datasets/your_dataset.jsonl test_solutions.jsonl
   ```
3. Verify all problems load correctly and tests execute

## Current Limitations

- **Metrics**: Only pass@k evaluation supported currently
- **Test Language**: Tests must be written in Python using assert statements  
- **Future**: LLM-as-judge and custom metrics planned for future releases

## Submit to Leaderboard

To include your dataset in the official Q evaluation leaderboard, submit a pull request with:

1. **Your dataset file** in the `./datasets/` directory
2. **Dataset submission form** (`DATASET_SUBMISSION.md`) with the following information:

### Dataset Submission Template

Create a file named `DATASET_SUBMISSION.md` with this template:

```markdown
# Dataset Submission: [Dataset Name]

## Basic Information
- **Name**: [Your dataset name]
- **Description**: [2-3 sentence description of what this dataset tests]
- **Size**: [Number of problems]
- **Domain**: [e.g., algorithms, data structures, math, string processing]

## Dataset Details
- **Difficulty Level**: [Beginner/Intermediate/Advanced]
- **Problem Types**: [Brief description of problem categories]

## Data Source & Quality
- **Original Source**: [Where did the problems come from? Original creation, adapted from X, etc.]
- **License**: [MIT, Apache 2.0, CC BY, etc.]
- **Quality Assurance**: [How were problems validated? Manual review, automated testing, etc.]
- **Test Coverage**: [How comprehensive are the test cases?]

## Maintainer
- **Contact**: [GitHub username or email]
- **Organization**: [Optional: company/institution]

## Additional Notes
[Any special considerations, limitations, or context reviewers should know]
```

### Review Process

1. Submit PR with dataset + submission form
2. Manual review for quality, appropriateness, and schema
3. Approval and inclusion in leaderboard

**Note**: Datasets must follow our schema requirements and pass basic quality checks to be accepted.