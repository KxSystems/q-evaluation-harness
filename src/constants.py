"""Configuration constants for the evaluation system."""

# Fixed configuration constants
DEFAULT_TIMEOUT = 5.0
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.95
DEFAULT_SEED = 1234
DEFAULT_MAX_TOKENS = 512
DEFAULT_MAX_THINKING_TOKENS = 2048 * 4
DEFAULT_REASONING_EFFORT = "medium"
MAX_RETRIES = 3

# Execution result constants
EXECUTION_PASSED = "passed"
EXECUTION_TIMED_OUT = "timed out"
EXECUTION_FAILED_PREFIX = "failed: "
