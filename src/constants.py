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

# Agent evaluation defaults
DEFAULT_AGENT_TIMEOUT = 300.0
DEFAULT_AGENT_MAX_TURNS = 20
DEFAULT_AGENT_CONCURRENCY = 4

# Execution result constants
EXECUTION_PASSED = "passed"
EXECUTION_TIMED_OUT = "timed out"
EXECUTION_FAILED_PREFIX = "failed: "
# Infrastructure error (NOT a wrong answer): the solution could not be scored
# because the q process/license/IPC could not be brought up. Excluded from
# pass/fail so a transient licensing/startup hiccup is never counted as a
# failing solution.
EXECUTION_ERRORED_PREFIX = "errored: "

# q process startup is subject to a transient license-acquisition race under
# rapid spawn/teardown churn ("license error: no license loaded"), so startup
# is retried more times and with a longer backoff than a normal op.
Q_STARTUP_MAX_RETRIES = 6
Q_STARTUP_BACKOFF_SECONDS = 1.0
# Substrings (lowercased) that mark an infra/startup error rather than a
# genuine test failure.
Q_INFRA_ERROR_MARKERS = (
    "license",
    "no license loaded",
    "failed to start q process",
    "q process failed to start",
    "failed to start ipc connection",
)
