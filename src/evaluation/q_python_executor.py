"""Q code execution against Python test assertions."""

import re
import time
import logging
import socket
import subprocess
import threading
import atexit
import os  # Added for os.environ
import signal
import psutil
from typing import Any, Tuple, Optional, Callable
import platform

try:
    if "QHOME" not in os.environ:
        import pykx as kx
        del os.environ['QHOME']
        os.environ["PATH"] += os.pathsep + "$HOME/.kx/bin/"
    else:

        original_qhome = os.environ["QHOME"]
        import pykx as kx
        # Restore QHOME if PyKX modified it
        os.environ["QHOME"] = original_qhome

        # Add platform-appropriate q executable to PATH
        system = platform.system().lower()
        if system == "linux":
            arch = "l64" if platform.machine() in ("x86_64", "amd64") else "l32"
            q_bin_path = os.path.join(original_qhome, arch)
        elif system == "darwin":  # macOS
            arch = "m64" if platform.machine() == "arm64" else "m64"
            q_bin_path = os.path.join(original_qhome, arch)
        elif system == "windows":
            arch = "w64"
            q_bin_path = os.path.join(original_qhome, arch)
        else:
            q_bin_path = os.path.join(original_qhome, "l64")  # fallback

        if os.path.exists(q_bin_path):
            os.environ["PATH"] += os.pathsep + q_bin_path
except ImportError:
    raise ImportError("pykx is required for Q code execution")

from .base import BaseTestExecutor
from ..constants import (
    EXECUTION_PASSED,
    EXECUTION_TIMED_OUT,
    EXECUTION_FAILED_PREFIX,
    MAX_RETRIES,
)

logger = logging.getLogger(__name__)


class QPythonExecutor(BaseTestExecutor):
    """Execute Q code against Python test assertions."""

    def __init__(self) -> None:
        """Initialize the executor with process management."""
        # Track spawned q processes for cleanup
        self._q_processes: list[subprocess.Popen] = []
        # Thread safety for process management
        self._lock = threading.Lock()
        # Track monitoring threads
        self._monitoring_threads: list[threading.Thread] = []
        # Register cleanup handler
        atexit.register(self._cleanup_processes)

    @property
    def supported_languages(self) -> Tuple[str, str]:
        """Return supported language combination."""
        return ("q", "python")

    def _find_available_port(self) -> int:
        """Find an available port for q process."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return int(port)

    def _start_q_process(self, port: int) -> subprocess.Popen:
        """Start a q process on specified port in new process group."""
        try:
            # Set up environment for Q to find q.k file
            env = os.environ.copy()

            # Start q process with specified port and minimal output
            # Use new process group for better cleanup control
            process = subprocess.Popen(
                ["q", "-p", str(port), "-q"],  # -q for quiet mode
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True,
                env=env,  # Use modified environment
                preexec_fn=os.setsid,  # Create new process group
            )

            # Track the process for cleanup
            with self._lock:
                self._q_processes.append(process)

            # Give q process more time to start properly
            time.sleep(1.0)

            # Check if process started successfully
            if process.poll() is not None:
                # Process died, get stderr for debugging
                stdout, stderr = process.communicate(timeout=2)
                logger.debug(f"Q process stdout: {stdout}")
                logger.debug(f"Q process stderr: {stderr}")
                raise RuntimeError(f"Q process failed to start: {stderr}")

            logger.debug(
                f"Started q process on port {port} with PID {process.pid}"
            )
            return process

        except FileNotFoundError:
            raise RuntimeError(
                "q executable not found. Please ensure q/kdb+ is installed "
                "and in PATH"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start q process: {str(e)}")

    def _start_process_monitor(
        self, process: subprocess.Popen, timeout: float
    ) -> threading.Thread:
        """Start background thread to monitor process memory and time."""

        def monitor_process() -> None:
            start_time = time.time()
            memory_limit_mb = 1024  # 1GB memory limit

            try:
                proc = psutil.Process(process.pid)
                while process.poll() is None:
                    current_time = time.time()
                    elapsed = current_time - start_time

                    # Check timeout
                    if elapsed > timeout:
                        logger.warning(
                            f"Process {process.pid} exceeded timeout "
                            f"{timeout}s, force killing"
                        )
                        self._force_kill_process(process)
                        break

                    # Check memory usage
                    try:
                        memory_mb = proc.memory_info().rss / (1024 * 1024)
                        if memory_mb > memory_limit_mb:
                            logger.warning(
                                f"Process {process.pid} exceeded memory "
                                f"limit {memory_mb:.1f}MB, force killing"
                            )
                            self._force_kill_process(process)
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        break  # Process already dead

                    time.sleep(0.1)  # Check every 100ms

            except Exception as e:
                logger.debug(f"Monitor thread error: {e}")

        monitor_thread = threading.Thread(target=monitor_process, daemon=True)
        monitor_thread.start()

        with self._lock:
            self._monitoring_threads.append(monitor_thread)

        return monitor_thread

    def _force_kill_process(self, process: subprocess.Popen) -> None:
        """Forcibly terminate process using escalating signals."""
        if process.poll() is not None:
            return  # Already dead

        pid = process.pid
        logger.debug(f"Force killing process {pid}")

        try:
            # Step 1: SIGTERM to process group (graceful)
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                logger.debug(f"Sent SIGTERM to process group {pid}")
            except (OSError, ProcessLookupError):
                pass

            # Wait 1 second for graceful shutdown
            try:
                process.wait(timeout=1.0)
                logger.debug(f"Process {pid} terminated gracefully")
                return
            except subprocess.TimeoutExpired:
                pass

            # Step 2: SIGKILL to process group (force)
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
                logger.debug(f"Sent SIGKILL to process group {pid}")
            except (OSError, ProcessLookupError):
                pass

            # Final wait
            try:
                process.wait(timeout=2.0)
                logger.debug(f"Process {pid} force killed")
            except subprocess.TimeoutExpired:
                logger.error(f"Failed to kill process {pid}")

        except Exception as e:
            logger.error(f"Error force killing process {pid}: {e}")

    def _cleanup_processes(self) -> None:
        """Clean up all spawned q processes with force kill."""
        with self._lock:
            for process in self._q_processes:
                try:
                    if process.poll() is None:  # Process still running
                        self._force_kill_process(process)
                except Exception as e:
                    logger.warning(f"Error cleaning up q process: {e}")
            self._q_processes.clear()
            self._monitoring_threads.clear()

    def _create_ipc_connection(
        self, port: int, timeout: float
    ) -> kx.SyncQConnection:
        """Create IPC connection to q process with timeout."""
        try:
            # Wait a bit for q process to be ready for connections
            time.sleep(0.5)

            # Create connection with timeout
            conn = kx.SyncQConnection(
                host="localhost", port=port, timeout=timeout
            )
            logger.debug(f"Created IPC connection to localhost:{port}")
            return conn

        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to q process on port {port}: {str(e)}"
            )

    def _create_ipc_function(
        self, conn: kx.SyncQConnection, q_code: str
    ) -> Callable[..., Any]:
        """Create IPC function wrapper that calls remote q function."""
        try:
            # Define function remotely - using the code directly as a function
            func_def = f"testfunc: {q_code}"
            conn(func_def)
            logger.debug("Defined function remotely via IPC")

            # Create wrapper function that converts strings and makes IPC calls
            def ipc_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Convert string arguments to bytes (same as embedded version)
                new_args = tuple(self._to_bytes(arg) for arg in args)
                new_kwargs = {k: self._to_bytes(v) for k, v in kwargs.items()}

                # Call remote function via IPC
                if new_args and new_kwargs:
                    return conn("testfunc", *new_args, **new_kwargs)
                elif new_args:
                    return conn("testfunc", *new_args)
                elif new_kwargs:
                    # Q doesn't support keyword args directly, so this would be
                    # an error
                    raise ValueError(
                        "Q functions don't support keyword arguments"
                    )
                else:
                    return conn("testfunc")

            return ipc_wrapper

        except Exception as e:
            raise RuntimeError(f"Failed to create IPC function: {str(e)}")

    def _execute_with_ipc(
        self, code: str, tests: str, setup_code: str = "", timeout: float = 5.0
    ) -> Tuple[bool, str]:
        """Execute Q code via IPC with reliable timeout and robust error
        handling."""
        process = None
        conn = None
        max_retries = MAX_RETRIES

        try:
            # Process the Q code string
            q_code = code.strip()

            # Handle edge case: empty or invalid code
            if not q_code:
                error_msg = "Empty Q code"
                logger.error(error_msg)
                return False, f"{EXECUTION_FAILED_PREFIX}{error_msg}"

            # Retry logic for port conflicts and startup issues
            for attempt in range(max_retries):
                try:
                    # Start q process with retry logic
                    port = self._find_available_port()
                    logger.debug(
                        f"Attempt {attempt + 1}: Starting q process on port "
                        f"{port}"
                    )
                    process = self._start_q_process(port)

                    # Start monitoring thread for this process
                    self._start_process_monitor(process, timeout + 5)

                    # Create IPC connection with timeout
                    conn = self._create_ipc_connection(port, timeout)

                    # Test the connection with a simple query
                    try:
                        # Simple test to verify connection
                        result = conn("1+1")
                        logger.debug(
                            f"IPC connection test successful: {result}"
                        )
                        break  # Connection successful, exit retry loop
                    except Exception as conn_test_error:
                        logger.warning(
                            f"IPC connection test failed: {conn_test_error}"
                        )
                        raise conn_test_error

                except Exception as startup_error:
                    logger.warning(
                        f"Startup attempt {attempt + 1} failed: "
                        f"{startup_error}"
                    )

                    # Clean up failed attempt
                    if conn:
                        try:
                            conn.close()
                        except Exception:
                            pass
                        conn = None

                    if process:
                        try:
                            if process.poll() is None:
                                process.terminate()
                                process.wait(timeout=2)
                            with self._lock:
                                if process in self._q_processes:
                                    self._q_processes.remove(process)
                        except Exception:
                            pass
                        process = None

                    # If this was the last attempt, raise the error
                    if attempt == max_retries - 1:
                        logger.error(
                            f"Failed to start IPC connection after "
                            f"{max_retries} attempts: {startup_error}"
                        )
                        raise RuntimeError(
                            f"Failed to start IPC connection after "
                            f"{max_retries} attempts: {startup_error}"
                        )

                    # Wait before retrying
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    logger.debug(
                        f"Retrying in {0.5 * (attempt + 1)} seconds..."
                    )

            # Create IPC function wrapper
            func = self._create_ipc_function(conn, q_code)

            # Execute the test function with timing
            start = time.perf_counter()

            # Create namespace with candidate pointing to IPC func
            # and smart_equal for enhanced comparisons
            namespace = {"candidate": func, "smart_equal": self._smart_equal}

            # Execute setup code if provided
            if setup_code.strip():
                exec(setup_code, namespace)

            # Preprocess tests to use smart equality
            processed_tests = self._preprocess_test_for_smart_equality(tests)
            logger.debug("Processed tests for IPC execution")

            # Execute the test function - IPC timeout handles hanging
            try:
                exec(processed_tests, namespace)

                # Call the check function
                if "check" in namespace:
                    check_func = namespace["check"]
                    if callable(check_func):
                        check_func(namespace["candidate"])
                    else:
                        error_msg = "'check' is not callable in tests"
                        logger.error(error_msg)
                        return False, f"{EXECUTION_FAILED_PREFIX}{error_msg}"
                else:
                    error_msg = "No 'check' function found in tests"
                    logger.error(error_msg)
                    return False, f"{EXECUTION_FAILED_PREFIX}{error_msg}"

            except AssertionError as test_error:
                error_msg = f"Test assertion failed: {str(test_error)}"
                logger.debug(error_msg)
                return False, f"{EXECUTION_FAILED_PREFIX}{error_msg}"
            except Exception as test_error:
                # Check if this might be a timeout error
                if (
                    "timeout" in str(test_error).lower()
                    or "time" in str(test_error).lower()
                ):
                    return False, EXECUTION_TIMED_OUT

                import traceback

                error_details = traceback.format_exc()
                logger.debug("IPC test failed with full traceback:")
                logger.debug(error_details)

                error_msg = f"Test execution failed: {str(test_error)}"
                return False, f"{EXECUTION_FAILED_PREFIX}{error_msg}"

            end = time.perf_counter()
            execution_time = end - start
            logger.debug("All IPC tests passed")
            logger.debug(f"Execution time: {execution_time} seconds")
            return True, EXECUTION_PASSED

        except Exception as e:
            # Handle IPC-specific errors with better context
            import traceback

            logger.debug("IPC execution error traceback:")
            logger.debug(traceback.format_exc())

            # Provide more specific error messages
            if "q executable not found" in str(e):
                error_msg = (
                    "Q/KDB+ executable not found. Please ensure q is "
                    "installed and in PATH."
                )
                logger.error(error_msg)
                return False, f"{EXECUTION_FAILED_PREFIX}{error_msg}"
            elif "port" in str(e).lower():
                error_msg = f"IPC port error: {str(e)}"
                logger.debug(error_msg)
                return False, f"{EXECUTION_FAILED_PREFIX}{error_msg}"
            elif "timeout" in str(e).lower() or "time" in str(e).lower():
                return False, EXECUTION_TIMED_OUT
            else:
                error_msg = f"IPC execution error: {str(e)}"
                logger.debug(error_msg)
                return False, f"{EXECUTION_FAILED_PREFIX}{error_msg}"

        finally:
            # Clean up connection and process
            if conn:
                try:
                    conn.close()
                    logger.debug("Closed IPC connection")
                except Exception as e:
                    logger.warning(f"Error closing IPC connection: {e}")

            if process:
                try:
                    if process.poll() is None:  # Process still running
                        process.terminate()
                        process.wait(timeout=2)
                        logger.debug(f"Terminated q process PID {process.pid}")
                    # Remove from tracked processes
                    with self._lock:
                        if process in self._q_processes:
                            self._q_processes.remove(process)
                except Exception as e:
                    logger.warning(f"Error cleaning up q process: {e}")

    def _check_q_available(self) -> None:
        """Check if q executable is available."""
        try:
            # Try to run q with version flag to check availability
            subprocess.run(
                ["q", "-?"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            logger.debug("Q executable availability check passed")
        except FileNotFoundError:
            raise RuntimeError(
                "Q/KDB+ executable not found. Please ensure q is installed "
                "and in PATH."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "Q executable check timed out. Q may not be properly "
                "installed."
            )
        except Exception as e:
            raise RuntimeError(f"Q executable check failed: {str(e)}")

    def execute(
        self, code: str, tests: str, setup_code: str = "", timeout: float = 5.0
    ) -> Tuple[bool, str]:
        """Execute Q code against Python test assertions using IPC.

        Args:
            code: Q code to test
            tests: Python test function with check(candidate) format
            setup_code: Code to execute before tests
            timeout: Maximum execution time in seconds

        Returns:
            Tuple of (success, message) where success is bool and message
            is one of: "passed", "timed out", or f"failed: {error}"
        """
        logger.debug(f"Executing Q code with IPC timeout {timeout}s")

        # Check q executable availability first
        try:
            self._check_q_available()
        except Exception as e:
            logger.error(f"Q availability check failed: {e}")
            return False, f"{EXECUTION_FAILED_PREFIX}{e}"

        return self._execute_with_ipc(code, tests, setup_code, timeout)

    def _to_bytes(self, obj: Any) -> Any:
        """Convert a string to a byte string."""
        try:
            if isinstance(obj, str):
                # Use UTF-8 encoding which is standard for KDB+/Q
                return obj.encode("utf-8")
            elif isinstance(obj, (list, tuple)):
                return type(obj)(self._to_bytes(x) for x in obj)
            elif isinstance(obj, dict):
                return {
                    self._to_bytes(k): self._to_bytes(v)
                    for k, v in obj.items()
                }
            return obj
        except (UnicodeEncodeError, AttributeError) as e:
            logger.warning(f"Failed to encode object {obj}: {e}")
            return obj

    def _extract_function_name_from_assertion(
        self, assertion: str
    ) -> Optional[str]:
        """Extract function name from assertion string."""
        # Pattern to match function calls in assertions:
        # assert function_name(...)
        pattern = r"assert\s+(\w+)\s*\("
        match = re.search(pattern, assertion)
        if match:
            function_name = match.group(1)
            logger.debug(
                f"Extracted function name from assertion: {function_name}"
            )
            return function_name
        logger.debug("Could not extract function name from assertion")
        return None

    def _preprocess_test_for_smart_equality(self, test_str: str) -> str:
        """Replace == operators with smart_equal() function calls using AST.

        Only processes simple assert statements with == comparisons.
        Preserves:
        - assert not statements
        - inequality operators (<, >, !=, <=, >=)
        - variable assignments
        - multi-line expressions
        - complex expressions with multiple operators
        """
        logger.debug("Preprocessing test for smart equality")

        # Only process if we have a check function with == comparisons
        if "def check(" not in test_str or "==" not in test_str:
            logger.debug("Test preprocessing not needed, returning original")
            return test_str

        try:
            import ast

            class EqualityTransformer(ast.NodeTransformer):
                """Transform assert x == y statements to
                assert smart_equal(x, y)."""

                def visit_Assert(self, node: ast.Assert) -> ast.AST:
                    """Visit assert nodes and transform == comparisons."""
                    # Only transform assert statements (not assert not)
                    if (
                        isinstance(node.test, ast.Compare)
                        and len(node.test.ops) == 1
                        and isinstance(node.test.ops[0], ast.Eq)
                        and len(node.test.comparators) == 1
                    ):
                        # Create smart_equal(left, right) function call
                        smart_equal_call = ast.Call(
                            func=ast.Name(id="smart_equal", ctx=ast.Load()),
                            args=[node.test.left, node.test.comparators[0]],
                            keywords=[],
                        )

                        # Replace the test with our function call
                        node.test = smart_equal_call
                        logger.debug(
                            "Transformed assert statement to use smart_equal"
                        )

                    return self.generic_visit(node)

            # Parse the code into AST
            tree = ast.parse(test_str)

            # Apply the transformation
            transformer = EqualityTransformer()
            new_tree = transformer.visit(tree)

            # Convert back to source code
            result = ast.unparse(new_tree)

            if result != test_str:
                logger.debug("Test preprocessed for smart equality")

            return result

        except (SyntaxError, ValueError) as e:
            # If parsing fails, return original (handles malformed code)
            logger.debug(f"Failed to parse test code: {e}, returning original")
            return test_str

    def _smart_equal(self, a: Any, b: Any) -> bool:
        """Robust equality function that handles arrays, lists, and scalars."""
        logger.debug("Performing smart equality comparison")
        logger.debug(f"a: {a}, b: {b}")

        # Convert both to bytes for pykx compatibility
        a = self._to_bytes(a)
        b = self._to_bytes(b)

        # First try direct equality for simple cases
        try:
            result = a == b

            # Handle array-like objects (numpy arrays, etc.) that return
            # boolean arrays
            if hasattr(result, "all") and callable(getattr(result, "all")):
                return bool(result.all())

            # Handle iterables that return lists of booleans
            if hasattr(result, "__iter__") and not isinstance(
                result, (str, bytes)
            ):
                try:
                    return all(result)
                except (TypeError, ValueError):
                    pass

            # If direct equality succeeded, return the result
            if isinstance(result, bool) and result:
                return True

        except Exception as e:
            logger.debug(
                f"Primary equality comparison failed: {e}, trying fallbacks"
            )

        # Handle mixed iterable types (list vs tuple, etc.)
        # But exclude strings and bytes which are also iterable
        if (
            hasattr(a, "__iter__")
            and hasattr(b, "__iter__")
            and not isinstance(a, (str, bytes))
            and not isinstance(b, (str, bytes))
        ):
            try:
                # Convert both to lists for comparison
                list_a = list(a)
                list_b = list(b)

                # Check if they have the same length first
                if len(list_a) != len(list_b):
                    return False

                # For nested structures, recursively apply smart_equal
                for item_a, item_b in zip(list_a, list_b):
                    if not self._smart_equal(item_a, item_b):
                        return False

                return True

            except (TypeError, ValueError, RecursionError):
                pass

        logger.warning("All equality comparison methods failed")
        return False
