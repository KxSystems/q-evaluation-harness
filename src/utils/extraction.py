"""Q function extraction utilities."""

import logging

logger = logging.getLogger(__name__)


def extract_q_code_block(content: str) -> str | None:
    """Extract the first Q code block from content.
    
    Args:
        content: Content that may contain ```q code blocks
        
    Returns:
        Content of the first Q code block, or None if not found
    """
    lines = content.split("\n")
    q_block_content = []
    in_q_block = False
    
    for line in lines:
        stripped_line = line.strip()
        
        if stripped_line == "```q":
            in_q_block = True
            continue
        elif stripped_line == "```" and in_q_block:
            # End of Q block found
            if q_block_content:
                return "\n".join(q_block_content)
            return None
        elif in_q_block:
            q_block_content.append(line)
    
    # If we reached end without closing ```, return what we have
    if q_block_content:
        return "\n".join(q_block_content)
    
    return None


def extract_function_from_lines(content: str, function_name: str) -> str | None:
    """Extract function from content using brace counting logic.
    
    Args:
        content: Content to search for function
        function_name: Name of the function to extract
        
    Returns:
        Extracted function body, or None if not found
    """
    lines = content.split("\n")
    function_body = ""
    brace_count = 0
    capturing = False

    for line in lines:
        stripped_line = line.strip()

        # Skip empty lines and comments when not capturing
        if not capturing and (
            not stripped_line or stripped_line.startswith("//")
        ):
            continue

        # Skip code block markers
        if stripped_line.startswith("```"):
            continue

        # Look for function definition start
        if not capturing and function_name + ":" in stripped_line:
            # Find the opening brace
            brace_start = stripped_line.find("{")
            if brace_start != -1:
                capturing = True
                # Extract from the opening brace onwards
                function_part = stripped_line[brace_start:]
                function_body = function_part
                brace_count = function_part.count("{") - function_part.count(
                    "}"
                )

                # If braces are balanced in this line, function is complete
                if brace_count == 0:
                    return function_body
                continue

        # If we're capturing, add the line and track braces
        if capturing:
            if stripped_line:  # Skip empty lines
                function_body += "\n" + line  # Preserve original indentation
                brace_count += stripped_line.count("{") - stripped_line.count(
                    "}"
                )

                # If braces are balanced, function is complete
                if brace_count == 0:
                    return function_body

    # If we captured something but braces aren't balanced, return what we have
    if function_body:
        return function_body

    # Fallback: look for any function-like pattern with braces
    for line in lines:
        line = line.strip()
        if "{" in line and "}" in line and function_name in line:
            # Try to extract a complete one-liner function
            start_idx = line.find("{")
            end_idx = line.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                return line[start_idx:end_idx + 1]

    return None


def extract_function_from_content(content: str, function_name: str) -> str:
    """Extract a single complete Q function implementation from { to }.

    Args:
        content: Generated content that may contain function definition
        function_name: Name of the function to extract (from entry_point)

    Returns:
        Extracted function body including braces, or original content if
        extraction fails
    """
    logger.debug(f"Extracting function '{function_name}' from content")
    
    # Phase 1: Try to extract from Q code block first
    q_block = extract_q_code_block(content)
    if q_block:
        logger.debug("Found Q code block, attempting function extraction within it")
        extracted_function = extract_function_from_lines(q_block, function_name)
        if extracted_function:
            logger.debug("Function extraction successful from Q block")
            return extracted_function
        logger.debug("Function not found in Q block, falling back to full content")
    else:
        logger.debug("No Q code block found, using full content")
    
    # Phase 2: Fallback to original logic on full content
    extracted_function = extract_function_from_lines(content, function_name)
    if extracted_function:
        logger.debug("Function extraction successful from full content")
        return extracted_function
    
    # Phase 3: Final fallback - return entire content
    logger.debug("Function extraction failed, returning entire content")
    return content.strip()
