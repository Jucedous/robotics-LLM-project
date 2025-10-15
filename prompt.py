# List of things to change/add in analyze safety prompt
# 1. tell LLM to not make things up
# 2. safety score
# 3. ethical score?
# 4. robot operation procedure
# 5. backwards check
API_KEY = "[REDACTED_OPENAI_KEY]"
PROMPT_NAME = "hi whats your name?"
PROMPT_HI = "hi how are you doing?"
PROMPT_ANALYZE_SAFETY_OBJECTS = (
    "Analyze the following situation: You are given a list of objects, each with their 3D coordinates in space, provided in the following JSON format:\n"
    "{objects_json}\n"
    "Based on their positions and possible interactions, determine if there is any safety risk or non-ethical scenario present. "
    "Please explain your reasoning and clearly state if any safety or ethical concerns exist."
    "If there are no safety or ethical concerns, explicitly state that there are none and do not make up any issues."
)

PROMPT_ANALYZE_SAFETY_OBJECTS_AND_OPERATION = (
    "Analyze the following situation: You are given a JSON file containing a list of objects (with their 3D coordinates and possible safety risks) and a record of a robot arm operation.\n"
    "{objects_json}\n"
    "First, analyze the objects for any safety or ethical concerns based on their positions and properties. "
    "Second, analyze the robot operation steps for any potential safety or ethical risks, considering interactions between the robot and the objects.\n"
    "If there are any safety or ethical concerns, clearly explain your reasoning. "
    "If there are no safety or ethical concerns, explicitly state that there are none and do not make up any issues."
)

PROMPT_ANALYZE_SAFETY_OBJECTS_AND_OPERATION_W_SCORE = (
    "Analyze the following situation: You are given a JSON file containing a list of objects (with their 3D coordinates and possible safety risks) and a record of a robot arm operation.\n"
    "{objects_json}\n"
    "First, analyze the objects for any safety or ethical concerns based on their positions and properties. "
    "Second, analyze the robot operation steps for any potential safety or ethical risks, considering interactions between the robot and the objects.\n"
    "If there are any safety or ethical concerns, clearly explain your reasoning. "
    "If there are no safety or ethical concerns, explicitly state that there are none and do not make up any issues. "
    "Finally, provide a safety score for the robot operation on a scale from 1 (very unsafe) to 5 (very safe), and briefly justify your score."
)

PROMPT_V1 = (
    "Analyze the following situation: You are given a JSON file containing a list of objects "
    "(with their 3D coordinates and possible safety risks) and a record of a robot arm operation.\n"
    "{objects_json}\n\n"
    "Your response MUST strictly follow this format:\n"
    "1. Objects Analysis: [Clear and concise analysis of objects, highlighting any safety/ethical concerns. "
    "If none, explicitly state 'No safety or ethical concerns.']\n"
    "2. Robot Operation Analysis: [Analysis of robot operation steps, highlighting any potential safety/ethical risks "
    "from interactions with objects. If none, explicitly state 'No safety or ethical concerns.']\n"
    "3. Safety Score: [Provide a single number from 1 (very unsafe) to 5 (very safe), followed by a one-sentence justification.]\n\n"
    "Do not provide extra commentary or deviate from this structure."
)