
# Maximum estimated API cost per workflow, in US dollars.
# 0.01 means one US cent.
MAX_REQUEST_COST_USD = 0.01

# Maximum input tokens we plan for per request.
# This is a planning assumption, not a tokenizer-enforced limit.
MAX_ESTIMATED_INPUT_TOKENS = 1000

# Maximum output tokens we plan for per provider attempt.
# This must later be passed to each provider's API request.
MAX_OUTPUT_TOKENS = 300

# Conservative assumed cost per one million input tokens.
# These are example planning rates, not live provider prices.
PLANNING_INPUT_USD_PER_MILLION = 5.00

# Conservative assumed cost per one million output tokens.
PLANNING_OUTPUT_USD_PER_MILLION = 15.00

# Allow for at most two paid provider attempts.
MAX_PROVIDER_ATTEMPTS = 2
