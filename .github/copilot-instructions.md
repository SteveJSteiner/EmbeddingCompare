For experimental/comparison code:
* Prefer assertions with descriptive messages over try/catch blocks
* Use explicit type checking without fallbacks + log actual vs expected types
* Include TODO comments for any assumption + variable state at assumption point
* Exit immediately on unexpected inputs + dump full context (params/state/location)
* Add trace-level logging for each major step with intermediate values
* Include stack traces AND semantic context ("Failed at model comparison step 2/5")

For running code:
* Use source .venv/bin/activate before running scripts