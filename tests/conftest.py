import sys
import os

# Ensure tools/submission is on the path so `import submission_checker` resolves.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "submission"))
