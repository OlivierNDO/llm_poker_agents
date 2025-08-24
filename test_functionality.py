"""
test_functionality.py

Executable script to manually test package functionality
"""
### Imports and configuration
###############################################################################
import os
from dotenv import load_dotenv


load_dotenv()



api_token = os.getenv('NLP_CLOUD_TOKEN')

print(api_token)








