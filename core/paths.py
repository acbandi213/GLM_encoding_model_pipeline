"""
paths.py
========
Single place that says where the data lives.

Every script reads and writes under BASE_DATA_PATH:

    {BASE_DATA_PATH}/{mouse_ID}/{date}/glm_data_stacked.pkl      (input)
    {BASE_DATA_PATH}/{mouse_ID}/{date}/{MODEL}_results/*.pkl     (output)

Set the environment variable GLM_BASE_DATA_PATH to point at your own copy:

    export GLM_BASE_DATA_PATH=/path/to/ProcessedData

If it is not set, the lab default below is used.
"""

import os

_DEFAULT_BASE_DATA_PATH = '/Volumes/Akhil Data/Akhil/ProcessedData'

BASE_DATA_PATH = os.environ.get('GLM_BASE_DATA_PATH', _DEFAULT_BASE_DATA_PATH)
