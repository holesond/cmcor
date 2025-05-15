"""
Search for and import an OnlineFlow optical flow estimator class.
"""

import os
import sys


folder_list = [
    "/home/user/path/to/movingcables/flow_predictors",
    ]

for folder in folder_list:
    if os.path.isdir(folder):
        sys.path.append(folder)

try:
    from online_flow import OnlineFlow
except ImportError as e:
    print("WARNING: Cannot import OnlineFlow flow predictor!", e)


class FlowPredictor():
    def __init__(self, gpu):
        try:
            self.mfnprob = OnlineFlow(
                gpu=gpu, probabilistic=True, finetuned=True)
        except AttributeError as e:
            new_error = RuntimeError(
                "OnlineFlow flow predictor is not available!")
            raise new_error from e

    def flow(self, img_target, img_reference):
        flow_result = self.mfnprob.flow(img_target, img_reference)
        return flow_result[0]
