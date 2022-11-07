import numpy as np

from evaluation.results_base import EvaluationResults

K_N_SUCCESS = "total_success"
K_N_FAIL = "total_fail"
K_N_SEG = "total_segments"
K_RATE = "success_rate"
K_DIST = "total_dist"
K_AVG_DIST = "avg_dist"

K_LAST_DIST = "last_dist"

K_MEDIAN_DIST = "median_dist"
K_ALL_DIST = "all_dist"


class ResultsNL(EvaluationResults):

    def __init__(self, success=None, end_dist=0):
        super(EvaluationResults, self).__init__()
        self.state = {
            K_N_SUCCESS: 1 if success else 0 if success is not None else 0,
            K_N_FAIL: 0 if success else 1 if success is not None else 0,
            K_N_SEG: 1 if success is not None else 0,
            K_RATE: 1.0 if success else 0.0 if success is not None else 0,
            K_DIST: end_dist,
            K_LAST_DIST: end_dist
        }
        self.metastate_distances = [end_dist]

    def __add__(self, past_results):
        self.state[K_N_SUCCESS] += past_results.state[K_N_SUCCESS]
        self.state[K_N_FAIL] += past_results.state[K_N_FAIL]
        self.state[K_N_SEG] += past_results.state[K_N_SEG]
        self.state[K_RATE] = self.state[K_N_SUCCESS] / (self.state[K_N_SEG] + 1e-9)

        self.state[K_DIST] += past_results.state[K_DIST]
        self.state[K_AVG_DIST] = self.state[K_DIST] / (self.state[K_N_SEG] + 1e-9)

        self.metastate_distances += past_results.metastate_distances

        return self

    def get_dict(self):
        self.state[K_MEDIAN_DIST] = np.median(np.asarray(self.metastate_distances[1:]))
        self.state[K_ALL_DIST] = self.metastate_distances
        return self.state
