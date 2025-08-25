import numpy as np

class SafetyMetricsCBF:
    def __init__(self, cbf_function):
        self.cbf_function = cbf_function

    def cbf_value(self, state):
        return self.cbf_function(state)

    def cbf_violation(self, trajectory):
        violations = [self.cbf_value(s) < 0 for s in trajectory]
        return np.sum(violations), len(violations)

    def min_cbf(self, trajectory):
        return min(self.cbf_value(s) for s in trajectory)

    def safety_score(self, trajectory):
        num_violations, total = self.cbf_violation(trajectory)
        if total == 0:
            return 1.0
        return 1.0 - (num_violations / total)

# def example_cbf(state):
#     # h(x) > 0 is safe, h(x) < 0 is unsafe
#     return state[0] - 1.0

# metrics = SafetyMetricsCBF(example_cbf)
# trajectory = [np.array([0.5]), np.array([1.2]), np.array([0.8])]
# print(metrics.safety_score(trajectory))