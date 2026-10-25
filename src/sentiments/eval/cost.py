class CostEstimator:
    @staticmethod
    def estimate(
        input_tokens: int,
        output_tokens: int,
        input_cost_per_m: float,
        output_cost_per_m: float,
    ) -> float:
        if input_tokens < 0 or output_tokens < 0 or input_cost_per_m < 0.0 or output_cost_per_m < 0.0:
            raise ValueError("tokens and cost rates must be non-negative")

        return (input_tokens * input_cost_per_m + output_tokens * output_cost_per_m) / 1_000_000.0
