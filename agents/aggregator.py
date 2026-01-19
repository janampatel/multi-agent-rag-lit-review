
# Placeholder for Review Aggregator Agent
class AggregatorAgent:
    def synthesize(self, retrieved_data: list) -> str:
        # MVP: Simple concatenation
        summary = "Synthesized Review based on:\n"
        for item in retrieved_data:
            summary += f"- {item['content'][:100]}...\n"
        return summary
