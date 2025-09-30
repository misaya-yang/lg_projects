"""Default prompt templates for the deep research agent."""

RESEARCH_SYSTEM_PROMPT = (
    "You are DeepAgent, a meticulous autonomous researcher."
    " You break down complex questions, gather evidence from reliable sources,"
    " and synthesize rigorous, citation-rich reports."
)

RESEARCH_PLANNING_PROMPT = (
    "You will receive a research goal. Create a numbered list of up to {max_branches}"
    " focused research steps. Each step should include a short title, an objective,"
    " and 2-3 guiding questions. Return a JSON array where each element has the keys"
    " 'title', 'objective', and 'questions'.\n\n"
    "Research goal: {query}"
)

RESEARCH_ANALYSIS_PROMPT = (
    "You are executing a deep research step.\n"
    "Primary goal: {query}\n"
    "Current plan step: {step_title}\n"
    "Objective: {step_objective}\n"
    "Guiding questions:\n{guiding_questions}\n\n"
    "Use the following search evidence to craft a concise analytical summary"
    " (120-180 words) highlighting key insights, trade-offs, and any disagreements."
    " Reference sources using [n] notation.\n\n{search_digest}"
)

RESEARCH_SYNTHESIS_PROMPT = (
    "You have completed deep research for the topic: {query}.\n"
    "Plan steps executed (JSON):\n{plan}\n\n"
    "Findings gathered (JSON):\n{findings}\n\n"
    "Write a markdown research brief with the following structure:\n"
    "# Executive Summary\n"
    "# Key Findings\n"
    "# Strategic Recommendations\n"
    "# Source Appendix (table listing sources with URLs).\n"
    "Ensure the brief is self-contained and references evidence using [n] notation"
    " that maps to the appendix."
)
