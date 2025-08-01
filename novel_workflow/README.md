# LangGraph Novel Workflow

This example implements a simple novel writing workflow using LangGraph. The graph
follows these steps:

1. **Outline Agent** – Generates a high level outline based on the user input.
2. **Schema Agent** – Produces a JSON schema describing each chapter.
3. **Chapter Writer** – Writes each chapter sequentially.
4. **Human Review** – Allows optional human feedback between chapters.

The schema and writer steps iterate until all chapters are produced.
