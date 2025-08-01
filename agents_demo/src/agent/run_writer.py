import argparse
from agent.writer import novel_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the novel writer workflow")
    parser.add_argument("title", help="Title of the novel")
    parser.add_argument("idea", help="Basic idea of the story")
    parser.add_argument("chapter_count", type=int, help="Number of chapters to generate")
    args = parser.parse_args()

    initial_state = {
        "user_input": {
            "title": args.title,
            "idea": args.idea,
            "chapter_cnt": args.chapter_count,
        },
        "outline": None,
        "outline_feedback": None,
        "chapter_index": 0,
        "chapter_text": None,
        "human_feedback": None,
    }

    novel_workflow.invoke(initial_state)


if __name__ == "__main__":
    main()
