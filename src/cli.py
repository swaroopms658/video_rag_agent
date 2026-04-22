import argparse
import json
import os
import pathlib
import sys
from itertools import zip_longest
from typing import Iterable, Optional

try:
    from rich.console import Console
    from rich.rule import Rule
    from rich.table import Table
except ImportError:  # pragma: no cover - fallback for minimal installs
    Console = None
    Rule = None
    Table = None


APP_NAME = "Agentic Video RAG"
DEFAULT_VECTOR_STORE = "data/vector_store.pkl"
ALT_VECTOR_STORE = "data/vector_store_7.pkl"


def _render_bot_image(width: int = 28) -> str:
    try:
        from PIL import Image
        img_path = pathlib.Path(__file__).parent / "assets" / "bot.png"
        if not img_path.exists():
            return ""
        img = Image.open(img_path).convert("RGBA")
        aspect = img.height / img.width
        height = max(2, int(width * aspect * 0.5) * 2)
        img = img.resize((width, height), Image.LANCZOS)
        pixels = img.load()
        lines = []
        for y in range(0, height, 2):
            line = ""
            for x in range(width):
                r1, g1, b1, a1 = pixels[x, y]
                r2, g2, b2, a2 = pixels[x, y + 1] if y + 1 < height else (18, 18, 18, 255)
                if a1 < 128:
                    r1, g1, b1 = 18, 18, 18
                if a2 < 128:
                    r2, g2, b2 = 18, 18, 18
                line += f"\033[38;2;{r1};{g1};{b1}m\033[48;2;{r2};{g2};{b2}m▀"
            lines.append(line + "\033[0m")
        return "\n".join(lines)
    except Exception:
        return ""


def _get_console():
    return Console() if Console else None


def _print_header(console, vector_store_path: str):
    bot_art = _render_bot_image()
    art_lines = bot_art.split("\n") if bot_art else []

    title_text = [
        "\033[1;34m  Agentic Video RAG\033[0m",
        "\033[2m  by M.S Swaroop · v0.1.0\033[0m",
        "",
        "\033[2m  Ask questions about your lecture.\033[0m",
        "\033[2m  /help for commands.\033[0m",
        f"\033[2m  Store: {vector_store_path}\033[0m",
    ]
    pad_top = max(0, (len(art_lines) - len(title_text)) // 2)
    padded_titles = [""] * pad_top + title_text

    print()
    if console:
        console.rule(style="blue")
    else:
        print("\033[34m" + "─" * 60 + "\033[0m")

    for art, title in zip_longest(art_lines, padded_titles, fillvalue=""):
        sys.stdout.write("  " + art + title + "\n")
    sys.stdout.flush()

    if console:
        console.rule(style="blue")
    else:
        print("\033[34m" + "─" * 60 + "\033[0m")
    print()


def _print_status(console, message: str):
    if console:
        console.print(f"[cyan]{message}[/cyan]")
    else:
        print(message)


def _print_answer(console, answer: str):
    if console:
        console.print(f"\n[bold blue]◈ Agentic Video RAG[/bold blue]")
        for line in answer.splitlines():
            console.print(f"  {line}")
        console.print()
    else:
        print(f"\n◈ Agentic Video RAG")
        for line in answer.splitlines():
            print(f"  {line}")
        print()


def _print_sources(console, contexts: Iterable[str], score: float):
    contexts = list(contexts)
    if console and Table:
        table = Table(title=f"Retrieved Transcript Chunks | Confidence {score:.1%}")
        table.add_column("#", style="bold")
        table.add_column("Preview")
        for index, context in enumerate(contexts, start=1):
            preview = context[:220].replace("\n", " ").strip()
            if len(context) > 220:
                preview += "..."
            table.add_row(str(index), preview)
        console.print(table)
    else:
        print(f"Retrieval confidence: {score:.1%}")
        print("Source transcript chunks:")
        for index, context in enumerate(contexts, start=1):
            preview = context[:220].replace("\n", " ").strip()
            if len(context) > 220:
                preview += "..."
            print(f"[{index}] {preview}")


def _resolve_vector_store(requested_path: Optional[str]) -> str:
    if requested_path:
        return requested_path
    if os.path.exists(DEFAULT_VECTOR_STORE):
        return DEFAULT_VECTOR_STORE
    if os.path.exists(ALT_VECTOR_STORE):
        return ALT_VECTOR_STORE
    return DEFAULT_VECTOR_STORE


def _log_feedback(query: str, answer: str, score: float, contexts, reward: int):
    from src.retrieval_memory import RetrievalMemory

    log_entry = {
        "mode": "TEXT",
        "query": query,
        "answer": answer,
        "score": score,
        "reward": reward,
        "context_ids": RetrievalMemory.make_context_ids(contexts),
    }
    os.makedirs("data", exist_ok=True)
    with open("data/rl_feedback.json", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(log_entry) + "\n")


def _create_agent(vector_store_path: str):
    from src.rag_chain import AgenticRAG
    from src.retrieval_memory import RetrievalMemory

    if not os.path.exists(vector_store_path):
        raise FileNotFoundError(
            f"Vector store not found at '{vector_store_path}'. Run build-store first."
        )
    agent = AgenticRAG(vector_store_path)
    memory = RetrievalMemory()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return agent, memory


def _run_query(agent, memory, query: str):
    verified_ids = memory.get_verified_contexts(query)
    answer, contexts, score = agent.get_answer_with_context(
        query,
        boost_ids=verified_ids,
    )
    return answer, contexts, score, verified_ids


def chat_command(args):
    console = _get_console()
    vector_store_path = _resolve_vector_store(args.vector_store)
    _print_header(console, vector_store_path)
    _print_status(console, "Commands: /help, /sources on|off, /feedback y|n, /exit")

    agent, memory = _create_agent(vector_store_path)
    show_sources = args.show_sources
    last_response = None

    while True:
        try:
            query = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue

        lowered = query.lower()
        if lowered in {"q", "quit", "exit", "/exit"}:
            break
        if lowered in {"/help", "help"}:
            _print_status(
                console,
                "Ask any lecture question. Use /sources on|off and /feedback y|n.",
            )
            continue
        if lowered.startswith("/sources"):
            parts = lowered.split()
            if len(parts) == 2 and parts[1] in {"on", "off"}:
                show_sources = parts[1] == "on"
                _print_status(console, f"Source previews {'enabled' if show_sources else 'disabled'}.")
            else:
                _print_status(console, "Usage: /sources on|off")
            continue
        if lowered.startswith("/feedback"):
            if not last_response:
                _print_status(console, "No answer available yet for feedback.")
                continue
            parts = lowered.split()
            if len(parts) != 2 or parts[1] not in {"y", "n"}:
                _print_status(console, "Usage: /feedback y|n")
                continue
            reward = 1 if parts[1] == "y" else -1
            _log_feedback(
                last_response["query"],
                last_response["answer"],
                last_response["score"],
                last_response["contexts"],
                reward,
            )
            _print_status(console, f"Feedback logged with reward {reward}.")
            continue

        _print_status(console, "Retrieving context and generating answer...")
        answer, contexts, score, verified_ids = _run_query(agent, memory, query)
        if verified_ids:
            _print_status(console, f"Memory boost reused {len(verified_ids)} verified chunks.")

        cache_status = "HIT" if agent.last_response_meta["cache_hit"] else "MISS"
        _print_status(console, f"Cache status: {cache_status}")
        _print_answer(console, answer)
        if show_sources:
            _print_sources(console, contexts, score)

        last_response = {
            "query": query,
            "answer": answer,
            "score": score,
            "contexts": contexts,
        }


def ask_command(args):
    console = _get_console()
    vector_store_path = _resolve_vector_store(args.vector_store)
    agent, memory = _create_agent(vector_store_path)
    answer, contexts, score, verified_ids = _run_query(agent, memory, args.question)
    if verified_ids:
        _print_status(console, f"Memory boost reused {len(verified_ids)} verified chunks.")
    _print_answer(console, answer)
    if args.show_sources:
        _print_sources(console, contexts, score)


def ingest_command(args):
    from src.transcribe import transcribe_audio

    transcribe_audio(args.media_path, args.output)


def build_store_command(args):
    from src.build_vectorstore import create_embeddings

    create_embeddings(args.transcript_path, args.output)


def eval_command(_args):
    from src.evaluate import run_evaluation

    run_evaluation()


def build_parser():
    parser = argparse.ArgumentParser(
        prog="agentic-video-rag",
        description="CLI for the Agentic Video RAG lecture assistant.",
    )
    parser.add_argument(
        "--vector-store",
        default=None,
        help="Path to the vector store. Defaults to data/vector_store.pkl when available.",
    )

    subparsers = parser.add_subparsers(dest="command")

    chat_parser = subparsers.add_parser("chat", help="Open interactive chat mode.")
    chat_parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Show retrieved transcript chunk previews after each answer.",
    )
    chat_parser.set_defaults(func=chat_command)

    ask_parser = subparsers.add_parser("ask", help="Ask a single question and exit.")
    ask_parser.add_argument("question", help="Question to ask the lecture assistant.")
    ask_parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Show retrieved transcript chunk previews.",
    )
    ask_parser.set_defaults(func=ask_command)

    ingest_parser = subparsers.add_parser("ingest", help="Transcribe lecture media into text.")
    ingest_parser.add_argument("media_path", help="Path to the input lecture media file.")
    ingest_parser.add_argument(
        "--output",
        default="data/lecture_transcript.txt",
        help="Transcript output path.",
    )
    ingest_parser.set_defaults(func=ingest_command)

    build_parser_cmd = subparsers.add_parser(
        "build-store",
        help="Build a vector store from a transcript.",
    )
    build_parser_cmd.add_argument("transcript_path", help="Path to the transcript file.")
    build_parser_cmd.add_argument(
        "--output",
        default=DEFAULT_VECTOR_STORE,
        help="Vector store output path.",
    )
    build_parser_cmd.set_defaults(func=build_store_command)

    eval_parser = subparsers.add_parser("eval", help="Run the evaluation pipeline.")
    eval_parser.set_defaults(func=eval_command)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        args.command = "chat"
        args.func = chat_command
        if not hasattr(args, "show_sources"):
            args.show_sources = False

    args.func(args)


if __name__ == "__main__":
    main()
