import argparse
import json
import os
import pathlib
import sys
import threading
import time
from contextlib import contextmanager
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
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
        aspect = img.height / img.width
        height = max(2, int(width * aspect * 0.5) * 2)
        img = img.resize((width, height), Image.LANCZOS)
        pixels = img.load()
        lines = []
        for y in range(0, height, 2):
            line = ""
            for x in range(width):
                r1, g1, b1, a1 = pixels[x, y]
                r2, g2, b2, a2 = pixels[x, y + 1] if y + 1 < height else (0, 0, 0, 0)
                top_vis = a1 >= 128
                bot_vis = a2 >= 128
                if top_vis and bot_vis:
                    line += f"\033[38;2;{r1};{g1};{b1}m\033[48;2;{r2};{g2};{b2}m▀"
                elif top_vis:
                    line += f"\033[0m\033[38;2;{r1};{g1};{b1}m▀"
                elif bot_vis:
                    line += f"\033[0m\033[38;2;{r2};{g2};{b2}m▄"
                else:
                    line += "\033[0m "
            lines.append(line + "\033[0m")
        return "\n".join(lines)
    except Exception:
        return ""


_SPARK_FRAMES = ["✶", "✸", "✺", "✸", "✶", " "]


@contextmanager
def _loading_spinner(message: str = "Retrieving context", suppress_output: bool = False):
    stop = threading.Event()
    start_time = time.time()
    real_stdout = getattr(sys, "__stdout__", sys.stdout)
    write_target = real_stdout if suppress_output else sys.stdout

    def _run():
        i = 0
        while not stop.is_set():
            elapsed = int(time.time() - start_time)
            frame = _SPARK_FRAMES[i % len(_SPARK_FRAMES)]
            write_target.write(f"\r  {frame}  {message}... {elapsed}s")
            write_target.flush()
            time.sleep(0.12)
            i += 1
        clear = " " * (len(message) + 30)
        write_target.write(f"\r{clear}\r")
        write_target.flush()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    if suppress_output:
        devnull = open(os.devnull, "w", encoding="utf-8")
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = devnull
    try:
        yield
    finally:
        stop.set()
        t.join()
        if suppress_output:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            devnull.close()
        elapsed = int(time.time() - start_time)
        write_target.write(f"  ·  {message} — {elapsed}s\n")
        write_target.flush()


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
        console.print(f"  [dim]· {message}[/dim]")
    else:
        print(f"  · {message}")


_PIPELINE_STEPS = {
    "received":       (" Received input",                            "bold white"),
    "memory_check":   (" Checking retrieval memory...",              "dim"),
    "memory_boost":   (" Memory boost applied — reusing verified chunks", "cyan"),
    "retrieving":     (" Retrieving transcript chunks...",           "dim"),
    "cache_check":    (" Checking answer cache...",                  "dim"),
    "cache_hit":      (" Cache HIT — skipping LLM call",            "green"),
    "cache_miss":     (" New query — no cache match",                "yellow"),
    "llm_call":       (" Sending to LLM...",                        "dim"),
    "llm_done":       (" Got response",                              "dim"),
    "verify":         (" Cross-verifying against transcript...",     "dim"),
    "faithful":       (" Faithful ✓ — sending final response",      "green"),
    "hallucination":  (" Caution: answer may not be grounded  ✗",   "red"),
    "cached_reply":   (" Sending cached response",                   "green"),
    "low_confidence": (" Low retrieval confidence — skipping LLM",  "yellow"),
}


def _print_step(console, step: str, override_msg: str = "", t0: float = None):
    msg, style = _PIPELINE_STEPS.get(step, (f" {step}", "dim"))
    if override_msg:
        msg = f" {override_msg}"
    elapsed = f"  [dim]{int(time.time() - t0)}s[/dim]" if t0 is not None else ""
    elapsed_plain = f"  {int(time.time() - t0)}s" if t0 is not None else ""
    if console:
        console.print(f"  [dim]↳[/dim][{style}]{msg}[/{style}]{elapsed}")
    else:
        print(f"  ↳{msg}{elapsed_plain}")


def _make_trace(console, t0: float):
    def trace(step):
        _print_step(console, step, t0=t0)
    return trace


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


def _print_accuracy(console, retrieval_score: float, faith):
    if faith is None:
        faith_label, faith_style = "Cached", "dim"
    elif faith is False:
        faith_label, faith_style = "—", "dim"
    elif faith == 1.0:
        faith_label, faith_style = "✓ Faithful", "green"
    elif faith == 0.0:
        faith_label, faith_style = "✗ Ungrounded", "red"
    else:
        faith_label, faith_style = "—", "dim"
    if console:
        console.print(
            f"  [dim]Retrieval confidence:[/dim] [bold]{retrieval_score:.1%}[/bold]"
            f"  [dim]|  Faithfulness:[/dim] [{faith_style}]{faith_label}[/{faith_style}]"
        )
    else:
        print(f"  Retrieval confidence: {retrieval_score:.1%}  |  Faithfulness: {faith_label}")


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
        "score": float(score),
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
    memory = RetrievalMemory(encoder=agent.embed_model)
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
    _print_status(console, (
        "/sources on|off — view transcript chunks"
        "  ·  /feedback y|n — rate last answer"
        "  ·  /help  ·  /exit"
    ))

    # Load models in background — prompt is available immediately
    _loaded: dict = {}
    _load_error: list = [None]
    _load_done = threading.Event()

    def _bg_load():
        try:
            from src.eval_utils import calculate_faithfulness as _cf
            _a, _m = _create_agent(vector_store_path)
            _loaded["agent"] = _a
            _loaded["memory"] = _m
            _loaded["faithfulness"] = _cf
        except Exception as exc:
            _load_error[0] = exc
        finally:
            _load_done.set()

    threading.Thread(target=_bg_load, daemon=True).start()

    show_sources = args.show_sources
    last_response = None
    agent = memory = calculate_faithfulness = None

    while True:
        try:
            query = input("\n➲ ").strip()
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

        # Ensure models are ready before first real query
        if agent is None:
            if not _load_done.is_set():
                with _loading_spinner("Loading models"):
                    _load_done.wait()
            if _load_error[0]:
                raise _load_error[0]
            agent = _loaded["agent"]
            memory = _loaded["memory"]
            calculate_faithfulness = _loaded["faithfulness"]

        t0 = time.time()
        _print_step(console, "received", override_msg=f'Received: "{query}"', t0=t0)

        _print_step(console, "memory_check", t0=t0)
        verified_ids = memory.get_verified_contexts(query)
        if verified_ids:
            _print_step(console, "memory_boost", t0=t0)

        trace = _make_trace(console, t0)
        answer, contexts, score = agent.get_answer_with_context(
            query, boost_ids=verified_ids, trace=trace
        )

        if agent.last_response_meta["cache_hit"]:
            _print_step(console, "cached_reply", t0=t0)
            faith = None
        elif agent.last_response_meta.get("low_confidence"):
            _print_step(console, "low_confidence", t0=t0)
            faith = False
        else:
            _print_step(console, "verify", t0=t0)
            raw_faith = calculate_faithfulness(agent.groq_try, answer, contexts)
            faith = raw_faith if raw_faith is not None else False
            if faith == 1.0:
                _print_step(console, "faithful", t0=t0)
            elif faith == 0.0:
                _print_step(console, "hallucination", t0=t0)

        _print_answer(console, answer)
        _print_accuracy(console, score, faith)
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
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")

    # Wrap stdout/stderr to drop HF Hub's direct-print auth warning (bypasses warnings module)
    _hf_msg = "unauthenticated requests to the HF Hub"

    class _HFFilter:
        def __init__(self, s): self._s = s
        def write(self, t):
            if _hf_msg not in t: self._s.write(t)
        def flush(self): self._s.flush()
        def __getattr__(self, n): return getattr(self._s, n)

    sys.stdout = _HFFilter(sys.stdout)
    sys.stderr = _HFFilter(sys.stderr)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HUGGINGFACE_HUB_VERBOSITY", "error")
    # Skip CUDA device enumeration — saves 3-5s on Windows even without a GPU
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    import logging
    import warnings
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
    warnings.filterwarnings("ignore", message=".*unauthenticated.*")
    warnings.filterwarnings("ignore", message=".*huggingface.*", category=UserWarning)

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
