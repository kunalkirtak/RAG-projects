#!/usr/bin/env python3
"""
main.py
-------
Command-line interface for the Multi-Document Research Agent.

Usage:
    python main.py ingest --path data/documents
    python main.py query "What is the refund policy?"
    python main.py chat
    python main.py evaluate
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from src.pipeline import RAGPipeline

console = Console()


def cmd_ingest(args: argparse.Namespace) -> None:
    pipeline = RAGPipeline(lazy_gemini=True)
    with console.status(f"Ingesting documents from '{args.path}'..."):
        stats = pipeline.ingest_directory(args.path)
    table = Table(title="Ingestion Summary")
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in stats.items():
        table.add_row(key, str(value))
    console.print(table)


def cmd_query(args: argparse.Namespace) -> None:
    pipeline = RAGPipeline(lazy_gemini=False)
    with console.status("Thinking..."):
        result = pipeline.query(args.question, top_k=args.top_k)
    console.print(Panel(Markdown(result["answer"]), title="Answer", border_style="cyan"))

    if result["sources"]:
        table = Table(title="Sources")
        table.add_column("Citation")
        table.add_column("File")
        table.add_column("Chunk #")
        table.add_column("Score")
        for src in result["sources"]:
            table.add_row(src["citation"], src["filename"], str(src["chunk_index"]), str(src["score"]))
        console.print(table)

    console.print(f"[dim]Total time: {result['timing']['total_pipeline_s']:.2f}s[/dim]")


def cmd_chat(args: argparse.Namespace) -> None:
    pipeline = RAGPipeline(lazy_gemini=False)
    console.print("[bold cyan]Multi-Document Research Agent — interactive chat[/bold cyan]")
    console.print("[dim]Type 'exit' to quit, 'reset' to clear memory.[/dim]\n")
    while True:
        try:
            question = console.input("[bold green]You:[/bold green] ")
        except (EOFError, KeyboardInterrupt):
            break
        if question.strip().lower() in {"exit", "quit"}:
            break
        if question.strip().lower() == "reset":
            pipeline.reset_memory()
            console.print("[dim]Memory cleared.[/dim]")
            continue
        with console.status("Thinking..."):
            result = pipeline.query(question, top_k=args.top_k)
        console.print(Panel(Markdown(result["answer"]), title="Assistant", border_style="cyan"))


def cmd_evaluate(args: argparse.Namespace) -> None:
    from src.evaluation import evaluate_retrieval

    pipeline = RAGPipeline(lazy_gemini=True)
    sample_queries = [
        {"query": "What is this project about?", "relevant_ids": []},
    ]
    df = evaluate_retrieval(pipeline, sample_queries, k=args.top_k)
    console.print(df.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research-agent", description="Multi-Document Research Agent CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_ingest = subparsers.add_parser("ingest", help="Ingest documents into the vector store")
    p_ingest.add_argument("--path", default=None, help="Directory of documents to ingest")
    p_ingest.set_defaults(func=cmd_ingest)

    p_query = subparsers.add_parser("query", help="Ask a single question")
    p_query.add_argument("question", help="The question to ask")
    p_query.add_argument("--top-k", type=int, default=5)
    p_query.set_defaults(func=cmd_query)

    p_chat = subparsers.add_parser("chat", help="Interactive chat session")
    p_chat.add_argument("--top-k", type=int, default=5)
    p_chat.set_defaults(func=cmd_chat)

    p_eval = subparsers.add_parser("evaluate", help="Run retrieval evaluation")
    p_eval.add_argument("--top-k", type=int, default=5)
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
