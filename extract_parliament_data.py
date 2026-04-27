#!/usr/bin/env python
"""Build a DPO preference dataset from CouncilX reward_data MongoDB records."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional

from pymongo import MongoClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract (prompt, chosen, rejected) preference pairs for DPO."
    )
    parser.add_argument(
        "--mongo-uri",
        default=os.getenv("MONGODB_URI"),
        help="MongoDB connection URI (recommended: set MONGODB_URI in .env).",
    )
    parser.add_argument(
        "--database",
        default=None,
        help="MongoDB database name (default: MONGODB_DB env var or parliament).",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="MongoDB collection name (default: MONGODB_COLLECTION env var or reward_data).",
    )
    parser.add_argument(
        "--output",
        default="dpo_dataset.jsonl",
        help="Output JSONL path for DPO training.",
    )
    parser.add_argument(
        "--legacy-output",
        default="parliament_preference_data.jsonl",
        help="Optional second JSONL path for compatibility with existing tooling.",
    )
    parser.add_argument(
        "--skip-legacy-output",
        action="store_true",
        help="Do not write the legacy compatibility output file.",
    )
    return parser.parse_args()


def load_dotenv(path: str = ".env") -> None:
    """Load simple KEY=VALUE entries from a local .env file into process env."""
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue    
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def resolve_mongo_uri(cli_mongo_uri: Optional[str]) -> str:
    uri = cli_mongo_uri or os.getenv("MONGODB_URI")
    if not uri:
        raise ValueError("MONGODB_URI is required. Set it in .env or pass --mongo-uri.")
    return uri


def resolve_database(cli_database: Optional[str]) -> str:
    return cli_database or os.getenv("MONGODB_DB", "parliament")


def resolve_collection(cli_collection: Optional[str]) -> str:
    return cli_collection or os.getenv("MONGODB_COLLECTION", "reward_data")


def score_value(answer: Dict[str, Any], metric: str) -> float:
    auto_scores = answer.get("auto_scores") or {}
    value = auto_scores.get(metric, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalize_answers(answers: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for answer in answers:
        if not isinstance(answer, dict):
            continue
        text = (answer.get("text") or "").strip()
        if not text:
            continue
        normalized.append(answer)
    return normalized


def pick_best(
    answers: List[Dict[str, Any]],
    metric: str,
    prefer_low: bool = False,
    avoid_text: Optional[str] = None,
) -> Optional[str]:
    ordered = sorted(
        answers,
        key=lambda item: score_value(item, metric),
        reverse=not prefer_low,
    )
    avoid_text = (avoid_text or "").strip()
    for answer in ordered:
        text = (answer.get("text") or "").strip()
        if not text:
            continue
        if avoid_text and text == avoid_text:
            continue
        return text
    return None


def has_explicit_feedback(feedback: Any) -> bool:
    if not isinstance(feedback, dict):
        return False
    if not feedback:
        return False
    return any(
        feedback.get(field) not in (None, "")
        for field in ("liked", "selected_answer_id", "report_bias", "feedback_text")
    )


def build_pair(doc: Dict[str, Any]) -> Dict[str, str]:
    question_id = str(doc.get("question_id") or "").strip()
    prompt = (doc.get("question") or "").strip()
    answers = normalize_answers(doc.get("answers") or [])
    final_text = ((doc.get("final_answer") or {}).get("final_text") or "").strip()
    feedback = doc.get("user_feedback")

    text_pool: List[str] = []
    if final_text:
        text_pool.append(final_text)
    for answer in answers:
        text = (answer.get("text") or "").strip()
        if text:
            text_pool.append(text)

    if not prompt:
        if question_id:
            prompt = f"Question ID: {question_id}"
        elif text_pool:
            prompt = f"Context-derived prompt: {text_pool[0][:280]}"
        else:
            prompt = "Unknown prompt"

    liked_value = feedback.get("liked") if isinstance(feedback, dict) else None
    feedback_exists = has_explicit_feedback(feedback)

    chosen: Optional[str] = None
    rejected: Optional[str] = None

    if liked_value is True:
        chosen = final_text or pick_best(answers, "neutrality")
        rejected = pick_best(answers, "bias", avoid_text=chosen)
    elif liked_value is False:
        rejected = final_text or pick_best(answers, "bias")
        chosen = pick_best(answers, "neutrality", avoid_text=rejected)
    elif not feedback_exists:
        chosen = pick_best(answers, "neutrality")
        rejected = pick_best(answers, "bias", avoid_text=chosen)
    else:
        chosen = pick_best(answers, "neutrality")
        rejected = pick_best(answers, "bias", avoid_text=chosen)

    if not chosen:
        chosen = pick_best(answers, "neutrality") or final_text

    if not rejected:
        rejected = pick_best(answers, "bias", avoid_text=chosen)

    if not rejected and final_text and final_text != chosen:
        rejected = final_text

    if not chosen and text_pool:
        chosen = text_pool[0]

    if not rejected:
        for candidate in text_pool:
            if candidate != chosen:
                rejected = candidate
                break

    if not chosen:
        chosen = "No chosen response available"

    if not rejected:
        rejected = chosen

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def write_jsonl(path: str, rows: List[Dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    load_dotenv()
    args = parse_args()

    mongo_uri = resolve_mongo_uri(args.mongo_uri)
    database_name = resolve_database(args.database)
    collection_name = resolve_collection(args.collection)
    client = MongoClient(mongo_uri)
    collection = client[database_name][collection_name]

    rows: List[Dict[str, str]] = []
    processed = 0

    print(f"Using database='{database_name}', collection='{collection_name}'")
    estimated_count = collection.count_documents({})
    print(f"Documents matched by query {{}}: {estimated_count}")
    if estimated_count == 0:
        print(
            "No records found with current database/collection. "
            "Verify MONGODB_DB and MONGODB_COLLECTION point to your Atlas data."
        )

    for doc in collection.find({}):
        processed += 1
        rows.append(build_pair(doc))

    write_jsonl(args.output, rows)
    if not args.skip_legacy_output:
        write_jsonl(args.legacy_output, rows)

    print(f"Processed={processed} | Written={len(rows)} | Output={args.output}")
    if not args.skip_legacy_output:
        print(f"Legacy output written to {args.legacy_output}")


if __name__ == "__main__":
    main()
