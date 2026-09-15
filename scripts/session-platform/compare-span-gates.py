#!/usr/bin/env python3
"""Compare bounded synthetic span-gate exports; not a semantic acceptance test."""
import argparse
import json
from pathlib import Path


def compare(left, right):
    task = left.get("task", "paraphrase")
    if task not in ("paraphrase", "fact_query") or right.get("task", "paraphrase") != task:
        raise ValueError("unsupported or mismatched task")
    for field in ("schema", "model_path", "tokenizer_path", "policy_sha256",
                  "projector_sha256", "groups"):
        if left[field] != right[field]:
            raise ValueError(f"mismatched {field}")
    if left["schema"] != "akh-span-paraphrase-gate/1":
        raise ValueError("unsupported schema")
    groups = left["groups"]
    if len(groups) != 3 or any(len(group) != 2 for group in groups):
        raise ValueError("expected three pairs")

    def vectors(report):
        groups = report["vectors"]
        if len(groups) != 3 or any(len(group) != 2 for group in groups):
            raise ValueError("expected three vector pairs")
        result = []
        for group in groups:
            for vector in group:
                if len(vector) != 1250 or any(type(x) is not int or not 0 <= x <= 255 for x in vector):
                    raise ValueError("expected 1250 bytes per vector")
                result.append(bytes(vector))
        return result

    a, b = vectors(left), vectors(right)

    def distance(x, y):
        return sum((i ^ j).bit_count() for i, j in zip(x, y))

    def neighbors(source, target):
        result = []
        for i, vector in enumerate(source):
            if task == "fact_query" and i % 2 == 0:
                continue
            candidates = [(distance(vector, other), j)
                          for j, other in enumerate(target)
                          if j != i and (task == "paraphrase" or j % 2 == 0)]
            minimum = min(d for d, _ in candidates)
            nearest = [j for d, j in candidates if d == minimum]
            result.append({"query": i, "nearest": nearest, "distance": minimum,
                           "unique_expected_neighbor": nearest == [i ^ 1]})
        return result

    flips = [distance(x, y) for x, y in zip(a, b)]
    forward, reverse = neighbors(a, b), neighbors(b, a)
    return {"task": task, "left_backend": left["backend"], "right_backend": right["backend"],
            "same_prompt_bit_flips": flips, "bit_identical": not any(flips),
            "left_queries_right_index": forward, "right_queries_left_index": reverse,
            "all_expected_neighbors_preserved": all(
                row["unique_expected_neighbor"] for row in forward + reverse)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    report = compare(json.loads(args.left.read_text()), json.loads(args.right.read_text()))
    print(json.dumps(report, indent=2))
    # Bit identity is reported separately: unequal bits alone do not establish
    # semantic incompatibility, and this small fixture cannot prove equivalence.
    return 0 if report["all_expected_neighbors_preserved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
