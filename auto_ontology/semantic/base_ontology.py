# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Align compiled Terms to a curated base ontology.

Compilation generates one Term per table and, on a name collision, *qualifies* the
name (``Card (financial.public.card)``) rather than sharing a concept — so the same
business entity ends up as several unrelated rows, and nothing is grounded in a
vocabulary the organisation already owns.

This module adds the missing layer. ``ontology/base_ontology.json`` holds curated
concepts (label, definition, synonyms, cited source, and the tables each concept is
expected to describe). Matching is deterministic — a table hint first, then an exact
label or synonym match — so aligning a catalog costs no model calls.

``plan_reconcile`` folds every Term that maps to the same concept into one Term that
spans tables and databases, and stamps the concept IRI on ``term.imported_id``.
``apply_reconcile`` writes the plan; both are pure over their inputs so a dry run and
an apply agree.

CLI (inside the app network, so Postgres is reachable):

    python -m auto_ontology.semantic.base_ontology --dry-run
    python -m auto_ontology.semantic.base_ontology --apply --report /tmp/report.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

_DEFAULT_PACK = Path(__file__).resolve().parents[2] / "ontology" / "base_ontology.json"


def pack_path() -> Path:
    """Where the base ontology lives, overridable for alternate packs."""
    override = os.environ.get("AUTO_ONTOLOGY_BASE_ONTOLOGY", "").strip()
    return Path(override) if override else _DEFAULT_PACK


@dataclass(frozen=True)
class Concept:
    id: str
    label: str
    definition: str
    synonyms: tuple[str, ...] = ()
    source: dict[str, Any] = field(default_factory=dict)
    tables: tuple[str, ...] = ()


@dataclass(frozen=True)
class BaseOntology:
    version: str
    concepts: tuple[Concept, ...]
    by_id: dict[str, Concept]
    by_table: dict[str, str]
    by_name: dict[str, str]

    def concept(self, concept_id: str) -> Concept:
        return self.by_id[concept_id]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def load_pack(path: Path | None = None) -> BaseOntology:
    """Parse the base ontology pack into lookup structures.

    ``by_table`` is keyed ``<database>.<table>`` lowercased, so a table hint
    resolves a Term without consulting its generated name. ``by_name`` holds every
    label and synonym, which is what a Term with no hint falls back to.
    """
    data = json.loads((path or pack_path()).read_text())
    concepts: list[Concept] = []
    by_table: dict[str, str] = {}
    by_name: dict[str, str] = {}

    for raw in data.get("concepts", []):
        concept = Concept(
            id=raw["id"],
            label=raw["label"],
            definition=raw.get("definition", ""),
            synonyms=tuple(raw.get("synonyms", ())),
            source=dict(raw.get("source", {})),
            tables=tuple(raw.get("tables", ())),
        )
        concepts.append(concept)
        for table in concept.tables:
            by_table[table.lower()] = concept.id
        for name in (concept.label, *concept.synonyms):
            key = _norm(name)
            if key:
                by_name.setdefault(key, concept.id)

    return BaseOntology(
        version=str(data.get("version", "")),
        concepts=tuple(concepts),
        by_id={c.id: c for c in concepts},
        by_table=by_table,
        by_name=by_name,
    )


@dataclass(frozen=True)
class Match:
    concept_id: str
    method: str  # table-hint | label | synonym | token
    evidence: str


def match_term(
    term_name: str,
    *,
    table_keys: Iterable[str],
    pack: BaseOntology,
) -> Match | None:
    """Resolve one Term to a concept, or ``None`` when nothing credible matches.

    A table hint wins because it is a statement about the data, not about the name
    the model happened to invent; a label match is the most common fallback, and a
    synonym match is a label match on an alternate name.
    """
    keys = [k.lower() for k in table_keys]
    for key in keys:
        concept_id = pack.by_table.get(key)
        if concept_id:
            return Match(concept_id, "table-hint", key)

    # Compilation qualifies a colliding name with the table identity
    # ("Card (financial.public.card)"); the qualifier is ours, not the concept's.
    bare = term_name.split(" (", 1)[0]
    name_key = _norm(bare)
    concept_id = pack.by_name.get(name_key)
    if concept_id:
        concept = pack.concept(concept_id)
        method = "label" if _norm(concept.label) == name_key else "synonym"
        return Match(concept_id, method, term_name)

    # Last resort: a generated name that contains a concept label as whole words,
    # e.g. "Historical Total Amount" for "Total Amount". Deliberately conservative
    # — one-sided containment on short labels produces false positives.
    for candidate_id in pack.by_name.values():
        label = _norm(pack.concept(candidate_id).label)
        if len(label) >= 8 and f" {label} " in f" {name_key} ":
            return Match(candidate_id, "token", term_name)
    return None


def candidate_concepts(
    database_name: str | None,
    table_name: str,
    column_names: Iterable[str] = (),
    *,
    limit: int = 6,
    pack: BaseOntology | None = None,
) -> list[Concept]:
    """Concepts worth offering the extractor for one table.

    A table hint is authoritative. Beyond it a concept qualifies when its label
    appears as whole words in the table name or a column name — enough to put the
    concept in front of the model without inviting false matches.
    """
    base = pack or load_pack()
    found: list[Concept] = []
    if database_name:
        concept_id = base.by_table.get(f"{database_name}.{table_name}".lower())
        if concept_id:
            found.append(base.concept(concept_id))

    haystacks = [_norm(table_name), *(_norm(name) for name in column_names)]
    for concept in base.concepts:
        if concept in found:
            continue
        label = _norm(concept.label)
        if len(label) < 4:
            continue
        padded = f" {label} "
        if any(padded in f" {text} " for text in haystacks):
            found.append(concept)
        if len(found) >= limit:
            break
    return found[:limit]


@dataclass
class TermRow:
    """One compiled Term and the tables that represent it."""

    id: str
    name: str
    description: str | None
    synonyms: list[str]
    imported_id: str | None
    tables: list[dict[str, str]]  # [{table_id, database, schema, table}]

    @property
    def table_keys(self) -> list[str]:
        return [f"{t['database']}.{t['table']}" for t in self.tables]


@dataclass
class MergePlan:
    concept_id: str
    concept_label: str
    method: str
    evidence: str
    survivor: TermRow
    folded: list[TermRow]
    new_name: str
    synonyms: list[str]
    concept_definition: str = ""

    @property
    def table_count(self) -> int:
        ids = {t["table_id"] for row in (self.survivor, *self.folded) for t in row.tables}
        return len(ids)

    def to_report(self) -> dict[str, Any]:
        return {
            "concept": self.concept_id,
            "concept_label": self.concept_label,
            "method": self.method,
            "evidence": self.evidence,
            "survivor": {"id": self.survivor.id, "name_before": self.survivor.name},
            "merged_terms": [{"id": t.id, "name": t.name} for t in self.folded],
            "tables": sorted(
                f"{t['database']}.{t['table']}"
                for row in [self.survivor, *self.folded]
                for t in row.tables
            ),
            "related_concepts_differing": [],
        }


def plan_reconcile(rows: Iterable[TermRow], pack: BaseOntology) -> tuple[list[MergePlan], list[dict[str, Any]]]:
    """Group Terms by concept and decide one survivor per concept.

    Returns the plans to apply plus the unmatched Terms (reported, never touched) so
    the caller can see what the base ontology does not yet cover.
    """
    grouped: dict[str, list[tuple[TermRow, Match]]] = {}
    unmatched: list[dict[str, Any]] = []

    for row in rows:
        found = match_term(row.name, table_keys=row.table_keys, pack=pack)
        if not found:
            unmatched.append({"term": row.name, "id": row.id, "tables": row.table_keys})
            continue
        grouped.setdefault(found.concept_id, []).append((row, found))

    plans: list[MergePlan] = []
    for concept_id, members in grouped.items():
        concept = pack.concept(concept_id)
        # Deterministic survivor: the term covering the most tables, then the
        # richest description, then term id so ties never churn between runs.
        members.sort(
            key=lambda pair: (
                -len(pair[0].tables),
                -len(pair[0].description or ""),
                pair[0].id,
            )
        )
        survivor, survivor_match = members[0]
        folded = [row for row, _ in members[1:]]
        synonyms = sorted(
            {*(survivor.synonyms or []), *concept.synonyms}
            - {concept.label}
        )
        plans.append(
            MergePlan(
                concept_id=concept_id,
                concept_label=concept.label,
                method=survivor_match.method,
                evidence=survivor_match.evidence,
                survivor=survivor,
                folded=folded,
                new_name=concept.label,
                synonyms=synonyms,
                concept_definition=concept.definition,
            )
        )

    plans.sort(key=lambda plan: plan.concept_label)
    return plans, unmatched


def _load_terms() -> list[TermRow]:
    """Every semantic Term with the tables that represent it."""
    from sqlalchemy import select

    from auto_ontology.dal import schema as s
    from auto_ontology.dal.session import store

    rows = store().query_read(
        select(
            s.term.c.id,
            s.term.c.name,
            s.term.c.description,
            s.term.c.synonyms,
            s.term.c.imported_id,
            s.catalog_table.c.id.label("table_id"),
            s.catalog_database.c.name.label("database"),
            s.catalog_schema.c.name.label("schema"),
            s.catalog_table.c.name.label("table"),
        )
        .select_from(
            s.term.join(s.table__term, s.table__term.c.term_id == s.term.c.id).join(
                s.catalog_table, s.catalog_table.c.id == s.table__term.c.table_id
            ).join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            ).join(
                s.catalog_database, s.catalog_database.c.id == s.catalog_schema.c.database_id
            )
        )
        .where(s.term.c.source == "semantic")
    )

    by_term: dict[str, TermRow] = {}
    for row in rows:
        term = by_term.get(row["id"])
        if term is None:
            term = TermRow(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                synonyms=list(row["synonyms"] or []),
                imported_id=row["imported_id"],
                tables=[],
            )
            by_term[row["id"]] = term
        term.tables.append(
            {
                "table_id": row["table_id"],
                "database": row["database"],
                "schema": row["schema"],
                "table": row["table"],
            }
        )
    return list(by_term.values())


def apply_reconcile(plans: list[MergePlan]) -> dict[str, int]:
    """Write the plans: rename, stamp provenance, fold tables, delete the rest."""
    from sqlalchemy import delete, update

    from auto_ontology.dal import schema as s
    from auto_ontology.dal.session import store

    stats = {"renamed": 0, "stamped": 0, "tables_moved": 0, "terms_removed": 0}

    # Two passes, because ``term.name`` is unique per source: a survivor cannot take
    # the concept label while a folded term still holds it. Fold first, rename second.
    for plan in plans:
        survivor = plan.survivor
        for row in plan.folded:
            for table in row.tables:
                store().query_write(
                    delete(s.table__term).where(
                        s.table__term.c.table_id == table["table_id"],
                        s.table__term.c.term_id == row.id,
                    )
                )
                store().query_write(
                    s.table__term.insert().values(
                        table_id=table["table_id"], term_id=survivor.id
                    )
                )
                stats["tables_moved"] += 1
            store().query_write(delete(s.term).where(s.term.c.id == row.id))
            stats["terms_removed"] += 1

    for plan in plans:
        survivor = plan.survivor
        table_ids = [
            t["table_id"] for row in [survivor, *plan.folded] for t in row.tables
        ]
        description = survivor.description or _concept_description(plan)
        values = {
            "name": plan.new_name,
            "imported_id": plan.concept_id,
            "synonyms": plan.synonyms,
            "description": description,
        }
        if (
            survivor.name != plan.new_name
            or survivor.imported_id != plan.concept_id
            or survivor.synonyms != plan.synonyms
        ):
            store().query_write(
                update(s.term).where(s.term.c.id == survivor.id).values(**values)
            )
            stats["renamed"] += 1
        stats["stamped"] += 1

        # The denormalised term_name on every attribute of those tables must follow
        # the surviving Term, or attribute lookups disagree with the glossary.
        store().query_write(
            update(s.column_attribute)
            .where(s.column_attribute.c.table_id.in_(table_ids))
            .values(term_name=plan.new_name)
        )
    return stats


def _concept_description(plan: MergePlan) -> str | None:
    """Fallback text when the surviving Term has no description of its own."""
    return plan.concept_definition or None


def reconcile(*, apply: bool, report: Path | None = None) -> dict[str, Any]:
    """Plan (and optionally apply) base-ontology alignment over the live catalog."""
    pack = load_pack()
    rows = _load_terms()
    plans, unmatched = plan_reconcile(rows, pack)

    result: dict[str, Any] = {
        "pack_version": pack.version,
        "concepts_in_pack": len(pack.concepts),
        "terms_examined": len(rows),
        "plans": [plan.to_report() for plan in plans],
        "unmatched": unmatched,
        "merged_terms": sum(len(plan.folded) for plan in plans),
        "applied": False,
    }
    if apply:
        result["stats"] = apply_reconcile(plans)
        result["applied"] = True
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Align compiled Terms to the base ontology.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", help="print the plan only (default)")
    group.add_argument("--apply", action="store_true", help="write the plan to the catalog")
    parser.add_argument("--report", type=Path, default=None, help="write a JSON report here")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = parser.parse_args(argv)

    result = reconcile(apply=args.apply, report=args.report)
    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(
        f"pack {result['pack_version']}: {result['concepts_in_pack']} concepts, "
        f"{result['terms_examined']} semantic terms"
    )
    for plan in result["plans"]:
        if not plan["merged_terms"] and plan["survivor"]["name_before"] == plan["concept_label"]:
            continue
        names = ", ".join(t["name"] for t in plan["merged_terms"]) or "-"
        print(
            f"  {plan['concept_label']:<28} <- {plan['survivor']['name_before']}"
            f"  [merge: {names}]  via {plan['method']} ({plan['evidence']})"
        )
    print(f"unmatched terms: {len(result['unmatched'])}")
    if result["applied"]:
        print("applied:", result["stats"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
