# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for base-ontology loading, matching and merge planning (no DB)."""

from __future__ import annotations

from auto_ontology.semantic.base_ontology import (
    BaseOntology,
    Concept,
    TermRow,
    candidate_concepts,
    load_pack,
    match_term,
    plan_reconcile,
)


def _pack() -> BaseOntology:
    concepts = (
        Concept(
            id="urn:test:Card",
            label="Card",
            definition="A payment card issued against an account.",
            synonyms=("Payment Card",),
            source={"kind": "curated"},
            tables=("bank.card",),
        ),
        Concept(
            id="urn:test:Transaction",
            label="Transaction",
            definition="A single monetary event.",
            synonyms=("Payment",),
            source={"kind": "curated"},
            tables=(),
        ),
    )
    return BaseOntology(
        version="test",
        concepts=concepts,
        by_id={c.id: c for c in concepts},
        by_table={t.lower(): c.id for c in concepts for t in c.tables},
        by_name={"card": "urn:test:Card", "payment card": "urn:test:Card",
                 "transaction": "urn:test:Transaction", "payment": "urn:test:Transaction"},
    )


def test_pack_on_disk_parses_with_sources() -> None:
    pack = load_pack()
    assert len(pack.concepts) >= 45
    for concept in pack.concepts:
        assert concept.id and concept.label and concept.definition
        assert concept.source.get("kind"), concept.id


def test_table_hint_beats_a_misleading_name() -> None:
    match = match_term(
        "Card (financial.public.card)", table_keys=["bank.card"], pack=_pack()
    )
    assert match is not None
    assert match.concept_id == "urn:test:Card"
    assert match.method == "table-hint"


def test_label_and_synonym_matches() -> None:
    pack = _pack()
    label = match_term("Transaction", table_keys=["finance.trans"], pack=pack)
    assert label is not None and label.method == "label"
    synonym = match_term("Payment", table_keys=["unknown.t"], pack=pack)
    assert synonym is not None and synonym.method == "synonym"


def test_unmatched_is_none_not_a_guess() -> None:
    assert match_term("Zebra Stripes", table_keys=["zoo.stripes"], pack=_pack()) is None


def _row(term_id: str, name: str, tables: list[tuple[str, str]], description: str = "") -> TermRow:
    return TermRow(
        id=term_id,
        name=name,
        description=description,
        synonyms=[],
        imported_id=None,
        tables=[
            {"table_id": f"{db}.{table}", "database": db, "schema": "public", "table": table}
            for db, table in tables
        ],
    )


def test_same_concept_across_databases_folds_into_one_term() -> None:
    pack = _pack()
    rows = [
        _row("t1", "Card", [("bank", "card")], description="A card."),
        _row("t2", "Card (games.cards)", [("games", "cards")], description="A collectible card."),
    ]
    plans, unmatched = plan_reconcile(rows, pack)
    assert not unmatched
    assert len(plans) == 1
    plan = plans[0]
    assert plan.new_name == "Card"
    # The longer description survives, so t1 is folded into it and renamed.
    assert plan.survivor.id == "t2"
    assert [t.id for t in plan.folded] == ["t1"]
    assert plan.table_count == 2
    assert "Payment Card" in plan.synonyms


def test_richest_term_survives() -> None:
    pack = _pack()
    rows = [
        _row("a", "Transaction", [("finance", "trans")], description="short"),
        _row("b", "Payment", [("bank", "payments")], description="a much longer description of a payment"),
    ]
    plans, _ = plan_reconcile(rows, pack)
    assert plans[0].survivor.id == "b"
    assert [t.id for t in plans[0].folded] == ["a"]


def test_plan_is_stable_for_identical_input() -> None:
    pack = _pack()
    rows = [_row("t1", "Card", [("bank", "card")]), _row("t2", "Card", [("games", "cards")])]
    first, _ = plan_reconcile(rows, pack)
    second, _ = plan_reconcile(rows, pack)
    assert [p.to_report() for p in first] == [p.to_report() for p in second]


def test_candidate_concepts_offers_the_table_hint_first() -> None:
    pack = _pack()
    offers = candidate_concepts("bank", "card", ["card_type"], pack=pack)
    assert [c.id for c in offers] == ["urn:test:Card"]


def test_candidate_concepts_falls_back_to_label_words() -> None:
    pack = _pack()
    offers = candidate_concepts("shop", "customer_transaction", ["amount"], pack=pack)
    assert "urn:test:Transaction" in [c.id for c in offers]


def test_candidate_concepts_ignores_short_labels_and_noise() -> None:
    pack = _pack()
    assert candidate_concepts("zoo", "zebra_stripes", ["stripe_count"], pack=pack) == []


def test_pack_covers_every_bird_table_once() -> None:
    pack = load_pack()
    hints = [
        table
        for concept in pack.concepts
        for table in concept.tables
        if not table.startswith("demo")
    ]
    assert len(hints) == 75
    assert len(set(hints)) == 75


def test_unmatched_terms_are_reported_not_dropped() -> None:
    pack = _pack()
    rows = [_row("t9", "Zebra Stripes", [("zoo", "stripes")])]
    plans, unmatched = plan_reconcile(rows, pack)
    assert plans == []
    assert unmatched == [{"term": "Zebra Stripes", "id": "t9", "tables": ["zoo.stripes"]}]
