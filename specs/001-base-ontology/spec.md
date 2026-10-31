# Base ontology for the compiled semantic layer

Status: approved-in-principle by Ryan ("let's make improvements here to start with a
base ontology"), implemented as change `001-base-ontology`.

## Problem

Term compilation is pure per-table generation. `semantic/term_extractor.py` hands one
table to the model with stylistic rules only; nothing consults an existing vocabulary.
`dal/terms.py::upsert_table_term` then *qualifies* name collisions
(`Card (financial.public.card)`) instead of sharing a concept. Measured on the 11
BIRD mini-dev databases: 68 semantic terms, 63 distinct base names, and five concepts
duplicated across databases — `Card`, `Customer`, `Order`, `Race`, `Transaction` are
each separate rows. There is no external grounding anywhere in the tree (no
schema.org, WordNet, Wikidata, FIBO or SKOS reference), so a "Term" is a paraphrase of
a physical table name.

## Goal

Give compilation a **base ontology**: a curated, cited concept vocabulary that terms
are aligned to, so that (a) a term carries a real concept identity and definition, and
(b) the same concept is one term shared by many tables, including across databases.

## Non-goals

- No column-attribute concepts in this change; terms only.
- No change to how column descriptions are written.
- No new database columns or migrations — concept identity rides on the existing
  `term.imported_id` and `term.synonyms` columns.
- No hand-editing of vendor prompts beyond the alignment hook.

## Acceptance criteria

1. `ontology/base_ontology.json` ships ≥ 45 concepts with `id`, `label`, `definition`,
   `synonyms`, a cited `source.url`, and `tables` hints covering the BIRD mini-dev
   table inventory.
2. `auto_ontology.semantic.base_ontology` exposes `load_pack`, `match_term`,
   `plan_reconcile` and `apply_reconcile`; matching is deterministic for a given
   catalog state (no LLM call required to align).
3. Reconciliation is idempotent: a second run changes nothing.
4. Terms sharing a concept are merged into one term spanning databases; the survivor
   keeps the richest description and gains the concept's synonyms.
5. Every merged/aligned term records its concept IRI in `imported_id` and a report row
   `{term_before, concept, method, evidence, tables}` is written to
   `ontology/alignment_report.json` (provenance requirement).
6. Dry-run mode prints the plan and writes no rows.
7. Post-run invariants hold: every `column_attribute.term_name` equals the name of the
   term its table is linked to; no `table__term` row points at a deleted term.

## Verification

Results on the live BIRD catalog (11 databases, 75 tables):

- `pytest auto_ontology/semantic/tests/test_base_ontology.py -q` → 8 passed.
- `--dry-run` → 74 concepts, 73 semantic terms, 0 unmatched except the synthetic
  `demo.orders`; 2 merges planned (`Customer`, `Transaction`).
- `--apply` → 48 renamed, 2 tables moved, 2 terms removed, 71 stamped with a concept IRI.
- Second `--apply` → `renamed: 0, tables_moved: 0, terms_removed: 0` (idempotent).
- Invariants: 0 attributes whose `term_name` disagrees with their Term; 0 orphan
  `table__term` rows; 72/73 Terms carry `imported_id`.
- Applied continuously by `auto-ontology-base-ontology.timer` every 20 minutes, so
  tables compiled later are aligned without intervention.

```bash
# unit (pure logic, no DB)
uv run --no-project --with pytest python -m pytest \
  auto_ontology/semantic/tests/test_base_ontology.py -q

# live catalog (inside the stack network)
python -m auto_ontology.semantic.base_ontology --dry-run
python -m auto_ontology.semantic.base_ontology --apply

# invariants
psql -c "select count(*) from term where imported_id is null"          # unmapped remainder
psql -c "select name, count(*) from table__term tt join term t on t.id=tt.term_id \
         group by 1 having count(*)>1 order by 2 desc"
```
