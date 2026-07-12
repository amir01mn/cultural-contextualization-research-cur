# Controlled Subset Guide: Food / Hospitality

## Purpose

This guide documents Step 6: controlled scalability before broad KG scaling.

The goal is not to build the largest possible KG yet. The goal is to build a small, interpretable food/hospitality subset that can be manually inspected before larger S4 experiments.

This controlled subset helps test whether CultureBank rows support complete cultural pathways:

```text
L3 deep value
-> L2 social norm / role expectation
-> L1 surface practice
```

## Why food / hospitality?

Food and hospitality are a useful first controlled topic because they often contain all three layers:

| Layer | Example content |
|---|---|
| L1 surface practice | offering tea, sharing a meal, serving coffee, accepting food, refusing food politely |
| L2 social norm | guest-host obligation, politeness, family meal expectation, welcoming visitors |
| L3 deep value | hospitality, respect, generosity, belonging, care, family unity |

This makes the topic easier to debug than a broad random sample.

## Script

Use:

```bash
python scripts/09_select_controlled_subset.py --scan_limit 2000 --max_records 100
```

This writes:

```text
data/controlled/food_hospitality_records.jsonl
data/controlled/food_hospitality_audit.csv
data/controlled/food_hospitality_selection_notes.md
```

To also find matching CulturalBench questions:

```bash
python scripts/09_select_controlled_subset.py \
  --scan_limit 2000 \
  --max_records 100 \
  --include_culturalbench \
  --cb_limit 250
```

This also writes:

```text
data/controlled/food_hospitality_culturalbench_questions.jsonl
```

## What the script searches

The script searches normalized CultureBank fields including:

```text
cultural_group
context
goal
actor
recipient
relation
actor_behavior
recipient_behavior
topic
source_text
raw
```

It selects rows that contain both:

1. food/hospitality content, and
2. social or cultural interaction content.

This prevents the subset from filling up with rows that only mention food in a generic way.

## Selection logic

The script scores rows using three term groups:

| Term group | Examples | Why it matters |
|---|---|---|
| Primary food terms | food, meal, tea, coffee, eat, drink, dinner | Finds the topic |
| Social/hospitality terms | guest, host, hospitality, offer, serve, welcome, etiquette | Finds cultural interaction |
| Value terms | respect, care, generosity, belonging, family, community | Helps identify possible L3 nodes |

A selected row must contain at least one primary food term and, by default, at least one social/hospitality term.

## Manual audit

After running the script, open:

```text
data/controlled/food_hospitality_audit.csv
```

For each row, fill in:

```text
manual_audit_status
manual_audit_notes
```

Use these statuses:

| Status | Meaning |
|---|---|
| keep | Clearly about food/hospitality as a cultural practice, social norm, or value |
| maybe | Relevant but weak, ambiguous, or missing enough context |
| remove | Only mentions food/hospitality without useful cultural reasoning value |

## What to look for during audit

For each selected row, ask:

```text
Does this row contain a concrete surface practice?          -> possible L1
Does this row contain a social role, norm, or expectation?  -> possible L2
Does this row imply or state a deeper value?                -> possible L3
Could this row help form an L3 -> L2 -> L1 pathway?
```

Good examples should support statements like:

```text
hospitality
-> guest-host obligation
-> offering food to guests
```

or:

```text
respect
-> polite acceptance/refusal norm
-> accepting tea or refusing food gently
```

## What not to keep

Remove rows that only describe:

- personal food preference,
- restaurant choice without cultural context,
- cooking instructions with no norm/value,
- nutrition/dieting without cultural meaning,
- generic mentions of food with no social interaction.

## After audit

Once rows are audited, the next controlled pipeline should be:

```text
food_hospitality_records.jsonl
-> triple extraction on the controlled subset
-> manual triple inspection
-> manual/coded canonicalization map
-> pathway construction
-> pathway retrieval analysis
-> S4 evaluation on matching CulturalBench questions if enough exist
```

Do not scale broadly until the audit shows that the subset has enough high-quality rows to support repeated L1, L2, and L3 nodes.

## Success criteria before broader scaling

Proceed only if:

1. Most selected rows are actually about food/hospitality.
2. Kept rows contain cultural/social meaning, not just food mentions.
3. Triple extraction produces meaningful L1, L2, and L3 nodes.
4. Obvious duplicate nodes can be canonicalized.
5. Some complete L3 -> L2 -> L1 pathways can be identified manually.
6. Matching CulturalBench questions exist, or the subset is clearly useful as a pathway-construction debug set.

## Recommended command sequence

```bash
# 1. Select controlled subset
python scripts/09_select_controlled_subset.py \
  --scan_limit 2000 \
  --max_records 100 \
  --include_culturalbench \
  --cb_limit 250

# 2. Manually audit the CSV
# Open: data/controlled/food_hospitality_audit.csv

# 3. Only after audit, run extraction/pathway construction on this controlled subset.
```
