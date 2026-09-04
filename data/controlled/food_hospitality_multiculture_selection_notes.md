# Culture-Matched Food/Hospitality Subset Notes

## Purpose

This file summarizes the CultureBank food/hospitality records selected for the cultures that appear in the selected CulturalBench food/hospitality questions.
The goal is to make the next S0-S4 evaluation culture-matched, rather than relying on an American-only KG corpus.

## Settings

- Topic: `food_hospitality`
- Questions file: `data/controlled/food_hospitality_culturalbench_questions.jsonl`
- CultureBank scan limit: 20000
- Per-culture maximum: 50
- Minimum record score: 3
- Require social/hospitality term: True
- Match full record text for culture aliases: True
- Combined selected records: 196

## Target cultures from selected CulturalBench questions

| Target culture | Selected CultureBank records | Status |
|---|---:|---|
| south korea | 15 | usable candidate coverage |
| iran | 3 | low coverage; inspect manually |
| bangladesh | 1 | low coverage; inspect manually |
| spain | 23 | usable candidate coverage |
| indonesia | 23 | usable candidate coverage |
| zimbabwe | 0 | no CultureBank coverage found |
| netherlands | 30 | usable candidate coverage |
| new zealand | 5 | usable candidate coverage |
| turkey | 16 | usable candidate coverage |
| china | 15 | usable candidate coverage |
| pakistan | 5 | usable candidate coverage |
| south africa | 1 | low coverage; inspect manually |
| japan | 33 | usable candidate coverage |
| philippines | 14 | usable candidate coverage |
| russia | 15 | usable candidate coverage |
| brazil | 7 | usable candidate coverage |

## Cultures with no or low selected coverage

- No selected records: zimbabwe
- Low selected coverage: iran, bangladesh, south africa

## Most common CultureBank cultural_group values scanned

| CultureBank cultural_group | Scanned count |
|---|---:|
| americans | 2321 |
| germans | 400 |
| canadians | 280 |
| netherlands | 253 |
| japan | 234 |
| french | 227 |
| filipinos | 211 |
| british | 200 |
| europeans | 197 |
| russians | 175 |
| indonesians | 165 |
| chinese | 165 |
| indians | 147 |
| italians | 146 |
| australians | 141 |
| south korea | 123 |
| brazilians | 109 |
| malaysians | 106 |
| mexicans | 75 |
| hungarians | 72 |
| united states | 69 |
| spaniards | 69 |
| irish | 64 |
| muslims | 63 |
| norwegians | 59 |
| québécois | 54 |
| african americans | 50 |
| asian americans | 50 |
| british people | 43 |
| swedish | 43 |

## Manual audit instructions

Open the audit CSV and set `manual_audit_status` to one of:

- `keep`: culturally meaningful food/hospitality evidence for the matched target culture
- `maybe`: relevant but weak, ambiguous, or missing enough social/value context
- `remove`: incidental food/hospitality mention, wrong culture, or not useful for cultural reasoning

During audit, check whether each row supports L1 surface practice, L2 social norm/role, L3 deep value, and a plausible pathway.
No outside cultural knowledge should be added during this audit; use the normalized fields and source text only.
