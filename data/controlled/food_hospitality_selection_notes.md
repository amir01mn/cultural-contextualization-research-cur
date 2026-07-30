# Controlled Subset Selection Notes

Topic: `food_hospitality`

## Purpose

This subset supports controlled scalability before broad KG expansion. Use it to debug extraction, canonicalization, pathway construction, and S4 retrieval behavior on one interpretable cultural topic.

## Selection settings

- CultureBank scan limit: 2000
- Max selected records: 100
- Minimum record score: 3
- Require social/hospitality term: True
- Selected CultureBank records: 100
- Selected CulturalBench questions: 40

## CultureBank culture distribution

| Cultural group | Count |
|---|---:|
| American | 100 |

## CultureBank topic distribution

| Topic | Count |
|---|---:|
| Food and Dining | 33 |
| Social Norms and Etiquette | 26 |
| Consumer Behavior | 10 |
| Cultural Exchange | 6 |
| Family Traditions and Heritage | 4 |
| Cultural Traditions and Festivals | 4 |
| Drinking and Alcohol | 3 |
| Social Interactions | 2 |
| Family Dynamics | 2 |
| Household and Daily Life | 2 |
| Finance and Economy | 2 |
| Miscellaneous | 1 |
| Community and Identity | 1 |
| Work-Life Balance | 1 |
| Cultural and Environmental Appreciation | 1 |
| Lifestyles | 1 |
| Health and Hygiene | 1 |

## Manual audit instructions

Open `food_hospitality_audit.csv` and set `manual_audit_status` to one of:

- `keep`: clearly about food/hospitality as a cultural practice, norm, or value
- `maybe`: relevant but weak, ambiguous, or missing social/value context
- `remove`: only mentions food/hospitality without cultural reasoning value

For kept rows, note whether they support L1 surface practice, L2 social norm, L3 deep value, and possible L3 -> L2 -> L1 pathways.
