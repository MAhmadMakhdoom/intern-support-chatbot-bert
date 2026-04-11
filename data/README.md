# Dataset Documentation

## Files

| File | Rows | Description |
|---|---|---|
| `intern_dataset.csv` | 98 | FAQ documents — 14 per intent |
| `support_tickets.csv` | 35 | Historical support tickets — 5 per intent |
| `intern_dataset_full.csv` | 133 | Combined training dataset |

## Schema

| Column | Description |
|---|---|
| user_input | Raw intern question |
| intent | One of 7 intent labels |
| answer | Official answer text |
| cleaned_input | Preprocessed lowercase question |
| intent_label | Numeric label (0–6) |

## Intent Distribution (perfectly balanced)

| Intent | Count |
|---|---|
| working_hours | 19 |
| leave_request | 19 |
| stipend_query | 19 |
| it_support | 19 |
| credential_issue | 19 |
| hr_policy | 19 |
| general_query | 19 |
| **Total** | **133** |
