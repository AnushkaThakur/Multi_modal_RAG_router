# Charge Creation and Safety

Create charge endpoint: POST /v1/charges

- Required fields: amount, currency, source
- Optional fields: customer_id, metadata

Operational guidance:

- Always attach Idempotency-Key when creating charges.
- On transient failures (5xx, timeout), retry with the same idempotency key.
- If rate limited (429), wait per Retry-After and then retry.
