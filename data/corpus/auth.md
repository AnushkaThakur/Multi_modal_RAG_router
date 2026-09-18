# Authentication and Authorization

Acme Payments API uses API-key based bearer auth.

- Header format: Authorization: Bearer <api_key>
- Keys are scoped as read_only, write_payments, or admin.
- Rotate keys every 90 days.

The API also supports idempotency keys for write operations.

- Header format: Idempotency-Key: <client_generated_uuid>
- Reusing the same idempotency key with the same payload returns the first successful result.
- Using the same key with a different payload returns HTTP 409.
