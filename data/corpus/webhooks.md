# Webhooks

Webhook events are delivered at-least-once.

- Verify X-Signature using HMAC-SHA256 over timestamp + raw body.
- Reject payloads older than 5 minutes to prevent replay attacks.
- De-duplicate events using event_id.

Delivery retries:

- Up to 8 retry attempts over 24 hours.
- A 2xx response marks delivery as successful.
