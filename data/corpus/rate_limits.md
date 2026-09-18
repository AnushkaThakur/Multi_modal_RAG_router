# Rate Limits and Retries

Acme Payments enforces account-level request limits.

- Default limit: 120 requests per minute
- Burst limit: 20 requests in 2 seconds
- Exceeded limit returns HTTP 429

Rate-limit headers:

- X-RateLimit-Limit
- X-RateLimit-Remaining
- Retry-After

Retry guidance:

- Use exponential backoff with jitter.
- Respect Retry-After when present.
