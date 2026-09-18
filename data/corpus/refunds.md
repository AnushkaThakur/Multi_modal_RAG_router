# Refund Lifecycle

Refund creation is asynchronous.

- Initial status: pending
- Terminal statuses: succeeded, failed, canceled
- Partial refunds are supported until the charge is fully refunded.

A charge cannot be refunded beyond its captured amount.
