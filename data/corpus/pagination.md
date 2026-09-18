# Pagination

List endpoints use cursor pagination.

- Request parameters: limit and cursor
- Default limit: 25
- Maximum limit: 100
- Response fields: data, next_cursor, has_more

Clients should continue fetching while has_more is true and pass next_cursor into the next request.
