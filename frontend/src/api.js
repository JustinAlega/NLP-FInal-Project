const API_BASE_URL = import.meta.env.VITE_API_URL || '/api'

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  })

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`

    try {
      const body = await response.json()
      message = body.detail || message
    } catch {
      // Keep the HTTP status message when the API does not return JSON.
    }

    throw new Error(message)
  }

  return response.json()
}

export function searchEntities({ query = '', entityType = '', limit = 25 }) {
  const params = new URLSearchParams({
    q: query,
    entity_type: entityType,
    limit: String(limit),
  })

  return request(`/entities?${params.toString()}`)
}

export function getNeighbors(entityId, { hops = 1, direction = 'both' } = {}) {
  const params = new URLSearchParams({
    hops: String(hops),
    direction,
  })

  return request(`/entities/${encodeURIComponent(entityId)}/neighbors?${params.toString()}`)
}

export function ingestUrl({ url, maxChunks = 3 }) {
  return request('/ingest/url', {
    method: 'POST',
    body: JSON.stringify({
      url,
      max_chunks: maxChunks,
    }),
  })
}
