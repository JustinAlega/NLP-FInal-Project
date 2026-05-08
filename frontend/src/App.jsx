import { useEffect, useMemo, useState } from 'react'
import { getNeighbors, ingestUrl, searchEntities } from './api'
import './App.css'

function Notice({ children, tone = 'info' }) {
  return <p className={`notice ${tone}`}>{children}</p>
}

function cleanType(type = '') {
  return type.replace('EntityType.', '')
}

function shortName(name = '', maxLength = 22) {
  return name.length > maxLength ? `${name.slice(0, maxLength)}...` : name
}

function buildPaperGraph(oneHop, twoHop) {
  const allowedNodeIds = new Set((oneHop.nodes || []).map((node) => node.id))
  const nodes = (oneHop.nodes || []).slice(0, 40)
  const edgesByKey = new Map()

  for (const edge of [...(oneHop.edges || []), ...(twoHop.edges || [])]) {
    const source = edge.source || edge.source_id
    const target = edge.target || edge.target_id

    if (!allowedNodeIds.has(source) || !allowedNodeIds.has(target)) {
      continue
    }

    edgesByKey.set(`${source}-${edge.relation_type}-${target}`, edge)
  }

  return {
    nodes,
    edges: Array.from(edgesByKey.values()).slice(0, 80),
  }
}

function GraphStats({ nodes = [], edges = [] }) {
  return (
    <div className="graph-stats" aria-label="Graph summary">
      <span>{nodes.length} nodes</span>
      <span>{edges.length} relationships</span>
    </div>
  )
}

function edgeSource(edge) {
  return edge.source || edge.source_id
}

function edgeTarget(edge) {
  return edge.target || edge.target_id
}

function NodeDetails({ node, nodes = [], edges = [] }) {
  const nodeById = useMemo(() => Object.fromEntries(nodes.map((item) => [item.id, item])), [nodes])

  if (!node) {
    return <Notice>Click a node in the graph to see details.</Notice>
  }

  const connectedEdges = edges.filter((edge) => edgeSource(edge) === node.id || edgeTarget(edge) === node.id)
  const aliases = node.aliases?.filter(Boolean) || []
  const attributes = Object.entries(node.attributes || {})

  return (
    <aside className="node-details">
      <div>
        <p className="eyebrow">Selected Node</p>
        <h2>{node.name}</h2>
        <span className="node-type">{cleanType(node.entity_type)}</span>
      </div>

      <dl>
        <div>
          <dt>Confidence</dt>
          <dd>{node.confidence ?? 'n/a'}</dd>
        </div>
        <div>
          <dt>Source</dt>
          <dd>{node.source_doc || 'n/a'}</dd>
        </div>
        {aliases.length > 0 && (
          <div>
            <dt>Aliases</dt>
            <dd>{aliases.join(', ')}</dd>
          </div>
        )}
        {attributes.slice(0, 4).map(([key, value]) => (
          <div key={key}>
            <dt>{key}</dt>
            <dd>{String(value)}</dd>
          </div>
        ))}
      </dl>

      <div>
        <h3>Relationships</h3>
        {connectedEdges.length > 0 ? (
          <ul className="relationship-list">
            {connectedEdges.slice(0, 8).map((edge, index) => {
              const source = nodeById[edgeSource(edge)]
              const target = nodeById[edgeTarget(edge)]
              return (
                <li key={`${edgeSource(edge)}-${edge.relation_type}-${edgeTarget(edge)}-${index}`}>
                  <strong>{source?.name || edgeSource(edge)}</strong>
                  <span>{edge.relation_type}</span>
                  <strong>{target?.name || edgeTarget(edge)}</strong>
                </li>
              )
            })}
          </ul>
        ) : (
          <p className="muted-text">No displayed relationships for this node.</p>
        )}
      </div>
    </aside>
  )
}

function SimpleGraph({
  nodes = [],
  edges = [],
  emptyMessage,
  focusNodeId = '',
  selectedNodeId = '',
  onSelectNode,
}) {
  const layout = useMemo(() => {
    const width = 1280
    const height = 780
    const centerX = width / 2
    const centerY = height / 2
    const radius = Math.min(width, height) / 2 - 120
    const focusNode = nodes.find((node) => node.id === focusNodeId)
    const outerNodes = focusNode ? nodes.filter((node) => node.id !== focusNodeId) : nodes

    const positionedNodes = [
      ...(focusNode ? [{ ...focusNode, x: centerX, y: centerY, focused: true }] : []),
      ...outerNodes.map((node, index) => {
        const angle = (2 * Math.PI * index) / Math.max(outerNodes.length, 1) - Math.PI / 2

        return {
          ...node,
          x: outerNodes.length === 1 && !focusNode ? centerX : centerX + radius * Math.cos(angle),
          y: outerNodes.length === 1 && !focusNode ? centerY : centerY + radius * Math.sin(angle),
        }
      }),
    ]

    return {
      width,
      height,
      nodes: positionedNodes,
      nodeById: Object.fromEntries(positionedNodes.map((node) => [node.id, node])),
    }
  }, [focusNodeId, nodes])

  if (!nodes.length) {
    return <Notice>{emptyMessage}</Notice>
  }

  return (
    <div className="graph-wrap">
      <svg viewBox={`0 0 ${layout.width} ${layout.height}`} role="img" aria-label="Knowledge graph">
        <defs>
          <linearGradient id="node-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#4f46e5" />
            <stop offset="100%" stopColor="#0ea5e9" />
          </linearGradient>
        </defs>
        {edges.map((edge, index) => {
          const source = layout.nodeById[edgeSource(edge)]
          const target = layout.nodeById[edgeTarget(edge)]

          if (!source || !target) {
            return null
          }

          return (
            <g
              key={`${source.id}-${target.id}-${index}`}
              className={
                source.id === selectedNodeId || target.id === selectedNodeId ? 'graph-edge selected' : 'graph-edge'
              }
            >
              <line x1={source.x} y1={source.y} x2={target.x} y2={target.y} />
              <text x={(source.x + target.x) / 2} y={(source.y + target.y) / 2}>
                {edge.relation_type}
              </text>
            </g>
          )
        })}

        {layout.nodes.map((node) => (
          <g
            key={node.id}
            className={`graph-node ${node.focused ? 'focused' : ''} ${
              node.id === selectedNodeId ? 'selected' : ''
            }`}
            role="button"
            tabIndex="0"
            onClick={() => onSelectNode?.(node)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault()
                onSelectNode?.(node)
              }
            }}
          >
            <circle cx={node.x} cy={node.y} r={node.focused ? '48' : '36'} />
            <text x={node.x} y={node.y - 38} textAnchor="middle">
              {cleanType(node.entity_type)}
            </text>
            <text x={node.x} y={node.y + 4} textAnchor="middle">
              {shortName(node.name, node.focused ? 28 : 22)}
            </text>
          </g>
        ))}
      </svg>
    </div>
  )
}

function ExploreGraphPage() {
  const [graph, setGraph] = useState({ nodes: [], edges: [] })
  const [papers, setPapers] = useState([])
  const [selectedPaperId, setSelectedPaperId] = useState('')
  const [selectedNodeId, setSelectedNodeId] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    async function loadPapers() {
      setLoading(true)
      setError('')

      try {
        const result = await searchEntities({ limit: 200 })
        const publications = (result.entities || [])
          .filter((entity) => cleanType(entity.entity_type) === 'PUBLICATION' || entity.entity_type === 'Publication')
          .slice(0, 12)

        setPapers(publications)
        setSelectedPaperId(publications[0]?.id || '')
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }

    loadPapers()
  }, [])

  useEffect(() => {
    async function loadSelectedPaperGraph() {
      if (!selectedPaperId) {
        setGraph({ nodes: [], edges: [] })
        return
      }

      setLoading(true)
      setError('')

      try {
        const [oneHop, twoHop] = await Promise.all([
          getNeighbors(selectedPaperId, { hops: 1, direction: 'both' }),
          getNeighbors(selectedPaperId, { hops: 2, direction: 'both' }),
        ])
        const nextGraph = buildPaperGraph(oneHop, twoHop)
        setGraph(nextGraph)
        setSelectedNodeId(selectedPaperId)
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }

    loadSelectedPaperGraph()
  }, [selectedPaperId])

  const selectedPaper = papers.find((paper) => paper.id === selectedPaperId)
  const selectedNode = graph.nodes.find((node) => node.id === selectedNodeId)

  return (
    <section className="page">
      <header className="page-header">
        <p className="eyebrow">Explore Graph</p>
        <h1>Choose a paper and explore its graph</h1>
        <p>
          Select a seeded paper to view the microplastics entities and relationships extracted from
          it.
        </p>
      </header>

      {error && <Notice tone="error">{error}</Notice>}
      <section className="control-card">
        {papers.length > 0 && (
          <label className="paper-picker">
            Paper
            <select value={selectedPaperId} onChange={(event) => setSelectedPaperId(event.target.value)}>
              {papers.map((paper) => (
                <option key={paper.id} value={paper.id}>
                  {paper.name}
                </option>
              ))}
            </select>
          </label>
        )}
      </section>

      {selectedPaper && (
        <div className="graph-heading">
          <h2>{selectedPaper.name}</h2>
          <GraphStats nodes={graph.nodes} edges={graph.edges} />
        </div>
      )}

      {loading ? (
        <Notice>Loading graph...</Notice>
      ) : (
        <SimpleGraph
          nodes={graph.nodes}
          edges={graph.edges}
          focusNodeId={selectedPaperId}
          selectedNodeId={selectedNodeId}
          onSelectNode={(node) => setSelectedNodeId(node.id)}
          emptyMessage="No graph data is available yet. Run the seed script first."
        />
      )}
      {!loading && <NodeDetails node={selectedNode} nodes={graph.nodes} edges={graph.edges} />}
    </section>
  )
}

function LinkGraphPage() {
  const [url, setUrl] = useState('')
  const [graph, setGraph] = useState(null)
  const [selectedNodeId, setSelectedNodeId] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleSubmit(event) {
    event.preventDefault()
    setLoading(true)
    setError('')
    setGraph(null)
    setSelectedNodeId('')

    try {
      const nextGraph = await ingestUrl({ url, maxChunks: 1 })
      setGraph(nextGraph)
      setSelectedNodeId(
        nextGraph.nodes?.find((node) => cleanType(node.entity_type) === 'PUBLICATION' || node.entity_type === 'Publication')
          ?.id || nextGraph.nodes?.[0]?.id || '',
      )
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const selectedNode = graph?.nodes?.find((node) => node.id === selectedNodeId)
  const focusNodeId = graph?.nodes?.find((node) => cleanType(node.entity_type) === 'PUBLICATION' || node.entity_type === 'Publication')?.id

  return (
    <section className="page">
      <header className="page-header">
        <p className="eyebrow">Build From Link</p>
        <h1>Paste a link and generate a graph</h1>
        <p>Use a PubMed, ACS, ScienceDirect, DOI, HTML article, or PDF URL to build a focused graph.</p>
      </header>

      <section className="control-card">
        <form className="link-form" onSubmit={handleSubmit}>
          <input
            type="url"
            value={url}
            onChange={(event) => setUrl(event.target.value)}
            placeholder="Paste a PubMed, ACS, ScienceDirect, DOI, article, or PDF link"
            required
          />
          <button type="submit" disabled={loading}>
            {loading ? 'Building...' : 'Build graph'}
          </button>
        </form>
      </section>

      {error && <Notice tone="error">{error}</Notice>}
      {graph?.title && (
        <div className="graph-heading">
          <h2>{graph.title}</h2>
          <GraphStats nodes={graph.nodes} edges={graph.edges} />
        </div>
      )}
      <SimpleGraph
        nodes={graph?.nodes || []}
        edges={graph?.edges || []}
        focusNodeId={focusNodeId}
        selectedNodeId={selectedNodeId}
        onSelectNode={(node) => setSelectedNodeId(node.id)}
        emptyMessage="Your generated graph will appear here."
      />
      {graph && <NodeDetails node={selectedNode} nodes={graph.nodes} edges={graph.edges} />}
    </section>
  )
}

function App() {
  const [page, setPage] = useState('explore')

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">MicroKG</p>
          <h1>Microplastics Knowledge Graph</h1>
          <p>Explore seeded research graphs or generate one from a research article link.</p>
        </div>
        <nav className="screen-nav" aria-label="App pages">
          <button
            type="button"
            className={page === 'explore' ? 'active' : ''}
            onClick={() => setPage('explore')}
          >
            Explore Graph
          </button>
          <button
            type="button"
            className={page === 'link' ? 'active' : ''}
            onClick={() => setPage('link')}
          >
            Build From Link
          </button>
        </nav>
      </header>

      {page === 'explore' ? <ExploreGraphPage /> : <LinkGraphPage />}
    </main>
  )
}

export default App
