from __future__ import annotations

import json
from typing import Any, Dict, List
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse

from core.auth import RoleChecker
from core.database import get_collection
from shared.constants import COLLECTION_CALLS, COLLECTION_MEMORY_LOGS, COLLECTION_STRATEGIES
from core.config import settings

from .live_calls import live_calls

router = APIRouter()


def _serialise(docs: Any) -> Any:
    return json.loads(json.dumps(docs, default=str))


@router.get("/ops", response_class=HTMLResponse)
async def ops_dashboard(_: Dict[str, Any] = Depends(RoleChecker(["admin"]))):
    html = """
    <!doctype html>
    <html>
    <head>
        <meta charset=\"utf-8\" />
        <title>Scrappy Singh Ops Console</title>
        <script crossorigin src=\"https://unpkg.com/react@18/umd/react.development.js\"></script>
        <script crossorigin src=\"https://unpkg.com/react-dom@18/umd/react-dom.development.js\"></script>
        <script src=\"https://unpkg.com/babel-standalone@6/babel.min.js\"></script>
        <style>
            body { font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 0; background: #0f172a; color: #e2e8f0; }
            header { padding: 24px; background: #1e293b; border-bottom: 1px solid #334155; }
            h1 { margin: 0; font-size: 24px; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; padding: 24px; }
            .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 16px; box-shadow: 0 10px 30px rgba(15,23,42,0.4); }
            .card h2 { margin-top: 0; font-size: 18px; }
            table { width: 100%; border-collapse: collapse; }
            th, td { text-align: left; padding: 6px; font-size: 13px; border-bottom: 1px solid #334155; }
            .badge { display: inline-block; padding: 2px 8px; border-radius: 8px; font-size: 11px; background: #2563eb; color: white; }
            .status-in_progress { background: #22c55e; }
            .status-completed { background: #64748b; }
            pre { background: rgba(148,163,184,0.12); padding: 8px; border-radius: 8px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <header>
            <h1>Scrappy Singh Ops Console</h1>
        </header>
        <div id=\"root\"></div>
        <script type=\"text/babel\">
            const { useEffect, useState } = React;

            function usePoll(url, intervalMs = 5000) {
                const [data, setData] = useState({ loading: true, payload: null, error: null });
                useEffect(() => {
                    let mounted = true;
                    async function fetchData() {
                        try {
                            const res = await fetch(url, { credentials: 'include' });
                            if (!res.ok) throw new Error(`Request failed: ${res.status}`);
                            const json = await res.json();
                            if (mounted) setData({ loading: false, payload: json, error: null });
                        } catch (err) {
                            if (mounted) setData({ loading: false, payload: null, error: err.message });
                        }
                    }
                    fetchData();
                    const id = setInterval(fetchData, intervalMs);
                    return () => { mounted = false; clearInterval(id); };
                }, [url, intervalMs]);
                return data;
            }

            function LiveCallsCard() {
                const { loading, payload, error } = usePoll('/api/v1/ops/live_calls', 3000);
                const calls = payload?.calls || [];
                return (
                    <div className=\"card\">
                        <h2>Active Calls</h2>
                        {loading && <p>Loading…</p>}
                        {error && <p style={{ color: '#f87171' }}>{error}</p>}
                        {!loading && !error && calls.length === 0 && <p>No live calls.</p>}
                        {calls.length > 0 && (
                            <table>
                                <thead>
                                    <tr>
                                        <th>Call ID</th>
                                        <th>Speaker</th>
                                        <th>Status</th>
                                        <th>Last Update</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {calls.map(call => (
                                        <tr key={call.call_id}>
                                            <td>{call.call_id}</td>
                                            <td>{call.current_speaker || '—'}</td>
                                            <td><span className={\`badge status-${call.status || 'in_progress'}\`}>{call.status || 'in_progress'}</span></td>
                                            <td>{call.updated_at}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        )}
                    </div>
                );
            }

            function FeedbackTimelineCard() {
                const { loading, payload, error } = usePoll('/api/v1/ops/feedback_timeline', 6000);
            const items = payload?.timeline || [];
            return (
                <div className=\"card\">
                    <h2>Feedback Timeline</h2>
                        {loading && <p>Loading…</p>}
                        {error && <p style={{ color: '#f87171' }}>{error}</p>}
                        {!loading && !error && items.length === 0 && <p>No feedback captured today.</p>}
                        <ul>
                            {items.map(item => (
                                <li key={item.id} style={{ marginBottom: 8 }}>
                                    <span className=\"badge\" style={{ marginRight: 8 }}>{item.event_type}</span>
                                    <strong>{item.timestamp}</strong> — {item.summary}
                                </li>
                            ))}
                        </ul>
                    </div>
                );
            }

            function PersonaSummaryCard() {
                const { loading, payload, error } = usePoll('/api/v1/ops/persona_summary', 10000);
                const persona = payload?.persona;
                return (
                    <div className=\"card\">
                        <h2>Persona Snapshot</h2>
                        {loading && <p>Loading…</p>}
                        {error && <p style={{ color: '#f87171' }}>{error}</p>}
                        {persona && (
                            <pre>{JSON.stringify(persona, null, 2)}</pre>
                        )}
                    </div>
                );
            }

            function ExcelLogsCard() {
                const { loading, payload, error } = usePoll('/api/v1/ops/excel_logs', 15000);
                const files = payload?.files || [];
                return (
                    <div className=\"card\">
                        <h2>Exported Excel Logs</h2>
                        {loading && <p>Loading…</p>}
                        {error && <p style={{ color: '#f87171' }}>{error}</p>}
                        {files.length === 0 && !loading && !error && <p>No exports generated.</p>}
                        <ul>
                            {files.map(file => (
                                <li key={file.name}>
                                    <a style={{ color: '#60a5fa' }} href={`/api/v1/ops/excel_logs/${file.name}`} target="_blank">{file.name}</a>
                                    <small style={{ display: 'block', opacity: 0.7 }}>{file.updated_at}</small>
                                </li>
                            ))}
                        </ul>
                    </div>
                );
            }

            function Dashboard() {
                return (
                    <div className=\"grid\">
                        <LiveCallsCard />
                        <FeedbackTimelineCard />
                        <PersonaSummaryCard />
                        <ExcelLogsCard />
                    </div>
                );
            }

            const root = ReactDOM.createRoot(document.getElementById('root'));
            root.render(<Dashboard />);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html)


@router.get("/api/v1/ops/live_calls")
async def get_live_calls(_: Dict[str, Any] = Depends(RoleChecker(["admin"]))):
    calls = await live_calls.snapshot()
    return {"calls": calls}


@router.websocket("/api/v1/ops/live_calls/ws")
async def live_calls_ws(websocket: WebSocket):
    await websocket.accept()
    queue = await live_calls.subscribe()
    try:
        while True:
            event = await queue.get()
            await websocket.send_text(json.dumps(event))
    except WebSocketDisconnect:
        pass
    finally:
        await live_calls.unsubscribe(queue)


@router.get("/api/v1/ops/agent_memory/{agent_id}")
async def get_agent_memory(agent_id: str, _: Dict[str, Any] = Depends(RoleChecker(["admin"]))):
    collection = get_collection(COLLECTION_MEMORY_LOGS)
    cursor = collection.find({"agent_id": agent_id}).sort("created_at", -1).limit(50)
    docs = await cursor.to_list(length=50)
    return {"memories": _serialise(docs)}


@router.get("/api/v1/ops/persona_summary")
async def persona_summary(_: Dict[str, Any] = Depends(RoleChecker(["admin"]))):
    collection = get_collection(COLLECTION_STRATEGIES)
    doc = await collection.find_one(sort=[("updated_at", -1)])
    if not doc:
        raise HTTPException(status_code=404, detail="No persona data found")
    persona = ((doc.get("payload") or {}).get("persona") or {})
    return {"persona": _serialise(persona)}


@router.get("/api/v1/ops/feedback_timeline")
async def feedback_timeline(_: Dict[str, Any] = Depends(RoleChecker(["admin"]))):
    collection = get_collection(COLLECTION_MEMORY_LOGS)
    cursor = collection.find({"event_type": {"$in": ["feedback", "train_loop", "fallback_triggered", "summary"]}}).sort("created_at", -1).limit(40)
    docs = await cursor.to_list(length=40)
    timeline: List[Dict[str, Any]] = []
    for doc in docs:
        timeline.append(
            {
                "id": doc.get("memory_id"),
                "event_type": doc.get("event_type", "feedback"),
                "timestamp": doc.get("created_at", datetime.utcnow()).isoformat() if isinstance(doc.get("created_at"), datetime) else str(doc.get("created_at")),
                "summary": doc.get("data") or doc.get("feedback") or doc.get("extra") or "",
            }
        )
    return {"timeline": _serialise(timeline)}


@router.get("/api/v1/ops/excel_logs")
async def excel_logs(_: Dict[str, Any] = Depends(RoleChecker(["admin"]))):
    base = Path(settings.CALL_LOG_DIR)
    if not base.exists():
        return {"files": []}
    files = sorted(base.glob("*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
    payload = [
        {
            "name": f.name,
            "size": f.stat().st_size,
            "updated_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
        }
        for f in files[:20]
    ]
    return {"files": payload}


@router.get("/api/v1/ops/excel_logs/{filename}")
async def download_excel(filename: str, _: Dict[str, Any] = Depends(RoleChecker(["admin"]))):
    base = Path(settings.CALL_LOG_DIR)
    path = base / filename
    if not path.exists() or path.suffix.lower() != ".xlsx":
        raise HTTPException(status_code=404, detail="Log not found")
    return FileResponse(path)
