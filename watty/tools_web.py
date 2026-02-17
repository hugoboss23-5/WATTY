"""
Watty Web Dashboard v2.1
========================
One tool: watty_web(action=start/stop/status).
Local HTTP server with live brain visualization + chat.
Apple-quality interactive dashboard.
February 2026
"""

import json
import threading
import uuid
import webbrowser
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

from mcp.types import Tool, TextContent

from watty.config import WATTY_HOME


# ── Chat helpers (shared with tools_comms) ────────────────
CHAT_FILE = WATTY_HOME / "chat.jsonl"

def _now_utc():
    return datetime.now(timezone.utc).isoformat()

def _now_local():
    return datetime.now().strftime("%I:%M:%S %p")

def _chat_append(entry: dict):
    CHAT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHAT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def _chat_read(last_n: int = 50) -> list[dict]:
    if not CHAT_FILE.exists():
        return []
    lines = CHAT_FILE.read_text(encoding="utf-8").strip().split("\n")
    lines = [l for l in lines if l.strip()]
    if last_n:
        lines = lines[-last_n:]
    msgs = []
    for line in lines:
        try:
            msgs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return msgs


# ── State ───────────────────────────────────────────────────

_server_instance = None
_server_thread = None
_server_port = 7777
_brain_cache = None

class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


# ── API Handler ─────────────────────────────────────────────

class _DashboardHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _json_response(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _html_response(self, html):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(html.encode())

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except Exception:
            return {}

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/" or path == "/dashboard":
            self._html_response(_DASHBOARD_HTML)
        elif path == "/brain":
            self._html_response(_BRAIN_VIEWER_HTML)
        elif path == "/api/stats":
            self._handle_stats()
        elif path == "/api/memories":
            self._handle_memories(params)
        elif path == "/api/memory":
            self._handle_memory_detail(params)
        elif path == "/api/graph":
            self._handle_graph()
        elif path == "/api/graph/full":
            self._handle_graph_full(params)
        elif path == "/api/contradictions":
            self._handle_contradictions()
        elif path == "/api/tiers":
            self._handle_tiers()
        elif path == "/api/search":
            self._handle_search(params)
        elif path == "/api/chat/history":
            self._handle_chat_history(params)
        elif path == "/api/chat/poll":
            self._handle_chat_poll(params)
        elif path == "/api/openapi.json":
            self._json_response(_OPENAPI_SPEC)
        elif path == "/api":
            self._html_response(_API_DOCS_HTML)
        elif path == "/navigator":
            self._html_response(_NAVIGATOR_HTML)
        elif path == "/api/navigate":
            self._handle_navigate(params)
        elif path == "/api/plasticity":
            self._handle_plasticity()
        elif path == "/watcher":
            self._html_response(_WATCHER_HTML)
        elif path == "/api/watcher/observations":
            self._handle_watcher_observations(params)
        elif path == "/api/watcher/status":
            self._handle_watcher_status()
        elif path == "/graph":
            self._html_response(_GRAPH_HTML)
        elif path == "/api/graph/entities":
            self._handle_graph_entities(params)
        elif path == "/api/graph/entity":
            self._handle_graph_entity_detail(params)
        elif path == "/api/graph/traverse":
            self._handle_graph_traverse(params)
        elif path == "/api/graph/stats":
            self._handle_graph_kg_stats()
        elif path == "/eval":
            self._html_response(_EVAL_HTML)
        elif path == "/api/eval/trends":
            self._handle_eval_trends(params)
        elif path == "/api/eval/alerts":
            self._handle_eval_alerts()
        elif path == "/api/eval/stats":
            self._handle_eval_stats()
        elif path == "/trading":
            self._html_response(_TRADING_HTML)
        elif path == "/api/trading/portfolio":
            self._handle_trading_portfolio()
        elif path == "/api/trading/market":
            self._handle_trading_market(params)
        elif path == "/api/trading/chain":
            self._handle_trading_chain(params)
        elif path == "/api/trading/analyze":
            self._handle_trading_analyze(params)
        else:
            self._json_response({"error": "Not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        body = self._read_body()

        if path == "/api/remember":
            self._handle_remember(body)
        elif path == "/api/forget":
            self._handle_forget(body)
        elif path == "/api/scan":
            self._handle_scan(body)
        elif path == "/api/dream":
            self._handle_dream()
        elif path == "/api/resolve":
            self._handle_resolve(body)
        elif path == "/api/chat/send":
            self._handle_chat_send(body)
        elif path == "/api/trading/buy":
            self._handle_trading_buy(body)
        elif path == "/api/trading/sell":
            self._handle_trading_sell(body)
        elif path == "/api/eval/ack_alert":
            self._handle_eval_ack_alert(body)
        else:
            self._json_response({"error": "Not found"}, 404)

    def _get_brain(self):
        global _brain_cache
        if _brain_cache is None:
            from watty.brain import Brain
            _brain_cache = Brain()
        return _brain_cache

    # ── Watcher endpoints ──────────────────────────────

    def _handle_watcher_observations(self, params):
        from watty.tools_watcher import _load_recent
        n = int(params.get("n", ["50"])[0])
        observations = _load_recent(n)
        self._json_response({"observations": observations, "count": len(observations)})

    def _handle_watcher_status(self):
        from watty.tools_watcher import _watcher, _load_config
        s = _watcher.status()
        s["config"] = _load_config()
        self._json_response(s)

    # ── Knowledge Graph endpoints ─────────────────────

    def _handle_graph_entities(self, params):
        brain = self._get_brain()
        if brain._kg is None:
            self._json_response({"error": "Knowledge graph not enabled"})
            return
        query = params.get("q", [""])[0]
        top_k = int(params.get("limit", ["50"])[0])
        try:
            results = brain._kg.entity_search(query, top_k=top_k) if query else []
            # If no query, get all entities
            if not query:
                conn = brain._kg._connect()
                try:
                    rows = conn.execute(
                        "SELECT id, name, entity_type, mention_count, description FROM entities ORDER BY mention_count DESC LIMIT ?",
                        (top_k,)
                    ).fetchall()
                    results = [dict(r) for r in rows]
                finally:
                    conn.close()
            self._json_response({"entities": results})
        except Exception as e:
            self._json_response({"error": str(e)})

    def _handle_graph_entity_detail(self, params):
        brain = self._get_brain()
        if brain._kg is None:
            self._json_response({"error": "Knowledge graph not enabled"})
            return
        entity_id = int(params.get("id", ["0"])[0])
        try:
            entity = brain._kg.get_entity(entity_id)
            if entity is None:
                self._json_response({"error": "Entity not found"}, 404)
                return
            self._json_response(entity)
        except Exception as e:
            self._json_response({"error": str(e)})

    def _handle_graph_traverse(self, params):
        brain = self._get_brain()
        if brain._kg is None:
            self._json_response({"error": "Knowledge graph not enabled"})
            return
        name = params.get("name", [""])[0]
        max_hops = int(params.get("hops", ["2"])[0])
        if not name:
            self._json_response({"error": "Need 'name' parameter"})
            return
        try:
            result = brain._kg.traverse_graph(name, max_hops=max_hops)
            self._json_response(result)
        except Exception as e:
            self._json_response({"error": str(e)})

    def _handle_graph_kg_stats(self):
        brain = self._get_brain()
        if brain._kg is None:
            self._json_response({"error": "Knowledge graph not enabled"})
            return
        try:
            conn = brain._kg._connect()
            try:
                entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
                rel_count = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
                types = conn.execute(
                    "SELECT entity_type, COUNT(*) as cnt FROM entities GROUP BY entity_type ORDER BY cnt DESC"
                ).fetchall()
                rel_types = conn.execute(
                    "SELECT relationship_type, COUNT(*) as cnt FROM relationships GROUP BY relationship_type ORDER BY cnt DESC"
                ).fetchall()
                self._json_response({
                    "entities": entity_count,
                    "relationships": rel_count,
                    "entity_types": {r[0]: r[1] for r in types},
                    "relationship_types": {r[0]: r[1] for r in rel_types},
                })
            finally:
                conn.close()
        except Exception as e:
            self._json_response({"error": str(e)})

    # ── Evaluation endpoints ──────────────────────────

    def _handle_eval_trends(self, params):
        brain = self._get_brain()
        if brain._eval is None:
            self._json_response({"error": "Evaluation engine not enabled"})
            return
        metric = params.get("metric", [""])[0]
        category = params.get("category", [""])[0]
        days = int(params.get("days", ["30"])[0])
        try:
            trends = brain._eval.get_trends(metric_name=metric or None, category=category or None, days=days)
            self._json_response({"trends": trends})
        except Exception as e:
            self._json_response({"error": str(e)})

    def _handle_eval_alerts(self):
        brain = self._get_brain()
        if brain._eval is None:
            self._json_response({"error": "Evaluation engine not enabled"})
            return
        try:
            alerts = brain._eval.get_alerts(include_acknowledged=False)
            self._json_response({"alerts": alerts})
        except Exception as e:
            self._json_response({"error": str(e)})

    def _handle_eval_stats(self):
        brain = self._get_brain()
        if brain._eval is None:
            self._json_response({"error": "Evaluation engine not enabled"})
            return
        try:
            stats = brain._eval.get_stats()
            self._json_response(stats)
        except Exception as e:
            self._json_response({"error": str(e)})

    def _handle_eval_ack_alert(self, body):
        brain = self._get_brain()
        if brain._eval is None:
            self._json_response({"error": "Evaluation engine not enabled"})
            return
        alert_id = body.get("alert_id")
        if not alert_id:
            self._json_response({"error": "Need 'alert_id'"})
            return
        try:
            brain._eval.acknowledge_alert(int(alert_id))
            self._json_response({"ok": True})
        except Exception as e:
            self._json_response({"error": str(e)})

    # ── Trading endpoints ──────────────────────────────

    def _get_trade_module(self):
        from watty.tools_trade import execute_trade, _load_portfolio
        return execute_trade, _load_portfolio

    def _handle_trading_portfolio(self):
        try:
            execute_trade, _load_portfolio = self._get_trade_module()
            portfolio = _load_portfolio()
            # Enrich open positions with current prices
            import yfinance as yf
            total_value = portfolio["cash"]
            for pos in portfolio["open_positions"]:
                try:
                    t = yf.Ticker(pos["ticker"])
                    chain = t.option_chain(pos["expiry"])
                    df = chain.calls if pos["type"] == "call" else chain.puts
                    match = df[df["strike"] == pos["strike"]]
                    if not match.empty:
                        row = match.iloc[0]
                        bid = row.get("bid", 0) or row.get("lastPrice", 0)
                        pos["current_price"] = float(bid) if bid else 0
                        pos["current_value"] = pos["current_price"] * 100 * pos["contracts"]
                        pos["pnl"] = pos["current_value"] - pos["cost"]
                        pos["pnl_pct"] = (pos["pnl"] / pos["cost"] * 100) if pos["cost"] else 0
                        total_value += pos["current_value"]
                    else:
                        pos["current_price"] = 0
                        pos["current_value"] = 0
                        pos["pnl"] = -pos["cost"]
                        pos["pnl_pct"] = -100
                except Exception:
                    pos["current_price"] = 0
                    pos["current_value"] = 0
                    pos["pnl"] = -pos["cost"]
                    pos["pnl_pct"] = -100

            closed_pnl = sum(c.get("pnl", 0) for c in portfolio.get("closed_positions", []))
            open_pnl = sum(p.get("pnl", 0) for p in portfolio["open_positions"])
            wins = sum(1 for c in portfolio.get("closed_positions", []) if c.get("pnl", 0) > 0)
            total_closed = len(portfolio.get("closed_positions", []))

            portfolio["total_value"] = total_value
            portfolio["total_pnl"] = closed_pnl + open_pnl
            portfolio["total_pnl_pct"] = ((total_value - 50000) / 50000 * 100)
            portfolio["win_rate"] = (wins / total_closed * 100) if total_closed else 0
            portfolio["wins"] = wins
            portfolio["losses"] = total_closed - wins
            self._json_response(portfolio)
        except Exception as e:
            self._json_response({"error": str(e)})

    def _handle_trading_market(self, params):
        try:
            execute_trade, _ = self._get_trade_module()
            ticker = params.get("ticker", ["SPY"])[0]
            result = execute_trade("market", {"ticker": ticker})
            self._json_response({"result": result})
        except Exception as e:
            self._json_response({"error": str(e)})

    def _handle_trading_chain(self, params):
        try:
            execute_trade, _ = self._get_trade_module()
            ticker = params.get("ticker", ["SPY"])[0]
            expiry = params.get("expiry", [None])[0]
            p = {"ticker": ticker}
            if expiry:
                p["expiry"] = expiry
            result = execute_trade("options_chain", p)
            self._json_response({"result": result})
        except Exception as e:
            self._json_response({"error": str(e)})

    def _handle_trading_analyze(self, params):
        try:
            execute_trade, _ = self._get_trade_module()
            ticker = params.get("ticker", ["SPY"])[0]
            result = execute_trade("analyze", {"ticker": ticker})
            self._json_response({"result": result})
        except Exception as e:
            self._json_response({"error": str(e)})

    def _handle_trading_buy(self, body):
        try:
            execute_trade, _ = self._get_trade_module()
            result = execute_trade("paper_buy", body)
            self._json_response({"result": result})
        except Exception as e:
            self._json_response({"error": str(e)})

    def _handle_trading_sell(self, body):
        try:
            execute_trade, _ = self._get_trade_module()
            result = execute_trade("paper_sell", body)
            self._json_response({"result": result})
        except Exception as e:
            self._json_response({"error": str(e)})

    # ── Read endpoints ──────────────────────────────────

    def _handle_stats(self):
        brain = self._get_brain()
        stats = brain.stats()
        conn = brain._connect()
        assoc_count = conn.execute("SELECT COUNT(*) FROM associations").fetchone()[0]
        contra_count = conn.execute("SELECT COUNT(*) FROM novelty_log WHERE is_contradiction = 1 AND resolved = 0").fetchone()[0]
        episodic = conn.execute("SELECT COUNT(*) FROM chunks WHERE memory_tier = 'episodic'").fetchone()[0]
        consolidated = conn.execute("SELECT COUNT(*) FROM chunks WHERE memory_tier = 'consolidated'").fetchone()[0]
        conn.close()
        stats["associations"] = assoc_count
        stats["contradictions"] = contra_count
        stats["episodic"] = episodic
        stats["consolidated"] = consolidated
        self._json_response(stats)

    def _handle_memories(self, params):
        brain = self._get_brain()
        conn = brain._connect()
        limit = int(params.get("limit", ["50"])[0])
        offset = int(params.get("offset", ["0"])[0])
        provider = params.get("provider", [None])[0]
        tier = params.get("tier", [None])[0]
        query = "SELECT id, content, provider, source_type, source_path, created_at, memory_tier, access_count, significance FROM chunks"
        conditions = []
        qparams = []
        if provider:
            conditions.append("provider = ?")
            qparams.append(provider)
        if tier:
            conditions.append("memory_tier = ?")
            qparams.append(tier)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        qparams.extend([limit, offset])
        rows = conn.execute(query, qparams).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM chunks" + (" WHERE " + " AND ".join(conditions) if conditions else ""), qparams[:-2] if conditions else []).fetchone()[0]
        conn.close()
        memories = [{
            "id": r["id"], "content": r["content"][:500],
            "provider": r["provider"], "source_type": r["source_type"],
            "source_path": r["source_path"], "created_at": r["created_at"],
            "tier": r["memory_tier"], "access_count": r["access_count"],
            "significance": r["significance"],
        } for r in rows]
        self._json_response({"memories": memories, "count": len(memories), "total": total})

    def _handle_memory_detail(self, params):
        chunk_id = params.get("id", [None])[0]
        if not chunk_id:
            self._json_response({"error": "No id"}, 400)
            return
        brain = self._get_brain()
        conn = brain._connect()
        row = conn.execute(
            "SELECT id, content, provider, source_type, source_path, created_at, memory_tier, access_count, significance, content_hash FROM chunks WHERE id = ?",
            (int(chunk_id),)
        ).fetchone()
        if not row:
            conn.close()
            self._json_response({"error": "Not found"}, 404)
            return
        assocs = conn.execute(
            "SELECT target_chunk_id, strength FROM associations WHERE source_chunk_id = ? UNION SELECT source_chunk_id, strength FROM associations WHERE target_chunk_id = ?",
            (int(chunk_id), int(chunk_id))
        ).fetchall()
        conn.close()
        self._json_response({
            "id": row["id"], "content": row["content"],
            "provider": row["provider"], "source_type": row["source_type"],
            "source_path": row["source_path"], "created_at": row["created_at"],
            "tier": row["memory_tier"], "access_count": row["access_count"],
            "significance": row["significance"],
            "associations": [{"chunk_id": a[0], "strength": round(a[1], 3)} for a in assocs],
        })

    def _handle_graph(self):
        brain = self._get_brain()
        conn = brain._connect()
        chunks = conn.execute(
            "SELECT id, content, provider, memory_tier, access_count FROM chunks ORDER BY access_count DESC LIMIT 150"
        ).fetchall()
        assocs = conn.execute(
            "SELECT source_chunk_id, target_chunk_id, strength FROM associations WHERE strength > 0.2 ORDER BY strength DESC LIMIT 500"
        ).fetchall()
        conn.close()
        chunk_ids = {r["id"] for r in chunks}
        nodes = [{"id": r["id"], "label": r["content"][:80].replace('"', "'"), "provider": r["provider"], "tier": r["memory_tier"], "access": r["access_count"]} for r in chunks]
        edges = [{"source": r["source_chunk_id"], "target": r["target_chunk_id"], "strength": round(r["strength"], 3)} for r in assocs if r["source_chunk_id"] in chunk_ids and r["target_chunk_id"] in chunk_ids]
        self._json_response({"nodes": nodes, "edges": edges})

    def _handle_graph_full(self, params):
        """Full graph for brain viewer — up to 600 nodes with cluster and compression info."""
        brain = self._get_brain()
        conn = brain._connect()
        limit = int(params.get("limit", ["600"])[0])
        # Get nodes: prioritize high-access and consolidated
        chunks = conn.execute(
            "SELECT id, content, provider, memory_tier, access_count, significance, "
            "created_at, source_type, compressed_content, compression_ratio "
            "FROM chunks ORDER BY "
            "CASE WHEN memory_tier = 'consolidated' THEN 0 "
            "WHEN memory_tier = 'schema' THEN 1 ELSE 2 END, "
            "access_count DESC LIMIT ?",
            (limit,)
        ).fetchall()
        chunk_ids = {r["id"] for r in chunks}
        # Get associations between visible nodes
        assocs = conn.execute(
            "SELECT source_chunk_id, target_chunk_id, strength "
            "FROM associations WHERE strength > 0.15 ORDER BY strength DESC LIMIT 2000"
        ).fetchall()
        # Get cluster info
        clusters_raw = conn.execute("SELECT label, chunk_ids FROM clusters").fetchall()
        # Compression stats
        compressed_count = conn.execute("SELECT COUNT(*) FROM chunks WHERE compressed_content IS NOT NULL").fetchone()[0]
        total_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        conn.close()

        # Build cluster membership map
        cluster_map = {}
        clusters_out = []
        for i, c in enumerate(clusters_raw):
            try:
                ids = json.loads(c["chunk_ids"])
                clusters_out.append({"id": i, "label": c["label"][:60], "size": len(ids)})
                for cid in ids:
                    cluster_map[cid] = i
            except (json.JSONDecodeError, TypeError):
                pass

        nodes = []
        for r in chunks:
            nodes.append({
                "id": r["id"],
                "label": (r["content"] or "")[:100].replace('"', "'").replace('\n', ' '),
                "provider": r["provider"],
                "tier": r["memory_tier"] or "episodic",
                "access": r["access_count"] or 0,
                "significance": round(r["significance"] or 0, 3),
                "created": r["created_at"],
                "source_type": r["source_type"],
                "compressed": r["compressed_content"] is not None,
                "ratio": round(r["compression_ratio"] or 1.0, 2),
                "cluster": cluster_map.get(r["id"], -1),
            })

        edges = [
            {"source": r["source_chunk_id"], "target": r["target_chunk_id"], "strength": round(r["strength"], 3)}
            for r in assocs
            if r["source_chunk_id"] in chunk_ids and r["target_chunk_id"] in chunk_ids
        ]

        self._json_response({
            "nodes": nodes,
            "edges": edges,
            "clusters": clusters_out,
            "meta": {
                "total_memories": total_count,
                "visible": len(nodes),
                "compressed": compressed_count,
                "total_edges": len(edges),
            }
        })

    def _handle_contradictions(self):
        brain = self._get_brain()
        contras = brain.get_contradictions()
        self._json_response({"contradictions": contras})

    def _handle_tiers(self):
        brain = self._get_brain()
        conn = brain._connect()
        tiers = conn.execute("SELECT memory_tier, COUNT(*) as count FROM chunks GROUP BY memory_tier").fetchall()
        providers = conn.execute("SELECT provider, COUNT(*) as count FROM chunks GROUP BY provider ORDER BY count DESC").fetchall()
        conn.close()
        self._json_response({
            "tiers": {r["memory_tier"]: r["count"] for r in tiers},
            "providers": {r["provider"]: r["count"] for r in providers},
        })

    def _handle_search(self, params):
        query = params.get("q", [""])[0]
        if not query:
            self._json_response({"results": [], "error": "No query"})
            return
        brain = self._get_brain()
        results = brain.recall(query, top_k=20)
        self._json_response({"results": results, "query": query})

    def _handle_navigate(self, params):
        query = params.get("q", [""])[0]
        if not query:
            self._json_response({"error": "No query", "results": []})
            return
        top_k = int(params.get("top_k", ["10"])[0])
        brain = self._get_brain()
        from watty.navigator import Navigator
        nav = Navigator(brain)
        result = nav.navigate(query, top_k=top_k)
        # Simplify organ readings for JSON serialization
        organ_summary = []
        for readings in result.get("organ_readings", []):
            organ_summary.append({
                "coherence": readings[0] if len(readings) > 0 else {},
                "depth": readings[1] if len(readings) > 1 else {},
                "bridge": readings[2] if len(readings) > 2 else {},
            })
        self._json_response({
            "query": query,
            "results": result.get("results", []),
            "circulations": result.get("circulations", 0),
            "final_coherence": result.get("final_coherence", 0),
            "heart_readings": result.get("heart_readings", []),
            "blood_strategies": result.get("blood_strategies", []),
            "organ_readings": organ_summary,
        })

    def _handle_plasticity(self):
        brain = self._get_brain()
        report = brain.chestahedron.plasticity_report()
        faces = ["TEMPORAL", "STRUCTURAL", "RELATIONAL", "SEMANTIC", "META", "INVERSE", "EXTERNAL"]
        report["face_names"] = faces
        report["hippocampus"] = brain.chesta_hippocampus.save_state()
        self._json_response(report)

    # ── Write endpoints ──────────────────────────────────

    def _handle_remember(self, body):
        content = body.get("content", "")
        provider = body.get("provider", "http_api")
        if not content.strip():
            self._json_response({"error": "No content"}, 400)
            return
        brain = self._get_brain()
        chunks = brain.store_memory(content, provider=provider)
        self._json_response({"stored": True, "chunks": chunks})

    def _handle_forget(self, body):
        brain = self._get_brain()
        result = brain.forget(
            query=body.get("query"), chunk_ids=body.get("chunk_ids"),
            provider=body.get("provider"), before=body.get("before"),
        )
        self._json_response(result)

    def _handle_scan(self, body):
        path = body.get("path", "")
        if not path:
            self._json_response({"error": "No path"}, 400)
            return
        brain = self._get_brain()
        result = brain.scan_directory(path, recursive=body.get("recursive", True))
        self._json_response(result)

    def _handle_dream(self):
        brain = self._get_brain()
        result = brain.dream()
        self._json_response(result)

    def _handle_resolve(self, body):
        chunk_id = body.get("chunk_id")
        keep = body.get("keep", "new")
        if chunk_id is None:
            self._json_response({"error": "No chunk_id"}, 400)
            return
        brain = self._get_brain()
        result = brain.resolve_contradiction(chunk_id, keep=keep)
        self._json_response(result)

    # ── Chat endpoints ──────────────────────────────────

    def _handle_chat_send(self, body):
        message = body.get("message", "").strip()
        if not message:
            self._json_response({"error": "No message"}, 400)
            return
        entry = {
            "id": f"web_{uuid.uuid4().hex[:8]}",
            "timestamp": _now_utc(),
            "time_local": _now_local(),
            "from": "web",
            "message": message,
        }
        _chat_append(entry)
        self._json_response({"sent": True, "id": entry["id"]})

    def _handle_chat_history(self, params):
        last_n = int(params.get("last_n", ["50"])[0])
        msgs = _chat_read(last_n)
        self._json_response({"messages": msgs})

    def _handle_chat_poll(self, params):
        after = params.get("after", [""])[0]
        msgs = _chat_read(100)
        if after:
            msgs = [m for m in msgs if m.get("timestamp", "") > after]
        self._json_response({"messages": msgs})


# ── OpenAPI Spec ──────────────────────────────────────────

_OPENAPI_SPEC = {
    "openapi": "3.1.0",
    "info": {"title": "Watty Brain API", "version": "2.1.0", "description": "HTTP API for Watty memory system."},
    "servers": [{"url": "http://localhost:7777"}],
    "paths": {
        "/api/search": {"get": {"operationId": "searchMemory", "summary": "Semantic search", "parameters": [{"name": "q", "in": "query", "required": True, "schema": {"type": "string"}}], "responses": {"200": {"description": "Results"}}}},
        "/api/stats": {"get": {"operationId": "getStats", "summary": "Brain stats", "responses": {"200": {"description": "Stats"}}}},
        "/api/memories": {"get": {"operationId": "listMemories", "summary": "List memories", "parameters": [{"name": "limit", "in": "query", "schema": {"type": "integer", "default": 50}}, {"name": "offset", "in": "query", "schema": {"type": "integer", "default": 0}}, {"name": "provider", "in": "query", "schema": {"type": "string"}}, {"name": "tier", "in": "query", "schema": {"type": "string"}}], "responses": {"200": {"description": "Memories"}}}},
        "/api/memory": {"get": {"operationId": "getMemory", "summary": "Single memory detail", "parameters": [{"name": "id", "in": "query", "required": True, "schema": {"type": "integer"}}], "responses": {"200": {"description": "Memory"}}}},
        "/api/remember": {"post": {"operationId": "remember", "summary": "Store memory", "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {"content": {"type": "string"}, "provider": {"type": "string", "default": "http_api"}}, "required": ["content"]}}}}, "responses": {"200": {"description": "OK"}}}},
        "/api/forget": {"post": {"operationId": "forget", "summary": "Delete memories", "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {"query": {"type": "string"}, "chunk_ids": {"type": "array", "items": {"type": "integer"}}, "provider": {"type": "string"}, "before": {"type": "string"}}}}}}, "responses": {"200": {"description": "OK"}}}},
        "/api/graph": {"get": {"operationId": "getGraph", "summary": "Knowledge graph", "responses": {"200": {"description": "Graph"}}}},
        "/api/tiers": {"get": {"operationId": "getTiers", "summary": "Tier breakdown", "responses": {"200": {"description": "Tiers"}}}},
        "/api/contradictions": {"get": {"operationId": "getContradictions", "summary": "Contradictions", "responses": {"200": {"description": "Contradictions"}}}},
        "/api/dream": {"post": {"operationId": "dream", "summary": "Run dream cycle", "responses": {"200": {"description": "Results"}}}},
        "/api/scan": {"post": {"operationId": "scan", "summary": "Scan directory", "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {"path": {"type": "string"}, "recursive": {"type": "boolean", "default": True}}, "required": ["path"]}}}}, "responses": {"200": {"description": "Results"}}}},
        "/api/chat/send": {"post": {"operationId": "chatSend", "summary": "Send chat message", "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}}}}, "responses": {"200": {"description": "OK"}}}},
        "/api/chat/history": {"get": {"operationId": "chatHistory", "summary": "Chat history", "parameters": [{"name": "last_n", "in": "query", "schema": {"type": "integer", "default": 50}}], "responses": {"200": {"description": "Messages"}}}},
        "/api/chat/poll": {"get": {"operationId": "chatPoll", "summary": "Poll chat", "parameters": [{"name": "after", "in": "query", "schema": {"type": "string"}}], "responses": {"200": {"description": "Messages"}}}},
    }
}


# ── API Docs Page ──────────────────────────────────────────

_API_DOCS_HTML = """<!DOCTYPE html><html><head><title>Watty API</title>
<style>body{background:#000;color:#f5f5f7;font-family:-apple-system,BlinkMacSystemFont,'SF Pro Display','Segoe UI',system-ui,sans-serif;padding:40px;max-width:800px;margin:0 auto;-webkit-font-smoothing:antialiased}
h1{font-size:32px;font-weight:700;letter-spacing:-0.5px}h2{font-size:18px;font-weight:600;color:#86868b;margin-top:36px;text-transform:uppercase;letter-spacing:1px;font-size:12px}
.e{background:#1c1c1e;border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:16px 20px;margin:10px 0;transition:background 200ms}.e:hover{background:#242424}
.m{display:inline-block;padding:3px 10px;border-radius:6px;font-weight:600;font-size:11px;margin-right:10px;letter-spacing:0.5px}
.get{background:rgba(48,209,88,0.12);color:#30d158}.post{background:rgba(110,86,207,0.12);color:#6e56cf}
.p{color:#f5f5f7;font-weight:500}.d{color:#86868b;font-size:13px;margin-top:6px}a{color:#6e56cf}</style></head>
<body><h1>Watty Brain API</h1><p style="color:#86868b">v2.1.0 · <a href="/api/openapi.json">OpenAPI spec</a> · <a href="/">Dashboard</a></p>
<h2>Read</h2>
<div class="e"><span class="m get">GET</span><span class="p">/api/search?q=query</span><div class="d">Semantic search across all memories</div></div>
<div class="e"><span class="m get">GET</span><span class="p">/api/stats</span><div class="d">Brain health stats</div></div>
<div class="e"><span class="m get">GET</span><span class="p">/api/memories?limit=50&offset=0&provider=&tier=</span><div class="d">List memories with filters</div></div>
<div class="e"><span class="m get">GET</span><span class="p">/api/memory?id=123</span><div class="d">Full memory detail with associations</div></div>
<div class="e"><span class="m get">GET</span><span class="p">/api/graph</span><div class="d">Knowledge graph nodes + edges</div></div>
<div class="e"><span class="m get">GET</span><span class="p">/api/tiers</span><div class="d">Memory tier and provider breakdown</div></div>
<div class="e"><span class="m get">GET</span><span class="p">/api/contradictions</span><div class="d">Unresolved contradictions</div></div>
<h2>Write</h2>
<div class="e"><span class="m post">POST</span><span class="p">/api/remember</span><div class="d">Store a memory. Body: {"content": "..."}</div></div>
<div class="e"><span class="m post">POST</span><span class="p">/api/forget</span><div class="d">Delete memories. Body: {"chunk_ids": [1,2]} or {"query": "..."}</div></div>
<div class="e"><span class="m post">POST</span><span class="p">/api/dream</span><div class="d">Run consolidation cycle</div></div>
<div class="e"><span class="m post">POST</span><span class="p">/api/resolve</span><div class="d">Resolve contradiction. Body: {"chunk_id": 1, "keep": "new"}</div></div>
<div class="e"><span class="m post">POST</span><span class="p">/api/scan</span><div class="d">Scan directory. Body: {"path": "/some/dir"}</div></div>
<h2>Chat</h2>
<div class="e"><span class="m post">POST</span><span class="p">/api/chat/send</span><div class="d">Send message. Body: {"message": "..."}</div></div>
<div class="e"><span class="m get">GET</span><span class="p">/api/chat/history?last_n=50</span><div class="d">Chat history</div></div>
<div class="e"><span class="m get">GET</span><span class="p">/api/chat/poll?after=ISO</span><div class="d">Poll for new messages</div></div>
</body></html>"""


# ── Trading Dashboard HTML ─────────────────────────────────

_TRADING_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>watty — trading desk</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:100%;min-height:100vh;background:#0a0a0f;font-family:-apple-system,BlinkMacSystemFont,'SF Pro','Segoe UI',sans-serif;color:#e0e0e0;-webkit-font-smoothing:antialiased}

/* Header */
.header{padding:20px 32px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid rgba(139,92,246,0.1)}
.header h1{font-size:22px;font-weight:700;letter-spacing:1px;background:linear-gradient(135deg,#34d399,#6ee7b7);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.header .sub{font-size:11px;color:#555;letter-spacing:1px;margin-top:2px}
.header .live-dot{width:8px;height:8px;border-radius:50%;background:#34d399;display:inline-block;margin-right:8px;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
.header-right{display:flex;align-items:center;gap:16px}
.refresh-label{font-size:11px;color:#555}

/* Stats bar */
.stats-bar{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;padding:20px 32px}
.stat-card{background:rgba(20,20,35,0.8);border:1px solid rgba(255,255,255,0.04);border-radius:14px;padding:16px 20px;transition:border-color 0.3s}
.stat-card:hover{border-color:rgba(139,92,246,0.2)}
.stat-card .label{font-size:11px;color:#666;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px}
.stat-card .value{font-size:24px;font-weight:700;letter-spacing:-0.5px}
.stat-card .sub{font-size:12px;margin-top:4px}
.green{color:#34d399}.red{color:#f87171}.dim{color:#555}.white{color:#f5f5f7}

/* Main layout */
.main{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:0 32px 32px}
.panel{background:rgba(20,20,35,0.8);border:1px solid rgba(255,255,255,0.04);border-radius:14px;padding:20px;overflow:hidden}
.panel-title{font-size:13px;font-weight:600;color:#888;text-transform:uppercase;letter-spacing:1px;margin-bottom:16px;display:flex;align-items:center;gap:8px}
.panel-title .icon{font-size:16px}

/* Positions table */
.pos-table{width:100%;border-collapse:collapse}
.pos-table th{font-size:10px;color:#555;text-transform:uppercase;letter-spacing:1px;padding:8px 12px;text-align:left;border-bottom:1px solid rgba(255,255,255,0.04)}
.pos-table td{font-size:13px;padding:10px 12px;border-bottom:1px solid rgba(255,255,255,0.02)}
.pos-table tr:hover td{background:rgba(139,92,246,0.03)}

/* Closed trades */
.trade-row{display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.02)}
.trade-row:last-child{border:none}
.trade-info{font-size:13px;color:#ccc}
.trade-badge{font-size:11px;font-weight:600;padding:3px 10px;border-radius:8px}
.trade-badge.win{background:rgba(52,211,153,0.12);color:#34d399}
.trade-badge.loss{background:rgba(248,113,113,0.12);color:#f87171}

/* Market data */
.market-line{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.02);font-size:13px}
.market-line:last-child{border:none}
.market-line .k{color:#888}
.market-line .v{color:#e0e0e0;font-weight:500}

/* Actions bar */
.actions-bar{padding:16px 32px;display:flex;gap:8px;flex-wrap:wrap}
.act-btn{background:rgba(20,20,35,0.8);border:1px solid rgba(139,92,246,0.15);border-radius:10px;padding:10px 20px;color:#a0a0b0;font-size:12px;cursor:pointer;transition:all 0.2s;font-family:inherit}
.act-btn:hover{border-color:rgba(139,92,246,0.4);color:#c4b5fd;background:rgba(30,30,50,0.8)}
.act-btn.primary{border-color:rgba(52,211,153,0.3);color:#34d399}
.act-btn.primary:hover{border-color:#34d399;background:rgba(52,211,153,0.08)}
.act-btn.danger{border-color:rgba(248,113,113,0.3);color:#f87171}
.act-btn.danger:hover{border-color:#f87171;background:rgba(248,113,113,0.08)}

/* Modal */
.modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.7);backdrop-filter:blur(8px);z-index:100;align-items:center;justify-content:center}
.modal-overlay.open{display:flex}
.modal{background:#16161f;border:1px solid rgba(139,92,246,0.15);border-radius:16px;padding:28px;width:420px;max-width:90vw}
.modal h2{font-size:18px;font-weight:600;margin-bottom:16px;color:#f5f5f7}
.modal label{font-size:12px;color:#888;display:block;margin:12px 0 4px;text-transform:uppercase;letter-spacing:0.5px}
.modal input,.modal select{width:100%;padding:10px 14px;background:rgba(30,30,50,0.8);border:1px solid rgba(255,255,255,0.06);border-radius:8px;color:#e0e0e0;font-size:14px;font-family:inherit;outline:none;transition:border-color 0.2s}
.modal input:focus,.modal select:focus{border-color:rgba(139,92,246,0.4)}
.modal-actions{display:flex;gap:8px;margin-top:20px;justify-content:flex-end}

/* Empty state */
.empty{text-align:center;padding:40px;color:#444;font-size:13px}

/* Full width panel */
.full-width{grid-column:1/-1}

/* Scrollable */
.scroll-y{max-height:400px;overflow-y:auto}
.scroll-y::-webkit-scrollbar{width:4px}
.scroll-y::-webkit-scrollbar-track{background:transparent}
.scroll-y::-webkit-scrollbar-thumb{background:rgba(139,92,246,0.2);border-radius:2px}

/* Status toast */
#toast{position:fixed;bottom:24px;left:50%;transform:translateX(-50%);background:rgba(20,20,40,0.95);backdrop-filter:blur(12px);border:1px solid rgba(139,92,246,0.25);border-radius:12px;padding:12px 24px;font-size:13px;color:#c4b5fd;z-index:200;opacity:0;transition:opacity 0.3s;pointer-events:none}
#toast.show{opacity:1}

/* Responsive */
@media(max-width:900px){.stats-bar{grid-template-columns:repeat(3,1fr)}.main{grid-template-columns:1fr}}
@media(max-width:600px){.stats-bar{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>WATTY TRADING DESK</h1>
    <div class="sub">PAPER TRADING &middot; OPTIONS &middot; LIVE</div>
  </div>
  <div class="header-right">
    <span class="refresh-label">Auto-refresh <span id="countdown">30</span>s</span>
    <span class="live-dot"></span>
    <span style="font-size:12px;color:#34d399;font-weight:600">LIVE</span>
  </div>
</div>

<div class="stats-bar">
  <div class="stat-card">
    <div class="label">Portfolio Value</div>
    <div class="value white" id="s-total">$50,000</div>
    <div class="sub dim">starting capital</div>
  </div>
  <div class="stat-card">
    <div class="label">Cash</div>
    <div class="value white" id="s-cash">$50,000</div>
    <div class="sub dim" id="s-cash-pct">100%</div>
  </div>
  <div class="stat-card">
    <div class="label">Open P&L</div>
    <div class="value" id="s-opnl">$0</div>
    <div class="sub dim" id="s-opnl-pct">0%</div>
  </div>
  <div class="stat-card">
    <div class="label">Total P&L</div>
    <div class="value" id="s-tpnl">$0</div>
    <div class="sub dim" id="s-tpnl-pct">0% from $50k</div>
  </div>
  <div class="stat-card">
    <div class="label">Win Rate</div>
    <div class="value white" id="s-wr">—</div>
    <div class="sub dim" id="s-wl">0W / 0L</div>
  </div>
  <div class="stat-card">
    <div class="label">Total Trades</div>
    <div class="value white" id="s-trades">0</div>
    <div class="sub dim" id="s-open-ct">0 open</div>
  </div>
</div>

<div class="actions-bar">
  <button class="act-btn primary" onclick="openBuyModal()">+ Buy Option</button>
  <button class="act-btn" onclick="refreshData()">Refresh Now</button>
  <button class="act-btn" onclick="showMarket()">Market Data</button>
  <button class="act-btn" onclick="showAnalysis()">Analysis</button>
  <a href="/" class="act-btn" style="text-decoration:none">Back to Brain</a>
</div>

<div class="main">
  <div class="panel">
    <div class="panel-title"><span class="icon">📊</span> Open Positions</div>
    <div class="scroll-y" id="open-positions">
      <div class="empty">No open positions. Click "Buy Option" to start trading.</div>
    </div>
  </div>

  <div class="panel">
    <div class="panel-title"><span class="icon">📈</span> Closed Trades</div>
    <div class="scroll-y" id="closed-trades">
      <div class="empty">No closed trades yet.</div>
    </div>
  </div>

  <div class="panel full-width" id="market-panel" style="display:none">
    <div class="panel-title"><span class="icon">🌎</span> Market Data</div>
    <div id="market-data" style="white-space:pre-wrap;font-family:monospace;font-size:13px;color:#ccc;line-height:1.8"></div>
  </div>

  <div class="panel full-width" id="analysis-panel" style="display:none">
    <div class="panel-title"><span class="icon">🧠</span> Analysis</div>
    <div id="analysis-data" style="white-space:pre-wrap;font-family:monospace;font-size:13px;color:#ccc;line-height:1.8"></div>
  </div>
</div>

<!-- Buy Modal -->
<div class="modal-overlay" id="buy-modal">
  <div class="modal">
    <h2>Buy Option (Paper)</h2>
    <label>Ticker</label>
    <input id="m-ticker" value="SPY" />
    <label>Type</label>
    <select id="m-type"><option value="call">Call</option><option value="put">Put</option></select>
    <label>Strike</label>
    <input id="m-strike" type="number" step="0.5" placeholder="e.g. 605" />
    <label>Expiry (YYYY-MM-DD)</label>
    <input id="m-expiry" placeholder="leave blank for nearest" />
    <label>Contracts</label>
    <input id="m-contracts" type="number" value="1" min="1" />
    <div class="modal-actions">
      <button class="act-btn" onclick="closeBuyModal()">Cancel</button>
      <button class="act-btn primary" onclick="executeBuy()">Buy</button>
    </div>
  </div>
</div>

<!-- Sell Modal -->
<div class="modal-overlay" id="sell-modal">
  <div class="modal">
    <h2>Sell Position</h2>
    <p id="sell-info" style="font-size:13px;color:#aaa;margin-bottom:12px"></p>
    <input type="hidden" id="sell-id" />
    <div class="modal-actions">
      <button class="act-btn" onclick="closeSellModal()">Cancel</button>
      <button class="act-btn danger" onclick="executeSell()">Sell</button>
    </div>
  </div>
</div>

<div id="toast"></div>

<script>
const API = '';
let refreshTimer;
let countdown = 30;

function toast(msg, dur=3000) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), dur);
}

function fmt(n) {
  if (n == null || isNaN(n)) return '$0';
  return '$' + Number(n).toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2});
}

function pnlClass(n) { return n >= 0 ? 'green' : 'red'; }

async function fetchPortfolio() {
  try {
    const r = await fetch(API + '/api/trading/portfolio');
    const d = await r.json();
    if (d.error) { toast('Error: ' + d.error); return; }
    updateUI(d);
  } catch(e) { toast('Failed to fetch portfolio'); }
}

function updateUI(p) {
  // Stats
  document.getElementById('s-total').textContent = fmt(p.total_value);
  document.getElementById('s-total').className = 'value ' + (p.total_value >= 50000 ? 'green' : 'red');
  document.getElementById('s-cash').textContent = fmt(p.cash);
  document.getElementById('s-cash-pct').textContent = ((p.cash/50000)*100).toFixed(0) + '% of starting';

  const opnl = p.open_positions.reduce((s,x) => s + (x.pnl||0), 0);
  document.getElementById('s-opnl').textContent = fmt(opnl);
  document.getElementById('s-opnl').className = 'value ' + pnlClass(opnl);
  document.getElementById('s-opnl-pct').textContent = (opnl >= 0 ? '+' : '') + ((opnl/50000)*100).toFixed(2) + '%';

  document.getElementById('s-tpnl').textContent = fmt(p.total_pnl);
  document.getElementById('s-tpnl').className = 'value ' + pnlClass(p.total_pnl);
  document.getElementById('s-tpnl-pct').textContent = (p.total_pnl >= 0 ? '+' : '') + p.total_pnl_pct.toFixed(2) + '% from $50k';

  if (p.wins + p.losses > 0) {
    document.getElementById('s-wr').textContent = p.win_rate.toFixed(0) + '%';
    document.getElementById('s-wr').className = 'value ' + (p.win_rate >= 50 ? 'green' : 'red');
    document.getElementById('s-wl').textContent = p.wins + 'W / ' + p.losses + 'L';
  }

  document.getElementById('s-trades').textContent = p.total_trades;
  document.getElementById('s-open-ct').textContent = p.open_positions.length + ' open';

  // Open positions
  const opDiv = document.getElementById('open-positions');
  if (!p.open_positions.length) {
    opDiv.innerHTML = '<div class="empty">No open positions. Click "Buy Option" to start.</div>';
  } else {
    let h = '<table class="pos-table"><thead><tr><th>ID</th><th>Position</th><th>Entry</th><th>Current</th><th>P&L</th><th></th></tr></thead><tbody>';
    for (const pos of p.open_positions) {
      const pnl = pos.pnl || 0;
      const pnlPct = pos.pnl_pct || 0;
      h += '<tr>';
      h += '<td style="color:#888;font-family:monospace">' + pos.id + '</td>';
      h += '<td>' + pos.contracts + 'x ' + pos.ticker + ' $' + pos.strike + ' ' + pos.type.toUpperCase() + '<br><span style="color:#555;font-size:11px">' + pos.expiry + '</span></td>';
      h += '<td>' + fmt(pos.entry_price) + '</td>';
      h += '<td>' + fmt(pos.current_price) + '</td>';
      h += '<td class="' + pnlClass(pnl) + '">' + fmt(pnl) + '<br><span style="font-size:11px">' + (pnl>=0?'+':'') + pnlPct.toFixed(1) + '%</span></td>';
      h += '<td><button class="act-btn danger" style="padding:6px 12px;font-size:11px" onclick="openSellModal(\'' + pos.id + '\',\'' + pos.contracts + 'x ' + pos.ticker + ' $' + pos.strike + ' ' + pos.type + '\')">Sell</button></td>';
      h += '</tr>';
    }
    h += '</tbody></table>';
    opDiv.innerHTML = h;
  }

  // Closed trades
  const clDiv = document.getElementById('closed-trades');
  const closed = p.closed_positions || [];
  if (!closed.length) {
    clDiv.innerHTML = '<div class="empty">No closed trades yet.</div>';
  } else {
    let h = '';
    for (const c of closed.slice().reverse().slice(0, 20)) {
      const pnl = c.pnl || 0;
      const isWin = pnl > 0;
      h += '<div class="trade-row">';
      h += '<div class="trade-info">' + c.contracts + 'x ' + c.ticker + ' $' + c.strike + ' ' + c.type.toUpperCase() + ' <span style="color:#555;font-size:11px">' + (c.closed||'').slice(0,10) + '</span></div>';
      h += '<div><span class="trade-badge ' + (isWin?'win':'loss') + '">' + (isWin?'+':'') + fmt(pnl) + ' (' + (pnl>=0?'+':'') + (c.pnl_pct||0).toFixed(1) + '%)</span></div>';
      h += '</div>';
    }
    clDiv.innerHTML = h;
  }
}

// Buy modal
function openBuyModal() { document.getElementById('buy-modal').classList.add('open'); }
function closeBuyModal() { document.getElementById('buy-modal').classList.remove('open'); }

async function executeBuy() {
  const body = {
    ticker: document.getElementById('m-ticker').value || 'SPY',
    option_type: document.getElementById('m-type').value,
    strike: parseFloat(document.getElementById('m-strike').value),
    contracts: parseInt(document.getElementById('m-contracts').value) || 1,
  };
  const exp = document.getElementById('m-expiry').value;
  if (exp) body.expiry = exp;

  if (!body.strike) { toast('Need a strike price'); return; }

  closeBuyModal();
  toast('Placing order...');
  try {
    const r = await fetch(API + '/api/trading/buy', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    const d = await r.json();
    toast(d.result || d.error || 'Done', 5000);
    setTimeout(fetchPortfolio, 500);
  } catch(e) { toast('Buy failed: ' + e.message); }
}

// Sell modal
function openSellModal(id, info) {
  document.getElementById('sell-id').value = id;
  document.getElementById('sell-info').textContent = 'Close position: ' + info + ' (ID: ' + id + ')';
  document.getElementById('sell-modal').classList.add('open');
}
function closeSellModal() { document.getElementById('sell-modal').classList.remove('open'); }

async function executeSell() {
  const id = document.getElementById('sell-id').value;
  closeSellModal();
  toast('Selling...');
  try {
    const r = await fetch(API + '/api/trading/sell', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({position_id: id})});
    const d = await r.json();
    toast(d.result || d.error || 'Done', 5000);
    setTimeout(fetchPortfolio, 500);
  } catch(e) { toast('Sell failed: ' + e.message); }
}

// Market data
async function showMarket() {
  const panel = document.getElementById('market-panel');
  panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
  if (panel.style.display === 'block') {
    document.getElementById('market-data').textContent = 'Loading...';
    try {
      const r = await fetch(API + '/api/trading/market');
      const d = await r.json();
      document.getElementById('market-data').textContent = d.result || d.error;
    } catch(e) { document.getElementById('market-data').textContent = 'Failed'; }
  }
}

async function showAnalysis() {
  const panel = document.getElementById('analysis-panel');
  panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
  if (panel.style.display === 'block') {
    document.getElementById('analysis-data').textContent = 'Analyzing...';
    try {
      const r = await fetch(API + '/api/trading/analyze');
      const d = await r.json();
      document.getElementById('analysis-data').textContent = d.result || d.error;
    } catch(e) { document.getElementById('analysis-data').textContent = 'Failed'; }
  }
}

function refreshData() {
  countdown = 30;
  fetchPortfolio();
  toast('Refreshed');
}

// Auto-refresh
function startTimer() {
  clearInterval(refreshTimer);
  countdown = 30;
  refreshTimer = setInterval(() => {
    countdown--;
    document.getElementById('countdown').textContent = countdown;
    if (countdown <= 0) {
      countdown = 30;
      fetchPortfolio();
    }
  }, 1000);
}

// Init
fetchPortfolio();
startTimer();
</script>
</body>
</html>"""


# ── Brain Viewer HTML ──────────────────────────────────────

_BRAIN_VIEWER_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Watty Brain Viewer</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#030308;overflow:hidden;font-family:-apple-system,BlinkMacSystemFont,'SF Pro Display',system-ui,sans-serif;color:#e0e0e0;-webkit-font-smoothing:antialiased}
canvas{display:block;cursor:grab}
canvas:active{cursor:grabbing}
#hud{position:fixed;top:20px;left:20px;z-index:10;pointer-events:none}
#hud>*{pointer-events:auto}
.title{font-size:28px;font-weight:700;letter-spacing:-0.5px;background:linear-gradient(135deg,#7c3aed,#a78bfa,#c4b5fd);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px}
.subtitle{font-size:11px;color:#666;letter-spacing:1px;text-transform:uppercase;margin-bottom:16px}
.stats{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px}
.stat{background:rgba(124,58,237,0.08);border:1px solid rgba(124,58,237,0.15);border-radius:10px;padding:8px 14px;min-width:90px}
.stat-val{font-size:20px;font-weight:700;color:#a78bfa}
.stat-label{font-size:10px;color:#666;text-transform:uppercase;letter-spacing:0.5px}
.controls{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap}
.btn{background:rgba(124,58,237,0.12);border:1px solid rgba(124,58,237,0.25);color:#c4b5fd;padding:6px 14px;border-radius:8px;font-size:12px;cursor:pointer;transition:all 200ms;font-family:inherit}
.btn:hover{background:rgba(124,58,237,0.25);border-color:rgba(124,58,237,0.5)}
.btn.active{background:rgba(124,58,237,0.35);border-color:#7c3aed}
.btn.dream{background:rgba(34,197,94,0.12);border-color:rgba(34,197,94,0.25);color:#86efac}
.btn.dream:hover{background:rgba(34,197,94,0.25)}
.btn.dream.running{animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.5}}
.search-box{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:6px 12px;color:#e0e0e0;font-size:12px;width:200px;font-family:inherit;outline:none;transition:border-color 200ms}
.search-box:focus{border-color:rgba(124,58,237,0.5)}
#detail{position:fixed;bottom:20px;left:20px;right:20px;max-width:600px;background:rgba(10,10,20,0.92);border:1px solid rgba(124,58,237,0.2);border-radius:14px;padding:16px 20px;z-index:10;display:none;backdrop-filter:blur(20px);max-height:40vh;overflow-y:auto}
#detail.show{display:block}
.detail-title{font-size:14px;font-weight:600;color:#c4b5fd;margin-bottom:6px;display:flex;justify-content:space-between;align-items:center}
.detail-close{cursor:pointer;color:#666;font-size:18px;padding:0 4px}
.detail-close:hover{color:#fff}
.detail-content{font-size:12px;color:#999;line-height:1.6;white-space:pre-wrap;word-break:break-word}
.detail-meta{display:flex;gap:8px;margin-top:8px;flex-wrap:wrap}
.detail-tag{font-size:10px;padding:2px 8px;border-radius:4px;background:rgba(255,255,255,0.05);color:#888}
.legend{position:fixed;top:20px;right:20px;z-index:10;background:rgba(10,10,20,0.8);border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:12px 16px;backdrop-filter:blur(10px)}
.legend-item{display:flex;align-items:center;gap:8px;margin:4px 0;font-size:11px;color:#888}
.legend-dot{width:10px;height:10px;border-radius:50%}
.toast{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:rgba(124,58,237,0.9);color:white;padding:16px 32px;border-radius:12px;font-size:16px;font-weight:600;z-index:100;opacity:0;transition:opacity 300ms;pointer-events:none}
.toast.show{opacity:1}
</style>
</head>
<body>
<canvas id="c"></canvas>

<div id="hud">
  <div class="title">Watty Brain</div>
  <div class="subtitle" id="sub">Loading...</div>
  <div class="stats" id="stats"></div>
  <div class="controls">
    <input type="text" class="search-box" id="search" placeholder="Search memories...">
    <button class="btn dream" id="dreamBtn" onclick="triggerDream()">Dream</button>
    <button class="btn" id="toggleEdges" onclick="toggleEdges()">Edges</button>
    <button class="btn" id="toggleLabels" onclick="toggleLabels()">Labels</button>
    <button class="btn" onclick="resetView()">Reset</button>
  </div>
</div>

<div class="legend" id="legend"></div>
<div id="detail"></div>
<div class="toast" id="toast"></div>

<script>
// ── Constants ──
const PROVIDERS = {
  file_scan: {color:'#3b82f6', label:'Files'},
  manual:    {color:'#22c55e', label:'Manual'},
  claude:    {color:'#a855f7', label:'Claude'},
  http_api:  {color:'#f59e0b', label:'API'},
  test:      {color:'#6b7280', label:'Test'},
  cognition: {color:'#ec4899', label:'Cognition'},
};
const TIER_SCALE = {episodic: 1, consolidated: 1.8, schema: 2.5};
const BG = '#030308';

// ── State ──
let nodes = [], edges = [], clusters = [], meta = {};
let showEdges = true, showLabels = false;
let cam = {x: 0, y: 0, zoom: 1};
let drag = null, hover = null, selected = null;
let searchQuery = '', searchResults = new Set();
let simRunning = true, simAlpha = 1;
let dreamRunning = false;
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
let W, H;

// ── Resize ──
function resize() {
  W = canvas.width = window.innerWidth;
  H = canvas.height = window.innerHeight;
}
window.addEventListener('resize', resize);
resize();

// ── Data Loading ──
async function loadData() {
  try {
    const [graphRes, statsRes, tiersRes] = await Promise.all([
      fetch('/api/graph/full').then(r => r.json()),
      fetch('/api/stats').then(r => r.json()),
      fetch('/api/tiers').then(r => r.json()),
    ]);

    meta = graphRes.meta || {};
    clusters = graphRes.clusters || [];

    // Build node map
    const nodeMap = new Map();
    const cx = W / 2, cy = H / 2;

    graphRes.nodes.forEach((n, i) => {
      // Initial position: cluster-based radial layout
      const cl = n.cluster >= 0 ? n.cluster : i;
      const angle = (cl / Math.max(clusters.length, 1)) * Math.PI * 2 + (Math.random() - 0.5) * 0.5;
      const radius = 150 + Math.random() * 250;
      n.x = cx + Math.cos(angle) * radius + (Math.random() - 0.5) * 80;
      n.y = cy + Math.sin(angle) * radius + (Math.random() - 0.5) * 80;
      n.vx = 0; n.vy = 0;
      n.radius = Math.max(2, Math.min(8, (n.access || 0) * 0.5 + 2)) * (TIER_SCALE[n.tier] || 1);
      n.color = (PROVIDERS[n.provider] || PROVIDERS.test).color;
      nodeMap.set(n.id, n);
    });
    nodes = graphRes.nodes;

    // Build edges with node references
    edges = graphRes.edges.filter(e => nodeMap.has(e.source) && nodeMap.has(e.target)).map(e => ({
      source: nodeMap.get(e.source),
      target: nodeMap.get(e.target),
      strength: e.strength,
    }));

    // Update HUD
    document.getElementById('sub').textContent =
      `${meta.total_memories || '?'} memories · ${meta.visible || nodes.length} visible · ${edges.length} connections · ${meta.compressed || 0} compressed`;

    document.getElementById('stats').innerHTML = `
      <div class="stat"><div class="stat-val">${statsRes.total_memories || 0}</div><div class="stat-label">Memories</div></div>
      <div class="stat"><div class="stat-val">${statsRes.associations || 0}</div><div class="stat-label">Links</div></div>
      <div class="stat"><div class="stat-val">${statsRes.episodic || 0}</div><div class="stat-label">Episodic</div></div>
      <div class="stat"><div class="stat-val">${statsRes.consolidated || 0}</div><div class="stat-label">Consolidated</div></div>
      <div class="stat"><div class="stat-val">${statsRes.contradictions || 0}</div><div class="stat-label">Conflicts</div></div>
    `;

    // Legend
    const legendHtml = Object.entries(PROVIDERS)
      .filter(([k]) => graphRes.nodes.some(n => n.provider === k))
      .map(([k, v]) => `<div class="legend-item"><div class="legend-dot" style="background:${v.color}"></div>${v.label}</div>`)
      .join('');
    document.getElementById('legend').innerHTML =
      '<div style="font-size:10px;color:#666;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">Providers</div>' + legendHtml +
      '<div style="margin-top:8px;font-size:10px;color:#666;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">Tiers</div>' +
      '<div class="legend-item"><div class="legend-dot" style="background:#fff;width:6px;height:6px"></div>Episodic</div>' +
      '<div class="legend-item"><div class="legend-dot" style="background:#fff;width:10px;height:10px"></div>Consolidated</div>' +
      '<div class="legend-item"><div class="legend-dot" style="background:#fff;width:14px;height:14px;box-shadow:0 0 6px rgba(255,255,255,0.4)"></div>Schema</div>';

    simAlpha = 1;
    simRunning = true;
    toast('Brain loaded');
  } catch (e) {
    document.getElementById('sub').textContent = 'Error loading brain data: ' + e.message;
  }
}

// ── Force Simulation ──
function simulate() {
  if (!simRunning || simAlpha < 0.001) return;
  simAlpha *= 0.995;

  const n = nodes.length;
  const cx = W / 2, cy = H / 2;

  // Center gravity
  for (let i = 0; i < n; i++) {
    const node = nodes[i];
    node.vx += (cx - node.x) * 0.0003;
    node.vy += (cy - node.y) * 0.0003;
  }

  // Node repulsion (Barnes-Hut approximation: grid-based)
  const repStr = 800;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dx = nodes[j].x - nodes[i].x;
      const dy = nodes[j].y - nodes[i].y;
      const d2 = dx * dx + dy * dy + 1;
      if (d2 > 90000) continue; // Skip far nodes (300px)
      const f = repStr / d2;
      const fx = dx * f, fy = dy * f;
      nodes[i].vx -= fx; nodes[i].vy -= fy;
      nodes[j].vx += fx; nodes[j].vy += fy;
    }
  }

  // Edge attraction
  for (const e of edges) {
    const dx = e.target.x - e.source.x;
    const dy = e.target.y - e.source.y;
    const d = Math.sqrt(dx * dx + dy * dy) + 0.1;
    const f = (d - 60) * 0.003 * e.strength;
    const fx = (dx / d) * f, fy = (dy / d) * f;
    e.source.vx += fx; e.source.vy += fy;
    e.target.vx -= fx; e.target.vy -= fy;
  }

  // Apply velocity with damping
  for (let i = 0; i < n; i++) {
    const node = nodes[i];
    node.vx *= 0.85;
    node.vy *= 0.85;
    node.x += node.vx * simAlpha;
    node.y += node.vy * simAlpha;
  }
}

// ── Rendering ──
function draw() {
  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, W, H);

  ctx.save();
  ctx.translate(cam.x, cam.y);
  ctx.scale(cam.zoom, cam.zoom);

  const t = performance.now() * 0.001;

  // Draw edges
  if (showEdges && edges.length < 3000) {
    for (const e of edges) {
      const alpha = Math.min(0.4, e.strength * 0.5);
      const isHighlight = selected && (e.source.id === selected.id || e.target.id === selected.id);
      ctx.strokeStyle = isHighlight
        ? `rgba(167,139,250,${Math.min(0.8, alpha * 3)})`
        : `rgba(100,100,140,${alpha})`;
      ctx.lineWidth = isHighlight ? 1.5 : 0.5;
      ctx.beginPath();
      ctx.moveTo(e.source.x, e.source.y);
      ctx.lineTo(e.target.x, e.target.y);
      ctx.stroke();

      // Animated pulse on strong edges
      if (e.strength > 0.6) {
        const progress = (t * 0.5 + e.source.id * 0.01) % 1;
        const px = e.source.x + (e.target.x - e.source.x) * progress;
        const py = e.source.y + (e.target.y - e.source.y) * progress;
        ctx.fillStyle = `rgba(167,139,250,${0.6 * (1 - progress)})`;
        ctx.beginPath();
        ctx.arc(px, py, 1.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  // Draw nodes
  for (const node of nodes) {
    const isSearch = searchResults.size > 0 && searchResults.has(node.id);
    const isSelected = selected && selected.id === node.id;
    const isHover = hover && hover.id === node.id;
    const isDimmed = searchResults.size > 0 && !isSearch;

    let r = node.radius * (isHover ? 1.4 : 1);
    let alpha = isDimmed ? 0.1 : (node.tier === 'consolidated' ? 1 : 0.7);
    let color = node.color;

    // Compressed indicator
    if (node.compressed) {
      // Draw compression ring
      ctx.strokeStyle = `rgba(34,197,94,${alpha * 0.4})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(node.x, node.y, r + 3, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Glow for selected/searched
    if (isSelected || isSearch) {
      const glow = ctx.createRadialGradient(node.x, node.y, r, node.x, node.y, r * 4);
      glow.addColorStop(0, color.replace(')', `,${alpha * 0.3})`).replace('rgb', 'rgba'));
      glow.addColorStop(1, 'transparent');
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(node.x, node.y, r * 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // Node body
    ctx.fillStyle = isDimmed ? `rgba(60,60,80,${alpha})` : color;
    ctx.globalAlpha = alpha;
    ctx.beginPath();
    ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;

    // Consolidated glow
    if (node.tier === 'consolidated' && !isDimmed) {
      ctx.shadowColor = color;
      ctx.shadowBlur = 8;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(node.x, node.y, r * 0.6, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    // Labels at zoom
    if (showLabels && cam.zoom > 1.5 && !isDimmed) {
      ctx.fillStyle = `rgba(200,200,220,${alpha * 0.7})`;
      ctx.font = `${9 / cam.zoom * 1.5}px system-ui`;
      ctx.fillText(node.label.substring(0, 40), node.x + r + 4, node.y + 3);
    }
  }

  ctx.restore();

  // FPS
  ctx.fillStyle = '#333';
  ctx.font = '10px monospace';
  ctx.fillText(`${nodes.length} nodes · ${edges.length} edges · zoom ${cam.zoom.toFixed(1)}x`, W - 250, H - 10);
}

// ── Animation Loop ──
function loop() {
  simulate();
  draw();
  requestAnimationFrame(loop);
}

// ── Interaction ──
function screenToWorld(sx, sy) {
  return {x: (sx - cam.x) / cam.zoom, y: (sy - cam.y) / cam.zoom};
}

function findNode(wx, wy) {
  let closest = null, minD = Infinity;
  for (const n of nodes) {
    const dx = n.x - wx, dy = n.y - wy;
    const d = dx * dx + dy * dy;
    const hitR = Math.max(n.radius, 8) * (1 / cam.zoom + 1);
    if (d < hitR * hitR && d < minD) {
      minD = d; closest = n;
    }
  }
  return closest;
}

canvas.addEventListener('mousedown', e => {
  const w = screenToWorld(e.clientX, e.clientY);
  const node = findNode(w.x, w.y);
  if (node) {
    selected = node;
    showDetail(node);
    drag = {node, offX: node.x - w.x, offY: node.y - w.y};
  } else {
    selected = null;
    hideDetail();
    drag = {panX: cam.x - e.clientX, panY: cam.y - e.clientY};
  }
});

canvas.addEventListener('mousemove', e => {
  if (drag && drag.node) {
    const w = screenToWorld(e.clientX, e.clientY);
    drag.node.x = w.x + drag.offX;
    drag.node.y = w.y + drag.offY;
    drag.node.vx = 0; drag.node.vy = 0;
  } else if (drag) {
    cam.x = e.clientX + drag.panX;
    cam.y = e.clientY + drag.panY;
  } else {
    const w = screenToWorld(e.clientX, e.clientY);
    hover = findNode(w.x, w.y);
    canvas.style.cursor = hover ? 'pointer' : 'grab';
  }
});

canvas.addEventListener('mouseup', () => { drag = null; });

canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const factor = e.deltaY > 0 ? 0.9 : 1.1;
  const mx = e.clientX, my = e.clientY;
  cam.x = mx - (mx - cam.x) * factor;
  cam.y = my - (my - cam.y) * factor;
  cam.zoom *= factor;
  cam.zoom = Math.max(0.1, Math.min(10, cam.zoom));
}, {passive: false});

// ── Detail Panel ──
async function showDetail(node) {
  const panel = document.getElementById('detail');
  try {
    const res = await fetch(`/api/memory?id=${node.id}`).then(r => r.json());
    const assocHtml = (res.associations || []).slice(0, 8)
      .map(a => `<span class="detail-tag">Link #${a.chunk_id} (${a.strength})</span>`).join('');
    panel.innerHTML = `
      <div class="detail-title">
        <span>Memory #${node.id}</span>
        <span class="detail-close" onclick="hideDetail()">&times;</span>
      </div>
      <div class="detail-content">${(res.content || '').substring(0, 800)}</div>
      <div class="detail-meta">
        <span class="detail-tag">${res.provider}</span>
        <span class="detail-tag">${res.tier}</span>
        <span class="detail-tag">Accessed: ${res.access_count || 0}x</span>
        <span class="detail-tag">${res.created_at ? res.created_at.substring(0, 10) : '?'}</span>
        ${node.compressed ? '<span class="detail-tag" style="color:#22c55e">Compressed</span>' : ''}
      </div>
      ${assocHtml ? '<div class="detail-meta" style="margin-top:4px">' + assocHtml + '</div>' : ''}
    `;
  } catch (e) {
    panel.innerHTML = `<div class="detail-title"><span>Memory #${node.id}</span><span class="detail-close" onclick="hideDetail()">&times;</span></div>
      <div class="detail-content">${node.label}</div>`;
  }
  panel.classList.add('show');
}

function hideDetail() {
  document.getElementById('detail').classList.remove('show');
  selected = null;
}

// ── Controls ──
function toggleEdges() {
  showEdges = !showEdges;
  document.getElementById('toggleEdges').classList.toggle('active', showEdges);
}
function toggleLabels() {
  showLabels = !showLabels;
  document.getElementById('toggleLabels').classList.toggle('active', showLabels);
}
function resetView() {
  cam = {x: 0, y: 0, zoom: 1};
  simAlpha = 0.5; simRunning = true;
}

// ── Search ──
let searchTimeout;
document.getElementById('search').addEventListener('input', e => {
  clearTimeout(searchTimeout);
  searchQuery = e.target.value.trim();
  if (!searchQuery) { searchResults.clear(); return; }
  searchTimeout = setTimeout(async () => {
    try {
      const res = await fetch(`/api/search?q=${encodeURIComponent(searchQuery)}`).then(r => r.json());
      searchResults = new Set((res.results || []).map(r => r.chunk_id));
    } catch (e) { searchResults.clear(); }
  }, 300);
});

// ── Dream ──
async function triggerDream() {
  if (dreamRunning) return;
  dreamRunning = true;
  const btn = document.getElementById('dreamBtn');
  btn.classList.add('running');
  btn.textContent = 'Dreaming...';
  toast('Dream cycle starting...');

  try {
    const res = await fetch('/api/dream', {method: 'POST'}).then(r => r.json());
    const parts = [];
    if (res.promoted) parts.push(`${res.promoted} promoted`);
    if (res.compressed) parts.push(`${res.compressed} compressed`);
    if (res.duplicates_pruned) parts.push(`${res.duplicates_pruned} pruned`);
    if (res.decayed) parts.push(`${res.decayed} decayed`);
    toast(parts.length ? `Dream: ${parts.join(', ')}` : 'Dream complete (no changes)');
    // Reload data to reflect changes
    setTimeout(() => loadData(), 1000);
  } catch (e) {
    toast('Dream failed: ' + e.message);
  }

  btn.classList.remove('running');
  btn.textContent = 'Dream';
  dreamRunning = false;
}

// ── Toast ──
function toast(msg) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 2500);
}

// ── Init ──
document.getElementById('toggleEdges').classList.add('active');
loadData();
loop();

// Auto-refresh stats every 30s
setInterval(async () => {
  try {
    const stats = await fetch('/api/stats').then(r => r.json());
    document.getElementById('stats').innerHTML = `
      <div class="stat"><div class="stat-val">${stats.total_memories || 0}</div><div class="stat-label">Memories</div></div>
      <div class="stat"><div class="stat-val">${stats.associations || 0}</div><div class="stat-label">Links</div></div>
      <div class="stat"><div class="stat-val">${stats.episodic || 0}</div><div class="stat-label">Episodic</div></div>
      <div class="stat"><div class="stat-val">${stats.consolidated || 0}</div><div class="stat-label">Consolidated</div></div>
      <div class="stat"><div class="stat-val">${stats.contradictions || 0}</div><div class="stat-label">Conflicts</div></div>
    `;
  } catch(e) {}
}, 30000);
</script>
</body>
</html>"""


# ── Watcher Live View HTML ─────────────────────────────────

_NAVIGATOR_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>watty — navigator</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:100%;height:100%;background:#000;font-family:-apple-system,BlinkMacSystemFont,'SF Pro','Segoe UI',sans-serif;color:#e0e0e0;overflow-x:hidden}
nav{position:fixed;top:0;left:0;right:0;z-index:50;display:flex;align-items:center;justify-content:space-between;padding:16px 24px;background:rgba(0,0,0,0.7);backdrop-filter:blur(16px);border-bottom:1px solid rgba(139,92,246,0.1)}
nav .brand{font-size:20px;font-weight:700;letter-spacing:2px;background:linear-gradient(135deg,#a78bfa,#6366f1);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
nav .links{display:flex;gap:12px}
nav .links a{color:#666;font-size:12px;text-decoration:none;padding:5px 12px;border-radius:14px;border:1px solid transparent;transition:all .2s;letter-spacing:.5px}
nav .links a:hover{color:#c4b5fd;border-color:rgba(139,92,246,0.2)}
nav .links a.active{color:#a78bfa;border-color:rgba(139,92,246,0.3);background:rgba(139,92,246,0.08)}
.container{padding:80px 24px 40px;max-width:900px;margin:0 auto}
.search-box{position:relative;margin:20px 0}
.search-box input{width:100%;padding:14px 20px;background:rgba(20,20,30,0.8);border:1px solid rgba(139,92,246,0.2);border-radius:16px;color:#e0e0e0;font-size:16px;outline:none;transition:border-color .2s}
.search-box input:focus{border-color:rgba(139,92,246,0.5)}
.search-box input::placeholder{color:#555}
.diagnostics{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0}
.organ-card{background:rgba(20,20,30,0.7);border:1px solid rgba(139,92,246,0.15);border-radius:16px;padding:20px;text-align:center}
.organ-card .label{font-size:11px;text-transform:uppercase;letter-spacing:1.5px;color:#666;margin-bottom:8px}
.organ-card .signal{font-size:18px;font-weight:600;color:#c4b5fd}
.organ-card .value{font-size:12px;color:#888;margin-top:4px}
.heart-bar{background:rgba(20,20,30,0.7);border:1px solid rgba(139,92,246,0.15);border-radius:16px;padding:16px 20px;margin:16px 0;display:flex;align-items:center;gap:16px}
.heart-bar .coherence-ring{width:52px;height:52px;border-radius:50%;border:3px solid;display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:700;flex-shrink:0;transition:all .3s}
.heart-bar .meta{flex:1}
.heart-bar .meta .title{font-size:13px;font-weight:600;color:#c4b5fd}
.heart-bar .meta .detail{font-size:12px;color:#888;margin-top:2px}
.strategy-pill{display:inline-block;padding:4px 12px;border-radius:12px;font-size:12px;font-weight:500;border:1px solid rgba(139,92,246,0.2);color:#a78bfa;background:rgba(139,92,246,0.08);margin:2px}
.results{margin:24px 0}
.result-card{background:rgba(20,20,30,0.5);border:1px solid rgba(139,92,246,0.1);border-radius:12px;padding:16px;margin:8px 0;transition:border-color .2s}
.result-card:hover{border-color:rgba(139,92,246,0.3)}
.result-card .header{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.result-card .rank{font-size:11px;font-weight:700;color:#a78bfa;background:rgba(139,92,246,0.1);padding:2px 8px;border-radius:8px}
.result-card .scores{display:flex;gap:8px;font-size:11px;color:#888}
.result-card .scores .act{color:#34d399}
.result-card .scores .sim{color:#60a5fa}
.result-card .content{font-size:13px;color:#ccc;line-height:1.5;max-height:120px;overflow:hidden;white-space:pre-wrap;word-break:break-word}
.result-card .source{font-size:11px;color:#555;margin-top:6px}
.empty{text-align:center;padding:60px 20px;color:#555;font-size:14px}
.loading{text-align:center;padding:40px;color:#666}
.circ-count{font-size:12px;color:#666;margin:8px 0}
.geometry-section{margin-top:32px;border:1px solid rgba(139,92,246,0.1);border-radius:16px;overflow:hidden}
.geo-title{padding:14px 20px;font-size:13px;font-weight:600;color:#888;letter-spacing:.5px;cursor:pointer;transition:color .2s}
.geo-title:hover{color:#c4b5fd}
.geo-body{display:none;padding:0 20px 20px}
.geometry-section.open .geo-body{display:block}
.geo-title::after{content:' +';float:right;color:#555}
.geometry-section.open .geo-title::after{content:' -'}
.face-bars{display:flex;gap:4px;align-items:flex-end;height:60px;margin:12px 0}
.face-bar{flex:1;display:flex;flex-direction:column;align-items:center;gap:4px}
.face-bar .bar{width:100%;border-radius:4px 4px 0 0;transition:height .3s}
.face-bar .lbl{font-size:9px;color:#666;writing-mode:vertical-lr;text-orientation:mixed;transform:rotate(180deg)}
.geo-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:12px}
.geo-stat{text-align:center}
.geo-stat .val{font-size:16px;font-weight:600;color:#c4b5fd}
.geo-stat .lbl{font-size:10px;color:#666;text-transform:uppercase;letter-spacing:.5px}
</style>
</head>
<body>
<nav>
  <div class="brand">WATTY</div>
  <div class="links">
    <a href="/dashboard">Dashboard</a>
    <a href="/brain">Brain</a>
    <a href="/navigator" class="active">Navigator</a>
    <a href="/watcher">Watcher</a>
    <a href="/api">API</a>
  </div>
</nav>
<div class="container">
  <div class="search-box">
    <input type="text" id="navQuery" placeholder="Navigate through memory... (Enter to search)" autofocus>
  </div>
  <div id="diagnostics" style="display:none">
    <div class="diagnostics">
      <div class="organ-card" id="organ-coherence">
        <div class="label">Coherence Organ</div>
        <div class="signal" id="coh-signal">—</div>
        <div class="value" id="coh-value"></div>
      </div>
      <div class="organ-card" id="organ-depth">
        <div class="label">Depth Organ</div>
        <div class="signal" id="depth-signal">—</div>
        <div class="value" id="depth-value"></div>
      </div>
      <div class="organ-card" id="organ-bridge">
        <div class="label">Bridge Organ</div>
        <div class="signal" id="bridge-signal">—</div>
        <div class="value" id="bridge-value"></div>
      </div>
    </div>
    <div class="heart-bar">
      <div class="coherence-ring" id="heart-ring">—</div>
      <div class="meta">
        <div class="title">Heart</div>
        <div class="detail" id="heart-detail">Waiting...</div>
      </div>
      <div id="strategies"></div>
    </div>
    <div class="circ-count" id="circ-info"></div>
  </div>
  <div class="results" id="results">
    <div class="empty">Enter a query to navigate through Watty's memory graph.<br>Unlike recall (flat similarity), Navigate follows association paths<br>and reads the geometric shape of activated memories.</div>
  </div>
  <div id="geometry" class="geometry-section">
    <div class="geo-title" onclick="this.parentElement.classList.toggle('open')">Chestahedron Geometry Status</div>
    <div class="geo-body" id="geo-body"></div>
  </div>
</div>
<script>
const input = document.getElementById('navQuery');
input.addEventListener('keydown', e => { if (e.key === 'Enter') doNavigate(); });
async function doNavigate() {
  const q = input.value.trim();
  if (!q) return;
  document.getElementById('results').innerHTML = '<div class="loading">Navigating...</div>';
  document.getElementById('diagnostics').style.display = 'none';
  try {
    const res = await fetch('/api/navigate?q=' + encodeURIComponent(q) + '&top_k=10');
    const data = await res.json();
    renderDiagnostics(data);
    renderResults(data.results || []);
  } catch(e) {
    document.getElementById('results').innerHTML = '<div class="empty">Error: ' + e.message + '</div>';
  }
}
function renderDiagnostics(data) {
  const diag = document.getElementById('diagnostics');
  diag.style.display = 'block';
  // Organ readings (last circulation)
  const organs = data.organ_readings || [];
  if (organs.length > 0) {
    const last = organs[organs.length - 1];
    const coh = last.coherence || {};
    const dep = last.depth || {};
    const bri = last.bridge || {};
    document.getElementById('coh-signal').textContent = (coh.signal || '—').replace(/_/g, ' ');
    document.getElementById('coh-value').textContent = 'mean: ' + (coh.mean_coherence || 0) + ' | pairs: ' + (coh.n_pairs || 0);
    document.getElementById('depth-signal').textContent = (dep.signal || '—').replace(/_/g, ' ');
    document.getElementById('depth-value').textContent = 'mean depth: ' + (dep.mean_depth || 0);
    document.getElementById('bridge-signal').textContent = (bri.signal || '—').replace(/_/g, ' ');
    const dom = bri.dominant_face;
    const faces = ['TEMPORAL','STRUCTURAL','RELATIONAL','SEMANTIC','META','INVERSE','EXTERNAL'];
    document.getElementById('bridge-value').textContent = dom !== null && dom !== undefined ? 'dominant: ' + faces[dom] : '';
  }
  // Heart
  const hearts = data.heart_readings || [];
  const ring = document.getElementById('heart-ring');
  const detail = document.getElementById('heart-detail');
  if (hearts.length > 0) {
    const last = hearts[hearts.length - 1];
    const coh = last.coherence || 0;
    ring.textContent = coh.toFixed(2);
    const hue = coh > 0.618 ? '160' : coh > 0.4 ? '270' : '0';
    ring.style.borderColor = 'hsl(' + hue + ', 70%, 55%)';
    ring.style.color = 'hsl(' + hue + ', 70%, 70%)';
    detail.textContent = (last.should_circulate ? 'Circulating...' : 'Converged') + ' | signals: ' + (last.signals || []).join(', ');
  }
  // Blood strategies
  const strats = data.blood_strategies || [];
  const stratEl = document.getElementById('strategies');
  stratEl.innerHTML = strats.map(s => '<span class="strategy-pill">' + s.replace(/_/g, ' ') + '</span>').join('');
  // Circulation count
  document.getElementById('circ-info').textContent = data.circulations + ' circulation(s) | final coherence: ' + (data.final_coherence || 0).toFixed(4);
}
function renderResults(results) {
  const el = document.getElementById('results');
  if (!results.length) { el.innerHTML = '<div class="empty">No memories activated.</div>'; return; }
  el.innerHTML = results.map((r, i) => {
    const src = r.source_path ? r.source_path.split(/[/\\]/).pop() : r.provider;
    return '<div class="result-card"><div class="header"><span class="rank">#' + (i+1) + '</span><div class="scores"><span class="act">activation: ' + r.activation + '</span><span class="sim">similarity: ' + r.similarity + '</span></div></div><div class="content">' + escHtml(r.content || '').slice(0, 500) + '</div><div class="source">' + escHtml(src) + ' | ' + (r.memory_tier || 'episodic') + (r.compressed ? ' | compressed' : '') + '</div></div>';
  }).join('');
}
function escHtml(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
// Load plasticity on page load
(async function() {
  try {
    const res = await fetch('/api/plasticity');
    const p = await res.json();
    const faces = p.face_names || [];
    const felt = p.felt_state || [];
    const maxFelt = Math.max(...felt.map(Math.abs), 0.01);
    let bars = '<div class="face-bars">';
    felt.forEach((v, i) => {
      const h = Math.round(Math.abs(v) / maxFelt * 50);
      const color = v >= 0 ? 'rgba(139,92,246,0.6)' : 'rgba(239,68,68,0.5)';
      bars += '<div class="face-bar"><div class="bar" style="height:' + h + 'px;background:' + color + '"></div><div class="lbl">' + (faces[i] || i) + '</div></div>';
    });
    bars += '</div>';
    const stats = '<div class="geo-stats">' +
      '<div class="geo-stat"><div class="val">' + (p.total_learn_steps || 0) + '</div><div class="lbl">learn steps</div></div>' +
      '<div class="geo-stat"><div class="val">' + (p.wout_deviation_from_identity || 0).toFixed(4) + '</div><div class="lbl">W_out drift</div></div>' +
      '<div class="geo-stat"><div class="val">' + (p.felt_magnitude || 0).toFixed(4) + '</div><div class="lbl">felt magnitude</div></div>' +
      '<div class="geo-stat"><div class="val">' + (p.n_processed || 0) + '</div><div class="lbl">processed</div></div>' +
      '<div class="geo-stat"><div class="val">' + (p.wout_spectral_radius || 0).toFixed(4) + '</div><div class="lbl">spectral radius</div></div>' +
      '<div class="geo-stat"><div class="val">' + (p.hippocampus?.history_len || 0) + '</div><div class="lbl">hippo history</div></div>' +
      '</div>';
    document.getElementById('geo-body').innerHTML = '<div style="font-size:12px;color:#888;margin-bottom:8px">Felt State (7D Chestahedron coordinates)</div>' + bars + stats;
  } catch(e) {}
})();
</script>
</body>
</html>"""

_WATCHER_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>watty — observer</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:100%;height:100%;background:#000;font-family:-apple-system,BlinkMacSystemFont,'SF Pro','Segoe UI',sans-serif;color:#e0e0e0;overflow-x:hidden}

/* ── NAV ── */
nav{position:fixed;top:0;left:0;right:0;z-index:50;display:flex;align-items:center;justify-content:space-between;padding:16px 24px;background:rgba(0,0,0,0.7);backdrop-filter:blur(16px);border-bottom:1px solid rgba(139,92,246,0.1)}
nav .brand{font-size:20px;font-weight:700;letter-spacing:2px;background:linear-gradient(135deg,#a78bfa,#6366f1);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
nav .links{display:flex;gap:12px}
nav .links a{color:#666;font-size:12px;text-decoration:none;padding:5px 12px;border-radius:14px;border:1px solid transparent;transition:all .2s;letter-spacing:.5px}
nav .links a:hover{color:#c4b5fd;border-color:rgba(139,92,246,0.2)}
nav .links a.active{color:#a78bfa;border-color:rgba(139,92,246,0.3);background:rgba(139,92,246,0.08)}

/* ── STATUS BAR ── */
.status-bar{position:fixed;top:60px;left:0;right:0;z-index:40;display:flex;justify-content:center;gap:12px;padding:12px 24px}
.pill{background:rgba(20,20,30,0.7);backdrop-filter:blur(12px);border:1px solid rgba(139,92,246,0.15);border-radius:20px;padding:6px 16px;font-size:12px;color:#a0a0b0;display:flex;align-items:center;gap:8px}
.pill .val{font-weight:600;font-size:14px;color:#c4b5fd}
.pill .dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.dot-live{background:#34d399;box-shadow:0 0 8px #34d399;animation:blink 2s infinite}
.dot-off{background:#ef4444;box-shadow:0 0 6px #ef4444}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.4}}

/* ── TIMELINE ── */
.timeline-wrap{padding:130px 24px 40px;max-width:720px;margin:0 auto}
.timeline{position:relative;padding-left:32px}
.timeline::before{content:'';position:absolute;left:11px;top:0;bottom:0;width:2px;background:linear-gradient(to bottom,rgba(139,92,246,0.4),rgba(139,92,246,0.05))}

/* ── EVENT CARD ── */
.ev{position:relative;margin-bottom:20px;padding:16px 20px;background:rgba(14,14,24,0.8);backdrop-filter:blur(8px);border:1px solid rgba(139,92,246,0.1);border-radius:14px;transition:border-color .3s,transform .3s;animation:fadeUp .4s ease both}
.ev:hover{border-color:rgba(139,92,246,0.3);transform:translateX(4px)}
.ev::before{content:'';position:absolute;left:-27px;top:22px;width:10px;height:10px;border-radius:50%;border:2px solid rgba(139,92,246,0.5);background:#000}
.ev.window_switch::before{background:#6366f1;border-color:#818cf8}
.ev.content_change::before{background:#a78bfa;border-color:#c4b5fd}
@keyframes fadeUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}

.ev-head{display:flex;align-items:center;gap:10px;margin-bottom:8px}
.ev-time{font-size:11px;color:#555;font-variant-numeric:tabular-nums}
.ev-app{font-size:15px;font-weight:600;color:#c4b5fd}
.ev-badge{font-size:10px;padding:2px 8px;border-radius:10px;font-weight:600;text-transform:uppercase;letter-spacing:.5px}
.badge-switch{background:rgba(99,102,241,0.15);color:#818cf8}
.badge-content{background:rgba(167,139,250,0.12);color:#a78bfa}
.ev-title{font-size:12px;color:#777;margin-bottom:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ev-summary{font-size:13px;color:#999}
.ev-text{font-size:11px;color:#555;margin-top:8px;padding:10px 12px;background:rgba(30,30,50,0.5);border-radius:10px;border:1px solid rgba(139,92,246,0.06);max-height:0;overflow:hidden;transition:max-height .3s ease,padding .3s ease,margin-top .3s ease;padding:0 12px;margin-top:0}
.ev.expanded .ev-text{max-height:200px;overflow-y:auto;padding:10px 12px;margin-top:8px}
.ev-expand{position:absolute;top:16px;right:16px;background:none;border:none;color:#555;font-size:10px;cursor:pointer;padding:4px 8px;border-radius:8px;transition:color .2s,background .2s}
.ev-expand:hover{color:#a78bfa;background:rgba(139,92,246,0.1)}

/* ── EMPTY STATE ── */
.empty{text-align:center;padding:80px 20px;color:#444}
.empty-icon{font-size:48px;margin-bottom:16px;opacity:.3}
.empty-text{font-size:16px}
.empty-sub{font-size:12px;color:#333;margin-top:8px}

/* ── SCROLLBAR ── */
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:rgba(139,92,246,0.2);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:rgba(139,92,246,0.4)}
</style>
</head>
<body>

<nav>
  <div class="brand">WATTY</div>
  <div class="links">
    <a href="/">brain</a>
    <a href="/brain">viewer</a>
    <a href="/navigator">navigator</a>
    <a href="/watcher" class="active">observer</a>
  </div>
</nav>

<div class="status-bar">
  <div class="pill"><div class="dot" id="dot"></div><span id="s-status">--</span></div>
  <div class="pill"><span class="val" id="s-count">--</span> observations</div>
  <div class="pill"><span class="val" id="s-uptime">--</span> uptime</div>
  <div class="pill"><span class="val" id="s-interval">--</span>s interval</div>
</div>

<div class="timeline-wrap">
  <div class="timeline" id="timeline"></div>
</div>

<script>
const $=s=>document.getElementById(s);
let lastCount=0;

function fmtUptime(s){
  if(!s||s<=0)return'--';
  const h=Math.floor(s/3600),m=Math.floor((s%3600)/60);
  return h>0?h+'h '+m+'m':m+'m';
}

function renderTimeline(obs){
  const tl=$('timeline');
  if(!obs.length){
    tl.innerHTML='<div class="empty"><div class="empty-icon">&#x25C9;</div><div class="empty-text">No observations yet</div><div class="empty-sub">Watcher is listening...</div></div>';
    return;
  }
  tl.innerHTML=obs.map((o,i)=>{
    const type=o.change_type||'unknown';
    const badgeCls=type==='window_switch'?'badge-switch':'badge-content';
    const badgeLabel=type==='window_switch'?'switch':'update';
    const hasText=o.text_digest&&o.text_digest.length>0;
    const textPreview=hasText?o.text_digest.substring(0,300):'';
    return`<div class="ev ${type}" style="animation-delay:${i*0.04}s" onclick="this.classList.toggle('expanded')">
      <div class="ev-head">
        <span class="ev-time">${o.timestamp_local||'?'}</span>
        <span class="ev-app">${o.app_name||'?'}</span>
        <span class="ev-badge ${badgeCls}">${badgeLabel}</span>
      </div>
      <div class="ev-title">${(o.window_title||'').substring(0,100)}</div>
      <div class="ev-summary">${o.change_summary||''}</div>
      ${hasText?`<div class="ev-text">${textPreview}</div><button class="ev-expand" onclick="event.stopPropagation();this.parentElement.classList.toggle('expanded')">&#x25BE; text</button>`:''}
    </div>`;
  }).join('');
}

async function refresh(){
  try{
    const[obsRes,statusRes]=await Promise.all([
      fetch('/api/watcher/observations?n=50'),
      fetch('/api/watcher/status')
    ]);
    const obsData=await obsRes.json();
    const status=await statusRes.json();

    // Status pills
    const running=status.running;
    $('dot').className='dot '+(running?'dot-live':'dot-off');
    $('s-status').textContent=running?'LIVE':'OFF';
    $('s-count').textContent=status.observation_count||0;
    $('s-uptime').textContent=fmtUptime(status.uptime_seconds);
    $('s-interval').textContent=(status.config&&status.config.interval_seconds)||'--';

    // Only re-render timeline if count changed
    const count=obsData.count||0;
    if(count!==lastCount){
      renderTimeline(obsData.observations||[]);
      lastCount=count;
    }
  }catch(e){console.error('refresh error',e)}
}

refresh();
setInterval(refresh,8000);
</script>
</body>
</html>"""


# ── Knowledge Graph HTML ───────────────────────────────────

_GRAPH_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Watty · Knowledge Graph</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0f0f1a;color:#e2e8f0;font-family:system-ui,-apple-system,sans-serif;overflow:hidden}
#header{position:fixed;top:0;left:0;right:0;height:48px;background:rgba(15,15,26,0.9);border-bottom:1px solid rgba(255,255,255,0.06);display:flex;align-items:center;padding:0 20px;z-index:100;backdrop-filter:blur(10px)}
#header h1{font-size:14px;font-weight:600;color:#818cf8}
#header a{color:#94a3b8;text-decoration:none;font-size:12px;margin-left:16px;padding:4px 10px;border-radius:6px;transition:all 0.2s}
#header a:hover,#header a.active{color:#e2e8f0;background:rgba(255,255,255,0.06)}
#header a.active{color:#818cf8}
.stats-bar{position:fixed;top:48px;left:0;right:0;height:36px;background:rgba(15,15,26,0.8);border-bottom:1px solid rgba(255,255,255,0.04);display:flex;align-items:center;padding:0 20px;gap:24px;font-size:11px;color:#64748b;z-index:99}
.stat-val{color:#818cf8;font-weight:600;margin-left:4px}
#search{position:fixed;top:96px;left:20px;z-index:98;display:flex;gap:8px}
#search input{background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:8px 14px;color:#e2e8f0;font-size:13px;width:260px;outline:none}
#search input:focus{border-color:#818cf8}
#search button{background:#818cf8;color:#fff;border:none;border-radius:8px;padding:8px 16px;font-size:12px;cursor:pointer}
#canvas{position:fixed;top:84px;left:0;right:0;bottom:0}
#detail-panel{position:fixed;top:84px;right:0;width:320px;bottom:0;background:rgba(15,15,26,0.95);border-left:1px solid rgba(255,255,255,0.06);padding:16px;overflow-y:auto;display:none;z-index:97;backdrop-filter:blur(10px)}
#detail-panel h3{color:#818cf8;font-size:14px;margin-bottom:12px}
#detail-panel .field{margin-bottom:10px}
#detail-panel .label{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1px}
#detail-panel .value{font-size:13px;color:#e2e8f0;margin-top:2px}
.rel-item{padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.04);font-size:12px}
.rel-item .rel-type{color:#34d399;font-weight:500}
.rel-item .rel-target{color:#818cf8;cursor:pointer}
.close-btn{position:absolute;top:12px;right:12px;background:none;border:none;color:#64748b;font-size:16px;cursor:pointer}
</style></head><body>
<div id="header">
  <h1>⊛ Knowledge Graph</h1>
  <a href="/dashboard">Dashboard</a>
  <a href="/brain">Brain</a>
  <a href="/navigator">Navigator</a>
  <a href="/watcher">Watcher</a>
  <a href="/graph" class="active">Graph</a>
  <a href="/eval">Metrics</a>
</div>
<div class="stats-bar">
  <span>Entities:<span class="stat-val" id="s-entities">—</span></span>
  <span>Relationships:<span class="stat-val" id="s-rels">—</span></span>
  <span>Types:<span class="stat-val" id="s-types">—</span></span>
</div>
<div id="search">
  <input id="search-input" placeholder="Search entities..." autocomplete="off">
  <button onclick="searchEntities()">Search</button>
</div>
<canvas id="canvas"></canvas>
<div id="detail-panel">
  <button class="close-btn" onclick="closeDetail()">✕</button>
  <h3 id="dp-name"></h3>
  <div class="field"><div class="label">Type</div><div class="value" id="dp-type"></div></div>
  <div class="field"><div class="label">Mentions</div><div class="value" id="dp-mentions"></div></div>
  <div class="field"><div class="label">Description</div><div class="value" id="dp-desc"></div></div>
  <div class="field"><div class="label">Relationships</div><div id="dp-rels"></div></div>
</div>
<script>
const C=document.getElementById('canvas'),ctx=C.getContext('2d');
let W,H,nodes=[],edges=[],dragging=null,offsetX=0,offsetY=0,scale=1,panX=0,panY=0;
const colors={person:'#818cf8',concept:'#34d399',project:'#fbbf24',technology:'#60a5fa',place:'#f472b6',organization:'#a78bfa'};
function resize(){W=C.width=window.innerWidth;H=C.height=window.innerHeight-84}
window.addEventListener('resize',resize);resize();

async function loadStats(){
  try{const r=await fetch('/api/graph/stats');const d=await r.json();
  document.getElementById('s-entities').textContent=d.entities||0;
  document.getElementById('s-rels').textContent=d.relationships||0;
  document.getElementById('s-types').textContent=Object.keys(d.entity_types||{}).length;
  }catch(e){}
}

async function loadEntities(q){
  try{const url=q?'/api/graph/entities?q='+encodeURIComponent(q)+'&limit=100':'/api/graph/entities?limit=100';
  const r=await fetch(url);const d=await r.json();
  const ents=d.entities||[];
  nodes=ents.map((e,i)=>{const a=2*Math.PI*i/ents.length;const r_=Math.min(W,H)*0.3;
    return{id:e.id,name:e.name,type:e.entity_type||'concept',mentions:e.mention_count||1,
    x:W/2+r_*Math.cos(a)+Math.random()*40,y:H/2+r_*Math.sin(a)+Math.random()*40,vx:0,vy:0}});
  // Load relationships for visible entities
  edges=[];
  for(const n of nodes.slice(0,30)){
    try{const tr=await fetch('/api/graph/entity?id='+n.id);const td=await tr.json();
    if(td.relationships){td.relationships.forEach(rel=>{
      if(nodes.find(x=>x.id===rel.target_id||x.id===rel.source_id)){
        edges.push({source:rel.source_id,target:rel.target_id,type:rel.type||'related_to',strength:rel.strength||1})}})}}catch(e){}
  }}catch(e){console.error(e)}
}

function searchEntities(){const q=document.getElementById('search-input').value;loadEntities(q)}
document.getElementById('search-input').addEventListener('keydown',e=>{if(e.key==='Enter')searchEntities()});

function simulate(){
  for(const n of nodes){n.vx*=0.9;n.vy*=0.9;n.vx+=(W/2-n.x)*0.0005;n.vy+=(H/2-n.y)*0.0005}
  for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++){
    let dx=nodes[j].x-nodes[i].x,dy=nodes[j].y-nodes[i].y,d=Math.sqrt(dx*dx+dy*dy)||1;
    if(d<120){const f=(120-d)*0.03/d;nodes[i].vx-=dx*f;nodes[i].vy-=dy*f;nodes[j].vx+=dx*f;nodes[j].vy+=dy*f}}
  for(const e of edges){const s=nodes.find(n=>n.id===e.source),t=nodes.find(n=>n.id===e.target);
    if(!s||!t)continue;let dx=t.x-s.x,dy=t.y-s.y,d=Math.sqrt(dx*dx+dy*dy)||1;
    const f=(d-150)*0.005/d;s.vx+=dx*f;s.vy+=dy*f;t.vx-=dx*f;t.vy-=dy*f}
  if(dragging)dragging.vx=dragging.vy=0;
  for(const n of nodes){n.x+=n.vx;n.y+=n.vy}
}

function draw(){
  ctx.clearRect(0,0,W,H);ctx.save();ctx.translate(panX,panY);ctx.scale(scale,scale);
  // edges
  for(const e of edges){const s=nodes.find(n=>n.id===e.source),t=nodes.find(n=>n.id===e.target);
    if(!s||!t)continue;ctx.beginPath();ctx.moveTo(s.x,s.y);ctx.lineTo(t.x,t.y);
    ctx.strokeStyle='rgba(129,140,248,0.15)';ctx.lineWidth=Math.max(0.5,e.strength);ctx.stroke()}
  // nodes
  for(const n of nodes){const r=Math.max(4,Math.min(16,Math.sqrt(n.mentions)*3));
    ctx.beginPath();ctx.arc(n.x,n.y,r,0,Math.PI*2);
    ctx.fillStyle=colors[n.type]||'#818cf8';ctx.globalAlpha=0.85;ctx.fill();ctx.globalAlpha=1;
    ctx.font='10px system-ui';ctx.fillStyle='#94a3b8';ctx.textAlign='center';
    ctx.fillText(n.name.length>18?n.name.slice(0,16)+'..':n.name,n.x,n.y+r+12)}
  ctx.restore()
}

function loop(){simulate();draw();requestAnimationFrame(loop)}

C.addEventListener('mousedown',e=>{const mx=(e.offsetX-panX)/scale,my=(e.offsetY-panY)/scale;
  for(const n of nodes){const d=Math.sqrt((n.x-mx)**2+(n.y-my)**2);
    if(d<20){dragging=n;return}}});
C.addEventListener('mousemove',e=>{if(dragging){dragging.x=(e.offsetX-panX)/scale;dragging.y=(e.offsetY-panY)/scale}});
C.addEventListener('mouseup',()=>dragging=null);
C.addEventListener('dblclick',async e=>{const mx=(e.offsetX-panX)/scale,my=(e.offsetY-panY)/scale;
  for(const n of nodes){const d=Math.sqrt((n.x-mx)**2+(n.y-my)**2);
    if(d<20){showDetail(n.id);return}}});
C.addEventListener('wheel',e=>{e.preventDefault();const z=e.deltaY>0?0.9:1.1;scale*=z;
  panX=e.offsetX-(e.offsetX-panX)*z;panY=e.offsetY-(e.offsetY-panY)*z},{passive:false});

async function showDetail(id){
  try{const r=await fetch('/api/graph/entity?id='+id);const d=await r.json();
  document.getElementById('dp-name').textContent=d.name||'?';
  document.getElementById('dp-type').textContent=d.entity_type||'concept';
  document.getElementById('dp-mentions').textContent=d.mention_count||0;
  document.getElementById('dp-desc').textContent=d.description||'(none)';
  const relsDiv=document.getElementById('dp-rels');relsDiv.innerHTML='';
  (d.relationships||[]).forEach(rel=>{const div=document.createElement('div');div.className='rel-item';
    div.innerHTML='<span class="rel-type">'+rel.type+'</span> → <span class="rel-target" onclick="showDetail('+
    (rel.target_id===id?rel.source_id:rel.target_id)+')">'+
    (rel.target_name||rel.source_name||'?')+'</span>';relsDiv.appendChild(div)});
  document.getElementById('detail-panel').style.display='block'}catch(e){console.error(e)}
}
function closeDetail(){document.getElementById('detail-panel').style.display='none'}

loadStats();loadEntities('');loop();
</script></body></html>"""


# ── Evaluation Metrics HTML ────────────────────────────────

_EVAL_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Watty · Evaluation Metrics</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0f0f1a;color:#e2e8f0;font-family:system-ui,-apple-system,sans-serif;overflow-y:auto}
#header{position:fixed;top:0;left:0;right:0;height:48px;background:rgba(15,15,26,0.9);border-bottom:1px solid rgba(255,255,255,0.06);display:flex;align-items:center;padding:0 20px;z-index:100;backdrop-filter:blur(10px)}
#header h1{font-size:14px;font-weight:600;color:#34d399}
#header a{color:#94a3b8;text-decoration:none;font-size:12px;margin-left:16px;padding:4px 10px;border-radius:6px;transition:all 0.2s}
#header a:hover,#header a.active{color:#e2e8f0;background:rgba(255,255,255,0.06)}
#header a.active{color:#34d399}
.container{margin-top:64px;padding:20px;max-width:1200px;margin-left:auto;margin-right:auto}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin-bottom:24px}
.card{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:20px}
.card h3{font-size:12px;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px}
.big-num{font-size:32px;font-weight:700;color:#34d399}
.sub-num{font-size:13px;color:#94a3b8;margin-top:4px}
.alert-card{background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.2);border-radius:10px;padding:14px;margin-bottom:10px;display:flex;justify-content:space-between;align-items:center}
.alert-card .msg{font-size:13px;color:#fbbf24;flex:1}
.alert-card .severity{font-size:10px;padding:2px 8px;border-radius:4px;background:rgba(251,191,36,0.15);color:#fbbf24;margin-right:12px}
.alert-card button{background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);color:#94a3b8;border-radius:6px;padding:4px 12px;font-size:11px;cursor:pointer}
.alert-card button:hover{color:#e2e8f0;background:rgba(255,255,255,0.1)}
.chart-wrap{position:relative;height:200px;margin-top:12px}
canvas.chart{width:100%;height:100%}
.no-data{color:#475569;font-size:13px;text-align:center;padding:40px 0}
.section-title{font-size:16px;font-weight:600;color:#e2e8f0;margin:24px 0 12px}
</style></head><body>
<div id="header">
  <h1>📊 Evaluation Metrics</h1>
  <a href="/dashboard">Dashboard</a>
  <a href="/brain">Brain</a>
  <a href="/navigator">Navigator</a>
  <a href="/watcher">Watcher</a>
  <a href="/graph">Graph</a>
  <a href="/eval" class="active">Metrics</a>
</div>
<div class="container">
  <div class="grid" id="stats-grid">
    <div class="card"><h3>Total Metrics</h3><div class="big-num" id="s-total">—</div><div class="sub-num" id="s-range"></div></div>
    <div class="card"><h3>Categories</h3><div class="big-num" id="s-cats">—</div><div class="sub-num" id="s-cat-list"></div></div>
    <div class="card"><h3>Active Alerts</h3><div class="big-num" id="s-alerts" style="color:#fbbf24">—</div></div>
  </div>
  <div class="section-title">Active Alerts</div>
  <div id="alerts-container"><div class="no-data">Loading alerts...</div></div>
  <div class="section-title">Retrieval Quality (30 days)</div>
  <div class="card"><div class="chart-wrap"><canvas class="chart" id="chart-retrieval"></canvas></div></div>
  <div class="section-title">Task Success Rate (30 days)</div>
  <div class="card"><div class="chart-wrap"><canvas class="chart" id="chart-tasks"></canvas></div></div>
  <div class="section-title">Memory Health (30 days)</div>
  <div class="card"><div class="chart-wrap"><canvas class="chart" id="chart-memory"></canvas></div></div>
</div>
<script>
async function loadStats(){
  try{const r=await fetch('/api/eval/stats');const d=await r.json();
  document.getElementById('s-total').textContent=d.total_metrics||0;
  document.getElementById('s-cats').textContent=(d.categories||[]).length;
  document.getElementById('s-cat-list').textContent=(d.categories||[]).join(', ');
  document.getElementById('s-range').textContent=(d.date_range||{}).earliest?
    'Since '+(d.date_range.earliest||'').slice(0,10):'No data yet';
  }catch(e){console.error(e)}
}

async function loadAlerts(){
  try{const r=await fetch('/api/eval/alerts');const d=await r.json();
  const c=document.getElementById('alerts-container');
  const alerts=d.alerts||[];
  document.getElementById('s-alerts').textContent=alerts.length;
  if(!alerts.length){c.innerHTML='<div class="no-data">No active alerts</div>';return}
  c.innerHTML='';
  alerts.forEach(a=>{const div=document.createElement('div');div.className='alert-card';
    div.innerHTML='<span class="severity">'+a.severity+'</span><span class="msg">'+a.message+
    '</span><button onclick="ackAlert('+a.id+',this)">Dismiss</button>';c.appendChild(div)});
  }catch(e){document.getElementById('alerts-container').innerHTML='<div class="no-data">Error loading alerts</div>'}
}

async function ackAlert(id,btn){
  try{await fetch('/api/eval/ack_alert',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({alert_id:id})});btn.parentElement.remove();
    const n=parseInt(document.getElementById('s-alerts').textContent)-1;
    document.getElementById('s-alerts').textContent=Math.max(0,n);
  }catch(e){alert('Failed: '+e)}
}

function drawChart(canvasId,data,color){
  const canvas=document.getElementById(canvasId);if(!canvas)return;
  const ctx=canvas.getContext('2d');
  const dpr=window.devicePixelRatio||1;
  canvas.width=canvas.offsetWidth*dpr;canvas.height=canvas.offsetHeight*dpr;
  ctx.scale(dpr,dpr);const W=canvas.offsetWidth,H=canvas.offsetHeight;
  ctx.clearRect(0,0,W,H);
  if(!data||!data.length){ctx.fillStyle='#475569';ctx.font='13px system-ui';ctx.textAlign='center';
    ctx.fillText('No data yet',W/2,H/2);return}
  const vals=data.map(d=>d.value);const mn=Math.min(...vals),mx=Math.max(...vals);
  const range=mx-mn||1;const pad=30;
  // Grid
  ctx.strokeStyle='rgba(255,255,255,0.04)';ctx.lineWidth=1;
  for(let i=0;i<5;i++){const y=pad+(H-2*pad)*i/4;ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(W-pad,y);ctx.stroke();
    ctx.fillStyle='#475569';ctx.font='9px system-ui';ctx.textAlign='right';
    ctx.fillText((mx-range*i/4).toFixed(2),pad-4,y+3)}
  // Line
  ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=2;
  data.forEach((d,i)=>{const x=pad+i*(W-2*pad)/(data.length-1||1);
    const y=pad+(1-(d.value-mn)/range)*(H-2*pad);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y)});ctx.stroke();
  // Fill under
  const last=data.length-1;
  ctx.lineTo(pad+last*(W-2*pad)/(data.length-1||1),H-pad);ctx.lineTo(pad,H-pad);ctx.closePath();
  ctx.fillStyle=color.replace('1)','0.08)');ctx.fill();
}

async function loadTrends(){
  const metrics=[
    {id:'chart-retrieval',metric:'precision_proxy',color:'rgba(129,140,248,1)'},
    {id:'chart-tasks',metric:'session_success_rate',color:'rgba(52,211,153,1)'},
    {id:'chart-memory',metric:'contradiction_rate',color:'rgba(251,191,36,1)'},
  ];
  for(const m of metrics){
    try{const r=await fetch('/api/eval/trends?metric='+m.metric+'&days=30');const d=await r.json();
    drawChart(m.id,d.trends||[],m.color)}catch(e){drawChart(m.id,[],m.color)}
  }
}

loadStats();loadAlerts();setTimeout(loadTrends,500);
window.addEventListener('resize',()=>setTimeout(loadTrends,100));
</script></body></html>"""


# ── Dashboard HTML (Immersive Galaxy Brain) ───────────────

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>watty — brain</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
html,body { width:100%; height:100%; overflow:hidden; background:#000; font-family:-apple-system,BlinkMacSystemFont,'SF Pro','Segoe UI',sans-serif; color:#e0e0e0; }
canvas { display:block; cursor:grab; }
canvas:active { cursor:grabbing; }

/* HUD overlay */
#hud { position:fixed; top:20px; left:20px; z-index:10; pointer-events:none; }
#hud h1 { font-size:28px; font-weight:700; letter-spacing:2px; background:linear-gradient(135deg,#a78bfa,#6366f1,#818cf8); -webkit-background-clip:text; -webkit-text-fill-color:transparent; text-shadow:0 0 40px rgba(139,92,246,0.3); }
#hud .sub { font-size:12px; color:#666; margin-top:2px; letter-spacing:1px; }

#stats-bar { position:fixed; top:20px; right:20px; z-index:10; display:flex; gap:16px; pointer-events:none; }
.stat-pill { background:rgba(20,20,30,0.7); backdrop-filter:blur(12px); border:1px solid rgba(139,92,246,0.15); border-radius:20px; padding:6px 14px; font-size:12px; color:#a0a0b0; display:flex; align-items:center; gap:6px; }
.stat-pill .num { color:#c4b5fd; font-weight:600; font-size:14px; }

/* Search */
#search-wrap { position:fixed; top:20px; left:50%; transform:translateX(-50%); z-index:10; }
#search-input { width:280px; padding:8px 16px 8px 36px; background:rgba(20,20,30,0.7); backdrop-filter:blur(12px); border:1px solid rgba(139,92,246,0.2); border-radius:20px; color:#e0e0e0; font-size:13px; outline:none; transition:border-color 0.3s,width 0.3s; }
#search-input:focus { border-color:rgba(139,92,246,0.5); width:360px; }
#search-icon { position:absolute; left:12px; top:50%; transform:translateY(-50%); color:#666; font-size:14px; pointer-events:none; }

/* Detail panel */
#detail { position:fixed; right:-420px; top:0; bottom:0; width:400px; background:rgba(10,10,20,0.85); backdrop-filter:blur(20px); border-left:1px solid rgba(139,92,246,0.15); z-index:20; transition:right 0.4s cubic-bezier(0.16,1,0.3,1); padding:24px; overflow-y:auto; }
#detail.open { right:0; }
#detail .close-btn { position:absolute; top:12px; right:16px; background:none; border:none; color:#888; font-size:22px; cursor:pointer; pointer-events:all; }
#detail .close-btn:hover { color:#fff; }
#detail .mem-provider { display:inline-block; padding:3px 10px; border-radius:12px; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px; }
#detail .mem-content { font-size:13px; line-height:1.7; color:#c8c8d0; white-space:pre-wrap; word-break:break-word; max-height:50vh; overflow-y:auto; margin:12px 0; padding:12px; background:rgba(30,30,50,0.5); border-radius:12px; border:1px solid rgba(139,92,246,0.08); }
#detail .mem-meta { font-size:11px; color:#666; line-height:1.8; }
#detail .mem-meta span { color:#a78bfa; }
#detail h3 { font-size:14px; color:#a0a0b0; margin:16px 0 8px; }
#detail .assoc-list { list-style:none; }
#detail .assoc-list li { padding:6px 10px; margin:4px 0; background:rgba(40,40,60,0.4); border-radius:8px; font-size:12px; color:#999; cursor:pointer; transition:background 0.2s; }
#detail .assoc-list li:hover { background:rgba(139,92,246,0.15); color:#c4b5fd; }

/* Bottom toolbar */
#toolbar { position:fixed; bottom:20px; left:50%; transform:translateX(-50%); z-index:10; display:flex; gap:8px; }
.tool-btn { background:rgba(20,20,30,0.7); backdrop-filter:blur(12px); border:1px solid rgba(139,92,246,0.15); border-radius:14px; padding:8px 16px; color:#a0a0b0; font-size:12px; cursor:pointer; transition:all 0.2s; pointer-events:all; display:flex; align-items:center; gap:6px; }
.tool-btn:hover { border-color:rgba(139,92,246,0.4); color:#c4b5fd; background:rgba(30,30,50,0.8); }
.tool-btn.active { border-color:#6366f1; color:#a78bfa; }

/* Toast */
#toast { position:fixed; bottom:80px; left:50%; transform:translateX(-50%); background:rgba(20,20,40,0.9); backdrop-filter:blur(12px); border:1px solid rgba(139,92,246,0.25); border-radius:12px; padding:10px 20px; font-size:13px; color:#c4b5fd; z-index:30; opacity:0; transition:opacity 0.3s; pointer-events:none; }
#toast.show { opacity:1; }

/* Legend */
#legend { position:fixed; bottom:60px; right:20px; z-index:10; display:flex; flex-direction:column; gap:4px; pointer-events:none; }
.legend-item { font-size:10px; color:#555; display:flex; align-items:center; gap:6px; }
.legend-dot { width:8px; height:8px; border-radius:50%; }
</style>
</head>
<body>
<canvas id="c"></canvas>

<div id="hud"><h1>WATTY</h1><div class="sub">NEURAL SPACE</div></div>

<div id="stats-bar">
  <div class="stat-pill"><span class="num" id="s-mem">—</span> memories</div>
  <div class="stat-pill"><span class="num" id="s-links">—</span> links</div>
  <div class="stat-pill"><span class="num" id="s-clusters">—</span> clusters</div>
</div>

<div id="search-wrap">
  <span id="search-icon">⌕</span>
  <input id="search-input" placeholder="search the brain..." autocomplete="off">
</div>

<div id="detail">
  <button class="close-btn" onclick="closeDetail()">✕</button>
  <div id="detail-body"></div>
</div>

<div id="toolbar">
  <button class="tool-btn" onclick="triggerDream()">✦ dream</button>
  <button class="tool-btn" id="btn-edges" onclick="toggleEdges()">◇ links</button>
  <button class="tool-btn" id="btn-labels" onclick="toggleLabels()">◈ labels</button>
  <button class="tool-btn" onclick="resetView()">⟲ reset</button>
  <a class="tool-btn" href="/navigator" style="text-decoration:none">⬡ navigator</a>
  <a class="tool-btn" href="/watcher" style="text-decoration:none">◉ observer</a>
  <a class="tool-btn" href="/graph" style="text-decoration:none">⊛ graph</a>
  <a class="tool-btn" href="/eval" style="text-decoration:none">📊 metrics</a>
</div>

<div id="toast"></div>

<div id="legend">
  <div class="legend-item"><div class="legend-dot" style="background:#818cf8"></div> claude</div>
  <div class="legend-item"><div class="legend-dot" style="background:#34d399"></div> manual</div>
  <div class="legend-item"><div class="legend-dot" style="background:#60a5fa"></div> file_scan</div>
  <div class="legend-item"><div class="legend-dot" style="background:#fbbf24"></div> chatgpt</div>
  <div class="legend-item"><div class="legend-dot" style="background:#f472b6"></div> discovery</div>
  <div class="legend-item"><div class="legend-dot" style="background:#a78bfa"></div> other</div>
</div>

<script>
// ── Config ──
const PROVIDERS = {
  claude:'#818cf8', manual:'#34d399', file_scan:'#60a5fa',
  chatgpt:'#fbbf24', grok:'#fb923c', discovery:'#f472b6',
  http_api:'#67e8f9', gemini:'#a3e635'
};
const TIERS = { consolidated:1.4, schema:1.2, episodic:1.0 };
const DEFAULT_COLOR = '#a78bfa';

// ── State ──
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
let W, H, dpr;
let nodes = [], edges = [], clusters = [];
let stars = [];
let cam = { x:0, y:0, zoom:1 };
let targetCam = { x:0, y:0, zoom:1 };
let dragging = false, dragStart = {x:0,y:0}, camStart = {x:0,y:0};
let hoveredNode = null, selectedNode = null;
let showEdges = true, showLabels = false;
let searchMatches = new Set();
let time = 0;
let particlePool = [];

// ── Resize ──
function resize() {
  dpr = window.devicePixelRatio || 1;
  W = window.innerWidth; H = window.innerHeight;
  canvas.width = W * dpr; canvas.height = H * dpr;
  canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
  ctx.setTransform(dpr,0,0,dpr,0,0);
}
window.addEventListener('resize', resize);
resize();

// ── Stars ──
function initStars() {
  stars = [];
  for (let i = 0; i < 400; i++) {
    stars.push({
      x: (Math.random() - 0.5) * 8000,
      y: (Math.random() - 0.5) * 8000,
      r: Math.random() * 1.2 + 0.3,
      a: Math.random() * 0.4 + 0.1,
      speed: Math.random() * 0.3 + 0.1,
      phase: Math.random() * Math.PI * 2,
    });
  }
}
initStars();

// ── Particles ──
function spawnParticle(x, y, color) {
  particlePool.push({
    x, y, vx: (Math.random()-0.5)*2, vy: (Math.random()-0.5)*2,
    life: 1, decay: 0.01 + Math.random()*0.02, r: Math.random()*3+1, color
  });
  if (particlePool.length > 200) particlePool.shift();
}

// ── Layout (force-directed) ──
function layoutNodes() {
  const N = nodes.length;
  if (N === 0) return;

  // Initial placement: by cluster in radial groups
  const clusterCenters = {};
  const numClusters = clusters.length || 1;
  clusters.forEach((c, i) => {
    const angle = (i / numClusters) * Math.PI * 2;
    const radius = 300 + numClusters * 20;
    clusterCenters[c.id] = { x: Math.cos(angle) * radius, y: Math.sin(angle) * radius };
  });

  nodes.forEach((n, i) => {
    if (n.x !== undefined) return; // already placed
    const cc = clusterCenters[n.cluster];
    if (cc) {
      n.x = cc.x + (Math.random()-0.5) * 150;
      n.y = cc.y + (Math.random()-0.5) * 150;
    } else {
      const angle = (i / N) * Math.PI * 2;
      n.x = Math.cos(angle) * 400 + (Math.random()-0.5)*100;
      n.y = Math.sin(angle) * 400 + (Math.random()-0.5)*100;
    }
    n.vx = 0; n.vy = 0;
  });

  // Run force simulation
  const edgeMap = {};
  edges.forEach(e => {
    if (!edgeMap[e.source]) edgeMap[e.source] = [];
    if (!edgeMap[e.target]) edgeMap[e.target] = [];
    edgeMap[e.source].push(e);
    edgeMap[e.target].push(e);
  });

  const nodeMap = {};
  nodes.forEach(n => nodeMap[n.id] = n);

  for (let iter = 0; iter < 120; iter++) {
    const alpha = 0.3 * (1 - iter/120);

    // Repulsion (Barnes-Hut approximate: only check nearby)
    for (let i = 0; i < N; i++) {
      for (let j = i+1; j < N; j++) {
        const dx = nodes[j].x - nodes[i].x;
        const dy = nodes[j].y - nodes[i].y;
        const dist = Math.sqrt(dx*dx + dy*dy) + 1;
        if (dist > 500) continue;
        const force = -800 / (dist * dist);
        const fx = (dx/dist) * force * alpha;
        const fy = (dy/dist) * force * alpha;
        nodes[i].vx -= fx; nodes[i].vy -= fy;
        nodes[j].vx += fx; nodes[j].vy += fy;
      }
    }

    // Edge attraction
    edges.forEach(e => {
      const a = nodeMap[e.source], b = nodeMap[e.target];
      if (!a || !b) return;
      const dx = b.x - a.x, dy = b.y - a.y;
      const dist = Math.sqrt(dx*dx + dy*dy) + 1;
      const force = (dist - 80) * 0.005 * e.strength * alpha;
      a.vx += (dx/dist)*force; a.vy += (dy/dist)*force;
      b.vx -= (dx/dist)*force; b.vy -= (dy/dist)*force;
    });

    // Center gravity
    nodes.forEach(n => {
      n.vx -= n.x * 0.0003 * alpha;
      n.vy -= n.y * 0.0003 * alpha;
    });

    // Apply + dampen
    nodes.forEach(n => {
      n.vx *= 0.85; n.vy *= 0.85;
      n.x += n.vx; n.y += n.vy;
    });
  }
}

// ── Node sizing ──
function nodeRadius(n) {
  const base = 4 + Math.min(n.access || 0, 30) * 0.5;
  const tierScale = TIERS[n.tier] || 1;
  return base * tierScale;
}

function nodeColor(n) {
  return PROVIDERS[n.provider] || DEFAULT_COLOR;
}

// ── Transform helpers ──
function worldToScreen(wx, wy) {
  return {
    x: (wx - cam.x) * cam.zoom + W/2,
    y: (wy - cam.y) * cam.zoom + H/2,
  };
}
function screenToWorld(sx, sy) {
  return {
    x: (sx - W/2) / cam.zoom + cam.x,
    y: (sy - H/2) / cam.zoom + cam.y,
  };
}

// ── Draw ──
function draw() {
  time += 0.016;

  // Smooth camera
  cam.x += (targetCam.x - cam.x) * 0.08;
  cam.y += (targetCam.y - cam.y) * 0.08;
  cam.zoom += (targetCam.zoom - cam.zoom) * 0.08;

  ctx.clearRect(0, 0, W, H);

  // Background gradient
  const grad = ctx.createRadialGradient(W/2, H/2, 0, W/2, H/2, Math.max(W,H)*0.7);
  grad.addColorStop(0, '#0a0a1a');
  grad.addColorStop(0.5, '#050510');
  grad.addColorStop(1, '#000005');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, W, H);

  // Stars (parallax)
  stars.forEach(s => {
    const parallax = 0.15;
    const sx = (s.x - cam.x * parallax) * cam.zoom * 0.3 + W/2;
    const sy = (s.y - cam.y * parallax) * cam.zoom * 0.3 + H/2;
    if (sx < -10 || sx > W+10 || sy < -10 || sy > H+10) return;
    const twinkle = Math.sin(time * s.speed + s.phase) * 0.3 + 0.7;
    ctx.beginPath();
    ctx.arc(sx, sy, s.r, 0, Math.PI*2);
    ctx.fillStyle = `rgba(200,200,255,${s.a * twinkle})`;
    ctx.fill();
  });

  // Cluster nebulae (soft glows behind nodes)
  if (clusters.length > 0 && cam.zoom > 0.15) {
    const clusterPositions = {};
    clusters.forEach(c => { clusterPositions[c.id] = { xs:[], ys:[] }; });
    nodes.forEach(n => {
      if (n.cluster >= 0 && clusterPositions[n.cluster]) {
        clusterPositions[n.cluster].xs.push(n.x);
        clusterPositions[n.cluster].ys.push(n.y);
      }
    });

    Object.entries(clusterPositions).forEach(([cid, pos]) => {
      if (pos.xs.length < 3) return;
      const cx = pos.xs.reduce((a,b)=>a+b,0)/pos.xs.length;
      const cy = pos.ys.reduce((a,b)=>a+b,0)/pos.ys.length;
      const sc = worldToScreen(cx, cy);
      const spread = Math.max(60, pos.xs.length * 8) * cam.zoom;
      const nebula = ctx.createRadialGradient(sc.x, sc.y, 0, sc.x, sc.y, spread);
      const baseColor = nodeColor(nodes.find(n => n.cluster == cid) || nodes[0]);
      nebula.addColorStop(0, baseColor.replace(')', ',0.06)').replace('rgb','rgba'));
      nebula.addColorStop(0.5, baseColor.replace(')', ',0.02)').replace('rgb','rgba'));
      nebula.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = nebula;
      ctx.fillRect(sc.x - spread, sc.y - spread, spread*2, spread*2);
    });
  }

  // Edges
  if (showEdges && cam.zoom > 0.2) {
    const nodeMap = {};
    nodes.forEach(n => nodeMap[n.id] = n);
    edges.forEach(e => {
      const a = nodeMap[e.source], b = nodeMap[e.target];
      if (!a || !b) return;
      const sa = worldToScreen(a.x, a.y);
      const sb = worldToScreen(b.x, b.y);
      // Cull off-screen
      if (Math.max(sa.x,sb.x) < -50 || Math.min(sa.x,sb.x) > W+50) return;
      if (Math.max(sa.y,sb.y) < -50 || Math.min(sa.y,sb.y) > H+50) return;
      const alpha = Math.min(0.25, e.strength * 0.4) * Math.min(1, cam.zoom);
      ctx.beginPath();
      ctx.moveTo(sa.x, sa.y);
      ctx.lineTo(sb.x, sb.y);
      ctx.strokeStyle = `rgba(139,92,246,${alpha})`;
      ctx.lineWidth = e.strength * 1.5;
      ctx.stroke();

      // Pulse on strong links
      if (e.strength > 0.5) {
        const t = (time * 0.5 + e.source * 0.01) % 1;
        const px = sa.x + (sb.x - sa.x) * t;
        const py = sa.y + (sb.y - sa.y) * t;
        ctx.beginPath();
        ctx.arc(px, py, 2, 0, Math.PI*2);
        ctx.fillStyle = `rgba(167,139,250,${0.6 * (1-Math.abs(t-0.5)*2)})`;
        ctx.fill();
      }
    });
  }

  // Nodes
  hoveredNode = null;
  const mouseWorld = screenToWorld(mouseX, mouseY);
  nodes.forEach(n => {
    const sc = worldToScreen(n.x, n.y);
    if (sc.x < -60 || sc.x > W+60 || sc.y < -60 || sc.y > H+60) return;

    const r = nodeRadius(n) * cam.zoom;
    if (r < 0.5) return; // too small to see

    const color = nodeColor(n);
    const isSelected = selectedNode && selectedNode.id === n.id;
    const isSearched = searchMatches.size > 0 && searchMatches.has(n.id);
    const isDimmed = searchMatches.size > 0 && !searchMatches.has(n.id);

    // Check hover
    const dx = mouseWorld.x - n.x, dy = mouseWorld.y - n.y;
    const hoverR = nodeRadius(n) + 5;
    if (dx*dx + dy*dy < hoverR*hoverR) hoveredNode = n;

    const isHovered = hoveredNode === n;

    // Outer glow
    if ((isSelected || isHovered || isSearched) && r > 2) {
      const glow = ctx.createRadialGradient(sc.x, sc.y, r*0.5, sc.x, sc.y, r*4);
      glow.addColorStop(0, color.slice(0,-1)+',0.3)'.replace('#','rgba('+parseInt(color.slice(1,3),16)+','+parseInt(color.slice(3,5),16)+','+parseInt(color.slice(5,7),16)+',0.3)'));
      glow.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.beginPath();
      ctx.arc(sc.x, sc.y, r*4, 0, Math.PI*2);
      ctx.fillStyle = glow;
      ctx.fill();
    }

    // Breathing animation
    const breathe = 1 + Math.sin(time * 1.5 + n.id * 0.3) * 0.08;
    const drawR = r * breathe;

    // Node body
    ctx.beginPath();
    ctx.arc(sc.x, sc.y, drawR, 0, Math.PI*2);
    const alpha = isDimmed ? 0.15 : (isSelected ? 1 : isHovered ? 0.95 : 0.7);
    ctx.fillStyle = hexToRgba(color, alpha);
    ctx.fill();

    // Inner bright core
    if (drawR > 3) {
      ctx.beginPath();
      ctx.arc(sc.x, sc.y, drawR * 0.4, 0, Math.PI*2);
      ctx.fillStyle = hexToRgba(color, alpha * 0.8);
      ctx.fill();
    }

    // Consolidated ring
    if (n.tier === 'consolidated' && drawR > 4) {
      ctx.beginPath();
      ctx.arc(sc.x, sc.y, drawR + 2, 0, Math.PI*2);
      ctx.strokeStyle = hexToRgba('#c4b5fd', 0.4);
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Compressed indicator
    if (n.compressed && drawR > 4) {
      ctx.beginPath();
      ctx.arc(sc.x, sc.y, drawR + 4, 0, Math.PI * 2 * (n.ratio || 0.5));
      ctx.strokeStyle = hexToRgba('#34d399', 0.4);
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // Labels (zoom-dependent)
    if (showLabels && cam.zoom > 0.6 && drawR > 5 && !isDimmed) {
      const label = n.label.slice(0,40);
      ctx.font = `${Math.max(9, Math.min(12, drawR))}px -apple-system,sans-serif`;
      ctx.fillStyle = `rgba(200,200,220,${Math.min(0.8, (cam.zoom-0.6)*2)})`;
      ctx.textAlign = 'center';
      ctx.fillText(label, sc.x, sc.y + drawR + 14);
    }
  });

  // Particles
  particlePool = particlePool.filter(p => p.life > 0);
  particlePool.forEach(p => {
    p.x += p.vx; p.y += p.vy;
    p.vx *= 0.98; p.vy *= 0.98;
    p.life -= p.decay;
    const sc = worldToScreen(p.x, p.y);
    ctx.beginPath();
    ctx.arc(sc.x, sc.y, p.r * p.life * cam.zoom, 0, Math.PI*2);
    ctx.fillStyle = hexToRgba(p.color, p.life * 0.6);
    ctx.fill();
  });

  // Hover tooltip
  if (hoveredNode && !dragging) {
    const sc = worldToScreen(hoveredNode.x, hoveredNode.y);
    const r = nodeRadius(hoveredNode) * cam.zoom;
    const text = hoveredNode.label.slice(0, 60);
    ctx.font = '12px -apple-system,sans-serif';
    const tw = ctx.measureText(text).width;
    const tx = sc.x - tw/2 - 8;
    const ty = sc.y - r - 28;
    ctx.fillStyle = 'rgba(15,15,30,0.9)';
    ctx.strokeStyle = 'rgba(139,92,246,0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(tx, ty, tw+16, 24, 8);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#d0d0e0';
    ctx.textAlign = 'left';
    ctx.fillText(text, tx+8, ty+16);
  }

  requestAnimationFrame(draw);
}

// ── Helpers ──
function hexToRgba(hex, a) {
  if (hex.startsWith('rgba') || hex.startsWith('rgb')) return hex;
  const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${a})`;
}

// ── Input ──
let mouseX = 0, mouseY = 0;
let dragNode = null;

canvas.addEventListener('mousemove', e => {
  mouseX = e.clientX; mouseY = e.clientY;
  if (dragNode) {
    const w = screenToWorld(e.clientX, e.clientY);
    dragNode.x = w.x; dragNode.y = w.y;
  } else if (dragging) {
    const dx = e.clientX - dragStart.x;
    const dy = e.clientY - dragStart.y;
    targetCam.x = camStart.x - dx / cam.zoom;
    targetCam.y = camStart.y - dy / cam.zoom;
  }
});

canvas.addEventListener('mousedown', e => {
  if (hoveredNode) {
    dragNode = hoveredNode;
    canvas.style.cursor = 'grabbing';
  } else {
    dragging = true;
    dragStart = { x: e.clientX, y: e.clientY };
    camStart = { x: targetCam.x, y: targetCam.y };
  }
});

canvas.addEventListener('mouseup', e => {
  if (dragNode) {
    // If barely moved, treat as click
    const sc = worldToScreen(dragNode.x, dragNode.y);
    const dist = Math.hypot(e.clientX - sc.x, e.clientY - sc.y);
    if (dist < 5) selectNode(dragNode);
    dragNode = null;
    canvas.style.cursor = hoveredNode ? 'pointer' : 'grab';
  } else {
    if (!dragging) return;
    dragging = false;
    canvas.style.cursor = 'grab';
    // If barely moved, deselect
    const dist = Math.hypot(e.clientX - dragStart.x, e.clientY - dragStart.y);
    if (dist < 3) { closeDetail(); selectedNode = null; }
  }
});

canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const zoomSpeed = 0.001;
  const delta = -e.deltaY * zoomSpeed;
  targetCam.zoom = Math.max(0.05, Math.min(5, targetCam.zoom * (1 + delta * 3)));
}, { passive: false });

// Touch support
let lastTouchDist = 0;
canvas.addEventListener('touchstart', e => {
  if (e.touches.length === 1) {
    mouseX = e.touches[0].clientX; mouseY = e.touches[0].clientY;
    if (hoveredNode) {
      dragNode = hoveredNode;
    } else {
      dragging = true;
      dragStart = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      camStart = { x: targetCam.x, y: targetCam.y };
    }
  } else if (e.touches.length === 2) {
    lastTouchDist = Math.hypot(e.touches[0].clientX-e.touches[1].clientX, e.touches[0].clientY-e.touches[1].clientY);
  }
}, { passive: false });

canvas.addEventListener('touchmove', e => {
  e.preventDefault();
  if (e.touches.length === 1) {
    mouseX = e.touches[0].clientX; mouseY = e.touches[0].clientY;
    if (dragNode) {
      const w = screenToWorld(mouseX, mouseY);
      dragNode.x = w.x; dragNode.y = w.y;
    } else if (dragging) {
      const dx = e.touches[0].clientX - dragStart.x;
      const dy = e.touches[0].clientY - dragStart.y;
      targetCam.x = camStart.x - dx / cam.zoom;
      targetCam.y = camStart.y - dy / cam.zoom;
    }
  } else if (e.touches.length === 2) {
    const dist = Math.hypot(e.touches[0].clientX-e.touches[1].clientX, e.touches[0].clientY-e.touches[1].clientY);
    const delta = (dist - lastTouchDist) * 0.005;
    targetCam.zoom = Math.max(0.05, Math.min(5, targetCam.zoom * (1 + delta)));
    lastTouchDist = dist;
  }
}, { passive: false });

canvas.addEventListener('touchend', e => {
  if (dragNode) {
    if (hoveredNode === dragNode) selectNode(dragNode);
    dragNode = null;
  }
  dragging = false;
});

// ── Node selection ──
function selectNode(n) {
  selectedNode = n;
  // Spawn particles
  for (let i = 0; i < 15; i++) spawnParticle(n.x, n.y, nodeColor(n));
  // Load detail
  loadDetail(n.id);
  document.getElementById('detail').classList.add('open');
}

function closeDetail() {
  document.getElementById('detail').classList.remove('open');
  selectedNode = null;
}

async function loadDetail(id) {
  try {
    const d = await fetch('/api/memory?id=' + id).then(r => r.json());
    if (d.error) { document.getElementById('detail-body').innerHTML = '<p style="color:#666">Memory not found</p>'; return; }
    const color = PROVIDERS[d.provider] || DEFAULT_COLOR;
    let html = `<div class="mem-provider" style="background:${hexToRgba(color,0.15)};color:${color}">${d.provider}</div>`;
    html += `<div class="mem-content">${escHtml(d.content)}</div>`;
    html += `<div class="mem-meta">
      <div>tier: <span>${d.tier}</span> · accessed: <span>${d.access_count}</span> times</div>
      <div>significance: <span>${(d.significance||0).toFixed(2)}</span></div>
      <div>created: <span>${(d.created_at||'').slice(0,19)}</span></div>
      ${d.source_path ? '<div>source: <span>'+escHtml(d.source_path)+'</span></div>' : ''}
    </div>`;
    if (d.associations && d.associations.length > 0) {
      html += '<h3>associations</h3><ul class="assoc-list">';
      d.associations.forEach(a => {
        html += `<li onclick="jumpToNode(${a.chunk_id})">→ #${a.chunk_id} (strength: ${a.strength})</li>`;
      });
      html += '</ul>';
    }
    document.getElementById('detail-body').innerHTML = html;
  } catch(e) {
    document.getElementById('detail-body').innerHTML = '<p style="color:#666">Error loading detail</p>';
  }
}

function jumpToNode(id) {
  const n = nodes.find(n => n.id === id);
  if (n) {
    targetCam.x = n.x; targetCam.y = n.y;
    targetCam.zoom = Math.max(1.5, targetCam.zoom);
    selectNode(n);
  }
}

function escHtml(s) { return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

// ── Search ──
let searchTimeout;
document.getElementById('search-input').addEventListener('input', e => {
  clearTimeout(searchTimeout);
  const q = e.target.value.trim();
  if (!q) { searchMatches = new Set(); return; }
  searchTimeout = setTimeout(async () => {
    try {
      const d = await fetch('/api/search?q=' + encodeURIComponent(q)).then(r => r.json());
      searchMatches = new Set((d.results||[]).map(r => r.id));
      if (searchMatches.size > 0) {
        // Pan to first result
        const firstId = [...searchMatches][0];
        const n = nodes.find(n => n.id === firstId);
        if (n) { targetCam.x = n.x; targetCam.y = n.y; }
      }
    } catch(e) { searchMatches = new Set(); }
  }, 300);
});

// ── Toolbar ──
function toggleEdges() {
  showEdges = !showEdges;
  document.getElementById('btn-edges').classList.toggle('active', showEdges);
}
function toggleLabels() {
  showLabels = !showLabels;
  document.getElementById('btn-labels').classList.toggle('active', showLabels);
}
function resetView() {
  targetCam = { x:0, y:0, zoom:1 };
  searchMatches = new Set();
  document.getElementById('search-input').value = '';
  closeDetail();
}
async function triggerDream() {
  toast('dreaming...');
  try {
    await fetch('/api/dream', { method:'POST' });
    toast('dream complete ✦');
    setTimeout(loadData, 1000);
  } catch(e) { toast('dream failed'); }
}

function toast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2500);
}

// ── Data Loading ──
async function loadData() {
  try {
    const [graphData, statsData] = await Promise.all([
      fetch('/api/graph/full').then(r => r.json()),
      fetch('/api/stats').then(r => r.json()),
    ]);

    // Preserve positions for existing nodes
    const oldPositions = {};
    nodes.forEach(n => { oldPositions[n.id] = { x:n.x, y:n.y }; });

    nodes = graphData.nodes || [];
    edges = graphData.edges || [];
    clusters = graphData.clusters || [];

    // Restore positions
    nodes.forEach(n => {
      if (oldPositions[n.id]) { n.x = oldPositions[n.id].x; n.y = oldPositions[n.id].y; }
    });

    layoutNodes();

    // Update stats
    document.getElementById('s-mem').textContent = statsData.total_memories || 0;
    document.getElementById('s-links').textContent = statsData.associations || 0;
    document.getElementById('s-clusters').textContent = clusters.length;
  } catch(e) {
    console.error('Load error:', e);
  }
}

// ── Init ──
document.getElementById('btn-edges').classList.add('active');
loadData();
setInterval(async () => {
  try {
    const s = await fetch('/api/stats').then(r=>r.json());
    document.getElementById('s-mem').textContent = s.total_memories || 0;
    document.getElementById('s-links').textContent = s.associations || 0;
  } catch(e){}
}, 30000);
draw();
</script>
</body>
</html>"""


# ── Tool Definition ─────────────────────────────────────────

WEB_ACTIONS = ["start", "stop", "status"]

TOOLS = [
    Tool(
        name="watty_web",
        description=(
            "Watty's brain dashboard. Local web UI with knowledge graph, memory search, tier breakdown.\n"
            "Actions: start (launch on port 7777), stop, status."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": WEB_ACTIONS, "description": "start/stop/status"},
                "port": {"type": "integer", "description": "Port (default: 7777)"},
                "open_browser": {"type": "boolean", "description": "Auto-open browser (default: true)"},
            },
            "required": ["action"],
        },
    ),
]


# ── Handler ─────────────────────────────────────────────────

async def _web_start(args):
    global _server_instance, _server_thread, _server_port
    if _server_instance is not None:
        return [TextContent(type="text", text=f"Dashboard already running at http://localhost:{_server_port}")]
    port = args.get("port", 7777)
    open_browser = args.get("open_browser", True)
    _server_port = port
    try:
        # Pre-warm brain cache before starting server
        global _brain_cache
        if _brain_cache is None:
            from watty.brain import Brain
            _brain_cache = Brain()

        _server_instance = _ThreadedHTTPServer(("127.0.0.1", port), _DashboardHandler)
        _server_thread = threading.Thread(target=_server_instance.serve_forever, daemon=True, name="watty-web")
        _server_thread.start()
        url = f"http://localhost:{port}"
        if open_browser:
            webbrowser.open(url)
        return [TextContent(type="text", text=f"Dashboard live at {url}")]
    except OSError as e:
        _server_instance = None
        return [TextContent(type="text", text=f"Failed to start: {e}")]

async def _web_stop(args):
    global _server_instance, _server_thread
    if _server_instance is None:
        return [TextContent(type="text", text="Dashboard not running.")]
    _server_instance.shutdown()
    _server_instance = None
    _server_thread = None
    return [TextContent(type="text", text="Dashboard stopped.")]

async def _web_status(args):
    if _server_instance is not None:
        return [TextContent(type="text", text=f"Dashboard running at http://localhost:{_server_port}")]
    return [TextContent(type="text", text="Dashboard not running. Use action=start.")]


_ACTION_MAP = {"start": _web_start, "stop": _web_stop, "status": _web_status}

async def handle_web(args: dict) -> list[TextContent]:
    action = args.get("action", "")
    if action not in _ACTION_MAP:
        return [TextContent(type="text", text=f"Unknown web action: {action}. Valid: {', '.join(WEB_ACTIONS)}")]
    return await _ACTION_MAP[action](args)


# ── Router ──────────────────────────────────────────────────

HANDLERS = {"watty_web": handle_web}
