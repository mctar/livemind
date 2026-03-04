"""
Deterministic graph reconciler for Live Mind Map.
Manages node lifecycle (active/parked/archived/hidden), scoring,
decay, budget enforcement, and user actions.
"""

import math, time
from dataclasses import dataclass, field


@dataclass
class NodeState:
    id: str
    label: str
    group: str
    state: str = "active"  # active | parked | archived | hidden
    importance: float = 0.5
    first_mentioned: float = 0.0
    last_mentioned: float = 0.0
    mention_count: int = 1
    edge_count: int = 0
    pinned: bool = False
    x: float = 0.0
    y: float = 0.0


class GraphReconciler:
    MAX_ACTIVE = 24
    MAX_VISIBLE = 30
    DECAY_SECONDS = 720        # 12 minutes → parked
    REACTIVATION_WINDOW = 180  # 3 minutes
    REACTIVATION_MENTIONS = 2
    POSITION_CLAMP = 30        # max px movement per update

    def __init__(self):
        self.nodes: dict[str, NodeState] = {}
        self.edges: list[dict] = []
        self._mention_log: list[tuple[str, float]] = []  # (node_id, timestamp)
        self._churn_log: list[tuple[float, int, int, int]] = []  # (ts, added, removed, edge_delta)

    def reconcile(self, proposed: dict) -> dict:
        """
        Reconcile Claude's proposed graph with server state.
        Returns the active graph suitable for the frontend.
        """
        now = time.time()
        proposed_nodes = {n["id"]: n for n in proposed.get("nodes", [])}
        proposed_edges = proposed.get("edges", [])
        old_edge_count = len(self.edges)

        added = 0
        removed = 0

        # 1. Update/create nodes from proposal
        seen_ids = set()
        for nid, pn in proposed_nodes.items():
            seen_ids.add(nid)
            if nid in self.nodes:
                ns = self.nodes[nid]
                ns.label = pn.get("label", ns.label)
                ns.group = pn.get("group", ns.group)
                ns.last_mentioned = now
                ns.mention_count += 1
                if ns.state == "parked":
                    ns.state = "active"  # re-proposed by Claude → reactivate
            else:
                self.nodes[nid] = NodeState(
                    id=nid,
                    label=pn.get("label", nid),
                    group=pn.get("group", ""),
                    state="active",
                    first_mentioned=now,
                    last_mentioned=now,
                    mention_count=1,
                    x=pn.get("x", 0),
                    y=pn.get("y", 0),
                )
                added += 1
            self._mention_log.append((nid, now))

        # 2. Reactivation check: parked nodes mentioned 2+ times in 3min
        cutoff = now - self.REACTIVATION_WINDOW
        recent_mentions: dict[str, int] = {}
        for nid, ts in self._mention_log:
            if ts >= cutoff:
                recent_mentions[nid] = recent_mentions.get(nid, 0) + 1
        for nid, count in recent_mentions.items():
            if nid in self.nodes and self.nodes[nid].state == "parked" and count >= self.REACTIVATION_MENTIONS:
                self.nodes[nid].state = "active"
                self.nodes[nid].last_mentioned = now

        # 3. Decay: active nodes not in proposal, last mentioned >12min ago → parked
        for nid, ns in self.nodes.items():
            if ns.state == "active" and nid not in seen_ids and not ns.pinned:
                age = now - ns.last_mentioned
                if age > self.DECAY_SECONDS:
                    ns.state = "parked"
                    removed += 1

        # 4. Update edges, filter to existing active/parked nodes
        active_ids = {nid for nid, ns in self.nodes.items() if ns.state in ("active", "parked")}
        self.edges = [
            {"source": e["source"], "target": e["target"], "label": e.get("label", "")}
            for e in proposed_edges
            if e["source"] in active_ids and e["target"] in active_ids
        ]

        # 5. Compute edge counts
        for ns in self.nodes.values():
            ns.edge_count = 0
        for e in self.edges:
            if e["source"] in self.nodes:
                self.nodes[e["source"]].edge_count += 1
            if e["target"] in self.nodes:
                self.nodes[e["target"]].edge_count += 1

        # 6. Score all active nodes
        active_nodes = [ns for ns in self.nodes.values() if ns.state == "active"]
        if active_nodes:
            max_mentions = max(ns.mention_count for ns in active_nodes) or 1
            max_edges = max(ns.edge_count for ns in active_nodes) or 1

            for ns in active_nodes:
                age_secs = now - ns.last_mentioned
                recency = math.pow(2, -age_secs / 300)  # 5-min half-life
                frequency = ns.mention_count / max_mentions
                centrality = ns.edge_count / max_edges
                pin_bonus = 0.15 if ns.pinned else 0.0
                ns.importance = 0.45 * recency + 0.35 * frequency + 0.20 * centrality + pin_bonus

        # 7. Budget enforcement: if >MAX_ACTIVE active, park lowest-scoring non-pinned
        active_nodes = sorted(
            [ns for ns in self.nodes.values() if ns.state == "active"],
            key=lambda ns: ns.importance,
        )
        while len([ns for ns in active_nodes if ns.state == "active"]) > self.MAX_ACTIVE:
            # Find lowest-scoring non-pinned
            for ns in active_nodes:
                if ns.state == "active" and not ns.pinned:
                    ns.state = "parked"
                    removed += 1
                    break
            else:
                break  # all remaining are pinned

        # 8. Track churn
        edge_delta = len(self.edges) - old_edge_count
        self._churn_log.append((now, added, removed, edge_delta))
        # Trim logs older than 10 minutes
        cutoff_log = now - 600
        self._mention_log = [(nid, ts) for nid, ts in self._mention_log if ts > cutoff_log]
        self._churn_log = [(ts, a, r, ed) for ts, a, r, ed in self._churn_log if ts > cutoff_log]

        return self.get_active_graph()

    def get_active_graph(self) -> dict:
        """Return only active nodes + their edges (max MAX_VISIBLE)."""
        active = [ns for ns in self.nodes.values() if ns.state == "active"]
        active.sort(key=lambda ns: ns.importance, reverse=True)
        active = active[:self.MAX_VISIBLE]
        active_ids = {ns.id for ns in active}

        nodes = [
            {
                "id": ns.id, "label": ns.label, "group": ns.group,
                "state": ns.state, "importance": round(ns.importance, 3),
                "pinned": ns.pinned, "x": ns.x, "y": ns.y,
            }
            for ns in active
        ]
        edges = [e for e in self.edges if e["source"] in active_ids and e["target"] in active_ids]

        return {"nodes": nodes, "edges": edges}

    def apply_action(self, action_type: str, payload: dict) -> dict:
        """Apply a user action and return updated active graph."""
        nid = payload.get("node_id", "")

        if action_type == "pin":
            if nid in self.nodes:
                self.nodes[nid].pinned = not self.nodes[nid].pinned

        elif action_type == "hide":
            if nid in self.nodes:
                self.nodes[nid].state = "hidden"

        elif action_type == "rename":
            if nid in self.nodes:
                self.nodes[nid].label = payload.get("label", self.nodes[nid].label)

        elif action_type == "merge":
            src_id = payload.get("source_id", "")
            tgt_id = payload.get("target_id", "")
            if src_id in self.nodes and tgt_id in self.nodes:
                src = self.nodes[src_id]
                tgt = self.nodes[tgt_id]
                tgt.mention_count += src.mention_count
                tgt.last_mentioned = max(tgt.last_mentioned, src.last_mentioned)
                tgt.first_mentioned = min(tgt.first_mentioned, src.first_mentioned)
                # Redirect edges
                for e in self.edges:
                    if e["source"] == src_id:
                        e["source"] = tgt_id
                    if e["target"] == src_id:
                        e["target"] = tgt_id
                # Remove self-loops
                self.edges = [e for e in self.edges if e["source"] != e["target"]]
                # Deduplicate edges
                seen = set()
                deduped = []
                for e in self.edges:
                    key = (e["source"], e["target"])
                    if key not in seen:
                        seen.add(key)
                        deduped.append(e)
                self.edges = deduped
                src.state = "archived"

        elif action_type == "promote":
            if nid in self.nodes:
                ns = self.nodes[nid]
                ns.importance = min(1.0, ns.importance + 0.2)
                ns.last_mentioned = time.time()
                if ns.state == "parked":
                    ns.state = "active"

        return self.get_active_graph()

    def clamp_positions(self, new_nodes: list[dict], old_positions: dict[str, tuple[float, float]]):
        """Cap x/y movement to ±POSITION_CLAMP px per update."""
        for n in new_nodes:
            nid = n["id"]
            if nid in old_positions:
                ox, oy = old_positions[nid]
                dx = n.get("x", ox) - ox
                dy = n.get("y", oy) - oy
                dx = max(-self.POSITION_CLAMP, min(self.POSITION_CLAMP, dx))
                dy = max(-self.POSITION_CLAMP, min(self.POSITION_CLAMP, dy))
                n["x"] = ox + dx
                n["y"] = oy + dy

    def get_churn_metrics(self) -> dict:
        """Return per-minute churn stats for the admin panel."""
        now = time.time()
        one_min_ago = now - 60
        recent = [(ts, a, r, ed) for ts, a, r, ed in self._churn_log if ts > one_min_ago]
        return {
            "nodes_added_per_min": sum(a for _, a, _, _ in recent),
            "nodes_removed_per_min": sum(r for _, _, r, _ in recent),
            "edge_churn_per_min": sum(abs(ed) for _, _, _, ed in recent),
        }

    def get_full_state(self) -> dict:
        """Serialize full state for snapshot storage."""
        return {
            "nodes": {
                nid: {
                    "id": ns.id, "label": ns.label, "group": ns.group,
                    "state": ns.state, "importance": ns.importance,
                    "first_mentioned": ns.first_mentioned,
                    "last_mentioned": ns.last_mentioned,
                    "mention_count": ns.mention_count,
                    "edge_count": ns.edge_count,
                    "pinned": ns.pinned, "x": ns.x, "y": ns.y,
                }
                for nid, ns in self.nodes.items()
            },
            "edges": self.edges,
        }

    def load_state(self, state: dict):
        """Restore from a snapshot."""
        self.nodes.clear()
        for nid, nd in state.get("nodes", {}).items():
            self.nodes[nid] = NodeState(
                id=nd["id"], label=nd["label"], group=nd["group"],
                state=nd.get("state", "active"),
                importance=nd.get("importance", 0.5),
                first_mentioned=nd.get("first_mentioned", 0),
                last_mentioned=nd.get("last_mentioned", 0),
                mention_count=nd.get("mention_count", 1),
                edge_count=nd.get("edge_count", 0),
                pinned=nd.get("pinned", False),
                x=nd.get("x", 0), y=nd.get("y", 0),
            )
        self.edges = state.get("edges", [])
