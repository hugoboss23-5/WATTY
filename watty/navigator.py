"""
Watty Navigator — Layer 2 of the Trinity Stack
=================================================
The felt reasoning engine. Makes the Chestahedron geometry active.

Instead of flat dot-product similarity (recall), queries *spread* through
the association graph, shaped by the geometry. Three organs read the shape
of activation. A heart decides if it's coherent. Blood negotiates when
organs disagree. The answer emerges from the geometry settling.

Organs:
  CoherenceOrgan — reads geometric coherence of activated memories in 7D
  DepthOrgan     — reads how deep activation has spread through associations
  BridgeOrgan    — reads which Chestahedron faces are active

Heart:
  Measures organ signal agreement. Decides whether to circulate again.
  Convergence threshold: PHI_INV (0.618...)

Blood:
  Pattern-matches on organ signals. Returns modulation dict that shapes
  the next spread pass.

Hugo & Rim & Claude — February 2026
"""

import numpy as np

from watty.brain import Brain
from watty.embeddings import embed_text
from watty.chestahedron import (
    Chestahedron as ChestahedronCore, embedding_to_signal,
    CHESTAHEDRON_DIM, PHI_INV,
)
from watty.config import (
    NAVIGATOR_MAX_CIRCULATIONS, NAVIGATOR_DECAY, NAVIGATOR_GEO_EDGE_WEIGHT,
    NAVIGATOR_FELT_MODULATION, NAVIGATOR_MIN_ACTIVATION,
    NAVIGATOR_SEED_TOP_N, NAVIGATOR_SEED_THRESHOLD,
    EMBEDDING_DIMENSION,
)


# ── Organs ───────────────────────────────────────────────────

class CoherenceOrgan:
    """Reads geometric coherence of activated memories in 7D space."""

    @staticmethod
    def read(activation: dict, coordinates: dict, chestahedron: ChestahedronCore) -> dict:
        # Collect top-20 activated nodes that have coordinates
        sorted_ids = sorted(activation, key=activation.get, reverse=True)
        active_with_coords = [
            cid for cid in sorted_ids
            if cid in coordinates and coordinates[cid] is not None
        ][:20]

        if len(active_with_coords) < 2:
            return {"signal": "balanced", "mean_coherence": 0.5, "std_coherence": 0.0, "n_pairs": 0}

        # Pairwise coherence
        coherences = []
        for i in range(len(active_with_coords)):
            for j in range(i + 1, len(active_with_coords)):
                c = chestahedron.coherence(
                    coordinates[active_with_coords[i]],
                    coordinates[active_with_coords[j]],
                )
                coherences.append(c)

        mean_coh = float(np.mean(coherences))
        std_coh = float(np.std(coherences))

        if mean_coh > 0.6:
            signal = "clustered"
        elif mean_coh < 0.2:
            signal = "scattered"
        elif std_coh > 0.25 and 0.2 <= mean_coh <= 0.6:
            signal = "multi_cluster"
        else:
            signal = "balanced"

        return {
            "signal": signal,
            "mean_coherence": round(mean_coh, 4),
            "std_coherence": round(std_coh, 4),
            "n_pairs": len(coherences),
        }


class DepthOrgan:
    """Reads how deep activation has spread through the association graph."""

    @staticmethod
    def read(activation: dict, hop_distances: dict, tiers: dict) -> dict:
        if not activation:
            return {"signal": "surface_only", "mean_depth": 0.0, "distribution": {}, "tier_distribution": {}}

        # Activation-weighted mean hop distance
        total_act = sum(activation.values())
        if total_act < 1e-12:
            return {"signal": "surface_only", "mean_depth": 0.0, "distribution": {}, "tier_distribution": {}}

        weighted_depth = sum(
            activation[cid] * hop_distances.get(cid, 0)
            for cid in activation
        )
        mean_depth = weighted_depth / total_act

        # Distribution per hop level
        hop_buckets = {0: 0.0, 1: 0.0, 2: 0.0, "3+": 0.0}
        for cid, act in activation.items():
            hop = hop_distances.get(cid, 0)
            if hop >= 3:
                hop_buckets["3+"] += act
            else:
                hop_buckets[hop] += act

        # Normalize to fractions
        distribution = {k: round(v / total_act, 4) for k, v in hop_buckets.items()}

        # Tier distribution
        tier_counts = {}
        for cid, act in activation.items():
            tier = tiers.get(cid, "episodic")
            tier_counts[tier] = tier_counts.get(tier, 0.0) + act
        tier_distribution = {k: round(v / total_act, 4) for k, v in tier_counts.items()}

        # Signal classification
        hop0_frac = distribution.get(0, 0.0)
        deep_frac = distribution.get(2, 0.0) + distribution.get("3+", 0.0)

        if hop0_frac > 0.8:
            signal = "surface_only"
        elif deep_frac > 0.4:
            signal = "deep"
        else:
            signal = "balanced_depth"

        return {
            "signal": signal,
            "mean_depth": round(mean_depth, 4),
            "distribution": distribution,
            "tier_distribution": tier_distribution,
        }


class BridgeOrgan:
    """Reads which Chestahedron faces are active."""

    @staticmethod
    def read(activation: dict, coordinates: dict) -> dict:
        # Accumulate activation × |coordinate| across 7 faces
        face_energy = np.zeros(CHESTAHEDRON_DIM)

        for cid, act in activation.items():
            coord = coordinates.get(cid)
            if coord is None:
                continue
            face_energy += act * np.abs(coord)

        total = float(np.sum(face_energy))
        if total < 1e-12:
            return {"signal": "balanced_bridging", "face_fractions": [0.0] * CHESTAHEDRON_DIM, "dominant_face": None}

        face_fractions = (face_energy / total).tolist()
        uniform_mean = 1.0 / CHESTAHEDRON_DIM

        dominant_face = int(np.argmax(face_fractions))
        max_frac = face_fractions[dominant_face]

        above_uniform = sum(1 for f in face_fractions if f > uniform_mean)

        if max_frac > 0.5:
            signal = "single_face"
        elif above_uniform >= 3:
            signal = "multi_face"
        else:
            signal = "balanced_bridging"

        return {
            "signal": signal,
            "face_fractions": [round(f, 4) for f in face_fractions],
            "dominant_face": dominant_face,
        }


# ── Heart ────────────────────────────────────────────────────

class Heart:
    """Measures organ signal agreement. Decides whether to circulate again."""

    NEUTRAL_SIGNALS = {"balanced", "balanced_depth", "balanced_bridging"}

    @staticmethod
    def beat(coherence_reading: dict, depth_reading: dict, bridge_reading: dict) -> dict:
        signals = []
        for reading in [coherence_reading, depth_reading, bridge_reading]:
            sig = reading.get("signal", "balanced")
            if sig not in Heart.NEUTRAL_SIGNALS:
                signals.append(sig)

        unique = len(set(signals))

        if unique == 0:
            coherence = 1.0
        elif unique == 1:
            coherence = 0.9
        else:
            coherence = 1.0 / (1.0 + unique)

        should_circulate = coherence < PHI_INV

        return {
            "coherence": round(coherence, 4),
            "should_circulate": should_circulate,
            "signals": signals,
            "unique_signal_count": unique,
        }


# ── Blood ────────────────────────────────────────────────────

class Blood:
    """Pattern-matches on organ signals. Returns modulation for the next spread."""

    @staticmethod
    def negotiate(coherence_reading: dict, depth_reading: dict, bridge_reading: dict) -> dict:
        coh_sig = coherence_reading.get("signal", "balanced")
        depth_sig = depth_reading.get("signal", "balanced_depth")
        bridge_sig = bridge_reading.get("signal", "balanced_bridging")

        modulation = {
            "depth_bias": 1.0,
            "coherence_bias": 1.0,
            "bridge_bias": 1.0,
            "focus_faces": None,
            "strategy": "amplify_aligned",
        }

        if coh_sig == "scattered" and bridge_sig == "multi_face":
            modulation.update({
                "strategy": "focus_bridges",
                "bridge_bias": 1.5,
                "coherence_bias": 0.7,
            })
        elif coh_sig == "clustered" and depth_sig == "surface_only":
            modulation.update({
                "strategy": "drill_deeper",
                "depth_bias": 1.5,
                "coherence_bias": 1.2,
            })
        elif coh_sig == "multi_cluster":
            modulation.update({
                "strategy": "bridge_clusters",
                "bridge_bias": 1.3,
                "depth_bias": 1.1,
            })
        elif coh_sig == "clustered" and bridge_sig == "single_face":
            dominant = bridge_reading.get("dominant_face")
            modulation.update({
                "strategy": "broaden_faces",
                "bridge_bias": 1.4,
                "focus_faces": [i for i in range(CHESTAHEDRON_DIM) if i != dominant] if dominant is not None else None,
            })
        elif coh_sig == "scattered" and depth_sig == "deep":
            modulation.update({
                "strategy": "tighten_surface",
                "depth_bias": 0.7,
                "coherence_bias": 1.4,
            })
        else:
            # Default: gentle boost to coherence and depth
            modulation.update({
                "strategy": "amplify_aligned",
                "coherence_bias": 1.1,
                "depth_bias": 1.05,
            })

        return modulation


# ── Navigator ────────────────────────────────────────────────

class Navigator:
    """
    The felt reasoning engine. Spreads activation through the association graph,
    shaped by geometry. Organs read the shape. Heart decides. Blood negotiates.
    The answer emerges from the geometry settling.
    """

    def __init__(self, brain: Brain):
        self.brain = brain
        self._assoc_graph: dict[int, list[tuple[int, float]]] = {}
        self._assoc_graph_version: bool = True  # tracks brain._index_dirty

    def navigate(self, query: str, top_k: int = 10) -> dict:
        """Main navigation loop. Returns results + organ diagnostics."""
        # Ensure brain index is fresh
        if self.brain._index_dirty:
            self.brain._build_index()
        if self.brain._vectors is None or len(self.brain._vectors) == 0:
            return {"results": [], "circulations": 0, "organ_readings": [],
                    "heart_readings": [], "blood_strategies": [], "final_coherence": 0.0}

        self._ensure_assoc_graph()

        # Embed query + compute geometric coordinate
        query_vec = embed_text(query)
        query_signal = embedding_to_signal(query_vec)
        query_coord, _ = self.brain.chestahedron.process(query_signal)

        # Build coordinate lookup from brain's cached coordinates
        coordinates = {}
        for i, cid in enumerate(self.brain._vector_ids):
            coord = self.brain._coordinates[i] if self.brain._coordinates else None
            if coord is not None:
                coordinates[cid] = coord

        # Seed: multi-resolution (keywords + embedding)
        activation, hop_distances = self._seed(query, query_vec)

        if not activation:
            return {"results": [], "circulations": 0, "organ_readings": [],
                    "heart_readings": [], "blood_strategies": [], "final_coherence": 0.0}

        # Load tiers for active chunk ids
        tiers = self._load_tiers(list(activation.keys()))

        organ_readings = []
        heart_readings = []
        blood_strategies = []
        circulations = 0

        for circ in range(NAVIGATOR_MAX_CIRCULATIONS):
            circulations = circ + 1

            # Organs read the shape
            coh_reading = CoherenceOrgan.read(activation, coordinates, self.brain.chestahedron)
            depth_reading = DepthOrgan.read(activation, hop_distances, tiers)
            bridge_reading = BridgeOrgan.read(activation, coordinates)

            organ_readings.append([coh_reading, depth_reading, bridge_reading])

            # Heart measures coherence
            heart = Heart.beat(coh_reading, depth_reading, bridge_reading)
            heart_readings.append(heart)

            if not heart["should_circulate"]:
                blood_strategies.append("converged")
                break

            # Blood negotiates modulation
            modulation = Blood.negotiate(coh_reading, depth_reading, bridge_reading)
            blood_strategies.append(modulation["strategy"])

            # Spread activation with modulation
            activation, hop_distances = self._spread(
                activation, hop_distances, coordinates, modulation, query_coord,
            )

            # Refresh tiers for newly activated nodes
            new_ids = [cid for cid in activation if cid not in tiers]
            if new_ids:
                tiers.update(self._load_tiers(new_ids))

        final_coherence = heart_readings[-1]["coherence"] if heart_readings else 0.0

        # Rank and format
        results = self._rank_and_format(activation, query_vec, query_coord, top_k)

        return {
            "results": results,
            "circulations": circulations,
            "organ_readings": organ_readings,
            "heart_readings": heart_readings,
            "blood_strategies": blood_strategies,
            "final_coherence": final_coherence,
        }

    def _ensure_assoc_graph(self):
        """Load association graph into RAM adjacency dict. Cache until index changes."""
        if self._assoc_graph and self._assoc_graph_version == self.brain._index_dirty:
            return

        conn = self.brain._connect()
        rows = conn.execute(
            "SELECT source_chunk_id, target_chunk_id, strength "
            "FROM associations WHERE strength >= 0.05"
        ).fetchall()
        conn.close()

        graph: dict[int, list[tuple[int, float]]] = {}
        for row in rows:
            src = row["source_chunk_id"]
            tgt = row["target_chunk_id"]
            strength = row["strength"]
            if src not in graph:
                graph[src] = []
            graph[src].append((tgt, strength))

        self._assoc_graph = graph
        self._assoc_graph_version = self.brain._index_dirty

    def _seed(self, query: str, query_vec: np.ndarray) -> tuple[dict, dict]:
        """
        Multi-resolution seeding (Blueprint Section 5.2):
        1. Keyword-based seeds from content search
        2. Embedding-based seeds from similarity
        3. Merge: union of both, max-score wins
        """
        activation = {}
        hop_distances = {}

        # Stage 1: Keyword seeds — each significant word gets its own lookup
        keywords = [w for w in query.lower().split() if len(w) >= 3]
        for kw in keywords[:5]:  # Cap at 5 keywords
            matches = self.brain.keyword_search(kw, limit=5)
            for cid, score in matches:
                adjusted = score * 0.7  # Keywords slightly lower weight than embedding
                if cid not in activation or adjusted > activation[cid]:
                    activation[cid] = adjusted
                    hop_distances[cid] = 0

        # Stage 2: Embedding seeds (original method)
        similarities = np.dot(self.brain._vectors, query_vec)
        top_indices = np.argsort(similarities)[::-1][:NAVIGATOR_SEED_TOP_N]

        for idx in top_indices:
            sim = float(similarities[idx])
            if sim < NAVIGATOR_SEED_THRESHOLD:
                break
            cid = self.brain._vector_ids[idx]
            if cid not in activation or sim > activation[cid]:
                activation[cid] = sim
                hop_distances[cid] = 0

        # Stage 3: Entity graph seeds — leverage knowledge graph relationships
        if hasattr(self.brain, '_kg') and self.brain._kg is not None:
            try:
                kg_chunks = self.brain._kg.graph_recall_chunks(query, top_k=10)
                for chunk_id, rrf_score in kg_chunks:
                    adjusted = rrf_score * 0.6  # KG seeds weighted below embeddings
                    if chunk_id not in activation or adjusted > activation[chunk_id]:
                        activation[chunk_id] = adjusted
                        hop_distances[chunk_id] = 0
            except Exception:
                pass  # KG unavailable — degrade silently

        return activation, hop_distances

    def _spread(self, activation: dict, hop_distances: dict,
                coordinates: dict, modulation: dict,
                query_coord: np.ndarray) -> tuple[dict, dict]:
        """One hop of spreading activation through the association graph."""
        new_activation = dict(activation)  # start with current
        new_hops = dict(hop_distances)

        depth_bias = modulation.get("depth_bias", 1.0)
        coherence_bias = modulation.get("coherence_bias", 1.0)
        bridge_bias = modulation.get("bridge_bias", 1.0)
        focus_faces = modulation.get("focus_faces")

        for src_id, src_act in list(activation.items()):
            neighbors = self._assoc_graph.get(src_id, [])
            src_hop = hop_distances.get(src_id, 0)

            for tgt_id, edge_strength in neighbors:
                # Base propagation
                propagated = src_act * edge_strength

                # Geometric weight: coherence between source and target in 7D
                src_coord = coordinates.get(src_id)
                tgt_coord = coordinates.get(tgt_id)
                if src_coord is not None and tgt_coord is not None and NAVIGATOR_GEO_EDGE_WEIGHT > 0:
                    geo_coh = self.brain.chestahedron.coherence(src_coord, tgt_coord)
                    geo_weight = max(0.05, geo_coh)
                    propagated *= geo_weight * NAVIGATOR_GEO_EDGE_WEIGHT

                # Felt factor: alignment of target coord with felt_state
                if NAVIGATOR_FELT_MODULATION and tgt_coord is not None:
                    felt = self.brain.chestahedron.felt_state
                    felt_norm = np.linalg.norm(felt)
                    if felt_norm > 1e-8:
                        felt_alignment = float(np.dot(tgt_coord, felt) / (np.linalg.norm(tgt_coord) * felt_norm + 1e-12))
                        felt_factor = max(0.1, (felt_alignment + 1.0) / 2.0)  # map [-1,1] to [0.1, 1.0]
                        propagated *= felt_factor

                # Blood modulation biases
                tgt_hop = src_hop + 1
                # Depth bias: scale by how deep we're going
                propagated *= depth_bias if tgt_hop > 0 else 1.0
                # Coherence bias: applied to geometric weight already implicitly
                propagated *= coherence_bias
                # Bridge bias: boost if target is on a focus face
                if focus_faces is not None and tgt_coord is not None:
                    dominant_face = int(np.argmax(np.abs(tgt_coord)))
                    if dominant_face in focus_faces:
                        propagated *= bridge_bias

                # Decay by hop distance
                propagated *= NAVIGATOR_DECAY ** tgt_hop

                # Max-merge: keep highest activation path
                if tgt_id not in new_activation or propagated > new_activation[tgt_id]:
                    new_activation[tgt_id] = propagated
                    new_hops[tgt_id] = tgt_hop

        # Prune below minimum activation
        pruned_activation = {
            cid: act for cid, act in new_activation.items()
            if act >= NAVIGATOR_MIN_ACTIVATION
        }
        pruned_hops = {cid: new_hops[cid] for cid in pruned_activation}

        return pruned_activation, pruned_hops

    def _load_tiers(self, chunk_ids: list[int]) -> dict:
        """Load memory_tier for a set of chunk IDs."""
        if not chunk_ids:
            return {}

        conn = self.brain._connect()
        placeholders = ",".join("?" * len(chunk_ids))
        rows = conn.execute(
            f"SELECT id, memory_tier FROM chunks WHERE id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        conn.close()

        return {row["id"]: (row["memory_tier"] or "episodic") for row in rows}

    def _rank_and_format(self, activation: dict, query_vec: np.ndarray,
                         query_coord: np.ndarray, top_k: int) -> list[dict]:
        """Rank by final activation score, format results matching brain.recall() output."""
        # Sort by activation
        ranked = sorted(activation.items(), key=lambda x: x[1], reverse=True)[:top_k]

        if not ranked:
            return []

        chunk_ids = [cid for cid, _ in ranked]
        placeholders = ",".join("?" * len(chunk_ids))

        conn = self.brain._connect()
        rows = conn.execute(
            f"SELECT c.id, c.content, c.compressed_content, c.provider, c.role, "
            f"c.created_at, c.source_type, c.source_path, c.memory_tier, c.coordinate "
            f"FROM chunks c WHERE c.id IN ({placeholders})",
            chunk_ids,
        ).fetchall()

        # Bump access counts
        from watty.config import EMBEDDING_DIMENSION as _  # noqa: already imported, just trigger
        self.brain._bump_access(chunk_ids, conn)
        conn.commit()
        conn.close()

        # Build lookup
        row_map = {row["id"]: row for row in rows}

        results = []
        for cid, act in ranked:
            row = row_map.get(cid)
            if not row:
                continue

            # Compute similarity to query for comparison
            idx = self.brain._vector_id_to_idx.get(cid)
            similarity = 0.0
            if idx is not None and self.brain._vectors is not None:
                similarity = float(np.dot(self.brain._vectors[idx], query_vec))

            content = row["content"]
            is_compressed = False
            if row["compressed_content"]:
                content = row["compressed_content"]
                is_compressed = True

            results.append({
                "chunk_id": cid,
                "content": content,
                "score": round(float(act), 4),
                "similarity": round(similarity, 4),
                "activation": round(float(act), 4),
                "provider": row["provider"],
                "role": row["role"],
                "created_at": row["created_at"],
                "source_type": row["source_type"],
                "source_path": row["source_path"],
                "memory_tier": row["memory_tier"] or "episodic",
                "compressed": is_compressed,
            })

        return results
