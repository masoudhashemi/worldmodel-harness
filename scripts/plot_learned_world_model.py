import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from worldmodel_harness.store import KnowledgeStore

def generate_world_model_plot(store_path: Path, output_path: Path):
    store = KnowledgeStore(store_path)
    claims = store.claims()
    
    # We focus on the two active audited domains: Operations (Sheets, Twilio) & Sales (Salesforce, Zoom)
    target_services = {"sheets", "twilio", "salesforce", "zoom"}
    
    G = nx.DiGraph()
    
    # Track node types and service groups
    node_colors = []
    node_sizes = []
    node_labels = {}
    
    endpoints = {}
    schema_fields = {}
    value_flows = []
    error_causes = []
    semantics = {}
    
    for c in claims:
        if c.relation == "endpoint_contract":
            ep_id = c.payload.get("id", "")
            svc = ep_id.split(".")[0] if "." in ep_id else "other"
            if svc in target_services:
                method = c.payload.get("method", "GET")
                endpoints[ep_id] = {
                    "svc": svc,
                    "method": method,
                    "url": c.payload.get("url", ""),
                    "statement": c.statement[:35],
                }
        elif c.relation == "observed_output_type":
            ep_id = c.payload.get("endpoint", "")
            svc = ep_id.split(".")[0] if "." in ep_id else "other"
            if svc in target_services:
                f_name = c.payload.get("field", "")
                f_type = c.payload.get("type", "")
                schema_fields.setdefault(ep_id, []).append((f_name, f_type))
        elif c.relation == "observed_value_flow":
            src = c.payload.get("source_endpoint", "")
            tgt = c.payload.get("target_endpoint", "")
            out_p = c.payload.get("output_path", "")
            in_p = c.payload.get("input_path", "")
            value_flows.append((src, tgt, out_p, in_p))
        elif c.relation == "documented_operation_semantics":
            quotes = c.payload.get("quotes", [])
            effect = c.payload.get("effect", "")
            for q in quotes:
                for ep_id in endpoints:
                    if ep_id in q.lower() or "sms" in q.lower() and "twilio" in ep_id:
                        semantics[ep_id] = effect
        elif c.relation == "possible_error_cause":
            error_causes.append(c.statement)

    print(f"Extracted: {len(endpoints)} endpoints, {sum(len(v) for v in schema_fields.values())} schema fields, {len(value_flows)} value flows")

    # Build NetworkX graph
    # 1. Add Service Clusters / Hubs
    for svc in target_services:
        G.add_node(f"SVC_{svc}", type="service", label=svc.upper())
    
    # 2. Add Endpoints
    for ep_id, ep_info in endpoints.items():
        ep_node = f"EP_{ep_id}"
        method = ep_info["method"]
        effect = semantics.get(ep_id, "read" if method == "GET" else "write")
        short_name = ep_id.split(".", 1)[-1]
        label = f"{method}\n{short_name}"
        G.add_node(ep_node, type="endpoint", label=label, service=ep_info["svc"], effect=effect)
        # Connect service to endpoint
        G.add_edge(f"SVC_{ep_info['svc']}", ep_node, kind="service_has")
        
        # 3. Add Schema Fields (collapsed or top fields)
        fields = schema_fields.get(ep_id, [])
        for f_name, f_type in fields[:4]: # Top 4 fields
            f_node = f"SCHEMA_{ep_id}_{f_name}"
            G.add_node(f_node, type="schema", label=f"{f_name}:{f_type}")
            G.add_edge(ep_node, f_node, kind="outputs")

    # 4. Add Value Flows
    for src, tgt, out_p, in_p in value_flows:
        src_node = f"EP_{src}"
        tgt_node = f"EP_{tgt}"
        if G.has_node(src_node) and G.has_node(tgt_node):
            G.add_edge(src_node, tgt_node, kind="value_flow", label=f"{out_p}->{in_p}")

    # Layout using Graphviz / Multipartite / Spring
    fig, ax = plt.subplots(figsize=(18, 12), dpi=200)
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    pos = nx.spring_layout(G, k=0.8, iterations=100, seed=42)

    # Color mapping
    colors = {
        "service": "#58a6ff",      # GitHub Blue
        "endpoint_read": "#3fb950", # GitHub Green
        "endpoint_write": "#f0883e",# GitHub Orange
        "schema": "#8b949e",        # GitHub Gray
    }

    # Draw edges
    service_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "service_has"]
    output_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "outputs"]
    flow_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "value_flow"]

    nx.draw_networkx_edges(G, pos, edgelist=service_edges, edge_color="#30363d", width=1.5, alpha=0.6, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=output_edges, edge_color="#238636", width=1.0, alpha=0.5, style="dashed", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=flow_edges, edge_color="#ec6547", width=2.5, alpha=0.9, arrowsize=18, ax=ax)

    # Draw nodes by type
    for n, d in G.nodes(data=True):
        ntype = d.get("type")
        if ntype == "service":
            nx.draw_networkx_nodes(G, pos, nodelist=[n], node_color="#1f6feb", node_size=3200, alpha=0.95, ax=ax, edgecolors="#58a6ff", linewidths=2)
            ax.text(pos[n][0], pos[n][1], d["label"], fontsize=11, fontweight="bold", color="white", ha="center", va="center")
        elif ntype == "endpoint":
            eff = d.get("effect", "read")
            c = colors["endpoint_read"] if eff == "read" else colors["endpoint_write"]
            nx.draw_networkx_nodes(G, pos, nodelist=[n], node_color=c, node_size=2000, alpha=0.9, ax=ax, edgecolors="white", linewidths=1.2)
            ax.text(pos[n][0], pos[n][1], d["label"], fontsize=7.5, fontweight="bold", color="#0d1117", ha="center", va="center")
        elif ntype == "schema":
            nx.draw_networkx_nodes(G, pos, nodelist=[n], node_color="#21262d", node_size=1000, alpha=0.85, ax=ax, edgecolors="#8b949e", linewidths=1)
            ax.text(pos[n][0], pos[n][1], d["label"], fontsize=6.5, color="#c9d1d9", ha="center", va="center")

    # Flow edge labels
    flow_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True) if d.get("kind") == "value_flow"}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=flow_labels, font_size=7.5, font_color="#f85149", bbox=dict(boxstyle="round,pad=0.2", facecolor="#161b22", edgecolor="#f85149", alpha=0.9), ax=ax)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#1f6feb", edgecolor="#58a6ff", label="Service Hub (Boundary Partition)"),
        mpatches.Patch(facecolor="#3fb950", edgecolor="white", label="Read Endpoint (Queries / Inspects State)"),
        mpatches.Patch(facecolor="#f0883e", edgecolor="white", label="Write Endpoint (Mutates External State)"),
        mpatches.Patch(facecolor="#21262d", edgecolor="#8b949e", label="Observed Output Schema (Field : Type)"),
        mpatches.Patch(facecolor="#ec6547", edgecolor="#ec6547", label="Value Flow / Composition Edge"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=9, framealpha=0.95)

    ax.set_title("Learned World Model (Environment Layer) — Entity, Schema & Composition Topology", fontsize=15, fontweight="bold", color="white", pad=20)
    ax.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Plot saved successfully to {output_path}")

if __name__ == "__main__":
    import sys
    store_file = Path("runs/focused-env-audit/store.sqlite")
    out_file = Path("/Users/masoud/.gemini/antigravity-cli/brain/0310ea49-122f-4371-b14b-e4c012e57910/world_model_graph.png")
    generate_world_model_plot(store_file, out_file)
