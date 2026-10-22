import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

def plot_state_transitions(out_path: Path):
    fig, ax = plt.subplots(figsize=(19, 11), dpi=220)
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # States definition (Nodes)
    states = [
        {
            "id": "S0",
            "x": 1.5, "y": 8.0,
            "title": "S₀: Initial Environment State",
            "desc": "• Spreadsheets & CRM unread\n• Badge expiry dates unknown\n• Notification not sent\n• Status: PENDING",
            "color": "#1f6feb",
            "border": "#58a6ff"
        },
        {
            "id": "S1",
            "x": 5.5, "y": 8.0,
            "title": "S₁: Interface Discovered",
            "desc": "• Endpoint contract cached:\n  GET /v4/spreadsheets/{id}\n• Spreadsheet schema known\n• Status: CAPABILITY DISCOVERED",
            "color": "#1f6feb",
            "border": "#58a6ff"
        },
        {
            "id": "S_err",
            "x": 5.5, "y": 4.5,
            "title": "S_err: Causal Error State",
            "desc": "• HTTP 400: Param in URL path\n• Invariant learned:\n  Separate path from query params\n• Status: RECOVERY TRIGGERED",
            "color": "#6e2a2a",
            "border": "#f85149"
        },
        {
            "id": "S2",
            "x": 9.5, "y": 8.0,
            "title": "S₂: Metadata Grounded",
            "desc": "• Title: 'Badge Tracker'\n• Sheet ID confirmed\n• Data range located: A1:Z100\n• Status: TARGET LOCATED",
            "color": "#238636",
            "border": "#3fb950"
        },
        {
            "id": "S3",
            "x": 13.5, "y": 8.0,
            "title": "S₃: Entity Data Grounded",
            "desc": "• Cell values extracted\n• 3 Expiring Badges identified\n• Phone: +15551234567\n• Status: MUTATION READY",
            "color": "#238636",
            "border": "#3fb950"
        },
        {
            "id": "S4",
            "x": 13.5, "y": 4.5,
            "title": "S₄: Notifier Discovered",
            "desc": "• Endpoint contract cached:\n  POST /Accounts/{Sid}/Messages\n• Required fields: To, Body\n• Status: ACTION AUTHORIZED",
            "color": "#1f6feb",
            "border": "#58a6ff"
        },
        {
            "id": "S5",
            "x": 9.5, "y": 1.5,
            "title": "S₅: World State Mutated",
            "desc": "• SMS Dispatched via Twilio\n• SID: SM8z0337aa0ga...\n• External state: NOTIFIED\n• Status: REQ SATISFIED",
            "color": "#bd5611",
            "border": "#f0883e"
        },
        {
            "id": "S_final",
            "x": 3.5, "y": 1.5,
            "title": "S_final: Verified Goal State",
            "desc": "• All expiring badges alerted\n• Goal preconditions verified\n• Top-level finish invoked\n• Status: TASK COMPLETE (100%)",
            "color": "#196c2e",
            "border": "#2ea043"
        }
    ]

    state_coords = {s["id"]: (s["x"], s["y"]) for s in states}

    # Draw State Boxes
    for s in states:
        box = mpatches.FancyBboxPatch(
            (s["x"] - 1.5, s["y"] - 1.0), 3.0, 2.0,
            boxstyle="round,pad=0.15,rounding_size=0.25",
            facecolor="#161b22",
            edgecolor=s["border"],
            linewidth=2.2,
            zorder=3
        )
        ax.add_patch(box)
        # Title badge
        ax.text(
            s["x"], s["y"] + 0.65, s["title"],
            fontsize=10, fontweight="bold", color="white",
            ha="center", va="center", zorder=4
        )
        # Separator line
        ax.plot([s["x"] - 1.3, s["x"] + 1.3], [s["y"] + 0.35, s["y"] + 0.35], color=s["border"], lw=1.0, zorder=4)
        # Description
        ax.text(
            s["x"], s["y"] - 0.3, s["desc"],
            fontsize=8.5, color="#c9d1d9",
            ha="center", va="center", zorder=4, linespacing=1.35
        )

    # Transitions (Action Edges: S_t -> S_{t+1})
    transitions = [
        ("S0", "S1", "api_search('spreadsheet ss_badges')", "#58a6ff", 0.0, "LOOKUP STAGE"),
        ("S1", "S_err", "api_fetch(GET ...?includeGridData)", "#f85149", 0.15, "FORMAT ERROR"),
        ("S_err", "S2", "api_fetch(GET .../ss_badges) [Fixed]", "#e3b341", -0.15, "CAUSAL RECOVERY"),
        ("S1", "S2", "api_fetch(GET .../ss_badges)", "#58a6ff", 0.0, "DIRECT READ"),
        ("S2", "S3", "api_fetch(GET .../values/Badge Tracker)", "#3fb950", 0.0, "DATAFLOW GROUNDING"),
        ("S3", "S4", "api_search('send text message sms')", "#58a6ff", 0.0, "NOTIFY LOOKUP"),
        ("S4", "S5", "api_fetch(POST .../Messages.json)", "#f0883e", 0.0, "STATE MUTATION (WRITE)"),
        ("S5", "S_final", "execute_decision -> finish()", "#2ea043", 0.0, "VERIFY & TERMINATE"),
    ]

    for src, tgt, label, col, rad, stage in transitions:
        x1, y1 = state_coords[src]
        x2, y2 = state_coords[tgt]

        # Arrow connection
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color=col,
                lw=2.5,
                shrinkA=38,
                shrinkB=38,
                connectionstyle=f"arc3,rad={rad}",
                mutation_scale=22
            ),
            zorder=2
        )

        # Label position
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        if rad > 0:
            mx -= 0.6
        elif rad < 0:
            mx += 0.6
        elif y1 == y2:
            my += 0.35
        elif x1 == x2:
            mx += 0.55

        ax.text(
            mx, my, f"{label}\n[{stage}]",
            fontsize=8, fontweight="bold", color=col,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#0d1117", edgecolor=col, alpha=0.95),
            zorder=5
        )

    # Macro Stages Header (from graph-grounded-automationbench motif)
    stage_boxes = [
        (3.5, 9.8, "1. LOOKUP / DISCOVERY", "#58a6ff"),
        (9.5, 9.8, "2. GROUNDING / DATAFLOW", "#3fb950"),
        (13.5, 9.8, "3. NOTIFICATION LOOKUP", "#58a6ff"),
        (9.5, 0.1, "4. STATE MUTATION", "#f0883e"),
        (3.5, 0.1, "5. VERIFICATION & FINISH", "#2ea043"),
    ]
    for sx, sy, stext, scol in stage_boxes:
        ax.text(
            sx, sy, stext,
            fontsize=10.5, fontweight="bold", color=scol,
            ha="center", va="center",
            bbox=dict(boxstyle="square,pad=0.35", facecolor="#161b22", edgecolor=scol, lw=1.5),
            zorder=5
        )

    ax.set_title(
        "World Model as State Transition System (S_t  —[Action / API Call]→  S_t+1)\n"
        "Grounded in Real Trajectory: operations.contractor_badge_expiration",
        fontsize=14, fontweight="bold", color="white", pad=32
    )

    ax.set_xlim(-0.5, 16.0)
    ax.set_ylim(-0.8, 10.8)
    ax.axis("off")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"State transition plot saved to {out_path}")

if __name__ == "__main__":
    out1 = Path("docs/state_transition_world_model.png")
    out2 = Path("/Users/masoud/.gemini/antigravity-cli/brain/0310ea49-122f-4371-b14b-e4c012e57910/state_transition_world_model.png")
    plot_state_transitions(out1)
    plot_state_transitions(out2)
