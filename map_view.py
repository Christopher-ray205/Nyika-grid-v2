"""
Nyika-Grid Predict AI — Regional Map View
------------------------------------------------
Addresses a limitation explicitly stated in MODEL_CARD.md: "no real map
view since no GPS-tagged multi-site data was available." Each region in
data_utils.REGIONS already carries real approximate lat/lon coordinates
(Lusaka, Chipata) — this renders them on an actual map, color-coded by
current risk status, instead of the comparison-card view used elsewhere.

Uses Plotly's scatter_mapbox with the open, tokenless "open-street-map"
style — no API key required, unlike Mapbox's own styles.
"""

import plotly.graph_objects as go
from typing import List, Dict


def build_region_map(region_points: List[Dict]) -> go.Figure:
    """
    region_points: list of {
        "label": str, "lat": float, "lon": float,
        "predicted_load_mw": float, "net_load_mw": float,
        "at_risk": bool
    }
    """
    lats = [p["lat"] for p in region_points]
    lons = [p["lon"] for p in region_points]
    labels = [p["label"] for p in region_points]
    colors = ["#A84432" if p["at_risk"] else "#6B8558" for p in region_points]
    hover_text = [
        f"{p['label']}<br>Net load: {p['net_load_mw']:.1f} MW<br>"
        f"{'⚠ Elevated risk' if p['at_risk'] else '✓ Stable'}"
        for p in region_points
    ]

    fig = go.Figure(go.Scattermap(
        lat=lats,
        lon=lons,
        mode="markers+text",
        marker=go.scattermap.Marker(size=22, color=colors),
        text=labels,
        textposition="top center",
        hovertext=hover_text,
        hoverinfo="text",
    ))

    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    fig.update_layout(
        map=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=5.2,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=420,
        showlegend=False,
    )
    return fig
