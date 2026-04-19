"""
Advisory report formatter.
Takes the raw JSON string from the LLM and returns a styled HTML report
with 4 sections: Valuation Summary, Market Context, Verdict, Disclaimer.
"""

import json

_VERDICT_STYLES = {
    "BUY":  {"color": "#16A34A", "bg": "#DCFCE7", "border": "#86EFAC", "arrow": "↑", "label": "BUY"},
    "HOLD": {"color": "#D97706", "bg": "#FEF9C3", "border": "#FDE68A", "arrow": "→", "label": "HOLD"},
    "SELL": {"color": "#DC2626", "bg": "#FEE2E2", "border": "#FCA5A5", "arrow": "↓", "label": "SELL"},
}

_RISK_COLORS = {
    "LOW":    "#16A34A",
    "MEDIUM": "#D97706",
    "HIGH":   "#DC2626",
}


def _parse(raw: str) -> dict:
    """Parse LLM JSON output, stripping any markdown fences."""
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        # parts[1] may start with 'json\n'
        text = parts[1][4:].strip() if parts[1].startswith("json") else parts[1].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "verdict": "HOLD",
            "reason": text,
            "risk": "MEDIUM",
            "summary": "Analysis complete. Please review the details above.",
            "disclaimer": "This is AI-generated analysis and not financial advice.",
        }


def format_report(agent_output_json: str) -> str:
    """
    Convert the LLM JSON advice string into a styled HTML advisory report.

    Sections:
      1. Valuation Summary
      2. Market Context & Analysis
      3. Verdict (color-coded: green=BUY, yellow=HOLD, red=SELL)
      4. Disclaimer
    """
    data = _parse(agent_output_json)

    verdict   = data.get("verdict", "HOLD").upper()
    reason    = data.get("reason", "")
    risk      = data.get("risk", "MEDIUM").upper()
    summary   = data.get("summary", "")
    market_ctx = data.get("market_context", "")
    disclaimer = data.get(
        "disclaimer", "This is AI-generated analysis and not financial advice."
    )

    style = _VERDICT_STYLES.get(verdict, _VERDICT_STYLES["HOLD"])
    risk_color = _RISK_COLORS.get(risk, _RISK_COLORS["MEDIUM"])

    market_section = ""
    if market_ctx:
        market_section = f"""
  <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;padding:1.5rem;margin-bottom:1rem;box-shadow:0 1px 3px rgba(0,0,0,0.05);">
    <div style="font-size:0.72rem;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.75rem;">COMPARABLE MARKET DATA (RAG)</div>
    <p style="color:#475569;font-size:0.88rem;line-height:1.7;margin:0;white-space:pre-line;">{market_ctx}</p>
  </div>"""

    return f"""
<div style="font-family:'Public Sans',sans-serif;max-width:780px;margin:0 auto;">

  <!-- 1. Valuation Summary -->
  <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;padding:1.5rem;margin-bottom:1rem;box-shadow:0 1px 3px rgba(0,0,0,0.05);">
    <div style="font-size:0.72rem;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.75rem;">VALUATION SUMMARY</div>
    <p style="color:#1E293B;font-size:0.95rem;line-height:1.6;margin:0;">{summary}</p>
  </div>

  <!-- 2. Market Context & Analysis -->
  <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;padding:1.5rem;margin-bottom:1rem;box-shadow:0 1px 3px rgba(0,0,0,0.05);">
    <div style="font-size:0.72rem;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.75rem;">MARKET CONTEXT &amp; ANALYSIS</div>
    <p style="color:#1E293B;font-size:0.95rem;line-height:1.6;margin:0;">{reason}</p>
  </div>

  {market_section}

  <!-- 3. Verdict -->
  <div style="background:{style['bg']};border:2px solid {style['border']};border-radius:12px;padding:1.5rem;margin-bottom:1rem;">
    <div style="font-size:0.72rem;font-weight:700;color:{style['color']};text-transform:uppercase;letter-spacing:1px;margin-bottom:0.6rem;">INVESTMENT VERDICT</div>
    <div style="display:flex;align-items:center;gap:1.2rem;flex-wrap:wrap;">
      <span style="font-size:2.8rem;font-weight:800;color:{style['color']};line-height:1;">
        {style['arrow']} {style['label']}
      </span>
      <span style="background:#FFFFFF;border:1.5px solid {style['color']};color:{risk_color};font-weight:700;font-size:0.8rem;padding:0.3rem 0.9rem;border-radius:999px;letter-spacing:0.5px;">
        RISK: {risk}
      </span>
    </div>
  </div>

  <!-- 4. Disclaimer -->
  <div style="background:#F8F9FB;border:1px solid #E2E8F0;border-radius:8px;padding:1rem;">
    <p style="color:#94A3B8;font-size:0.78rem;margin:0;font-style:italic;">&#9888;&#65039; {disclaimer}</p>
  </div>

</div>
"""
