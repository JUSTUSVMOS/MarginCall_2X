from typing import Dict, Any, List
import engine_portfolio as portfolio
import engine_market as market


class _RiskFallback:
    @staticmethod
    def get_global_risk_snapshot() -> Dict[str, Any]:
        return {"state": "risk_unavailable", "riskScore": "n/a"}


try:
    import engine_risk as risk
except ModuleNotFoundError:
    risk = _RiskFallback()

def derive_morning_action_items(*, alerts: List[Dict[str, Any]], events: List[Dict[str, Any]], risk_snapshot: Dict[str, Any]) -> List[str]:
    items: List[tuple[int, str]] = []
    for alert in alerts:
        if alert["alert_type"] == "stop_hit":
            items.append((0, f"{alert['symbol']} stop_hit - review or exit immediately"))
        elif alert["alert_type"] == "thesis_invalid":
            items.append((1, f"{alert['symbol']} thesis_invalid - reassess the original trade thesis"))
        elif alert["alert_type"] == "holding_expiry":
            items.append((2, f"{alert['symbol']} holding_expiry - time-box has been exceeded"))
        elif alert["alert_type"] == "monitor_degraded":
            items.append((3, f"{alert['symbol']} monitor_degraded - {alert['payload'].get('error', 'unknown error')}"))
    for event in events:
        items.append((4, f"{event['symbol']} {event['label']} today - avoid adding risk before the event"))
    items.sort(key=lambda row: row[0])
    return [item for _, item in items]

def build_morning_briefing() -> str:
    holdings = portfolio.get_current_portfolio_symbols()
    alerts = portfolio.get_open_trade_plan_alerts()
    events = market.get_market_calendar_events(symbols=holdings, days=1)
    risk_snapshot = risk.get_global_risk_snapshot()
    status_summary = portfolio.build_trade_plan_status_summary()
    action_items = derive_morning_action_items(alerts=alerts, events=events, risk_snapshot=risk_snapshot)
    lines = [
        "🗞️ 【Morning Briefing】",
        f"一句話：{'今天觀望' if not action_items else action_items[0]}",
        f"Risk: {risk_snapshot.get('state')} ({risk_snapshot.get('riskScore')})",
        f"Open Alerts: {status_summary['open_alert_count']} | Missing Plans: {status_summary['missing_plan_count']}",
        "Action Items:",
    ]
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(action_items, start=1))
    return "\n".join(lines)
