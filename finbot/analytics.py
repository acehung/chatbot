from __future__ import annotations

import pandas as pd
from finbot.db import db_connect


def get_financial_kpis(days: int | None = 30) -> dict:
    with db_connect() as con:
        total_invoiced = con.execute("SELECT COALESCE(SUM(amount_vnd), 0) FROM Invoices").fetchone()[0]
        paid_ar = con.execute("SELECT COALESCE(SUM(amount_vnd), 0) FROM Invoices WHERE status='PAID'").fetchone()[0]
        total_ar = con.execute(
            "SELECT COALESCE(SUM(amount_vnd), 0) FROM Invoices WHERE status IN ('PENDING','OVERDUE')"
        ).fetchone()[0]
        overdue_ar = con.execute(
            "SELECT COALESCE(SUM(amount_vnd), 0) FROM Invoices WHERE status='OVERDUE'"
        ).fetchone()[0]
        overdue_count = con.execute("SELECT COUNT(*) FROM Invoices WHERE status='OVERDUE'").fetchone()[0]
        pending_count = con.execute("SELECT COUNT(*) FROM Invoices WHERE status='PENDING'").fetchone()[0]

        latest_cf = con.execute("SELECT * FROM Cashflow ORDER BY period_month DESC LIMIT 1").fetchone()
        avg_out = con.execute("SELECT COALESCE(AVG(outflow), 1) FROM Cashflow").fetchone()[0]
        avg_in = con.execute("SELECT COALESCE(AVG(inflow), 1) FROM Cashflow").fetchone()[0]

        if days:
            rev_period = con.execute(
                "SELECT COALESCE(SUM(amount_vnd), 0) FROM Revenue WHERE date >= date('now', ?)",
                (f"-{int(days)} day",),
            ).fetchone()[0]
            rev_days = max(days, 1)
        else:
            rev_period = con.execute("SELECT COALESCE(SUM(amount_vnd), 0) FROM Revenue").fetchone()[0]
            span = con.execute(
                "SELECT CAST(julianday(MAX(date)) - julianday(MIN(date)) + 1 AS INTEGER) FROM Revenue"
            ).fetchone()[0]
            rev_days = max(span or 30, 1)

    cash = latest_cf["ending_cash_balance"] if latest_cf else 0
    latest_net_cash = latest_cf["net_cash"] if latest_cf else 0
    latest_month = latest_cf["period_month"] if latest_cf else "N/A"

    daily_rev = rev_period / rev_days if rev_period else 0
    dso = (total_ar / daily_rev) if daily_rev > 0 else 0.0
    quick_ratio = (cash + total_ar) / avg_out if avg_out > 0 else 0.0
    runway = cash / avg_out if avg_out > 0 else 0.0
    overdue_ratio = (overdue_ar / total_ar * 100.0) if total_ar > 0 else 0.0
    collection_rate = (paid_ar / total_invoiced * 100.0) if total_invoiced > 0 else 0.0

    return {
        "total_invoiced": total_invoiced,
        "paid_ar": paid_ar,
        "total_ar": total_ar,
        "overdue_ar": overdue_ar,
        "overdue_count": overdue_count,
        "pending_count": pending_count,
        "cash": cash,
        "latest_net_cash": latest_net_cash,
        "latest_month": latest_month,
        "avg_out": avg_out,
        "avg_in": avg_in,
        "rev_period": rev_period,
        "rev_days": rev_days,
        "dso": dso,
        "quick_ratio": quick_ratio,
        "runway": runway,
        "overdue_ratio": overdue_ratio,
        "collection_rate": collection_rate,
    }


def get_aging_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (aging_buckets_df, detailed_ar_df)."""
    with db_connect() as con:
        detail_sql = """
        SELECT
            invoice_id,
            customer_name,
            issue_date,
            due_date,
            amount_vnd,
            vat_amount,
            status,
            CAST(julianday('now') - julianday(due_date) AS INTEGER) AS overdue_days,
            CASE
                WHEN status = 'PAID' THEN 'Đã thanh toán'
                WHEN julianday('now') - julianday(due_date) <= 0 THEN '0. Chưa đến hạn'
                WHEN julianday('now') - julianday(due_date) BETWEEN 1 AND 30 THEN '1. Quá hạn 1-30 ngày'
                WHEN julianday('now') - julianday(due_date) BETWEEN 31 AND 60 THEN '2. Quá hạn 31-60 ngày'
                ELSE '3. Quá hạn > 60 ngày'
            END AS aging_bucket
        FROM Invoices
        WHERE status IN ('PENDING', 'OVERDUE')
        ORDER BY overdue_days DESC
        """
        detail_df = pd.read_sql_query(detail_sql, con)

    if detail_df.empty:
        buckets_df = pd.DataFrame(columns=["aging_bucket", "invoice_count", "total_amount_vnd", "share_pct"])
    else:
        total = detail_df["amount_vnd"].sum() or 1
        buckets_df = (
            detail_df.groupby("aging_bucket", as_index=False)
            .agg(
                invoice_count=("invoice_id", "count"),
                total_amount_vnd=("amount_vnd", "sum"),
            )
            .sort_values("aging_bucket")
        )
        buckets_df["share_pct"] = (buckets_df["total_amount_vnd"] / total * 100).round(1)

    return buckets_df, detail_df


def get_pareto_revenue(days: int | None = 30) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (pareto_category_df, channel_df, daily_trend_df)."""
    where_clause = f"WHERE date >= date('now', '-{int(days)} day')" if days else ""
    with db_connect() as con:
        cat_df = pd.read_sql_query(
            f"""
            SELECT category, SUM(amount_vnd) AS revenue_vnd, COUNT(*) AS transactions
            FROM Revenue
            {where_clause}
            GROUP BY category
            ORDER BY revenue_vnd DESC
            """,
            con,
        )
        channel_df = pd.read_sql_query(
            f"""
            SELECT channel, SUM(amount_vnd) AS revenue_vnd, COUNT(*) AS transactions
            FROM Revenue
            {where_clause}
            GROUP BY channel
            ORDER BY revenue_vnd DESC
            """,
            con,
        )
        trend_df = pd.read_sql_query(
            f"""
            SELECT date, SUM(amount_vnd) AS revenue_vnd, COUNT(*) AS transactions
            FROM Revenue
            {where_clause}
            GROUP BY date
            ORDER BY date ASC
            """,
            con,
        )

    if not cat_df.empty:
        total_rev = cat_df["revenue_vnd"].sum() or 1
        cat_df["share_pct"] = (cat_df["revenue_vnd"] / total_rev * 100).round(1)
        cat_df["cumulative_pct"] = cat_df["share_pct"].cumsum().round(1)
        cat_df["pareto_group"] = cat_df["cumulative_pct"].apply(
            lambda x: "Nhóm A (Chủ lực ≤80%)" if x <= 85.0 else "Nhóm B (Bổ trợ)"
        )

    if not channel_df.empty:
        total_ch = channel_df["revenue_vnd"].sum() or 1
        channel_df["share_pct"] = (channel_df["revenue_vnd"] / total_ch * 100).round(1)

    return cat_df, channel_df, trend_df


def get_cashflow_history() -> pd.DataFrame:
    with db_connect() as con:
        df = pd.read_sql_query(
            "SELECT period_month, inflow, outflow, net_cash, ending_cash_balance FROM Cashflow ORDER BY period_month ASC",
            con,
        )
    return df


def forecast_cashflow(months: int = 3) -> pd.DataFrame:
    """Simple linear projection from recent monthly closing cash balances."""
    history = get_cashflow_history().tail(6).reset_index(drop=True)
    if history.empty:
        return pd.DataFrame(columns=["period_month", "projected_balance"])

    latest_period = pd.Period(history.iloc[-1]["period_month"], freq="M")
    balances = history["ending_cash_balance"].astype(float)
    # Fit a straight line to recent balances; keep a flat forecast with one point.
    if len(balances) > 1:
        x = pd.Series(range(len(balances)), dtype=float)
        slope = float(((x - x.mean()) * (balances - balances.mean())).sum() / ((x - x.mean()) ** 2).sum())
    else:
        slope = 0.0
    return pd.DataFrame([
        {
            "period_month": str(latest_period + offset),
            "projected_balance": max(0, round(float(balances.iloc[-1] + slope * offset))),
        }
        for offset in range(1, months + 1)
    ])
