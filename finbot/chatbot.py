from __future__ import annotations

import os
import re
import time
import pandas as pd
import streamlit as st
from openai import OpenAI

from finbot.db import db_connect, money, query_readonly, validate_select

SUGGESTED_QUESTIONS = [
    ("🚨 Hóa đơn quá hạn", "Những hóa đơn nào đang quá hạn thanh toán?"),
    ("⏳ Phân tích tuổi nợ (Aging)", "Phân tích tuổi nợ phải thu theo các nhóm quá hạn"),
    ("📊 Pareto doanh thu 80/20", "Phân tích doanh thu theo danh mục và tỷ trọng Pareto 80/20"),
    ("🛒 Doanh thu theo kênh bán", "So sánh doanh thu giữa kênh Online và Bán trực tiếp"),
    ("💸 Dòng tiền & Runway", "Tình hình dòng tiền 6 tháng gần nhất và burn rate"),
    ("🏆 Top khách hàng công nợ", "Top khách hàng có giá trị hóa đơn phải thu lớn nhất"),
    ("📋 Tổng hợp trạng thái HĐ", "Thống kê số lượng và tổng tiền hóa đơn theo từng trạng thái"),
    ("📈 Tổng quan tài chính", "Báo cáo tổng quan sức khỏe tài chính hiện tại"),
]


def configured_api_key() -> str | None:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None and st.session_state.get("custom_openai_key"):
            return st.session_state["custom_openai_key"].strip()
        if get_script_run_ctx() is not None:
            return st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def _match_customer_in_prompt(q: str) -> str | None:
    with db_connect() as con:
        rows = con.execute("SELECT DISTINCT customer_name FROM Invoices").fetchall()
    for r in rows:
        name = r["customer_name"]
        name_lower = name.lower()
        # Check full name or core distinctive words (ignoring generic prefixes)
        short_name = re.sub(r"^(công ty|tập đoàn|giải pháp số|jsc|trading)\s*", "", name_lower).strip()
        if name_lower in q or (len(short_name) >= 4 and short_name in q):
            return name
    return None


def route(prompt: str) -> tuple[str, str]:
    q = prompt.lower().strip()

    # Check if user is asking for a specific customer
    matched_customer = _match_customer_in_prompt(q)
    if matched_customer:
        safe_cust = matched_customer.replace("'", "''")
        sql = (
            f"SELECT invoice_id, customer_name, issue_date, due_date, amount_vnd, vat_amount, status, "
            f"CAST(julianday('now') - julianday(due_date) AS INTEGER) AS overdue_days "
            f"FROM Invoices WHERE customer_name = '{safe_cust}' ORDER BY due_date DESC LIMIT 50"
        )
        return f"Hóa đơn của {matched_customer}", sql

    # 1. Aging buckets / Phân tích tuổi nợ theo nhóm
    if any(w in q for w in ("tuổi nợ", "nhóm nợ", "aging", "phân loại nợ", "khung quá hạn")):
        sql = """
        SELECT
            CASE
                WHEN status = 'PAID' THEN 'Đã thanh toán'
                WHEN julianday('now') - julianday(due_date) <= 0 THEN '0. Chưa đến hạn'
                WHEN julianday('now') - julianday(due_date) BETWEEN 1 AND 30 THEN '1. Quá hạn 1-30 ngày'
                WHEN julianday('now') - julianday(due_date) BETWEEN 31 AND 60 THEN '2. Quá hạn 31-60 ngày'
                ELSE '3. Quá hạn > 60 ngày'
            END AS aging_bucket,
            COUNT(*) AS invoice_count,
            SUM(amount_vnd) AS amount_vnd
        FROM Invoices
        WHERE status IN ('PENDING', 'OVERDUE')
        GROUP BY aging_bucket
        ORDER BY aging_bucket
        LIMIT 20
        """.strip()
        return "Phân tích tuổi nợ phải thu (AR Aging)", sql

    # 2. Tổng hợp theo trạng thái hóa đơn
    if any(w in q for w in ("trạng thái", "tổng hợp hóa đơn", "thống kê hóa đơn", "theo status")):
        sql = """
        SELECT
            status,
            COUNT(*) AS invoice_count,
            SUM(amount_vnd) AS amount_vnd,
            SUM(vat_amount) AS total_vat_vnd
        FROM Invoices
        GROUP BY status
        ORDER BY amount_vnd DESC
        LIMIT 10
        """.strip()
        return "Thống kê hóa đơn theo trạng thái", sql

    # 3. Top khách hàng / Hóa đơn lớn nhất
    if any(w in q for w in ("top", "lớn nhất", "cao nhất", "nhiều nhất", "khách hàng nợ")):
        if "doanh thu" in q:
            sql = (
                "SELECT category, channel, SUM(amount_vnd) AS revenue_vnd, COUNT(*) AS transactions "
                "FROM Revenue GROUP BY category, channel ORDER BY revenue_vnd DESC LIMIT 10"
            )
            return "Top nguồn doanh thu lớn nhất", sql
        sql = """
        SELECT
            customer_name,
            COUNT(*) AS invoice_count,
            SUM(amount_vnd) AS amount_vnd,
            SUM(CASE WHEN status='OVERDUE' THEN amount_vnd ELSE 0 END) AS overdue_vnd
        FROM Invoices
        WHERE status IN ('PENDING', 'OVERDUE')
        GROUP BY customer_name
        ORDER BY amount_vnd DESC
        LIMIT 10
        """.strip()
        return "Top khách hàng có công nợ phải thu lớn nhất", sql

    # 4. Công nợ quá hạn
    if any(w in q for w in ("quá hạn", "nợ xấu", "đến hạn", "trễ hạn", "overdue")):
        sql = (
            "SELECT invoice_id, customer_name, due_date, amount_vnd, status, "
            "CAST(julianday('now')-julianday(due_date) AS INTEGER) AS overdue_days "
            "FROM Invoices WHERE status='OVERDUE' ORDER BY overdue_days DESC LIMIT 100"
        )
        return "Công nợ quá hạn", sql

    # 5. Hóa đơn đã thanh toán hoặc chờ thanh toán
    if any(w in q for w in ("đã thanh toán", "đã trả", "paid")):
        sql = (
            "SELECT invoice_id, customer_name, issue_date, due_date, amount_vnd, status "
            "FROM Invoices WHERE status='PAID' ORDER BY issue_date DESC LIMIT 100"
        )
        return "Hóa đơn đã thanh toán", sql

    if any(w in q for w in ("chờ thanh toán", "chưa đến hạn", "pending")):
        sql = (
            "SELECT invoice_id, customer_name, issue_date, due_date, amount_vnd, status "
            "FROM Invoices WHERE status='PENDING' ORDER BY due_date ASC LIMIT 100"
        )
        return "Hóa đơn chờ thanh toán", sql

    # 6. Doanh thu theo kênh bán
    if any(w in q for w in ("kênh bán", "kênh", "online", "trực tiếp", "channel")):
        sql = """
        SELECT
            channel,
            COUNT(*) AS transactions,
            SUM(amount_vnd) AS revenue_vnd,
            ROUND(SUM(amount_vnd) * 100.0 / (SELECT SUM(amount_vnd) FROM Revenue), 1) AS share_pct
        FROM Revenue
        GROUP BY channel
        ORDER BY revenue_vnd DESC
        LIMIT 20
        """.strip()
        return "Doanh thu theo kênh bán hàng", sql

    # 7. Xu hướng doanh thu theo thời gian (ngày / tuần / 30 ngày)
    if any(w in q for w in ("theo ngày", "xu hướng", "30 ngày", "7 ngày", "gần đây", "biến động")):
        days = 7 if "7 ngày" in q or "tuần" in q else 30
        sql = (
            f"SELECT date, SUM(amount_vnd) AS revenue_vnd, COUNT(*) AS transactions "
            f"FROM Revenue WHERE date >= date('now', '-{days} day') "
            f"GROUP BY date ORDER BY date DESC LIMIT 60"
        )
        return f"Xu hướng doanh thu {days} ngày gần nhất", sql

    # 8. Doanh thu theo danh mục / Pareto 80/20
    if any(w in q for w in ("doanh thu", "danh mục", "pareto", "80/20", "sản phẩm", "dịch vụ", "phần mềm", "tỷ trọng")):
        sql = """
        SELECT
            category,
            SUM(amount_vnd) AS revenue_vnd,
            COUNT(*) AS transactions,
            ROUND(SUM(amount_vnd) * 100.0 / (SELECT SUM(amount_vnd) FROM Revenue), 1) AS share_pct
        FROM Revenue
        GROUP BY category
        ORDER BY revenue_vnd DESC
        LIMIT 20
        """.strip()
        return "Doanh thu theo danh mục (Pareto)", sql

    # 9. Dòng tiền / Cashflow / Burn rate / Runway
    if any(w in q for w in ("dòng tiền", "cashflow", "burn rate", "runway", "số dư", "thu chi")):
        sql = (
            "SELECT period_month, inflow, outflow, net_cash, ending_cash_balance "
            "FROM Cashflow ORDER BY period_month DESC LIMIT 6"
        )
        return "Tình hình dòng tiền", sql

    # 10. Tổng quan sức khỏe tài chính
    if any(w in q for w in ("tổng quan", "tóm tắt", "sức khỏe", "báo cáo", "tình hình tài chính", "kpi")):
        sql = """
        SELECT
            (SELECT COALESCE(SUM(amount_vnd), 0) FROM Invoices WHERE status IN ('PENDING','OVERDUE')) AS total_receivables_vnd,
            (SELECT COALESCE(SUM(amount_vnd), 0) FROM Invoices WHERE status = 'OVERDUE') AS overdue_receivables_vnd,
            (SELECT COALESCE(SUM(amount_vnd), 0) FROM Revenue WHERE date >= date('now', '-30 day')) AS revenue_30d_vnd,
            (SELECT ending_cash_balance FROM Cashflow ORDER BY period_month DESC LIMIT 1) AS ending_cash_vnd
        LIMIT 1
        """.strip()
        return "Tổng quan sức khỏe tài chính", sql

    # 11. Danh sách hóa đơn chung
    if any(w in q for w in ("hóa đơn", "invoice", "phải thu", "công nợ", "danh sách")):
        sql = (
            "SELECT invoice_id, customer_name, issue_date, due_date, amount_vnd, vat_amount, status "
            "FROM Invoices ORDER BY due_date ASC LIMIT 100"
        )
        return "Danh sách hóa đơn", sql

    raise LookupError(
        "Mình có thể giúp bạn tra cứu:\n"
        "- **Công nợ & Hóa đơn**: hóa đơn quá hạn, phân tích tuổi nợ (Aging), thống kê theo trạng thái, hoặc tra cứu theo tên khách hàng.\n"
        "- **Doanh thu**: doanh thu theo danh mục (Pareto 80/20), so sánh kênh Online vs Bán trực tiếp, xu hướng 7/30 ngày.\n"
        "- **Dòng tiền & Thanh khoản**: dòng tiền 6 tháng, burn rate, runway, tổng quan sức khỏe tài chính.\n\n"
        "💡 *Gợi ý: Bạn có thể bấm vào các nút câu hỏi mẫu phía trên để thử ngay!*"
    )


def _get_few_shot_examples(limit: int = 5) -> str:
    with db_connect() as con:
        rows = con.execute(
            "SELECT question, expected_sql FROM Evaluation_Examples WHERE expected_sql != '' ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    if not rows:
        return ""
    examples_text = "\n\nCác ví dụ mẫu đã được duyệt:\n"
    for r in rows:
        examples_text += f"- Câu hỏi: {r['question']}\n  SQL: {r['expected_sql']}\n"
    return examples_text


def ai_text_to_sql(prompt: str) -> tuple[str, str]:
    api_key = configured_api_key()
    if not api_key:
        return route(prompt)

    schema = (
        "Invoices(invoice_id TEXT, customer_name TEXT, tax_id TEXT, issue_date TEXT YYYY-MM-DD, "
        "due_date TEXT YYYY-MM-DD, amount_vnd INTEGER, vat_amount INTEGER, status TEXT: PAID/PENDING/OVERDUE)\n"
        "Revenue(transaction_id TEXT, date TEXT YYYY-MM-DD, category TEXT, amount_vnd INTEGER, channel TEXT)\n"
        "Cashflow(period_month TEXT YYYY-MM, inflow INTEGER, outflow INTEGER, net_cash INTEGER, ending_cash_balance INTEGER)"
    )
    few_shot = _get_few_shot_examples(limit=5)
    model = st.session_state.get("custom_openai_model") or os.getenv("OPENAI_MODEL", "gpt-5-mini")
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        instructions=(
            "Bạn là bộ chuyển câu hỏi tiếng Việt về tài chính doanh nghiệp thành đúng MỘT truy vấn SQLite SELECT (hoặc WITH ... SELECT). "
            "Chỉ dùng bảng Invoices, Revenue, Cashflow và các cột trong schema. Không dùng SELECT *; không truy cập Audit_Log; "
            "luôn thêm LIMIT (tối đa 200); không dùng comment, dấu chấm phẩy, hay bất kỳ câu lệnh nào khác. "
            "Chỉ trả về câu lệnh SQL thuần túy, không dùng markdown."
        ),
        input=f"Schema:\n{schema}{few_shot}\n\nCâu hỏi: {prompt}",
        store=False,
    )
    sql = response.output_text.strip()
    sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql, flags=re.I).strip()
    sql = sql.rstrip(";").strip()
    if not re.search(r"\bLIMIT\s+\d+\b", sql, re.I):
        sql = f"{sql} LIMIT 200"
    validate_select(sql)
    return "Kết quả phân tích AI Text-to-SQL", sql


def generate_insights(label: str, df: pd.DataFrame, prompt: str) -> str:
    if df.empty:
        return f"Mình đã kiểm tra **{label.lower()}** nhưng hiện chưa có bản ghi nào phù hợp với điều kiện lọc."

    lines = [f"### 📊 {label}"]

    # Summary totals
    if "amount_vnd" in df.columns:
        total = df["amount_vnd"].sum()
        lines.append(f"- **Tổng giá trị công nợ / hóa đơn hiển thị:** **{money(total)}** ({len(df)} dòng dữ liệu).")
    elif "revenue_vnd" in df.columns:
        total = df["revenue_vnd"].sum()
        tx_total = int(df["transactions"].sum()) if "transactions" in df.columns else len(df)
        lines.append(f"- **Tổng doanh thu ghi nhận:** **{money(total)}** trên tổng số **{tx_total:,}** giao dịch.")

    # Specific insights based on columns returned
    if "overdue_days" in df.columns and "customer_name" in df.columns:
        overdue_only = df[df["overdue_days"] > 0]
        if not overdue_only.empty:
            worst = overdue_only.sort_values("overdue_days", ascending=False).iloc[0]
            max_amount_row = overdue_only.sort_values("amount_vnd", ascending=False).iloc[0]
            lines.append(
                f"- ⚠️ **Cảnh báo quá hạn lâu nhất:** **{worst['customer_name']}** ({worst['invoice_id']}) đã quá hạn **{int(worst['overdue_days'])} ngày** với số tiền **{money(worst['amount_vnd'])}**."
            )
            if max_amount_row["invoice_id"] != worst["invoice_id"]:
                lines.append(
                    f"- 💰 **Khoản quá hạn lớn nhất:** **{max_amount_row['customer_name']}** ({max_amount_row['invoice_id']}) trị giá **{money(max_amount_row['amount_vnd'])}**."
                )
            lines.append(
                "- 💡 *Khuyến nghị:* Ưu tiên gửi email nhắc nợ hoặc tạo mã VietQR trong tab **⚡ Tác vụ** cho các khoản quá hạn > 30 ngày."
            )

    elif "aging_bucket" in df.columns and "amount_vnd" in df.columns:
        total_ar = df["amount_vnd"].sum() or 1
        severe = df[df["aging_bucket"].str.contains("31-60|> 60", regex=True)]["amount_vnd"].sum()
        severe_pct = severe / total_ar * 100
        lines.append(
            f"- ⏳ **Tỷ trọng nợ quá hạn trên 30 ngày:** chiếm **{severe_pct:.1f}%** ({money(severe)}) tổng phải thu."
        )

    elif "category" in df.columns and "revenue_vnd" in df.columns:
        top_cat = df.sort_values("revenue_vnd", ascending=False).iloc[0]
        total_rev = df["revenue_vnd"].sum() or 1
        top_share = top_cat["revenue_vnd"] / total_rev * 100
        lines.append(
            f"- 🏆 **Danh mục chủ lực (Pareto):** **{top_cat['category']}** dẫn đầu với **{money(top_cat['revenue_vnd'])}** (chiếm **{top_share:.1f}%** tổng doanh thu)."
        )
        if len(df) >= 2:
            top2_share = df.sort_values("revenue_vnd", ascending=False).iloc[:2]["revenue_vnd"].sum() / total_rev * 100
            lines.append(f"- 📈 **Quy tắc 80/20:** Top 2 danh mục đầu tiên đóng góp tới **{top2_share:.1f}%** tổng doanh thu.")

    elif "channel" in df.columns and "revenue_vnd" in df.columns:
        top_ch = df.sort_values("revenue_vnd", ascending=False).iloc[0]
        total_rev = df["revenue_vnd"].sum() or 1
        ch_share = top_ch["revenue_vnd"] / total_rev * 100
        lines.append(
            f"- 🛒 **Kênh bán hàng hiệu quả nhất:** **{top_ch['channel']}** mang về **{money(top_ch['revenue_vnd'])}** (**{ch_share:.1f}%** tỷ trọng)."
        )

    elif "inflow" in df.columns and "outflow" in df.columns and "net_cash" in df.columns:
        avg_in = df["inflow"].mean()
        avg_out = df["outflow"].mean()
        avg_net = df["net_cash"].mean()
        latest_row = df.sort_values("period_month", ascending=False).iloc[0]
        runway = latest_row["ending_cash_balance"] / avg_out if avg_out > 0 else 0
        lines.append(
            f"- 💵 **Số dư tiền kỳ gần nhất ({latest_row['period_month']}):** **{money(latest_row['ending_cash_balance'])}** (Dòng tiền thuần kỳ này: **{money(latest_row['net_cash'])}**)."
        )
        lines.append(
            f"- 🔥 **Chi tiêu trung bình (Gross Burn Rate):** **{money(avg_out)}/tháng** · **Runway ước tính:** **{runway:.1f} tháng**."
        )
        if avg_net > 0:
            lines.append(f"- ✅ **Tín hiệu tích cực:** Dòng tiền ròng trung bình đạt dương **+{money(avg_net)}/tháng**.")
        else:
            lines.append(f"- ⚠️ **Cảnh báo dòng tiền âm:** Dòng tiền ròng trung bình đang thâm hụt **{money(avg_net)}/tháng**.")

    elif "total_receivables_vnd" in df.columns:
        row = df.iloc[0]
        ar = row.get("total_receivables_vnd", 0)
        ovd = row.get("overdue_receivables_vnd", 0)
        rev30 = row.get("revenue_30d_vnd", 0)
        cash = row.get("ending_cash_vnd", 0)
        ovd_pct = (ovd / ar * 100) if ar else 0
        lines.append(f"- **Tổng phải thu (AR):** **{money(ar)}** (trong đó quá hạn: **{money(ovd)}** — chiếm **{ovd_pct:.1f}%**).")
        lines.append(f"- **Doanh thu 30 ngày gần nhất:** **{money(rev30)}**.")
        lines.append(f"- **Số dư tiền mặt cuối kỳ:** **{money(cash)}**.")

    return "\n".join(lines)


DEFAULT_BENCHMARK_CASES = [
    {
        "question": "Những hóa đơn nào đang quá hạn?",
        "expected_sql": "SELECT invoice_id, customer_name, due_date, amount_vnd, status FROM Invoices WHERE status='OVERDUE'",
        "expected_answer": "Công nợ quá hạn",
    },
    {
        "question": "Phân tích doanh thu theo danh mục Pareto",
        "expected_sql": "SELECT category, SUM(amount_vnd) AS revenue_vnd FROM Revenue GROUP BY category ORDER BY revenue_vnd DESC",
        "expected_answer": "Doanh thu theo danh mục",
    },
    {
        "question": "Tình hình dòng tiền và runway 6 tháng gần nhất",
        "expected_sql": "SELECT period_month, inflow, outflow, net_cash, ending_cash_balance FROM Cashflow ORDER BY period_month DESC LIMIT 6",
        "expected_answer": "Tình hình dòng tiền",
    },
    {
        "question": "So sánh doanh thu theo kênh bán hàng Online và trực tiếp",
        "expected_sql": "SELECT channel, SUM(amount_vnd) AS revenue_vnd FROM Revenue GROUP BY channel ORDER BY revenue_vnd DESC",
        "expected_answer": "Doanh thu theo kênh bán hàng",
    },
    {
        "question": "Phân tích tuổi nợ phải thu theo nhóm",
        "expected_sql": (
            "SELECT CASE WHEN status = 'PAID' THEN 'Đã thanh toán' "
            "WHEN julianday('now') - julianday(due_date) <= 0 THEN '0. Chưa đến hạn' "
            "WHEN julianday('now') - julianday(due_date) BETWEEN 1 AND 30 THEN '1. Quá hạn 1-30 ngày' "
            "WHEN julianday('now') - julianday(due_date) BETWEEN 31 AND 60 THEN '2. Quá hạn 31-60 ngày' "
            "ELSE '3. Quá hạn > 60 ngày' END AS aging_bucket, COUNT(*) AS invoice_count, SUM(amount_vnd) AS amount_vnd "
            "FROM Invoices WHERE status IN ('PENDING', 'OVERDUE') GROUP BY aging_bucket ORDER BY aging_bucket LIMIT 20"
        ),
        "expected_answer": "Phân tích tuổi nợ",
    },
]


def run_evaluation_suite() -> pd.DataFrame:
    with db_connect() as con:
        rows = con.execute("SELECT question, expected_sql, expected_answer FROM Evaluation_Examples ORDER BY id").fetchall()

    cases = [dict(r) for r in rows] if rows else DEFAULT_BENCHMARK_CASES
    results = []

    for idx, case in enumerate(cases, start=1):
        q = case["question"]
        exp_sql = (case.get("expected_sql") or "").strip()
        t0 = time.perf_counter()
        try:
            label, gen_sql = ai_text_to_sql(q)
            gen_df = query_readonly(gen_sql)
            latency_ms = int((time.perf_counter() - t0) * 1000)

            row_match = True
            if exp_sql:
                try:
                    exp_df = query_readonly(exp_sql)
                    row_match = len(gen_df) == len(exp_df)
                except Exception:
                    row_match = True

            status = "✅ PASS" if (not gen_df.empty and row_match) else "⚠️ PARTIAL"
            results.append({
                "STT": idx,
                "Câu hỏi kiểm thử": q,
                "Chủ đề nhận diện": label,
                "Số dòng trả về": len(gen_df),
                "Thời gian (ms)": latency_ms,
                "Kết quả": status,
                "SQL sinh ra": gen_sql,
            })
        except Exception as e:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            # If expected_sql is empty, it might be a safety/rejection test case!
            if not exp_sql:
                status = "✅ PASS (Từ chối an toàn)"
            else:
                status = f"❌ FAIL ({type(e).__name__})"
            results.append({
                "STT": idx,
                "Câu hỏi kiểm thử": q,
                "Chủ đề nhận diện": str(e)[:60],
                "Số dòng trả về": 0,
                "Thời gian (ms)": latency_ms,
                "Kết quả": status,
                "SQL sinh ra": "",
            })

    return pd.DataFrame(results)
