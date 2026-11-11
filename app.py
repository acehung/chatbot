from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import date, datetime, timedelta
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from finbot.analytics import (
    get_aging_analysis,
    get_cashflow_history,
    get_financial_kpis,
    get_pareto_revenue,
    forecast_cashflow,
)
from finbot.chatbot import (
    SUGGESTED_QUESTIONS,
    ai_text_to_sql,
    configured_api_key,
    generate_insights,
    route,
    run_evaluation_suite,
)
from finbot.db import (
    DB_PATH,
    EVAL_PATH,
    ROOT,
    db_connect,
    init_db,
    log_event,
    money,
    query_readonly,
    refresh_evaluation_jsonl,
    save_feedback,
    import_invoice_csv,
    seed_demo_data,
    validate_select,
)
from finbot.ocr import extract_invoice_text, parse_invoice, parse_invoice_with_ai_vision

st.set_page_config(page_title="FinBot | Trợ lý tài chính AI", page_icon="💹", layout="wide")
init_db()

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("### ⚙️ Cấu hình & Công cụ")
    dark_mode = st.toggle("🌙 Giao diện tối", key="dark_mode")

    with st.expander("🔑 Cấu hình AI (OpenAI)", expanded=False):
        st.text_input(
            "OpenAI API Key (tùy chọn)",
            type="password",
            key="custom_openai_key",
            help="Để trống nếu muốn dùng biến môi trường OPENAI_API_KEY hoặc chế độ Offline thông minh.",
        )
        st.selectbox(
            "Mô hình AI",
            ["gpt-5-mini", "gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"],
            key="custom_openai_model",
        )

    ai_enabled = bool(configured_api_key())
    st.info(
        "🟢 **Chế độ:** AI Text-to-SQL & Vision"
        if ai_enabled
        else "🔵 **Chế độ:** Định tuyến thông minh (Offline)"
    )

    st.divider()
    st.markdown("### 🛠️ Quản lý phiên & Dữ liệu")
    if st.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

    if st.button("🔄 Nạp lại dữ liệu Demo chuẩn", use_container_width=True):
        with db_connect() as con:
            seed_demo_data(con, force=True)
        st.toast("Đã làm mới hóa đơn và doanh thu từ file CSV demo trong dự án.", icon="✅")
        st.rerun()

    with st.expander("📤 Thêm dữ liệu từ file CSV", expanded=False):
        uploaded_csv = st.file_uploader(
            "Chọn CSV hóa đơn",
            type=["csv"],
            help="CSV cần có các cột theo mẫu data/du_lieu_demo_hoa_don.csv.",
            key="invoice_csv_upload",
        )
        st.download_button(
            "Tải file CSV mẫu",
            (ROOT / "data" / "du_lieu_demo_hoa_don.csv").read_bytes(),
            file_name="mau_du_lieu_hoa_don.csv",
            mime="text/csv",
            use_container_width=True,
        )
        import_mode = st.radio(
            "Cách nhập dữ liệu",
            ["Gộp và cập nhật hóa đơn trùng số", "Thay toàn bộ hóa đơn và doanh thu"],
            help="Gộp giữ dữ liệu hiện có; số hóa đơn trùng sẽ được cập nhật. Thay toàn bộ sẽ xóa hai bảng hóa đơn và doanh thu trước khi nạp.",
            key="invoice_csv_mode",
        )
        if st.button("Nạp CSV vào ứng dụng", disabled=uploaded_csv is None, use_container_width=True):
            try:
                with db_connect() as con:
                    added, updated = import_invoice_csv(
                        con,
                        uploaded_csv.getvalue(),
                        replace_existing=import_mode.startswith("Thay toàn bộ"),
                    )
                st.toast(f"Đã nạp {added} hóa đơn mới và cập nhật {updated} hóa đơn.", icon="✅")
                st.rerun()
            except (ValueError, OSError, sqlite3.Error) as exc:
                st.error(f"Không thể nạp CSV: {exc}")

    with db_connect() as con:
        all_inv_df = pd.read_sql_query("SELECT * FROM Invoices ORDER BY due_date", con)
    st.download_button(
        "📥 Xuất sổ cái Hóa đơn (.csv)",
        all_inv_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"so_cai_hoa_don_{date.today().isoformat()}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        "🔒 **Bảo mật SQLite:** Chỉ cho phép truy vấn `SELECT` qua `set_authorizer`. "
        "Mọi truy vấn đều được ghi vào nhật ký kiểm toán."
    )

# --- Theme Styling ---
if dark_mode:
    page_bg, panel_bg, input_bg, text_color, muted, border = (
        "#0b1220",
        "#151f30",
        "#101a2b",
        "#e5eaf2",
        "#a7b3c5",
        "#2b394e",
    )
    hero_bg = "linear-gradient(115deg,#101b30,#07584d)"
else:
    page_bg, panel_bg, input_bg, text_color, muted, border = (
        "#f4f7fb",
        "#ffffff",
        "#ffffff",
        "#172033",
        "#64748b",
        "#e6ebf2",
    )
    hero_bg = "linear-gradient(115deg,#10233f,#137c70)"

st.markdown(
    f"""<style>
  .stApp {{background:{page_bg}; color:{text_color}}}
  .block-container {{padding-top:1.1rem; max-width:1440px}}
  [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{background:{page_bg}}}
  [data-testid="stSidebar"] {{background:{panel_bg}}}
  div[data-testid="stMetric"] {{background:{panel_bg};border:1px solid {border};padding:16px;border-radius:14px}}
  div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricValue"] {{color:{text_color}}}
  .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, label {{color:{text_color}}}
  [data-testid="stCaptionContainer"] {{color:{muted}}}
  [data-testid="stTabs"] button {{color:{text_color}}}
  [data-testid="stChatMessage"] {{background:{panel_bg};border:1px solid {border};border-radius:14px;padding:12px 16px}}
  [data-testid="stChatInput"] textarea, [data-testid="stTextInput"] input,
  [data-testid="stTextArea"] textarea, [data-testid="stNumberInput"] input,
  [data-testid="stSelectbox"] div[data-baseweb="select"] > div {{background:{input_bg};color:{text_color};border-color:{border}}}
  div[data-testid="stDataFrame"] {{border:1px solid {border};border-radius:10px}}
  .stButton button, .stDownloadButton button, [data-testid="stLinkButton"] a {{border-radius:10px}}
  .hero {{padding:22px 26px;border-radius:18px;background:{hero_bg};color:white;margin-bottom:16px}}
  .hero h1 {{color:white;margin:0;font-size:30px}} .hero p {{color:#d8e9e7;margin:6px 0 0}}
</style>""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>FinBot · Trợ lý Tài chính & Quản trị Thanh khoản AI</h1>'
    "<p>Truy vấn dữ liệu tự nhiên, phân tích tuổi nợ (Aging), Pareto 80/20, dòng tiền và tự động hóa tác vụ kế toán.</p></div>",
    unsafe_allow_html=True,
)

# --- Top Executive Summary Metrics ---
kpis_30d = get_financial_kpis(days=30)
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Tổng phải thu (AR)",
    money(kpis_30d["total_ar"]),
    f"{kpis_30d['pending_count'] + kpis_30d['overdue_count']} hóa đơn mở",
    delta_color="off",
)
c2.metric(
    "Công nợ quá hạn",
    money(kpis_30d["overdue_ar"]),
    f"{kpis_30d['overdue_ratio']:.1f}% tổng phải thu",
    delta_color="inverse",
)
c3.metric(
    "Doanh thu 30 ngày",
    money(kpis_30d["rev_period"]),
    f"Tỷ lệ thu nợ: {kpis_30d['collection_rate']:.1f}%",
)
c4.metric(
    "Số dư tiền cuối kỳ",
    money(kpis_30d["cash"]) if kpis_30d["latest_month"] != "N/A" else "Chưa có dữ liệu",
    f"Runway: {kpis_30d['runway']:.1f} tháng" if kpis_30d["latest_month"] != "N/A" else "CSV demo không có số dư tiền",
)

tab_chat, tab_fin, tab_docs, tab_actions, tab_audit, tab_feedback = st.tabs([
    "💬 Chat tài chính",
    "📊 Sức khỏe tài chính",
    "🧾 Chứng từ",
    "⚡ Tác vụ",
    "🛡️ Kiểm toán",
    "🗳️ Đánh giá AI",
])


def render_inline_chat_chart(df: pd.DataFrame) -> None:
    if df is None or df.empty or len(df) < 2:
        return
    cols = set(df.columns)
    if {"category", "revenue_vnd"}.issubset(cols):
        st.bar_chart(df.set_index("category")["revenue_vnd"], height=220)
    elif {"aging_bucket", "amount_vnd"}.issubset(cols):
        st.bar_chart(df.set_index("aging_bucket")["amount_vnd"], height=220)
    elif {"channel", "revenue_vnd"}.issubset(cols):
        st.bar_chart(df.set_index("channel")["revenue_vnd"], height=220)
    elif {"period_month", "inflow", "outflow"}.issubset(cols):
        st.bar_chart(df.sort_values("period_month").set_index("period_month")[["inflow", "outflow", "net_cash"]], height=230)
    elif {"date", "revenue_vnd"}.issubset(cols):
        st.line_chart(df.sort_values("date").set_index("date")["revenue_vnd"], height=220)


def process_user_prompt(user_prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    try:
        label, sql = ai_text_to_sql(user_prompt)
        result = query_readonly(sql)
        log_event(user_prompt, sql, "SUCCESS")
        answer = generate_insights(label, result, user_prompt)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "data": result,
            "prompt": user_prompt,
            "sql": sql,
            "feedback_id": uuid.uuid4().hex,
        })
    except LookupError as e:
        log_event(user_prompt, None, "SUCCESS")
        st.session_state.messages.append({
            "role": "assistant",
            "content": str(e),
            "prompt": user_prompt,
            "feedback_id": uuid.uuid4().hex,
        })
    except ValueError as e:
        log_event(user_prompt, None, "BLOCKED")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⛔ {e}",
            "prompt": user_prompt,
            "feedback_id": uuid.uuid4().hex,
        })
    except Exception as e:
        log_event(user_prompt, None, "ERROR")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Có lỗi khi truy vấn dữ liệu: {e}",
            "prompt": user_prompt,
            "feedback_id": uuid.uuid4().hex,
        })


# ==============================================================================
# TAB 1: CHAT TÀI CHÍNH
# ==============================================================================
with tab_chat:
    st.subheader("Hỏi đáp & Phân tích Tài chính Thông minh")
    st.caption(
        "🟢 Đang dùng **AI Text-to-SQL** (tự động học từ bộ ví dụ đã duyệt)."
        if ai_enabled
        else "🔵 Đang dùng **Bộ định tuyến thông minh cục bộ** (hỗ trợ 11+ chủ đề phân tích tài chính). Nhập `OPENAI_API_KEY` ở thanh bên để bật AI Text-to-SQL."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Xin chào! Mình là **FinBot**. Mình có thể phân tích **công nợ quá hạn**, **tuổi nợ (AR Aging)**, "
                    "**doanh thu Pareto 80/20**, **kênh bán hàng**, **dòng tiền & runway**, hoặc tra cứu theo từng **khách hàng**.\n\n"
                    "👇 *Hãy chọn nhanh một câu hỏi mẫu bên dưới hoặc nhập câu hỏi của bạn:*"
                ),
            }
        ]

    # Quick Suggestion Buttons
    st.markdown("**💡 Câu hỏi gợi ý nhanh:**")
    q_cols = st.columns(4)
    triggered_question = None
    for idx, (btn_label, sample_q) in enumerate(SUGGESTED_QUESTIONS):
        if q_cols[idx % 4].button(btn_label, key=f"suggest_{idx}", use_container_width=True, help=sample_q):
            triggered_question = sample_q

    st.divider()

    # Display Chat History
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "data" in msg and isinstance(msg["data"], pd.DataFrame) and not msg["data"].empty:
                render_inline_chat_chart(msg["data"])
                st.dataframe(msg["data"], width="stretch", hide_index=True)

            if msg.get("sql"):
                with st.expander("🔍 Xem truy vấn SQL & Tải dữ liệu CSV"):
                    st.code(msg["sql"], language="sql")
                    if "data" in msg and isinstance(msg["data"], pd.DataFrame) and not msg["data"].empty:
                        csv_bytes = msg["data"].to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            "📥 Tải kết quả (.csv)",
                            csv_bytes,
                            file_name=f"finbot_query_{idx}.csv",
                            mime="text/csv",
                            key=f"dl_csv_{msg.get('feedback_id', idx)}",
                        )

            if msg["role"] == "assistant" and msg.get("feedback_id"):
                vote = st.feedback("thumbs", key=f"feedback_{msg['feedback_id']}")
                if vote is not None:
                    if msg.get("feedback_vote") != vote:
                        save_feedback(msg, vote)
                        msg["feedback_vote"] = vote
                    st.caption("Đã ghi nhận đánh giá. Người quản lý sẽ duyệt trước khi thêm vào bộ đánh giá.")

    prompt = st.chat_input("Ví dụ: Phân tích tuổi nợ phải thu, hoặc Hóa đơn của Công ty An Phát...")
    active_prompt = triggered_question or prompt
    if active_prompt:
        process_user_prompt(active_prompt)
        st.rerun()


# ==============================================================================
# TAB 2: SỨC KHỎE TÀI CHÍNH
# ==============================================================================
with tab_fin:
    filter_col1, filter_col2 = st.columns([3, 1.5], vertical_alignment="center")
    with filter_col1:
        st.subheader("Bảng điều khiển Sức khỏe Tài chính & Thanh khoản")
    with filter_col2:
        period_choice = st.selectbox(
            "Khoảng thời gian phân tích doanh thu",
            options=[("7 ngày gần nhất", 7), ("30 ngày gần nhất", 30), ("60 ngày gần nhất", 60), ("Toàn bộ thời gian", None)],
            index=1,
            format_func=lambda x: x[0],
        )
    selected_days = period_choice[1]
    fin_kpis = get_financial_kpis(days=selected_days)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("DSO ước tính", f"{fin_kpis['dso']:.1f} ngày", help="Days Sales Outstanding: Số ngày thu hồi nợ trung bình")
    k2.metric("Quick Ratio (proxy)", f"{fin_kpis['quick_ratio']:.2f}x" if fin_kpis["latest_month"] != "N/A" else "Chưa có dữ liệu", help="(Tiền mặt + Phải thu) / Chi phí dòng tiền ra trung bình tháng")
    k3.metric("Runway ước tính", f"{fin_kpis['runway']:.1f} tháng" if fin_kpis["latest_month"] != "N/A" else "Chưa có dữ liệu", help="Số tháng duy trì hoạt động với số dư tiền hiện tại")
    k4.metric("Tỷ lệ nợ quá hạn", f"{fin_kpis['overdue_ratio']:.1f}%", help="Nợ quá hạn / Tổng phải thu")
    k5.metric("Tỷ lệ thu hồi HĐ", f"{fin_kpis['collection_rate']:.1f}%", help="Hóa đơn đã thanh toán / Tổng giá trị hóa đơn phát hành")
    k6.metric("Dòng tiền ròng kỳ gần nhất", money(fin_kpis["latest_net_cash"]) if fin_kpis["latest_month"] != "N/A" else "Chưa có dữ liệu", f"Kỳ {fin_kpis['latest_month']}")

    st.caption(
        "Lưu ý: Quick Ratio là chỉ số proxy minh họa dựa trên Tiền + Phải thu so với dòng tiền chi ra bình quân tháng."
    )

    sub_aging, sub_pareto, sub_cashflow = st.tabs([
        "⏳ Phân tích Tuổi nợ (AR Aging)",
        "📊 Doanh thu & Pareto 80/20",
        "💸 Xu hướng Dòng tiền (Cashflow)",
    ])

    with sub_aging:
        buckets_df, detail_ar_df = get_aging_analysis()
        ag_left, ag_right = st.columns([1.15, 1.35])
        with ag_left:
            st.markdown("**Phân bổ công nợ theo nhóm tuổi nợ (VND)**")
            if not buckets_df.empty:
                st.bar_chart(buckets_df.set_index("aging_bucket")["total_amount_vnd"])
                st.dataframe(
                    buckets_df.rename(columns={
                        "aging_bucket": "Nhóm tuổi nợ",
                        "invoice_count": "Số HĐ",
                        "total_amount_vnd": "Tổng tiền (VND)",
                        "share_pct": "Tỷ trọng (%)",
                    }),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("Hiện không có công nợ phải thu.")
        with ag_right:
            st.markdown("**Chi tiết các khoản phải thu & quá hạn**")
            status_filter = st.multiselect(
                "Lọc trạng thái",
                options=["OVERDUE", "PENDING"],
                default=["OVERDUE", "PENDING"],
                key="ar_status_filter",
            )
            filtered_ar = detail_ar_df[detail_ar_df["status"].isin(status_filter)] if not detail_ar_df.empty else detail_ar_df
            st.dataframe(filtered_ar, width="stretch", hide_index=True)

    with sub_pareto:
        cat_df, channel_df, trend_df = get_pareto_revenue(days=selected_days)
        p_left, p_right = st.columns(2)
        with p_left:
            st.markdown(f"**Phân tích Pareto 80/20 theo danh mục ({period_choice[0].lower()})**")
            if not cat_df.empty:
                st.bar_chart(cat_df.set_index("category")["revenue_vnd"])
                st.dataframe(
                    cat_df.rename(columns={
                        "category": "Danh mục",
                        "revenue_vnd": "Doanh thu (VND)",
                        "transactions": "Số GD",
                        "share_pct": "Tỷ trọng (%)",
                        "cumulative_pct": "Lũy kế (%)",
                        "pareto_group": "Phân nhóm Pareto",
                    }),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("Chưa có dữ liệu doanh thu trong kỳ này.")
        with p_right:
            st.markdown(f"**Cơ cấu theo Kênh bán & Xu hướng theo ngày ({period_choice[0].lower()})**")
            if not channel_df.empty:
                st.dataframe(
                    channel_df.rename(columns={
                        "channel": "Kênh bán",
                        "revenue_vnd": "Doanh thu (VND)",
                        "transactions": "Số GD",
                        "share_pct": "Tỷ trọng (%)",
                    }),
                    width="stretch",
                    hide_index=True,
                )
            if not trend_df.empty:
                st.markdown("**Biến động doanh thu theo ngày**")
                st.line_chart(trend_df.set_index("date")["revenue_vnd"])

    with sub_cashflow:
        cf_df = get_cashflow_history()
        cf_left, cf_right = st.columns([1.3, 1])
        with cf_left:
            st.markdown("**Biểu đồ Dòng tiền vào (Inflow) · Dòng tiền ra (Outflow) · Dòng tiền ròng (Net Cash)**")
            if not cf_df.empty:
                st.bar_chart(cf_df.set_index("period_month")[["inflow", "outflow", "net_cash"]])
                st.markdown("**Xu hướng Số dư tiền cuối kỳ (Ending Cash Balance)**")
                st.line_chart(cf_df.set_index("period_month")["ending_cash_balance"])
            else:
                st.info("File CSV hóa đơn không có dữ liệu chi phí hoặc số dư tiền; biểu đồ và dự báo dòng tiền sẽ trống cho đến khi có dữ liệu Cashflow thực tế.")
        with cf_right:
            st.markdown("**Bảng cân đối dòng tiền 6 tháng gần nhất**")
            st.dataframe(cf_df.sort_values("period_month", ascending=False), width="stretch", hide_index=True)
        forecast_df = forecast_cashflow(3)
        if not forecast_df.empty:
            st.markdown("**Dự báo số dư tiền 3 tháng (xu hướng tuyến tính từ tối đa 6 kỳ gần nhất)**")
            st.line_chart(forecast_df.set_index("period_month")["projected_balance"])
            st.caption("Đây là ước tính xu hướng, không thay thế kế hoạch ngân quỹ hoặc dự báo theo kịch bản.")


# ==============================================================================
# TAB 3: CHỨNG TỪ (OCR & ĐỐI SOÁT)
# ==============================================================================
with tab_docs:
    st.subheader("Bóc tách chứng từ & Đối soát số học tự động")
    uploaded = st.file_uploader("Tải hóa đơn JPG, PNG hoặc PDF", type=["jpg", "jpeg", "png", "pdf"])
    extracted = extract_invoice_text(uploaded) if uploaded and uploaded.name.lower().endswith(".pdf") else ""
    if uploaded:
        st.caption(f"Đã nhận {uploaded.name}.")
        if uploaded.name.lower().endswith(".pdf") and not extracted:
            st.info("PDF chưa có lớp văn bản hoặc OCR chưa khả dụng. Với PDF scan, cài PyMuPDF, pytesseract, Pillow và Tesseract OCR (gói ngôn ngữ vie+eng), rồi tải lại file.")
        if uploaded.name.lower().endswith(("jpg", "jpeg", "png")):
            img_col1, img_col2 = st.columns([1, 1.5])
            with img_col1:
                st.image(uploaded, width=380)
            with img_col2:
                if ai_enabled:
                    if st.button("🤖 Bóc tách ảnh hóa đơn bằng AI Vision", type="primary"):
                        try:
                            with st.spinner("AI Vision đang đọc hóa đơn..."):
                                model_name = st.session_state.get("custom_openai_model") or "gpt-4o-mini"
                                vision_data = parse_invoice_with_ai_vision(uploaded, configured_api_key(), model=model_name)
                                st.session_state.parsed_invoice = vision_data
                                if vision_data.get("raw_text"):
                                    st.session_state["ocr_text"] = vision_data["raw_text"]
                                st.success("Đã bóc tách thành công bằng AI Vision!")
                        except Exception as e:
                            st.error(f"Không thể gọi AI Vision: {e}")
                else:
                    st.info("💡 Nhập `OPENAI_API_KEY` ở thanh bên để bật tính năng đọc ảnh hóa đơn tự động bằng **AI Vision**, hoặc dán văn bản OCR vào ô bên dưới.")

    sample_col1, sample_col2 = st.columns([3, 1])
    with sample_col2:
        if st.button("📄 Dán mẫu hóa đơn thử nghiệm", use_container_width=True):
            st.session_state["ocr_text"] = (
                "HÓA ĐƠN GIÁ TRỊ GIA TĂNG\n"
                "Số hóa đơn: HD-2026-009\n"
                "Đơn vị mua: Công ty Cổ phần Công nghệ Ánh Dương\n"
                "Mã số thuế: 0316677889\n"
                "Ngày lập: 05/11/2026\n"
                "Cộng tiền hàng: 45.000.000\n"
                "Thuế GTGT: 4.500.000\n"
                "Tổng tiền thanh toán: 49.500.000"
            )
            st.session_state.parsed_invoice = parse_invoice(st.session_state["ocr_text"])
            st.rerun()

    if extracted and not st.session_state.get("ocr_text"):
        st.session_state["ocr_text"] = extracted

    ocr_text = st.text_area(
        "Văn bản OCR (dán kết quả nhận dạng hoặc nội dung hóa đơn)",
        height=140,
        key="ocr_text",
    )
    if st.button("🔍 Bóc tách trường dữ liệu từ văn bản", type="primary"):
        st.session_state.parsed_invoice = parse_invoice(ocr_text)

    parsed = st.session_state.get("parsed_invoice", {})
    if parsed:
        with st.form("invoice_review"):
            a, b, c = st.columns(3)
            inv_id = a.text_input("Số hóa đơn", parsed.get("invoice_id", ""))
            tax_id = b.text_input("Mã số thuế", parsed.get("tax_id", ""))
            customer = c.text_input("Đơn vị bán / đối tác", parsed.get("customer_name", ""))
            issue = a.text_input("Ngày lập (YYYY-MM-DD)", parsed.get("issue_date", date.today().isoformat()))
            subtotal = b.number_input("Tiền trước thuế (VND)", min_value=0, value=int(parsed.get("subtotal_vnd", 0)), step=100000)
            vat = c.number_input("Thuế GTGT (VND)", min_value=0, value=int(parsed.get("vat_amount", 0)), step=10000)
            amount = a.number_input("Tổng thanh toán (VND)", min_value=0, value=int(parsed.get("amount_vnd", 0)), step=100000)
            due = b.date_input("Ngày đến hạn", value=date.today() + timedelta(days=30))
            status = c.selectbox("Trạng thái", ["PENDING", "PAID", "OVERDUE"])
            matched = subtotal + vat == amount and amount > 0
            if matched:
                st.success(f"✓ Đối soát số học khớp: {money(subtotal)} + {money(vat)} = {money(amount)}.")
            else:
                st.warning("Chưa khớp hoặc thiếu số tiền (Tiền trước thuế + VAT phải bằng Tổng thanh toán). Cần kiểm tra trước khi lưu.")
            save = st.form_submit_button("💾 Ghi hóa đơn vào sổ cái", disabled=not matched or not inv_id or not customer)
            if save:
                with db_connect() as con:
                    con.execute(
                        """INSERT OR REPLACE INTO Invoices(
                            invoice_id,customer_name,tax_id,issue_date,due_date,amount_vnd,vat_amount,status,
                            customer_code,channel,category,subtotal_vnd,paid_amount_vnd,payment_date,outstanding_vnd,overdue_days
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (inv_id, customer, tax_id, issue, due.isoformat(), amount, vat, status,
                         None, None, None, subtotal, amount if status == "PAID" else 0, None,
                         0 if status == "PAID" else amount,
                         max(0, (date.today() - due).days) if status == "OVERDUE" else 0),
                    )
                st.success(f"Đã lưu hóa đơn **{inv_id}** ({customer}) vào cơ sở dữ liệu.")


# ==============================================================================
# TAB 4: TÁC VỤ HỖ TRỢ (VIETQR, EMAIL NHẮC NỢ & CẬP NHẬT THU NỢ)
# ==============================================================================
with tab_actions:
    st.subheader("Tác vụ Thu hồi Công nợ & Thanh toán")
    qr_col, mail_col = st.columns(2)

    with db_connect() as con:
        unpaid_invoices = con.execute(
            "SELECT invoice_id, customer_name, due_date, amount_vnd, status FROM Invoices WHERE status IN ('OVERDUE','PENDING') ORDER BY due_date"
        ).fetchall()
        overdue_invoices = [r for r in unpaid_invoices if r["status"] == "OVERDUE"]

    with qr_col:
        st.markdown("#### 🏧 Tạo mã thanh toán VietQR")
        inv_choices = {"-- Nhập thủ công --": None}
        for r in unpaid_invoices:
            inv_choices[f"{r['invoice_id']} · {r['customer_name']} ({money(r['amount_vnd'])})"] = r
        picked_qr_label = st.selectbox("Điền nhanh từ hóa đơn chưa thu", list(inv_choices.keys()))
        picked_qr_row = inv_choices[picked_qr_label]

        default_amount = int(picked_qr_row["amount_vnd"]) if picked_qr_row else 0
        default_info = f"Thanh toan {picked_qr_row['invoice_id']}" if picked_qr_row else "Thanh toan hoa don"

        bank = st.selectbox(
            "Ngân hàng thụ hưởng (Mã VietQR)",
            ["ICB (VietinBank)", "VCB (Vietcombank)", "BIDV", "TCB (Techcombank)", "MB (MBBank)", "ACB", "VPB (VPBank)"],
        )
        bank_code = bank.split()[0]
        account = st.text_input("Số tài khoản nhận tiền", value="102000999888")
        account_name = st.text_input("Tên chủ tài khoản (không dấu)", value="CONG TY FINBOT")
        qr_amount = st.number_input("Số tiền (VND)", min_value=0, value=default_amount, step=100000)
        qr_info = st.text_input("Nội dung chuyển khoản", value=default_info)

        if st.button("Tạo link VietQR", type="primary"):
            if bank_code and account:
                safe_info = re.sub(r"[^A-Za-z0-9 ]", "", qr_info)[:25]
                base = f"https://img.vietqr.io/image/{bank_code}-{account}-compact2.png"
                params = {"amount": int(qr_amount), "addInfo": safe_info}
                if account_name:
                    params["accountName"] = account_name
                link = base + "?" + urlencode(params)
                st.image(link, width=250)
                st.link_button("Mở / tải VietQR", link)
                st.caption("Ảnh QR được tải từ VietQR Quick Link; cần internet để hiển thị.")
            else:
                st.error("Nhập mã ngân hàng và số tài khoản để tạo QR.")

    with mail_col:
        st.markdown("#### ✉️ Soạn email nhắc nợ tự động")
        if overdue_invoices:
            labels = {
                f"{row['invoice_id']} · {row['customer_name']} · {money(row['amount_vnd'])}": row
                for row in overdue_invoices
            }
            selected = st.selectbox("Chọn hóa đơn quá hạn", list(labels))
            tone = st.radio(
                "Giọng văn email",
                ["Nhắc nhở lịch sự", "Đôn đốc khẩn cấp", "Đề xuất lịch thanh toán"],
                horizontal=True,
            )
            row = labels[selected]
            days = max(0, (date.today() - date.fromisoformat(row["due_date"])).days)

            if tone == "Nhắc nhở lịch sự":
                subject = f"[Nhắc thanh toán] Hóa đơn {row['invoice_id']} - {row['customer_name']}"
                body = (
                    f"Kính gửi Quý đối tác {row['customer_name']},\n\n"
                    f"Theo biên bản đối chiếu công nợ định kỳ, hóa đơn {row['invoice_id']} (đến hạn ngày {row['due_date']}) "
                    f"với giá trị {money(row['amount_vnd'])} hiện đã quá hạn {days} ngày.\n\n"
                    f"Kính mong Quý đối tác kiểm tra và sắp xếp lịch thanh toán trong tuần này. "
                    f"Nếu Quý đối tác đã thực hiện chuyển khoản, vui lòng gửi lại ủy nhiệm chi để bộ phận Kế toán cập nhật.\n\n"
                    f"Trân trọng,\nBộ phận Tài chính - Kế toán"
                )
            elif tone == "Đôn đốc khẩn cấp":
                subject = f"[KHẨN] Đề nghị tất toán công nợ quá hạn {days} ngày - HĐ {row['invoice_id']}"
                body = (
                    f"Kính gửi Ban Giám đốc & Bộ phận Kế toán {row['customer_name']},\n\n"
                    f"Chúng tôi xin thông báo khoản công nợ theo hóa đơn {row['invoice_id']} trị giá {money(row['amount_vnd'])} "
                    f"đã quá hạn thanh toán {days} ngày (hạn thanh toán: {row['due_date']}).\n\n"
                    f"Để không làm gián đoạn tiến độ cung cấp dịch vụ/đơn hàng tiếp theo, đề nghị Quý đối tác hoàn tất thanh toán "
                    f"hoặc phản hồi văn bản xác nhận ngày thanh toán chính xác trước 17h00 ngày mai.\n\n"
                    f"Trân trọng,\nBộ phận Quản trị Công nợ"
                )
            else:
                subject = f"[Đề xuất phương án thanh toán] Hóa đơn {row['invoice_id']} - {row['customer_name']}"
                body = (
                    f"Kính gửi Quý đối tác {row['customer_name']},\n\n"
                    f"Liên quan đến hóa đơn {row['invoice_id']} trị giá {money(row['amount_vnd'])} (quá hạn {days} ngày), "
                    f"chúng tôi hiểu rằng Quý đối tác có thể đang trong kỳ cân đối dòng tiền.\n\n"
                    f"Chúng tôi đề xuất phương án thanh toán thành 2 đợt (50% trong tuần này và 50% còn lại trong vòng 15 ngày kế tiếp). "
                    f"Rất mong nhận được phản hồi xác nhận từ Quý đối tác.\n\n"
                    f"Trân trọng,\nBộ phận Tài chính"
                )

            edited_subject = st.text_input("Tiêu đề", value=subject)
            edited_body = st.text_area("Bản nháp email", value=body, height=220)
            st.download_button(
                "📥 Tải email dạng .eml (bản nháp)",
                f"Subject: {edited_subject}\nContent-Type: text/plain; charset=utf-8\n\n{edited_body}",
                file_name=f"nhac_no_{row['invoice_id']}.eml",
                mime="message/rfc822",
            )
        else:
            st.info("Không có hóa đơn quá hạn trong dữ liệu.")

    st.divider()
    st.markdown("#### ✅ Ghi nhận thu hồi nợ (Cập nhật trạng thái Hóa đơn)")
    if unpaid_invoices:
        pay_col1, pay_col2 = st.columns([3, 1], vertical_alignment="bottom")
        with pay_col1:
            pay_map = {
                f"{r['invoice_id']} · {r['customer_name']} · {money(r['amount_vnd'])} ({r['status']})": r
                for r in unpaid_invoices
            }
            selected_pay_label = st.selectbox("Chọn hóa đơn đã nhận thanh toán", list(pay_map.keys()))
        with pay_col2:
            if st.button("✔️ Đánh dấu ĐÃ THANH TOÁN", use_container_width=True):
                target_inv = pay_map[selected_pay_label]
                with db_connect() as con:
                    con.execute("UPDATE Invoices SET status='PAID' WHERE invoice_id=?", (target_inv["invoice_id"],))
                st.toast(f"Đã cập nhật {target_inv['invoice_id']} thành PAID!", icon="✅")
                st.rerun()
    else:
        st.success("Tất cả hóa đơn hiện tại đều đã được thanh toán (PAID).")


# ==============================================================================
# TAB 5: NHẬT KÝ KIỂM TOÁN
# ==============================================================================
with tab_audit:
    st.subheader("Nhật ký kiểm toán (Audit Log)")
    status_opt = st.selectbox("Lọc theo trạng thái thực thi", ["Tất cả", "SUCCESS", "BLOCKED", "ERROR"])
    with db_connect() as con:
        if status_opt == "Tất cả":
            audit = pd.read_sql_query(
                "SELECT id, timestamp, user_role, original_prompt, generated_sql, execution_status FROM Audit_Log ORDER BY id DESC LIMIT 100",
                con,
            )
        else:
            audit = pd.read_sql_query(
                "SELECT id, timestamp, user_role, original_prompt, generated_sql, execution_status FROM Audit_Log WHERE execution_status=? ORDER BY id DESC LIMIT 100",
                con,
                params=(status_opt,),
            )
    st.dataframe(audit, width="stretch", hide_index=True)
    st.caption(
        "Ứng dụng chỉ cho chạy SELECT khi chatbot truy vấn dữ liệu. Mọi prompt và trạng thái xử lý được ghi cục bộ vào Audit_Log."
    )


# ==============================================================================
# TAB 6: ĐÁNH GIÁ AI & AUTO-EVALUATION
# ==============================================================================
with tab_feedback:
    st.subheader("Duyệt phản hồi & Chạy kiểm thử tự động (Auto-Eval)")
    st.caption(
        "Đánh giá 👍/👎 trong tab Chat tạo phản hồi chờ duyệt. Sau khi duyệt, ví dụ sẽ được đưa vào bộ Evaluation và làm mẫu Few-shot cho AI."
    )
    with db_connect() as con:
        feedback_rows = con.execute("SELECT * FROM Chat_Feedback ORDER BY created_at DESC").fetchall()
        eval_rows = con.execute("SELECT * FROM Evaluation_Examples ORDER BY id DESC").fetchall()

    pending = [row for row in feedback_rows if row["review_status"] == "PENDING"]
    approved = [row for row in feedback_rows if row["review_status"] == "APPROVED"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Chờ duyệt", len(pending))
    m2.metric("Đã duyệt", len(approved))
    m3.metric("Ví dụ trong bộ Eval", len(eval_rows))

    eval_col1, eval_col2 = st.columns([1.15, 1.25])

    with eval_col1:
        st.markdown("#### 1. Duyệt phản hồi từ người dùng")
        if pending:
            labels = {
                f"{'👍' if row['vote'] else '👎'} · {row['created_at'][:16]} · {row['prompt'][:72]}": row
                for row in pending
            }
            selected_label = st.selectbox("Chọn phản hồi chờ duyệt", list(labels))
            selected_feedback = labels[selected_label]
            with st.form(f"review_{selected_feedback['feedback_id']}"):
                reviewed_question = st.text_area(
                    "Câu hỏi đã rà soát / ẩn danh hóa", value=selected_feedback["prompt"], height=80
                )
                expected_sql = st.text_area(
                    "SQL mong đợi (để trống nếu đây là ví dụ từ chối/an toàn)",
                    value=selected_feedback["generated_sql"] or "",
                    height=110,
                )
                expected_answer = st.text_area(
                    "Câu trả lời mong đợi đã được duyệt",
                    value=selected_feedback["assistant_response"],
                    height=110,
                )
                pii_confirmed = st.checkbox("Tôi đã kiểm tra và loại thông tin cá nhân/tài chính không cần thiết")
                approve = st.form_submit_button("Duyệt và thêm vào bộ đánh giá", type="primary")
                reject = st.form_submit_button("Đánh dấu không dùng")
            if approve:
                if not reviewed_question.strip() or not expected_answer.strip():
                    st.error("Cần có câu hỏi và câu trả lời mong đợi.")
                elif not pii_confirmed:
                    st.error("Hãy xác nhận đã rà soát thông tin nhạy cảm trước khi lưu.")
                else:
                    try:
                        clean_sql = validate_select(expected_sql) if expected_sql.strip() else ""
                        with db_connect() as con:
                            con.execute(
                                "INSERT OR REPLACE INTO Evaluation_Examples(feedback_id,question,expected_sql,expected_answer,created_at) VALUES(?,?,?,?,?)",
                                (
                                    selected_feedback["feedback_id"],
                                    reviewed_question.strip(),
                                    clean_sql,
                                    expected_answer.strip(),
                                    datetime.now().isoformat(timespec="seconds"),
                                ),
                            )
                            con.execute(
                                "UPDATE Chat_Feedback SET review_status='APPROVED',reviewed_at=? WHERE feedback_id=?",
                                (datetime.now().isoformat(timespec="seconds"), selected_feedback["feedback_id"]),
                            )
                        refresh_evaluation_jsonl()
                        st.success(f"Đã duyệt. Bộ đánh giá được lưu tại {EVAL_PATH.relative_to(ROOT)}.")
                        st.rerun()
                    except ValueError as e:
                        st.error(f"SQL mong đợi không hợp lệ: {e}")
            if reject:
                with db_connect() as con:
                    con.execute(
                        "UPDATE Chat_Feedback SET review_status='REJECTED',reviewed_at=? WHERE feedback_id=?",
                        (datetime.now().isoformat(timespec="seconds"), selected_feedback["feedback_id"]),
                    )
                st.rerun()
        else:
            st.info("Chưa có phản hồi chờ duyệt. Hãy bấm 👍/👎 dưới câu trả lời trong tab Chat tài chính.")

    with eval_col2:
        st.markdown("#### 2. Chạy kiểm thử tự động (Auto-Eval Runner)")
        st.caption(
            "Kiểm tra tự động toàn bộ các câu hỏi trong bộ Evaluation (hoặc 5 câu hỏi chuẩn mặc định nếu bộ Eval đang trống)."
        )
        if st.button("▶️ Chạy kiểm thử bộ Đánh giá ngay", type="primary"):
            with st.spinner("Đang thực thi bộ kiểm thử..."):
                st.session_state["eval_run_df"] = run_evaluation_suite()

        if "eval_run_df" in st.session_state:
            run_df = st.session_state["eval_run_df"]
            passed = sum(run_df["Kết quả"].str.startswith("✅"))
            st.success(f"Kết quả kiểm thử: Đạt **{passed}/{len(run_df)}** câu hỏi ({passed/len(run_df)*100:.0f}%).")
            st.dataframe(run_df, width="stretch", hide_index=True)

    if eval_rows:
        st.divider()
        st.markdown("**Danh sách các ví dụ chuẩn đã duyệt**")
        st.dataframe(pd.DataFrame([dict(row) for row in eval_rows]), width="stretch", hide_index=True)

    with db_connect() as con:
        examples = con.execute("SELECT question,expected_sql,expected_answer FROM Evaluation_Examples ORDER BY id").fetchall()
    jsonl_download = "\n".join(
        json.dumps(
            {
                "input": {"question": row["question"]},
                "expected_output": {"sql": row["expected_sql"], "answer": row["expected_answer"]},
            },
            ensure_ascii=False,
        )
        for row in examples
    )
    st.download_button(
        "📥 Tải bộ đánh giá JSONL",
        jsonl_download + ("\n" if jsonl_download else ""),
        file_name="evaluation_examples.jsonl",
        mime="application/jsonl",
        disabled=not examples,
    )
