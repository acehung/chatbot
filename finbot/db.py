from __future__ import annotations

import json
from io import BytesIO
import re
import sqlite3
from datetime import date, datetime
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "finbot.db"
EVAL_PATH = ROOT / "data" / "evaluation_examples.jsonl"
DEMO_CSV_PATH = ROOT / "data" / "du_lieu_demo_hoa_don.csv"
DB_PATH.parent.mkdir(exist_ok=True)

BLOCKED_WORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|REPLACE|ATTACH|DETACH|VACUUM|PRAGMA|CREATE)\b",
    re.I,
)

ALLOWED_READ_TABLES = {"Invoices", "Revenue", "Cashflow"}
DEMO_COLUMNS = {
    "so_hoa_don", "ngay_phat_hanh", "ngay_den_han", "ma_khach_hang", "ten_khach_hang",
    "kenh_ban", "danh_muc", "doanh_thu_chua_vat", "thue_vat", "tong_tien",
    "da_thanh_toan", "ngay_thanh_toan", "trang_thai", "con_no", "so_ngay_qua_han",
}


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _read_invoice_csv(source) -> pd.DataFrame:
    try:
        frame = pd.read_csv(source, dtype={"so_hoa_don": "string", "ma_khach_hang": "string"})
    except Exception as exc:
        raise ValueError(f"Không đọc được CSV: {exc}") from exc
    missing = DEMO_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"CSV thiếu cột bắt buộc: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("CSV không có dòng dữ liệu.")

    numeric_columns = [
        "doanh_thu_chua_vat", "thue_vat", "tong_tien", "da_thanh_toan", "con_no", "so_ngay_qua_han",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any() or (frame[column] < 0).any():
            raise ValueError(f"Cột {column} phải chỉ chứa số nguyên không âm.")
        if (frame[column] % 1 != 0).any():
            raise ValueError(f"Cột {column} phải chứa số nguyên.")

    for column in ("ngay_phat_hanh", "ngay_den_han"):
        dates = pd.to_datetime(frame[column], errors="coerce", format="mixed")
        if dates.isna().any():
            raise ValueError(f"Cột {column} có ngày không hợp lệ; dùng định dạng YYYY-MM-DD.")
        frame[column] = dates.dt.strftime("%Y-%m-%d")
    payment_dates = frame["ngay_thanh_toan"].fillna("").astype(str).str.strip()
    nonempty_payment = payment_dates.ne("")
    parsed_payment = pd.to_datetime(payment_dates[nonempty_payment], errors="coerce", format="mixed")
    if parsed_payment.isna().any():
        raise ValueError("Cột ngay_thanh_toan có ngày không hợp lệ; dùng YYYY-MM-DD hoặc để trống.")
    payment_dates.loc[nonempty_payment] = parsed_payment.dt.strftime("%Y-%m-%d")
    frame["ngay_thanh_toan"] = payment_dates

    for column in ("so_hoa_don", "ten_khach_hang", "ma_khach_hang", "kenh_ban", "danh_muc"):
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    if (frame["so_hoa_don"] == "").any() or (frame["ten_khach_hang"] == "").any():
        raise ValueError("Số hóa đơn và tên khách hàng không được để trống.")
    if frame["so_hoa_don"].duplicated().any():
        raise ValueError("CSV có số hóa đơn bị trùng lặp.")
    if (frame["doanh_thu_chua_vat"] + frame["thue_vat"] != frame["tong_tien"]).any():
        raise ValueError("Có dòng không khớp: doanh thu chưa VAT + thuế VAT phải bằng tổng tiền.")
    if (frame["da_thanh_toan"] + frame["con_no"] != frame["tong_tien"]).any():
        raise ValueError("Có dòng không khớp: đã thanh toán + còn nợ phải bằng tổng tiền.")
    return frame


def import_invoice_csv(con: sqlite3.Connection, source, replace_existing: bool = False) -> tuple[int, int]:
    """Import invoice CSV rows. Return (inserted, updated) counts."""
    frame = _read_invoice_csv(BytesIO(source) if isinstance(source, bytes) else source)
    ids = frame["so_hoa_don"].tolist()
    existing = set()
    if not replace_existing:
        current_ids = {row[0] for row in con.execute("SELECT invoice_id FROM Invoices").fetchall()}
        existing = current_ids.intersection(ids)
    else:
        con.execute("DELETE FROM Invoices")
        con.execute("DELETE FROM Revenue")

    invoices = []
    revenues = []
    today = date.today().isoformat()
    for row in frame.to_dict("records"):
        outstanding = int(row["con_no"])
        due_date = row["ngay_den_han"]
        status = "PAID" if outstanding == 0 else ("OVERDUE" if due_date < today else "PENDING")
        invoices.append((
            row["so_hoa_don"], row["ten_khach_hang"], None, row["ngay_phat_hanh"], due_date,
            int(row["tong_tien"]), int(row["thue_vat"]), status, row["ma_khach_hang"],
            row["kenh_ban"], row["danh_muc"], int(row["doanh_thu_chua_vat"]),
            int(row["da_thanh_toan"]), row["ngay_thanh_toan"] or None, outstanding,
            int(row["so_ngay_qua_han"]),
        ))
        revenues.append((
            row["so_hoa_don"], row["ngay_phat_hanh"], row["danh_muc"],
            int(row["doanh_thu_chua_vat"]), row["kenh_ban"],
        ))

    con.executemany(
        """INSERT INTO Invoices(invoice_id,customer_name,tax_id,issue_date,due_date,amount_vnd,vat_amount,status,
        customer_code,channel,category,subtotal_vnd,paid_amount_vnd,payment_date,outstanding_vnd,overdue_days)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(invoice_id) DO UPDATE SET
        customer_name=excluded.customer_name,tax_id=excluded.tax_id,issue_date=excluded.issue_date,
        due_date=excluded.due_date,amount_vnd=excluded.amount_vnd,vat_amount=excluded.vat_amount,status=excluded.status,
        customer_code=excluded.customer_code,channel=excluded.channel,category=excluded.category,
        subtotal_vnd=excluded.subtotal_vnd,paid_amount_vnd=excluded.paid_amount_vnd,
        payment_date=excluded.payment_date,outstanding_vnd=excluded.outstanding_vnd,overdue_days=excluded.overdue_days""",
        invoices,
    )
    con.executemany(
        """INSERT INTO Revenue(transaction_id,date,category,amount_vnd,channel) VALUES(?,?,?,?,?)
        ON CONFLICT(transaction_id) DO UPDATE SET date=excluded.date,category=excluded.category,
        amount_vnd=excluded.amount_vnd,channel=excluded.channel""",
        revenues,
    )
    return len(ids) - len(existing), len(existing)


def seed_demo_data(con: sqlite3.Connection, force: bool = False) -> None:
    if force:
        con.execute("DELETE FROM Cashflow")

    if force or con.execute("SELECT COUNT(*) FROM Invoices").fetchone()[0] == 0:
        if not DEMO_CSV_PATH.exists():
            raise FileNotFoundError(f"Không tìm thấy dữ liệu demo: {DEMO_CSV_PATH}")
        import_invoice_csv(con, DEMO_CSV_PATH, replace_existing=force)


def init_db() -> None:
    with db_connect() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS Invoices (
            invoice_id TEXT PRIMARY KEY, customer_name TEXT NOT NULL, tax_id TEXT,
            issue_date TEXT NOT NULL, due_date TEXT NOT NULL, amount_vnd INTEGER NOT NULL,
            vat_amount INTEGER NOT NULL, status TEXT NOT NULL CHECK(status IN ('PAID','PENDING','OVERDUE')),
            customer_code TEXT, channel TEXT, category TEXT, subtotal_vnd INTEGER,
            paid_amount_vnd INTEGER NOT NULL DEFAULT 0, payment_date TEXT,
            outstanding_vnd INTEGER NOT NULL DEFAULT 0, overdue_days INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS Revenue (
            transaction_id TEXT PRIMARY KEY, date TEXT NOT NULL, category TEXT NOT NULL,
            amount_vnd INTEGER NOT NULL, channel TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS Cashflow (
            period_month TEXT PRIMARY KEY, inflow INTEGER NOT NULL, outflow INTEGER NOT NULL,
            net_cash INTEGER NOT NULL, ending_cash_balance INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS Audit_Log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
            user_role TEXT NOT NULL, original_prompt TEXT NOT NULL, generated_sql TEXT,
            execution_status TEXT NOT NULL CHECK(execution_status IN ('SUCCESS','BLOCKED','ERROR'))
        );
        CREATE TABLE IF NOT EXISTS Chat_Feedback (
            feedback_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, prompt TEXT NOT NULL,
            generated_sql TEXT, assistant_response TEXT NOT NULL,
            vote INTEGER NOT NULL CHECK(vote IN (0,1)),
            review_status TEXT NOT NULL DEFAULT 'PENDING' CHECK(review_status IN ('PENDING','APPROVED','REJECTED')),
            reviewed_at TEXT
        );
        CREATE TABLE IF NOT EXISTS Evaluation_Examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT, feedback_id TEXT NOT NULL UNIQUE,
            question TEXT NOT NULL, expected_sql TEXT NOT NULL DEFAULT '',
            expected_answer TEXT NOT NULL, created_at TEXT NOT NULL
        );
        """)
        invoice_columns = {row[1] for row in con.execute("PRAGMA table_info(Invoices)")}
        optional_columns = {
            "customer_code": "TEXT", "channel": "TEXT", "category": "TEXT", "subtotal_vnd": "INTEGER",
            "paid_amount_vnd": "INTEGER NOT NULL DEFAULT 0", "payment_date": "TEXT",
            "outstanding_vnd": "INTEGER NOT NULL DEFAULT 0", "overdue_days": "INTEGER NOT NULL DEFAULT 0",
        }
        for name, declaration in optional_columns.items():
            if name not in invoice_columns:
                con.execute(f"ALTER TABLE Invoices ADD COLUMN {name} {declaration}")
        seed_demo_data(con, force=False)


def validate_select(sql: str) -> str:
    clean = re.sub(r"--[^\n]*|/\*.*?\*/", " ", sql, flags=re.S).strip()
    if ";" in clean or not re.match(r"^\s*(SELECT|WITH)\b", clean, re.I) or BLOCKED_WORDS.search(clean):
        raise ValueError("Chỉ cho phép một truy vấn SELECT đọc dữ liệu.")
    return clean


def query_readonly(sql: str) -> pd.DataFrame:
    clean = validate_select(sql)
    con = db_connect()
    denied_actions = {
        getattr(sqlite3, name)
        for name in dir(sqlite3)
        if name.startswith("SQLITE_")
        and name
        in {
            "SQLITE_INSERT",
            "SQLITE_UPDATE",
            "SQLITE_DELETE",
            "SQLITE_CREATE_INDEX",
            "SQLITE_CREATE_TABLE",
            "SQLITE_DROP_INDEX",
            "SQLITE_DROP_TABLE",
            "SQLITE_ALTER_TABLE",
            "SQLITE_ATTACH",
            "SQLITE_DETACH",
            "SQLITE_PRAGMA",
            "SQLITE_TRANSACTION",
            "SQLITE_SAVEPOINT",
        }
    }

    def authorizer(action, arg1, arg2, dbname, source):
        if action == sqlite3.SQLITE_READ and arg1 not in ALLOWED_READ_TABLES:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_DENY if action in denied_actions else sqlite3.SQLITE_OK

    con.set_authorizer(authorizer)
    try:
        return pd.read_sql_query(clean, con)
    finally:
        con.close()


def log_event(prompt: str, sql: str | None, status: str, role: str = "manager") -> None:
    with db_connect() as con:
        con.execute(
            "INSERT INTO Audit_Log(timestamp,user_role,original_prompt,generated_sql,execution_status) VALUES(?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), role, prompt[:1000], (sql or "")[:2000], status),
        )


def save_feedback(message: dict, vote: int) -> None:
    with db_connect() as con:
        con.execute(
            """INSERT INTO Chat_Feedback(feedback_id,created_at,prompt,generated_sql,assistant_response,vote)
            VALUES(?,?,?,?,?,?) ON CONFLICT(feedback_id) DO UPDATE SET
            vote=excluded.vote,
            review_status=CASE WHEN Chat_Feedback.review_status='APPROVED' THEN 'APPROVED' ELSE 'PENDING' END""",
            (
                message["feedback_id"],
                datetime.now().isoformat(timespec="seconds"),
                message.get("prompt", ""),
                message.get("sql", "") or "",
                message["content"],
                int(vote),
            ),
        )


def refresh_evaluation_jsonl() -> None:
    with db_connect() as con:
        rows = con.execute("SELECT question,expected_sql,expected_answer FROM Evaluation_Examples ORDER BY id").fetchall()
    lines = [
        json.dumps(
            {
                "input": {"question": row["question"]},
                "expected_output": {"sql": row["expected_sql"], "answer": row["expected_answer"]},
            },
            ensure_ascii=False,
        )
        for row in rows
    ]
    temporary = EVAL_PATH.with_suffix(".jsonl.tmp")
    temporary.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    temporary.replace(EVAL_PATH)


def money(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "0 ₫"
    return f"{float(value):,.0f} ₫".replace(",", ".")
