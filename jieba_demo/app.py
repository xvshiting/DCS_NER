import io
import os
import sqlite3
import tempfile
import zipfile
from collections import Counter
from functools import wraps
from hashlib import md5
from typing import Optional

import chardet
import jieba
from flask import (Flask, Response, g, jsonify, render_template,
                   request, send_file, session)
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "jieba-demo-secret-key-change-in-prod")

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "jieba_demo.db")

ADMIN_USERNAME = "admin"
ADMIN_DEFAULT_PW = "bigmoney@123"

# ─── Database ────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT    NOT NULL UNIQUE,
    pw_hash  TEXT    NOT NULL,
    is_admin INTEGER NOT NULL DEFAULT 0,
    created  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS dictionaries (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT    NOT NULL,
    owner_id   INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content    TEXT    NOT NULL DEFAULT '',
    is_public  INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_public_dict_name
    ON dictionaries(name) WHERE is_public=1;

CREATE UNIQUE INDEX IF NOT EXISTS uq_private_dict_name
    ON dictionaries(name, owner_id) WHERE is_public=0;
"""


def get_db():
    if "db" not in g:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    # Migration: add is_admin column for existing databases
    try:
        conn.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.close()


def init_admin():
    """Ensure the admin user exists with is_admin=1."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT id, is_admin FROM users WHERE username=?", (ADMIN_USERNAME,)
    ).fetchone()
    if not row:
        conn.execute(
            "INSERT INTO users (username, pw_hash, is_admin) VALUES (?, ?, 1)",
            (ADMIN_USERNAME, generate_password_hash(ADMIN_DEFAULT_PW)),
        )
        conn.commit()
    elif not row["is_admin"]:
        conn.execute("UPDATE users SET is_admin=1 WHERE username=?", (ADMIN_USERNAME,))
        conn.commit()
    conn.close()


# ─── Auth helpers ─────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "login required"}), 401
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "login required"}), 401
        user = current_user()
        if not user or not user.get("is_admin"):
            return jsonify({"error": "admin required"}), 403
        return f(*args, **kwargs)
    return decorated


def current_user():
    uid = session.get("user_id")
    if uid is None:
        return None
    db = get_db()
    row = db.execute(
        "SELECT id, username, is_admin FROM users WHERE id=?", (uid,)
    ).fetchone()
    return dict(row) if row else None


# ─── Jieba tokenizer cache ────────────────────────────────────────────────────

_tok_cache: dict = {}


def get_tokenizer(dict_content: Optional[str]) -> jieba.Tokenizer:
    key = md5((dict_content or "").encode()).hexdigest()
    if key not in _tok_cache:
        tok = jieba.Tokenizer()
        tok.initialize()
        if dict_content and dict_content.strip():
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", encoding="utf-8", delete=False
            ) as tmp:
                tmp.write(dict_content)
                tmp_path = tmp.name
            try:
                tok.load_userdict(tmp_path)
            finally:
                os.unlink(tmp_path)
        _tok_cache[key] = tok
    return _tok_cache[key]


# ─── Segmentation logic ───────────────────────────────────────────────────────

def segment_text(text: str, dict_content: Optional[str], method: str, separator: str) -> dict:
    tok = get_tokenizer(dict_content)
    all_words = []
    lines_out = []
    for line in text.splitlines():
        if method == "full":
            words = list(tok.cut(line, cut_all=True))
        elif method == "search":
            words = list(tok.cut_for_search(line))
        else:
            words = list(tok.cut(line, cut_all=False))
        words = [w for w in words if w.strip()]
        all_words.extend(words)
        lines_out.append(separator.join(words))
    freq = Counter(all_words)
    return {
        "words": all_words,
        "lines": lines_out,
        "joined": "\n".join(lines_out),
        "count": len(all_words),
        "unique_count": len(freq),
        "freq": freq.most_common(),
    }


def decode_file(raw: bytes) -> str:
    detected = chardet.detect(raw)
    enc = detected.get("encoding") or "utf-8"
    return raw.decode(enc, errors="replace")


def get_dict_content(dict_id: Optional[int], user_id: int) -> Optional[str]:
    if not dict_id:
        return None
    db = get_db()
    row = db.execute(
        "SELECT content, owner_id, is_public FROM dictionaries WHERE id=?",
        (dict_id,),
    ).fetchone()
    if not row:
        return None
    if row["is_public"] or row["owner_id"] == user_id:
        return row["content"]
    return None


# ─── Routes: pages ────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return render_template("index.html")


# ─── Routes: auth ─────────────────────────────────────────────────────────────

@app.get("/api/me")
def api_me():
    user = current_user()
    return jsonify({"user": user})


@app.post("/api/login")
def api_login():
    data = request.get_json(force=True)
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    db = get_db()
    row = db.execute(
        "SELECT id, pw_hash, is_admin FROM users WHERE username=?", (username,)
    ).fetchone()
    if not row or not check_password_hash(row["pw_hash"], password):
        return jsonify({"error": "用户名或密码错误"}), 401
    session["user_id"] = row["id"]
    return jsonify({"username": username, "is_admin": bool(row["is_admin"])})


@app.post("/api/logout")
def api_logout():
    session.clear()
    return jsonify({"ok": True})


# ─── Routes: admin – user management ─────────────────────────────────────────

@app.get("/api/admin/users")
@admin_required
def api_admin_users():
    db = get_db()
    rows = db.execute(
        "SELECT id, username, is_admin, created FROM users ORDER BY id"
    ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.post("/api/admin/users")
@admin_required
def api_admin_create_user():
    data = request.get_json(force=True)
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    is_admin = int(bool(data.get("is_admin", False)))
    if not username or not password:
        return jsonify({"error": "用户名和密码不能为空"}), 400
    if len(username) > 64:
        return jsonify({"error": "用户名最多64个字符"}), 400
    if len(password) < 6:
        return jsonify({"error": "密码至少6个字符"}), 400
    db = get_db()
    try:
        cur = db.execute(
            "INSERT INTO users (username, pw_hash, is_admin) VALUES (?, ?, ?)",
            (username, generate_password_hash(password), is_admin),
        )
        db.commit()
    except sqlite3.IntegrityError:
        return jsonify({"error": "用户名已存在"}), 409
    row = db.execute(
        "SELECT id, username, is_admin, created FROM users WHERE id=?", (cur.lastrowid,)
    ).fetchone()
    return jsonify(dict(row)), 201


@app.put("/api/admin/users/<int:user_id>/password")
@admin_required
def api_admin_change_password(user_id):
    data = request.get_json(force=True)
    password = data.get("password") or ""
    if len(password) < 6:
        return jsonify({"error": "密码至少6个字符"}), 400
    db = get_db()
    row = db.execute("SELECT id FROM users WHERE id=?", (user_id,)).fetchone()
    if not row:
        return jsonify({"error": "用户不存在"}), 404
    db.execute(
        "UPDATE users SET pw_hash=? WHERE id=?",
        (generate_password_hash(password), user_id),
    )
    db.commit()
    return jsonify({"ok": True})


@app.delete("/api/admin/users/<int:user_id>")
@admin_required
def api_admin_delete_user(user_id):
    me = current_user()
    if me["id"] == user_id:
        return jsonify({"error": "不能删除当前登录的管理员账户"}), 400
    db = get_db()
    row = db.execute("SELECT id FROM users WHERE id=?", (user_id,)).fetchone()
    if not row:
        return jsonify({"error": "用户不存在"}), 404
    db.execute("DELETE FROM users WHERE id=?", (user_id,))
    db.commit()
    return jsonify({"ok": True})


# ─── Routes: dictionaries ─────────────────────────────────────────────────────

@app.get("/api/dicts")
@login_required
def api_dicts_list():
    uid = session["user_id"]
    db = get_db()
    rows = db.execute(
        """
        SELECT d.id, d.name, d.owner_id, d.is_public, d.updated_at,
               u.username AS owner_name
        FROM dictionaries d
        JOIN users u ON u.id = d.owner_id
        WHERE d.owner_id = ? OR d.is_public = 1
        ORDER BY d.is_public ASC, d.name ASC
        """,
        (uid,),
    ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.post("/api/dicts")
@login_required
def api_dicts_create():
    uid = session["user_id"]
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    content = data.get("content") or ""
    if not name:
        return jsonify({"error": "词典名称不能为空"}), 400
    db = get_db()
    try:
        cur = db.execute(
            "INSERT INTO dictionaries (name, owner_id, content, is_public) VALUES (?, ?, ?, 0)",
            (name, uid, content),
        )
        db.commit()
    except sqlite3.IntegrityError:
        return jsonify({"error": "已存在同名私有词典"}), 409
    row = db.execute("SELECT * FROM dictionaries WHERE id=?", (cur.lastrowid,)).fetchone()
    return jsonify(dict(row)), 201


@app.get("/api/dicts/<int:dict_id>")
@login_required
def api_dicts_get(dict_id):
    uid = session["user_id"]
    db = get_db()
    row = db.execute(
        "SELECT * FROM dictionaries WHERE id=? AND (owner_id=? OR is_public=1)",
        (dict_id, uid),
    ).fetchone()
    if not row:
        return jsonify({"error": "词典不存在或无权访问"}), 404
    return jsonify(dict(row))


@app.put("/api/dicts/<int:dict_id>")
@login_required
def api_dicts_update(dict_id):
    uid = session["user_id"]
    db = get_db()
    row = db.execute(
        "SELECT * FROM dictionaries WHERE id=? AND owner_id=? AND is_public=0",
        (dict_id, uid),
    ).fetchone()
    if not row:
        return jsonify({"error": "词典不存在或无权修改"}), 404
    data = request.get_json(force=True)
    name = (data.get("name") or row["name"]).strip()
    content = data.get("content") if "content" in data else row["content"]
    try:
        db.execute(
            "UPDATE dictionaries SET name=?, content=?, updated_at=datetime('now') WHERE id=?",
            (name, content, dict_id),
        )
        db.commit()
    except sqlite3.IntegrityError:
        return jsonify({"error": "已存在同名私有词典"}), 409
    updated = db.execute("SELECT * FROM dictionaries WHERE id=?", (dict_id,)).fetchone()
    return jsonify(dict(updated))


@app.delete("/api/dicts/<int:dict_id>")
@login_required
def api_dicts_delete(dict_id):
    uid = session["user_id"]
    db = get_db()
    row = db.execute(
        "SELECT id, owner_id FROM dictionaries WHERE id=? AND owner_id=?",
        (dict_id, uid),
    ).fetchone()
    if not row:
        return jsonify({"error": "词典不存在或无权删除"}), 404
    db.execute("DELETE FROM dictionaries WHERE id=?", (dict_id,))
    db.commit()
    return jsonify({"ok": True})


@app.post("/api/dicts/<int:dict_id>/publish")
@login_required
def api_dicts_publish(dict_id):
    uid = session["user_id"]
    db = get_db()
    row = db.execute(
        "SELECT * FROM dictionaries WHERE id=? AND owner_id=? AND is_public=0",
        (dict_id, uid),
    ).fetchone()
    if not row:
        return jsonify({"error": "词典不存在或无权发布"}), 404

    name = row["name"]
    content = row["content"]

    existing = db.execute(
        "SELECT id FROM dictionaries WHERE name=? AND is_public=1", (name,)
    ).fetchone()
    if existing:
        db.execute(
            "UPDATE dictionaries SET content=?, owner_id=?, updated_at=datetime('now') WHERE id=?",
            (content, uid, existing["id"]),
        )
    else:
        db.execute(
            "INSERT INTO dictionaries (name, owner_id, content, is_public) VALUES (?, ?, ?, 1)",
            (name, uid, content),
        )
    db.commit()
    return jsonify({"ok": True})


@app.post("/api/dicts/<int:dict_id>/fork")
@login_required
def api_dicts_fork(dict_id):
    """Copy a public dictionary into the current user's private space."""
    uid = session["user_id"]
    db = get_db()
    row = db.execute(
        "SELECT * FROM dictionaries WHERE id=? AND is_public=1", (dict_id,)
    ).fetchone()
    if not row:
        return jsonify({"error": "公共词典不存在"}), 404

    name = row["name"]
    content = row["content"]

    existing = db.execute(
        "SELECT id FROM dictionaries WHERE name=? AND owner_id=? AND is_public=0",
        (name, uid),
    ).fetchone()
    if existing:
        db.execute(
            "UPDATE dictionaries SET content=?, updated_at=datetime('now') WHERE id=?",
            (content, existing["id"]),
        )
        db.commit()
        new_id = existing["id"]
    else:
        cur = db.execute(
            "INSERT INTO dictionaries (name, owner_id, content, is_public) VALUES (?, ?, ?, 0)",
            (name, uid, content),
        )
        db.commit()
        new_id = cur.lastrowid

    new_row = db.execute("SELECT * FROM dictionaries WHERE id=?", (new_id,)).fetchone()
    return jsonify(dict(new_row)), 201


@app.post("/api/dicts/<int:dict_id>/append")
@login_required
def api_dicts_append(dict_id):
    """Append new words to a private dictionary, skipping duplicates."""
    uid = session["user_id"]
    db = get_db()
    row = db.execute(
        "SELECT * FROM dictionaries WHERE id=? AND owner_id=? AND is_public=0",
        (dict_id, uid),
    ).fetchone()
    if not row:
        return jsonify({"error": "词典不存在或无权修改"}), 404

    data = request.get_json(force=True)
    words = [w.strip() for w in (data.get("words") or []) if str(w).strip()]
    if not words:
        return jsonify({"error": "无有效词条"}), 400

    content = row["content"] or ""
    # Build set of existing first-tokens (the actual word part of each line)
    existing = set()
    for line in content.splitlines():
        parts = line.strip().split()
        if parts:
            existing.add(parts[0])

    new_lines = []
    for w in words:
        if w not in existing:
            new_lines.append(w)
            existing.add(w)

    if new_lines:
        sep = "\n" if content.rstrip("\n") else ""
        new_content = content.rstrip("\n") + sep + "\n".join(new_lines)
        db.execute(
            "UPDATE dictionaries SET content=?, updated_at=datetime('now') WHERE id=?",
            (new_content, dict_id),
        )
        db.commit()

    return jsonify({"added": len(new_lines), "skipped": len(words) - len(new_lines)})


# ─── Routes: segmentation ─────────────────────────────────────────────────────

@app.post("/api/segment")
@login_required
def api_segment():
    uid = session["user_id"]

    if request.content_type and "multipart" in request.content_type:
        f = request.files.get("file")
        if not f:
            return jsonify({"error": "未上传文件"}), 400
        text = decode_file(f.read())
        dict_id = request.form.get("dict_id") or None
        method = request.form.get("method", "precise")
        separator = request.form.get("separator", " ")
    else:
        data = request.get_json(force=True)
        text = data.get("text", "")
        dict_id = data.get("dict_id") or None
        method = data.get("method", "precise")
        separator = data.get("separator", " ")

    if dict_id:
        try:
            dict_id = int(dict_id)
        except (ValueError, TypeError):
            dict_id = None

    dict_content = get_dict_content(dict_id, uid)
    result = segment_text(text, dict_content, method, separator)
    return jsonify(result)


@app.post("/api/segment/download")
@login_required
def api_segment_download():
    uid = session["user_id"]

    if request.content_type and "multipart" in request.content_type:
        f = request.files.get("file")
        if not f:
            return jsonify({"error": "未上传文件"}), 400
        text = decode_file(f.read())
        # Keep original filename as-is
        download_name = f.filename or "result.txt"
        dict_id = request.form.get("dict_id") or None
        method = request.form.get("method", "precise")
        separator = request.form.get("separator", " ")
    else:
        data = request.get_json(force=True)
        text = data.get("text", "")
        download_name = data.get("filename") or "result.txt"
        dict_id = data.get("dict_id") or None
        method = data.get("method", "precise")
        separator = data.get("separator", " ")

    if dict_id:
        try:
            dict_id = int(dict_id)
        except (ValueError, TypeError):
            dict_id = None

    dict_content = get_dict_content(dict_id, uid)
    result = segment_text(text, dict_content, method, separator)
    out = result["joined"].encode("utf-8")
    return Response(
        out,
        mimetype="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{download_name}"'},
    )


@app.post("/api/segment/batch")
@login_required
def api_segment_batch():
    uid = session["user_id"]
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "未上传文件"}), 400

    dict_id = request.form.get("dict_id") or None
    method = request.form.get("method", "precise")
    separator = request.form.get("separator", " ")

    if dict_id:
        try:
            dict_id = int(dict_id)
        except (ValueError, TypeError):
            dict_id = None

    dict_content = get_dict_content(dict_id, uid)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            text = decode_file(f.read())
            result = segment_text(text, dict_content, method, separator)
            # Keep original filename for each file in the zip
            out_name = f.filename or "file.txt"
            zf.writestr(out_name, result["joined"].encode("utf-8"))

    zip_buf.seek(0)
    return send_file(
        zip_buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name="batch_segmented.zip",
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    init_admin()
    app.run(host="0.0.0.0", port=5001, debug=True)
