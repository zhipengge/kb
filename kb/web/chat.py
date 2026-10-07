"""网页端的 AI 会话。

**为什么不直接复用 ``/api/v1`` 的会话接口**（三个理由，缺一不可）：

1. 那些端点走 ``Authorization: Bearer <key>``，而网页端根本没有 Key 体系
   （浏览器里放 API Key 等于把它公开）；
2. ``CSRFProtect`` 全局启用且 ``api_bp`` 未豁免——浏览器 fetch ``/api/v1`` 会被
   直接拦掉，要同时带 API Key 和 ``X-CSRFToken`` 两套凭证；
3. ``EventSource`` 无法自定义请求头，带鉴权的 SSE 从浏览器里根本走不通。

所以这一组路由直接调 ``services/rag.py``，与其余 web 视图一致：
同源 + CSRF 保护，不需要第二套鉴权。

**响应格式**：JSON 与 SSE（``data: {...}\\n\\n``）。前端用
``fetch`` + ``ReadableStream`` 手工解析，而不是 ``EventSource``——
后者只能发 GET，带不了 CSRF 头，也发不了请求体。
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from flask import (
    Response,
    current_app,
    jsonify,
    render_template,
    request,
    stream_with_context,
)

from ..extensions import db
from ..models import ROLE_USER, Conversation, Message
from . import web_bp

log = logging.getLogger(__name__)

# 附件类型白名单。**必须限制**：上传目录在磁盘上，接受任意扩展名等于
# 让这个接口变成「往服务器写任意文件」的入口。
IMAGE_TYPES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
PDF_TYPES = {"application/pdf": ".pdf"}

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_PDF_BYTES = 64 * 1024 * 1024
# 附件正文注入上下文的上限。整篇论文塞进去会挤掉检索到的资料。
MAX_PDF_CHARS = 20000
MAX_PDF_PAGES = 40


# --------------------------------------------------------------------------
# 辅助
# --------------------------------------------------------------------------


def _conversation_or_404(conversation_id: str) -> Conversation | None:
    return db.session.get(Conversation, conversation_id)


def _history_of(conversation: Conversation) -> list[dict]:
    """取会话历史，供模型使用。

    历史里带的是 ``content`` 内容块（可能含图片），不是纯文本副本——
    多模态会话的「上一轮那张图」就靠它。**引用标记要去掉**：把上一轮的
    ``[1]`` 原样传给模型，它会以为那是本轮的编号，导致引用错乱。
    """
    history: list[dict] = []
    for message in conversation.messages:
        if message.role not in {"user", "assistant"}:
            continue
        content = message.content
        if not content and message.text:
            content = message.text
        if isinstance(content, str):
            # 助手消息里剔掉引用标记，避免模型把旧编号当成新编号
            content = _strip_markers(content) if message.role == "assistant" else content
        history.append({"role": message.role, "content": content})
    return history


def _strip_markers(text: str) -> str:
    import re

    return re.sub(r"\[\d+\]", "", text or "")


def _scope_filters(conversation: Conversation) -> dict | None:
    ids = list(conversation.scope_paper_ids or [])
    return {"paper_ids": ids} if ids else None


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _message_dict(message: Message) -> dict:
    return {
        "id": message.id,
        "role": message.role,
        "text": message.text or "",
        "content": message.content or [],
        "citations": message.citations or [],
        "created_at": message.created_at.isoformat() if message.created_at else None,
    }


def _conversation_dict(conversation: Conversation) -> dict:
    return {
        "id": conversation.id,
        "title": conversation.title,
        "pinned": bool(conversation.pinned),
        "message_count": len(conversation.messages or []),
        "updated_at": conversation.updated_at.isoformat() if conversation.updated_at else None,
    }


def _upload_dir() -> Path:
    cfg = current_app.extensions["kb_boot_config"]
    path = Path(cfg.uploads_dir) / "chat"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _extract_pdf_text(path: Path) -> str:
    """把 PDF 抽成文本。

    当前端点的 ``pdf_native`` 能力是关的（实测它接受 document 块但不解析内容），
    所以附件必须先本地抽取再作为文本注入。抽取结果会随消息存进
    ``Message.content``，刷新或追问时不必重抽。
    """
    text = ""
    try:
        import pymupdf4llm

        pages = pymupdf4llm.to_markdown(
            str(path), page_chunks=True, show_progress=False
        )
        text = "\n\n".join(chunk.get("text", "") for chunk in pages[:MAX_PDF_PAGES])
    except Exception:
        log.warning("pymupdf4llm 抽取失败，退回纯文本", exc_info=True)
        try:
            import pymupdf

            with pymupdf.open(path) as doc:
                text = "\n\n".join(
                    doc[index].get_text() for index in range(min(len(doc), MAX_PDF_PAGES))
                )
        except Exception as exc:
            return f"（附件解析失败：{type(exc).__name__}）"
    return text[:MAX_PDF_CHARS]


def _blocks_from_attachments(attachments: list[dict]) -> tuple[list[dict], list[str]]:
    """把附件转成内容块。返回 (块列表, 人类可读的说明)。"""
    blocks: list[dict] = []
    notes: list[str] = []
    for item in attachments:
        kind = item.get("kind")
        path = item.get("path")
        if not path:
            continue
        file_path = Path(path)
        if not file_path.is_file():
            log.warning("附件文件不存在：%s", file_path)
            continue

        if kind == "image":
            import base64

            data = base64.standard_b64encode(file_path.read_bytes()).decode()
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": item.get("media_type") or "image/png",
                        "data": data,
                    },
                }
            )
            notes.append(f"图片 {item.get('name') or ''}".strip())
        elif kind == "pdf":
            text = _extract_pdf_text(file_path)
            blocks.append(
                {
                    "type": "text",
                    "text": f"［附件：{item.get('name') or 'PDF'}］\n\n{text}",
                }
            )
            notes.append(f"PDF 附件 {item.get('name') or ''}".strip())
    return blocks, notes


# --------------------------------------------------------------------------
# 页面
# --------------------------------------------------------------------------


@web_bp.get("/chat")
def chat_page():
    """会话页。

    **热启动**：服务端直接把最近一次会话连同消息一起渲染出来，
    首屏就能读，不用等前端再拉一次接口。左侧会话列表同理。
    """
    conversations = (
        db.session.query(Conversation)
        .order_by(Conversation.pinned.desc(), Conversation.updated_at.desc())
        .limit(100)
        .all()
    )
    # ?c=<id> 指定要打开哪个会话；没有指定（或指定了不存在的）就回到最近一个。
    # 「最近一个」是热启动的默认行为：打开 /chat 应该直接续上刚才的对话。
    current = conversations[0] if conversations else None
    wanted = (request.args.get("c") or "").strip()
    if wanted:
        picked = _conversation_or_404(wanted)
        if picked is not None:
            current = picked
    return render_template(
        "chat.html",
        conversations=conversations,
        current=current,
        messages=list(current.messages) if current else [],
    )


@web_bp.get("/chat/conversations")
def chat_conversations():
    query = (request.args.get("q") or "").strip()
    rows = db.session.query(Conversation)
    if query:
        rows = rows.filter(Conversation.title.ilike(f"%{query}%"))
    rows = rows.order_by(Conversation.pinned.desc(), Conversation.updated_at.desc()).limit(100)
    return jsonify({"ok": True, "conversations": [_conversation_dict(c) for c in rows]})


@web_bp.get("/chat/conversations/<conversation_id>")
def chat_conversation(conversation_id: str):
    conversation = _conversation_or_404(conversation_id)
    if conversation is None:
        return jsonify({"ok": False, "error": "会话不存在"}), 404
    return jsonify(
        {
            "ok": True,
            "conversation": _conversation_dict(conversation),
            "messages": [_message_dict(m) for m in conversation.messages],
        }
    )


@web_bp.post("/chat/conversations")
def chat_create():
    conversation = Conversation(title="新会话")
    db.session.add(conversation)
    db.session.commit()
    return jsonify({"ok": True, "conversation": _conversation_dict(conversation)})


@web_bp.patch("/chat/conversations/<conversation_id>")
def chat_rename(conversation_id: str):
    conversation = _conversation_or_404(conversation_id)
    if conversation is None:
        return jsonify({"ok": False, "error": "会话不存在"}), 404
    payload = request.get_json(silent=True) or {}
    title = (payload.get("title") or "").strip()
    if title:
        conversation.title = title[:200]
    if "pinned" in payload:
        conversation.pinned = bool(payload["pinned"])
    db.session.commit()
    return jsonify({"ok": True, "conversation": _conversation_dict(conversation)})


@web_bp.delete("/chat/conversations/<conversation_id>")
def chat_delete(conversation_id: str):
    conversation = _conversation_or_404(conversation_id)
    if conversation is None:
        return jsonify({"ok": False, "error": "会话不存在"}), 404
    db.session.delete(conversation)
    db.session.commit()
    return jsonify({"ok": True})


# --------------------------------------------------------------------------
# 附件上传
# --------------------------------------------------------------------------


@web_bp.post("/chat/upload")
def chat_upload():
    """上传图片或 PDF 附件。

    按**内容哈希**命名存盘：同一张图重复粘贴不会存出多份，
    而且天然去重。文件名不参与路径，避免用户提供的名字跑到路径里。
    """
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"ok": False, "error": "没有收到文件"}), 400

    media_type = (upload.mimetype or "").lower()
    if media_type in IMAGE_TYPES:
        kind, suffix, limit = "image", IMAGE_TYPES[media_type], MAX_IMAGE_BYTES
    elif media_type in PDF_TYPES:
        kind, suffix, limit = "pdf", PDF_TYPES[media_type], MAX_PDF_BYTES
    else:
        return jsonify(
            {"ok": False, "error": f"不支持的文件类型：{media_type or '未知'}"}
        ), 415

    data = upload.read()
    if len(data) > limit:
        return jsonify(
            {"ok": False, "error": f"文件太大（上限 {limit // 1024 // 1024} MB）"}
        ), 413

    digest = hashlib.sha256(data).hexdigest()[:32]
    target = _upload_dir() / f"{digest}{suffix}"
    if not target.exists():
        target.write_bytes(data)

    return jsonify(
        {
            "ok": True,
            "attachment": {
                "kind": kind,
                "name": upload.filename[:120],
                "media_type": media_type,
                "path": str(target),
                "size": len(data),
            },
        }
    )


# --------------------------------------------------------------------------
# 流式问答（含落库）
# --------------------------------------------------------------------------


@web_bp.post("/chat/conversations/<conversation_id>/stream")
def chat_stream(conversation_id: str):
    """流式回答并落库。

    落库放在这里而不是交给前端二次调用：**流式中断（刷新、关标签页）时
    前端那条路不会执行**，由服务端在生成结束时统一写，才能保证
    「页面上看到过的回答」和「历史里存着的回答」是一致的。
    """
    conversation = _conversation_or_404(conversation_id)
    if conversation is None:
        return jsonify({"ok": False, "error": "会话不存在"}), 404

    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    attachments = payload.get("attachments") or []
    if not question and not attachments:
        return jsonify({"ok": False, "error": "问题为空"}), 400

    blocks, notes = _blocks_from_attachments(attachments[:6])
    # 提问本身作为文本块放最前面，附件跟在后面
    user_blocks = [{"type": "text", "text": question or "请看这些附件。"}, *blocks]

    history = _history_of(conversation)
    filters = _scope_filters(conversation)

    # 先落用户消息：模型调用可能失败，但那句话确实说过，历史里应该有
    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        text=question or "（附件）",
        content=user_blocks,
    )
    db.session.add(user_message)
    if conversation.title in {"新会话", ""} and question:
        conversation.title = question[:60]
    db.session.commit()

    # **只把纯值带进生成器**，不要闭包 ORM 实例。
    #
    # 视图函数一返回，Flask-SQLAlchemy 就会回收这个请求的 session，
    # 之前加载的对象变成 detached；生成器里再碰它们会抛 DetachedInstanceError
    # （属性甚至包括 .id 都可能触发刷新）。所以下面全部用字符串 ID，
    # 在生成器内部按 ID 重查——那时 stream_with_context 已经给了新的 app context。
    user_message_id = user_message.id
    conv_id = conversation.id
    note_suffix = ("　［" + "、".join(notes) + "］") if notes else ""

    return _sse_response(
        _answer_stream(
            conv_id=conv_id,
            question=question,
            user_blocks=user_blocks,
            history=history,
            filters=filters,
            note_suffix=note_suffix,
            user_message_id=user_message_id,
        )
    )


def _answer_stream(
    *,
    conv_id: str,
    question: str,
    user_blocks: list[dict],
    history: list[dict],
    filters: dict | None,
    note_suffix: str,
    user_message_id: str,
    removed: int = 0,
):
    """流式回答并落库。新提问与「编辑后重发」两个入口共用。

    抽出来是因为两个入口的区别只在**提问之前**做了什么（一个新建消息、
    一个截断后改写），从提问开始的流程完全一样。各写一份的话，
    「落库」「错误兜底」这些细节迟早会分叉。
    """
    from ..services.rag import answer_streaming

    accumulated: list[str] = []
    citations: list[dict] = []
    grounded = True
    error: str | None = None

    # removed 告诉前端「这次重发清掉了多少条消息」，它据此把界面上多余的
    # 消息节点删掉。不告诉它的话页面会继续显示已经不存在的回答。
    yield _sse({"type": "start", "user_message_id": user_message_id, "removed": removed})

    try:
        for event in answer_streaming(
            question or "请说明这些附件的内容。",
            filters=filters,
            history=history,
            question_blocks=user_blocks,
            ref=conv_id,
        ):
            kind = event.get("type")
            if kind == "text":
                accumulated.append(event.get("text", ""))
            elif kind == "done":
                citations = event.get("citations") or []
                grounded = bool(event.get("grounded", True))
            elif kind == "error":
                error = event.get("message")
            yield _sse(event)
    except Exception as exc:  # pragma: no cover - 兜底，避免流断在半路
        log.exception("流式回答失败")
        error = f"{type(exc).__name__}: {exc}"
        yield _sse({"type": "error", "message": error})

    # 落库。即使出错也存——否则用户看到半截回答，刷新后却什么都找不到。
    text = "".join(accumulated).strip()
    try:
        assistant = Message(
            conversation_id=conv_id,
            role="assistant",
            text=text or (f"（生成失败：{error}）" if error else "（空回答）"),
            content=[{"type": "text", "text": text}],
            citations=citations,
        )
        db.session.add(assistant)
        if note_suffix:
            stored = db.session.get(Message, user_message_id)
            if stored is not None:
                stored.text = (stored.text or "") + note_suffix
        db.session.commit()
        yield _sse({"type": "saved", "assistant_message_id": assistant.id,
                    "grounded": grounded})
    except Exception as exc:
        log.exception("保存回答失败")
        yield _sse({"type": "error", "message": f"回答已生成但保存失败：{exc}"})


def _sse_response(generator):
    return Response(
        stream_with_context(generator),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 让 nginx 之类的反代不要缓冲 SSE
        },
    )


@web_bp.post("/chat/conversations/<conversation_id>/messages/<message_id>/restream")
def chat_restream(conversation_id: str, message_id: str):
    """编辑一条历史提问，并从那里重新发起问答。

    **语义是截断：被编辑那条之后的全部消息会被删除。**

    为什么不做分支：模型里确实留了 ``parent_id`` 做分支的余地，但历史组装
    是按时间顺序取全部消息的，真做分支要改成沿 parent 链回溯；更麻烦的是
    「编辑之后旧回答还留在页面上」会让上下文变得难以理解——用户看到的是一条
    线，模型看到的却是分叉，追问时指代指到哪一支全凭运气。

    截断也与主流对话产品的行为一致（编辑即替换）。删除是不可逆的，
    所以前端必须先明确告知会影响多少条消息再让用户确认。
    """
    conversation = _conversation_or_404(conversation_id)
    if conversation is None:
        return jsonify({"ok": False, "error": "会话不存在"}), 404

    target = db.session.get(Message, message_id)
    if target is None or target.conversation_id != conversation.id:
        return jsonify({"ok": False, "error": "消息不存在"}), 404
    if target.role != ROLE_USER:
        return jsonify({"ok": False, "error": "只能编辑自己的提问"}), 400

    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"ok": False, "error": "问题为空"}), 400

    # 删除这条**之后**的所有消息。用 created_at 排序会因同秒消息的顺序不稳定，
    # 所以按 (created_at, id) 排——id 是 ULID，同秒内也保持时间序。
    #
    # 比较必须用**严格大于**：写成 >= 会把被编辑的那条自己也删掉，
    # 表现是「编辑后只剩一条回答」，而提问凭空消失。
    later = (
        db.session.query(Message)
        .filter(
            Message.conversation_id == conversation.id,
            db.tuple_(Message.created_at, Message.id) > (target.created_at, target.id),
        )
        .order_by(Message.created_at, Message.id)
        .all()
    )
    removed = len(later)
    for message in later:
        db.session.delete(message)

    # 改写这条提问本身
    target.text = question
    target.content = [{"type": "text", "text": question}]
    target.citations = []
    db.session.commit()

    # 截断之后的历史（不含被改写那条）
    history = _history_of(conversation)

    return _sse_response(
        _answer_stream(
            conv_id=conversation.id,
            question=question,
            user_blocks=[{"type": "text", "text": question}],
            history=history,
            filters=_scope_filters(conversation),
            note_suffix="",
            user_message_id=target.id,
            removed=removed,
        )
    )


@web_bp.patch("/chat/conversations/<conversation_id>/scope")
def chat_set_scope(conversation_id: str):
    """设置会话的论文范围（「只在这几篇里聊」）。"""
    conversation = _conversation_or_404(conversation_id)
    if conversation is None:
        return jsonify({"ok": False, "error": "会话不存在"}), 404
    payload = request.get_json(silent=True) or {}
    ids = [str(x) for x in (payload.get("paper_ids") or [])][:20]
    conversation.scope_paper_ids = ids
    db.session.commit()
    return jsonify({"ok": True, "paper_ids": ids})


@web_bp.get("/chat/papers")
def chat_paper_search():
    """@ 引用论文时用的搜索。只返回轻量字段——这是个补全接口，不是列表接口。"""
    from ..services import papers as papers_service

    query = (request.args.get("q") or "").strip()
    rows, _, _ = papers_service.list_papers(query=query or None, limit=12)
    return jsonify(
        {
            "ok": True,
            "papers": [
                {"id": p.id, "title": p.title or "（未命名）", "year": p.year}
                for p in rows
            ],
        }
    )
