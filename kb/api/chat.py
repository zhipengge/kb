"""问答与会话接口。

对外的核心是 ``POST /ask``：一次调用完成「检索知识库 + 生成带引用的回答」，
是其他 agent 最常用的入口。要比它更细的控制时，用 ``/chat/*`` 那组
管理多轮会话，或者直接用 ``/search`` 自己组装。

流式版本用 SSE。之所以不把流式做成默认：agent 调用方通常等一个完整结果，
而 SSE 的解析成本不低。两条路径并存，各取所需。
"""

from __future__ import annotations

import json
import logging

from flask import Response, request

from ..extensions import db
from . import api_bp
from .auth import require_scope
from .envelope import error_response, ok

log = logging.getLogger(__name__)


def _sse_event(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@api_bp.post("/ask")
@require_scope("read")
def ask_endpoint():
    """基于知识库回答问题，返回带引用的答案。

    请求体：

        {"question": "...", "limit": 8, "paper_ids": [...], "stream": false}

    回答里的每条引用都带 ``check`` 字段说明校验结果：
    ``verified``（引文对得上）/ ``unverified``（模型没给引文）/
    ``mismatched``（给了引文但对不上，**最可疑**）。
    """
    from ..services.rag import answer

    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or payload.get("q") or "").strip()
    if not question:
        return error_response("invalid_argument", "缺少 question", 400)

    filters = {}
    if payload.get("paper_ids"):
        filters["paper_ids"] = payload["paper_ids"]
    if payload.get("paper_id"):
        filters["paper_id"] = payload["paper_id"]

    result = answer(
        question,
        limit=payload.get("limit"),
        filters=filters,
        history=payload.get("history"),
    )

    if result.error:
        return error_response("llm_error", result.error, 502)

    return ok(result.to_dict())


@api_bp.get("/ask/stream")
@api_bp.post("/ask/stream")
@require_scope("read")
def ask_stream_endpoint():
    """流式问答（SSE）。

    事件序列：``sources``（检索到哪些片段，立刻可显示）→
    ``thinking`` / ``text``（增量输出）→ ``done``（带引用与校验结果）。
    """
    from ..services.rag import answer_streaming

    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
    else:
        payload = {"question": request.args.get("q", "")}

    question = (payload.get("question") or payload.get("q") or "").strip()
    if not question:
        return error_response("invalid_argument", "缺少 question", 400)

    filters = {}
    if payload.get("paper_ids"):
        filters["paper_ids"] = payload["paper_ids"]

    def generate():
        try:
            for event in answer_streaming(question, limit=payload.get("limit"), filters=filters):
                yield _sse_event(event)
        except Exception as exc:
            log.exception("流式问答失败")
            yield _sse_event({"type": "error", "message": str(exc)})

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# --------------------------------------------------------------------------
# 会话管理
# --------------------------------------------------------------------------


def _conversation_dict(conversation) -> dict:
    from ..utils.time import iso

    return {
        "id": conversation.id,
        "title": conversation.title,
        "model": conversation.model,
        "pinned": conversation.pinned,
        "created_at": iso(conversation.created_at),
        "updated_at": iso(conversation.updated_at),
        "message_count": len(conversation.messages),
    }


def _message_dict(message) -> dict:
    from ..utils.time import iso

    return {
        "id": message.id,
        "role": message.role,
        "text": message.text,
        "citations": message.citations or [],
        "usage": message.usage or {},
        "created_at": iso(message.created_at),
    }


@api_bp.get("/chat/conversations")
@require_scope("read")
def list_conversations():
    from ..models import Conversation

    rows = (
        db.session.query(Conversation)
        .order_by(Conversation.pinned.desc(), Conversation.updated_at.desc())
        .limit(100)
        .all()
    )
    return ok([_conversation_dict(row) for row in rows])


@api_bp.post("/chat/conversations")
@require_scope("write")
def create_conversation():
    from ..models import Conversation

    payload = request.get_json(silent=True) or {}
    conversation = Conversation(
        title=(payload.get("title") or "新会话")[:512],
        scope_paper_ids=payload.get("paper_ids") or [],
    )
    db.session.add(conversation)
    db.session.commit()
    return ok(_conversation_dict(conversation), status=201)


@api_bp.get("/chat/conversations/<conversation_id>")
@require_scope("read")
def get_conversation(conversation_id: str):
    from ..models import Conversation

    conversation = db.session.get(Conversation, conversation_id)
    if conversation is None:
        return error_response("not_found", "会话不存在", 404)

    data = _conversation_dict(conversation)
    data["messages"] = [_message_dict(m) for m in conversation.messages]
    return ok(data)


@api_bp.delete("/chat/conversations/<conversation_id>")
@require_scope("write")
def delete_conversation(conversation_id: str):
    from ..models import Conversation

    conversation = db.session.get(Conversation, conversation_id)
    if conversation is None:
        return error_response("not_found", "会话不存在", 404)
    db.session.delete(conversation)
    db.session.commit()
    return ok({"id": conversation_id, "deleted": True})


@api_bp.post("/chat/conversations/<conversation_id>/messages")
@require_scope("write")
def post_message(conversation_id: str):
    """在一个会话里提问，返回回答并把它存进历史。

    历史只带**文本**，不带引用标记——把上一轮的 ``[1]`` 原样传给模型，
    它会以为那是本轮的编号，导致引用错乱。
    """
    from ..models import Conversation, Message
    from ..services.rag import answer

    conversation = db.session.get(Conversation, conversation_id)
    if conversation is None:
        return error_response("not_found", "会话不存在", 404)

    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or payload.get("message") or "").strip()
    if not question:
        return error_response("invalid_argument", "缺少 question", 400)

    history = [
        {"role": m.role, "content": m.text or ""}
        for m in conversation.messages
        if m.role in {"user", "assistant"} and m.text
    ]

    filters = {}
    if conversation.scope_paper_ids:
        filters["paper_ids"] = list(conversation.scope_paper_ids)

    db.session.add(Message(conversation_id=conversation.id, role="user", text=question))
    db.session.commit()

    result = answer(question, filters=filters or None, history=history)
    if result.error:
        return error_response("llm_error", result.error, 502)

    assistant = Message(
        conversation_id=conversation.id,
        role="assistant",
        text=result.answer,
        citations=[c.to_dict() for c in result.citations],
        usage={"tokens": result.tokens_used, "model": result.model},
        model=result.model,
    )
    db.session.add(assistant)

    # 首条消息用作会话标题，省得用户在列表里看到一堆「新会话」
    if conversation.title in {"新会话", ""} and question:
        conversation.title = question[:60]
    db.session.commit()

    return ok(
        {
            "message": _message_dict(assistant),
            "citations": [c.to_dict() for c in result.citations],
            "grounded": result.grounded,
            "elapsed_ms": result.elapsed_ms,
        }
    )
