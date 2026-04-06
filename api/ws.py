import json
import logging
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from voice_chatbot.config import VoiceChatConfig
from voice_chatbot.conversation import ConversationMemory
from voice_chatbot.llm.mock import MockLLM
from voice_chatbot.pipeline import VoicePipeline
from voice_chatbot.stt.mock import MockSTT
from voice_chatbot.tts.mock import MockTTS

log = logging.getLogger(__name__)
ws_router = APIRouter()

_config = VoiceChatConfig()


def _create_mock_pipeline() -> VoicePipeline:
    return VoicePipeline(
        stt=MockSTT(),
        llm=MockLLM(),
        tts=MockTTS(),
        conversation=ConversationMemory(
            system_prompt=_config.system_prompt,
            max_turns=_config.max_conversation_turns,
        ),
    )


# TODO: benchmark async generator vs chunked response for streaming
@ws_router.websocket("/chat/stream")
async def voice_stream(ws: WebSocket):
    """WebSocket endpoint for streaming voice conversation.

    Protocol:
      Client sends: {"type": "audio", "data": "<base64 audio>", "format": "wav"}
                 or: {"type": "text", "data": "hello"}
      Server sends: {"type": "transcript", "data": "..."}
                    {"type": "response_text", "data": "..."}
                    {"type": "audio_chunk", "data": "<base64 audio>"}
                    {"type": "done"}
    """
    await ws.accept()
    session_id = str(uuid.uuid4())
    pipeline = _create_mock_pipeline()
    log.info("WebSocket session started: %s", session_id)

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type", "text")
            data = msg.get("data", "")

            if msg_type == "text":
                response = await pipeline.process_text(data)
                await ws.send_json({"type": "response_text", "data": response})
                await ws.send_json({"type": "done"})

            elif msg_type == "audio":
                import base64

                audio_bytes = base64.b64decode(data)
                fmt = msg.get("format", "wav")

                transcript = await pipeline.stt.transcribe(audio_bytes, fmt)
                await ws.send_json({"type": "transcript", "data": transcript})

                pipeline.conversation.add_user_message(transcript)
                response = await pipeline.llm.generate(pipeline.conversation.get_messages())
                pipeline.conversation.add_assistant_message(response)
                await ws.send_json({"type": "response_text", "data": response})

                audio_out = await pipeline.tts.synthesize(response)
                encoded = base64.b64encode(audio_out).decode()
                await ws.send_json({"type": "audio_chunk", "data": encoded})
                await ws.send_json({"type": "done"})

    except WebSocketDisconnect:
        log.info("WebSocket session ended: %s", session_id)
    except Exception:
        log.exception("WebSocket error in session %s", session_id)
        await ws.close(code=1011)
