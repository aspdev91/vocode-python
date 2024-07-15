import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Optional, Tuple, Union
import wave
import aiohttp
from opentelemetry.trace import Span

from vocode import getenv

from vocode.streaming.models.synthesizer import OpenAISynthesizerConfig, SynthesizerType


from vocode.streaming.synthesizer.base_synthesizer import (
    BaseSynthesizer,
    SynthesisResult,
    encode_as_wav,
    tracer,
)

from vocode.streaming.agent.bot_sentiment_analyser import BotSentiment
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.utils.mp3_helper import decode_mp3


class OpenAISynthesizer(BaseSynthesizer[OpenAISynthesizerConfig]):
    def __init__(
        self,
        synthesizer_config: OpenAISynthesizerConfig,
        logger: Optional[logging.Logger] = None,
        aiohttp_session: Optional[aiohttp.ClientSession] = None,
    ):
        super().__init__(synthesizer_config, aiohttp_session)
        self.api_key = synthesizer_config.api_key or getenv("OPENAI_API_KEY")
        self.voice_id = synthesizer_config.voice_id
        self.model = synthesizer_config.model_id

    async def create_speech(
        self,
        message: BaseMessage,
        chunk_size: int,
        bot_sentiment: Optional[BotSentiment] = None,
    ) -> SynthesisResult:
        url = "https://api.openai.com/v1/audio/speech"

        headers = {"Authorization": "Bearer " + self.api_key}

        session = self.aiohttp_session

        body = {"model": self.model, "input": message.text, "voice": self.voice_id}

        create_speech_span = tracer.start_span(
            f"synthesizer.{SynthesizerType.OPENAI.value.split('_', 1)[-1]}.create_total",
        )

        response = await session.request(
            "POST",
            url,
            json=body,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=15),
        )

        if not response.ok:
            raise Exception(f"OPEN AI TTS API returned {response.status} status code")

        audio_data = await response.read()
        create_speech_span.end()
        convert_span = tracer.start_span(
            f"synthesizer.{SynthesizerType.ELEVEN_LABS.value.split('_', 1)[-1]}.convert",
        )
        output_bytes_io = decode_mp3(audio_data)

        result = self.create_synthesis_result_from_wav(
            synthesizer_config=self.synthesizer_config,
            file=output_bytes_io,
            message=message,
            chunk_size=chunk_size,
        )
        convert_span.end()

        return result
