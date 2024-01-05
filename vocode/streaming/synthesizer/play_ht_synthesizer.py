import json
import asyncio
import logging
import sys
from typing import Optional
import base64

from aiohttp import ClientSession, ClientTimeout
from vocode import getenv
from vocode.streaming.agent.bot_sentiment_analyser import BotSentiment
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import PlayHtSynthesizerConfig, SynthesizerType
from vocode.streaming.synthesizer.base_synthesizer import (
    BaseSynthesizer,
    SynthesisResult,
    tracer,
)
from vocode.streaming.utils.mp3_helper import decode_mp3

from vocode.streaming.synthesizer.eleven_labs_synthesizer import ElevenLabsSynthesizer
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig

TTS_ENDPOINT = "https://play.ht/api/v2/tts/stream"

play_ht_backup = getenv("PLAY_HT_BACKUP")
if play_ht_backup is None:
    print("PLAY_HT_BACKUP environment variable is not set")
    sys.exit("Missing PLAY_HT_BACKUP environment variable")

eleven_labs_api_key = getenv("ELEVEN_LABS_API_KEY")
if eleven_labs_api_key is None:
    print("ELEVEN_LABS_API_KEY environment variable is not set")
    sys.exit("Missing ELEVEN_LABS_API_KEY environment variable")

try:
    backup_credentials = json.loads(base64.b64decode(play_ht_backup).decode())
except json.JSONDecodeError:
    print("Failed to parse PLAY_HT_BACKUP as JSON")
    sys.exit("Invalid JSON in PLAY_HT_BACKUP environment variable")


class PlayHtSynthesizer(BaseSynthesizer[PlayHtSynthesizerConfig]):
    def __init__(
        self,
        synthesizer_config: PlayHtSynthesizerConfig,
        logger: Optional[logging.Logger] = None,
        aiohttp_session: Optional[ClientSession] = None,
    ):
        super().__init__(synthesizer_config, aiohttp_session)
        self.logger = logger or logging.getLogger(__name__)

        self.synthesizer_config = synthesizer_config
        self.api_key = synthesizer_config.api_key or getenv("PLAY_HT_API_KEY")
        self.user_id = synthesizer_config.user_id or getenv("PLAY_HT_USER_ID")
        self.backup_eleven_labs_synthesizer_api_key = getenv("ELEVEN_LABS_API_KEY")
        if not self.api_key or not self.user_id:
            raise ValueError(
                "You must set the PLAY_HT_API_KEY and PLAY_HT_USER_ID environment variables"
            )
        self.words_per_minute = 150
        self.experimental_streaming = synthesizer_config.experimental_streaming

    async def create_speech(
        self,
        message: BaseMessage,
        chunk_size: int,
        bot_sentiment: Optional[BotSentiment] = None,
    ) -> SynthesisResult:
        self.logger.info("Attempting with Eleven Labs Synthesizer")

        # Configure Eleven Labs synthesizer here
        eleven_labs_config = ElevenLabsSynthesizerConfig(
            # Set your Eleven Labs configuration parameters here
            api_key=self.backup_eleven_labs_synthesizer_api_key,
            voice_id="Y8ljZnfwX14S0JfyoEFq",
            synthesizer_config=self.synthesizer_config,
            sampling_rate=self.synthesizer_config.sampling_rate,
            audio_encoding=self.synthesizer_config.audio_encoding,
            should_encode_as_wav=True
            # ... Other configuration parameters ...
        )

        eleven_labs_synthesizer = ElevenLabsSynthesizer(
            synthesizer_config=eleven_labs_config,
            logger=self.logger,
            aiohttp_session=self.aiohttp_session,
        )

        # Attempt to create speech using Eleven Labs
        try:
            return await eleven_labs_synthesizer.create_speech(message, chunk_size)
        except Exception as e:
            self.logger.error(
                f"Trying PlayHT.Failed to create speech with Eleven Labs: {e}"
            )

        credential_combinations = [
            {
                "apiKey": self.api_key,
                "userId": self.user_id,
                "voiceId": self.synthesizer_config.voice_id,
            }
        ] + backup_credentials

        for credentials in credential_combinations:
            print(f"Trying with userId: {credentials['userId']}")

            headers = {
                "AUTHORIZATION": f"Bearer {credentials['apiKey']}",
                "X-USER-ID": credentials["userId"],
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
            }
            body = {
                "quality": "draft",
                "voice": credentials["voiceId"],
                "text": message.text,
                "sample_rate": self.synthesizer_config.sampling_rate,
            }
            if self.synthesizer_config.speed:
                body["speed"] = self.synthesizer_config.speed
            if self.synthesizer_config.seed:
                body["seed"] = self.synthesizer_config.seed
            if self.synthesizer_config.temperature:
                body["temperature"] = self.synthesizer_config.temperature

            create_speech_span = tracer.start_span(
                f"synthesizer.{SynthesizerType.PLAY_HT.value.split('_', 1)[-1]}.create_total",
            )

            try:
                response = await self.aiohttp_session.post(
                    TTS_ENDPOINT,
                    headers=headers,
                    json=body,
                    timeout=ClientTimeout(total=15),
                )

                if response.ok:
                    if self.experimental_streaming:
                        return SynthesisResult(
                            self.experimental_mp3_streaming_output_generator(
                                response, chunk_size, create_speech_span
                            ),
                            lambda seconds: self.get_message_cutoff_from_voice_speed(
                                message, seconds, self.words_per_minute
                            ),
                        )
                    else:
                        read_response = await response.read()
                        create_speech_span.end()
                        convert_span = tracer.start_span(
                            f"synthesizer.{SynthesizerType.PLAY_HT.value.split('_', 1)[-1]}.convert",
                        )
                        output_bytes_io = decode_mp3(read_response)

                        result = self.create_synthesis_result_from_wav(
                            synthesizer_config=self.synthesizer_config,
                            file=output_bytes_io,
                            message=message,
                            chunk_size=chunk_size,
                        )
                        convert_span.end()
                        return result

            except Exception as e:
                self.logger.error(f"Failed to create speech with Play.ht: {e}")

        raise Exception("Failed to create speech with all credential combinations")
