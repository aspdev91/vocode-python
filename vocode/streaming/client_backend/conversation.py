import os
import jwt
import logging
from typing import Callable, Optional
import typing
import httpx

from fastapi import APIRouter, WebSocket
from vocode.streaming.agent.base_agent import BaseAgent
from vocode.streaming.models.audio_encoding import AudioEncoding
from vocode.streaming.models.client_backend import InputAudioConfig, OutputAudioConfig
from vocode.streaming.models.synthesizer import AzureSynthesizerConfig
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
    TimeEndpointingConfig,
)
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.models.websocket import (
    AudioConfigStartMessage,
    AudioMessage,
    ReadyMessage,
    WebSocketMessage,
    WebSocketMessageType,
)

from vocode.streaming.output_device.websocket_output_device import WebsocketOutputDevice
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.synthesizer.azure_synthesizer import AzureSynthesizer
from vocode.streaming.synthesizer.base_synthesizer import BaseSynthesizer
from vocode.streaming.transcriber.base_transcriber import BaseTranscriber
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber
from vocode.streaming.utils.base_router import BaseRouter

from vocode.streaming.models.events import Event, EventType
from vocode.streaming.models.transcript import TranscriptEvent
from vocode.streaming.utils import events_manager
from vocode.streaming.models.events import Sender

from vocode.streaming.models.message import BaseMessage
import asyncio


def get_env_var(env_var_name):
    value = os.environ.get(env_var_name)
    if value is None:
        raise EnvironmentError(f"Environment variable '{env_var_name}' is not set.")
    return value


# Set environment variables as Python variables
CATALYST_API_SERVER_URL = get_env_var("CATALYST_API_SERVER_URL")
CATALYST_API_SECRET = get_env_var("CATALYST_API_SECRET")
CATALYST_SUPABASE_JWT_SECRET = get_env_var("CATALYST_SUPABASE_JWT_SECRET")


BASE_CONVERSATION_ENDPOINT = "/conversation"


class ConversationRouter(BaseRouter):
    def __init__(
        self,
        agent_thunk: Callable[
            [], ChatGPTAgent
        ] = lambda prompt_preamble, initial_message: ChatGPTAgent(
            ChatGPTAgentConfig(
                initial_message=initial_message, prompt_preamble=prompt_preamble
            )
        ),
        transcriber_thunk: Callable[
            [InputAudioConfig], BaseTranscriber
        ] = lambda input_audio_config: DeepgramTranscriber(
            DeepgramTranscriberConfig.from_input_audio_config(
                _audio_config=input_audio_config,
                endpointing_config=TimeEndpointingConfig(1.5),
            )
        ),
        synthesizer_thunk: Callable[
            [OutputAudioConfig], BaseSynthesizer
        ] = lambda output_audio_config: AzureSynthesizer(
            AzureSynthesizerConfig.from_output_audio_config(
                output_audio_config=output_audio_config
            )
        ),
        logger: Optional[logging.Logger] = None,
        conversation_endpoint: str = BASE_CONVERSATION_ENDPOINT,
    ):
        super().__init__()
        self.transcriber_thunk = transcriber_thunk
        self.agent_thunk = agent_thunk
        self.synthesizer_thunk = synthesizer_thunk
        self.logger = logger or logging.getLogger(__name__)
        self.router = APIRouter()
        self.router.websocket(conversation_endpoint)(self.conversation)
        self.users_data = {}
        self.user_uuid = None
        asyncio.create_task(self.update_users_data_periodically())

    def get_conversation(
        self,
        output_device: WebsocketOutputDevice,
        start_message: AudioConfigStartMessage,
    ) -> StreamingConversation:
        print("prompt_premable")
        print(type(start_message.prompt_preamble))
        print(start_message.prompt_preamble)
        print("initial message")
        print(type(start_message.initial_message))
        transcriber = self.transcriber_thunk(start_message.input_audio_config)
        synthesizer = self.synthesizer_thunk(start_message.output_audio_config)
        synthesizer.synthesizer_config.should_encode_as_wav = True
        return StreamingConversation(
            output_device=output_device,
            transcriber=transcriber,
            agent=self.agent_thunk(
                start_message.prompt_preamble, start_message.initial_message
            ),
            synthesizer=synthesizer,
            conversation_id=start_message.conversation_id,
            events_manager=TranscriptEventManager(output_device, self.logger)
            if start_message.subscribe_transcript
            else None,
            logger=self.logger,
        )

    async def start_periodic_update(self):
        # This method should be called after the event loop has started
        self.update_task = asyncio.create_task(self.update_users_data_periodically())

    async def update_users_data_periodically(self):
        while True:
            try:
                newUsersData = await fetch_users_data()
                if newUsersData:
                    self.users_data = newUsersData
            except Exception as e:
                print(f"Error occurred while fetching user data: {e}")
                # You can log the error or handle it as needed
            finally:
                await asyncio.sleep(30)  # Wait for 30 seconds before fetching again

    async def conversation(self, websocket: WebSocket):
        await websocket.accept()
        start_message: AudioConfigStartMessage = AudioConfigStartMessage.parse_obj(
            await websocket.receive_json()
        )

        # Extract the token and find the corresponding user

        if not start_message.auth_token:
            self.logger.error("No auth token provided in the first message.")
            await websocket.close()
            return

        user_uuid = get_user_uuid_from_token(start_message.auth_token)
        user_id = await fetch_user_id_by_uuid(user_uuid)
        self.user_uuid = user_uuid

        if not user_uuid:
            self.logger.error("Invalid or expired auth token.")
            await websocket.close()
            return

        user_data = self.users_data.get(user_uuid)

        if not user_data:
            self.logger.error("User doesn't exist in user data.")
            await websocket.close()
            return

        if (
            user_data
            and user_data["availableCharacterCount"] - user_data["usedCharacterCount"]
            < 10
        ):
            print("Closing connection because user has insufficient balance.")
            await websocket.send_json({"error": "INSUFFICIENT_CHARACTER_BALANCE"})
            await websocket.close()  # Terminate the conversation
            return

        self.logger.debug(f"Conversation started")
        output_device = WebsocketOutputDevice(
            websocket,
            start_message.output_audio_config.sampling_rate,
            start_message.output_audio_config.audio_encoding,
        )
        conversation = self.get_conversation(output_device, start_message)
        print("CONVERSATION: ", conversation.id)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{CATALYST_API_SERVER_URL}/chats/create-chat",
                headers={"Authorization": f"Bearer {CATALYST_API_SECRET}"},
                json={
                    "coachIds": [],  # Add actual coach IDs as required
                    "cloneIds": [1],
                    "userIds": [user_id],
                    "uuid": conversation.id,
                },
            )
            if response.status_code == 200:
                chat_data = response.json()
            else:
                print("Request failed:", response.status_code, response.text)
                await websocket.send_json({"error": "CHAT_DB_INITIATION_FAILED"})
                await websocket.close()
                return

        await conversation.start(lambda: websocket.send_text(ReadyMessage().json()))
        while conversation.is_active():
            message: WebSocketMessage = WebSocketMessage.parse_obj(
                await websocket.receive_json()
            )
            if message.type == WebSocketMessageType.STOP:
                break
            user_data = self.users_data.get(self.user_uuid)
            if (
                user_data
                and user_data["availableCharacterCount"]
                - user_data["usedCharacterCount"]
                < 10
            ):
                print(
                    "Closing connection during conversation because user has insufficient balance."
                )
                await websocket.send_json({"error": "INSUFFICIENT_CHARACTER_BALANCE"})
                await websocket.close()  # Terminate the conversation
                return

            audio_message = typing.cast(AudioMessage, message)
            conversation.receive_audio(audio_message.get_bytes())
        output_device.mark_closed()
        await conversation.terminate()

    def get_router(self) -> APIRouter:
        return self.router


class TranscriptEventManager(events_manager.EventsManager):
    def __init__(
        self,
        output_device: WebsocketOutputDevice,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(subscriptions=[EventType.TRANSCRIPT])
        self.output_device = output_device
        self.logger = logger or logging.getLogger(__name__)

    async def handle_event(self, event: Event):
        if event.type == EventType.TRANSCRIPT:
            # self.logger.debug("HANDLING TRANSCRIPT", event)
            transcript_event = typing.cast(TranscriptEvent, event)
            self.output_device.consume_transcript(transcript_event)
            self.logger.debug(event.dict())
            await create_message_and_count_characters(event)

    def restart(self, output_device: WebsocketOutputDevice):
        self.output_device = output_device


async def create_message_and_count_characters(event: Event):
    if event.sender == Sender.HUMAN:
        sender_type = "USER"
    elif event.sender == Sender.BOT:
        sender_type = "CLONE"
    print(
        {
            "text": event.text,
            "timestamp": event.timestamp,
            "chatUUID": event.conversation_id,
            "senderType": sender_type,
        }
    )
    async with httpx.AsyncClient() as client:
        try:
            headers = {"Authorization": f"Bearer {CATALYST_API_SECRET}"}
            response = await client.post(
                f"{CATALYST_API_SERVER_URL}/messages",
                headers=headers,
                json={
                    "text": event.text,
                    "timestamp": event.timestamp,
                    "chatUUID": event.conversation_id,
                    "senderType": sender_type,
                    "messageType": "VOICE",
                },
            )
            response.raise_for_status()
        except httpx.RequestError as e:
            print(f"Error creating a new message: {e}")
            return None


async def fetch_user_id_by_uuid(uuid: str):
    headers = {"Authorization": f"Bearer {CATALYST_API_SECRET}"}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{CATALYST_API_SERVER_URL}/users/by-uuid",
                headers=headers,
                json={"uuid": uuid},
            )
            response.raise_for_status()
            return response.json().get("id")
        except httpx.RequestError as e:
            print(f"Error fetching user ID: {e}")
            return None


async def fetch_users_data():
    headers = {"Authorization": f"Bearer {CATALYST_API_SECRET}"}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{CATALYST_API_SERVER_URL}/users/all", headers=headers
            )
            response.raise_for_status()  # Raises an HTTPError for error responses
            return response.json()
        except httpx.RequestError as e:
            print(f"Error fetching users data: {e}")
            raise


def get_user_uuid_from_token(token: str) -> str:
    secret = CATALYST_SUPABASE_JWT_SECRET
    try:
        # Decode the token
        payload = jwt.decode(
            token, secret, algorithms=["HS256"], options={"verify_aud": False}
        )
        # Extract user ID from the token payload
        user_id = payload.get("sub")  # 'sub' usually contains the user ID in JWT tokens
        return user_id
    except jwt.ExpiredSignatureError:
        print("Token expired. Get a new one.")
    except jwt.InvalidTokenError:
        print("Invalid Token. Please check.")
    return None
