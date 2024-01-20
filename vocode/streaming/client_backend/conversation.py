import os
import jwt
import logging
from typing import Callable, Optional
import typing
import httpx
import time

from starlette.websockets import WebSocketDisconnect
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

from vocode.streaming.compa_api.types import StoredMessage
from typing import List


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
        self.user_id = None
        self.user_uuid = None
        self.conversation_id = None
        asyncio.create_task(self.update_users_data_periodically())

    def get_conversation(
        self,
        output_device: WebsocketOutputDevice,
        start_message: AudioConfigStartMessage,
        prompt_preamble: str,
        initial_message: str,
        past_transcript: Optional[List[StoredMessage]],
    ) -> StreamingConversation:
        transcriber = self.transcriber_thunk(start_message.input_audio_config)
        synthesizer = self.synthesizer_thunk(start_message.output_audio_config)
        synthesizer.synthesizer_config.should_encode_as_wav = True
        return StreamingConversation(
            output_device=output_device,
            transcriber=transcriber,
            agent=self.agent_thunk(prompt_preamble, BaseMessage(text=initial_message)),
            synthesizer=synthesizer,
            conversation_id=start_message.conversation_id,
            events_manager=TranscriptEventManager(output_device, self.logger)
            if start_message.subscribe_transcript
            else None,
            logger=self.logger,
            past_transcript=past_transcript,
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

    async def start_conversation(
        self,
        conversation: StreamingConversation,
        websocket: WebSocket,
        output_device: WebsocketOutputDevice,
    ):
        await conversation.start(lambda: websocket.send_text(ReadyMessage().json()))
        try:
            while conversation.is_active():
                message: WebSocketMessage = WebSocketMessage.parse_obj(
                    await websocket.receive_json()
                )
                if message.type == WebSocketMessageType.STOP:
                    break
                user_data = self.users_data.get(self.user_uuid)
                if user_data["remainingTalkTime"] < 1:
                    print(
                        "Closing connection during conversation because user has insufficient balance."
                    )
                    await websocket.send_json(
                        {"error": "INSUFFICIENT_CHARACTER_BALANCE"}
                    )
                    await websocket.close()  # Terminate the conversation
                    return

                audio_message = typing.cast(AudioMessage, message)
                conversation.receive_audio(audio_message.get_bytes())
        except WebSocketDisconnect as e:
            self.logger.error(f"WebSocket disconnected: {e.code}")
        finally:
            output_device.mark_closed()
            await update_conversation(
                self.conversation_id, int(time.time()), self.user_id
            )
            await conversation.terminate()

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

        if not start_message.conversation_id:
            self.logger.error("No conversation id provided in the first message.")
            await websocket.close()
            return

        # Assuming start_message is an object that contains auth_token
        auth_token = start_message.auth_token

        print(start_message.auth_token)

        if auth_token.startswith("Bearer guest-user-id-"):
            self.user_uuid, self.user_id = await fetch_user_by_guest_id(
                auth_token.replace("Bearer ", "")
            )
        else:
            # Original logic
            user_uuid = get_user_uuid_from_token(auth_token)
            user_id = await fetch_user_id_by_uuid(user_uuid)
            self.user_id = user_id
            self.user_uuid = user_uuid

        conversation = await fetch_conversation_by_uuid(
            start_message.conversation_id, self.user_id
        )

        self.conversation_id = conversation.get("id")

        if not self.user_uuid:
            self.logger.error("Invalid or expired auth token.")
            await websocket.close()
            return

        user_data = self.users_data.get(self.user_uuid)

        if not user_data:
            try:
                newUsersData = await fetch_users_data()
                if newUsersData:
                    self.users_data = newUsersData
                user_data = self.users_data.get(self.user_uuid)
            except Exception as e:
                print(f"Error occurred while fetching user data: {e}")
                # You can log the error or handle it as needed

        if not conversation.get("promptPreamble") or not conversation.get(
            "initialMessage"
        ):
            self.logger.error(
                "No prompt preamble or initial message found in conversation.",
                conversation,
            )
            await websocket.close()
            return

        if not user_data:
            self.logger.error("User doesn't exist in user data.")
            await websocket.close()
            return

        if user_data["remainingTalkTime"] < 1:
            print("Closing connection because user has insufficient balance.")
            await websocket.send_json({"error": "INSUFFICIENT_TALK_TIME"})
            await websocket.close()  # Terminate the conversation
            return

        output_device = WebsocketOutputDevice(
            websocket,
            start_message.output_audio_config.sampling_rate,
            start_message.output_audio_config.audio_encoding,
        )

        promptPreamble = conversation.get("promptPreamble")
        past_transcript = conversation.get("pastTranscripts")

        self.logger.debug(f"Conversation started")

        conversation = self.get_conversation(
            output_device,
            start_message,
            promptPreamble,
            conversation.get("initialMessage"),
            past_transcript=past_transcript,
        )
        await self.start_conversation(conversation, websocket, output_device)

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
        if event.type == EventType.TRANSCRIPT_COMPLETE:
            self.logger.debug("TRANSCRIPT COMPLETE", event)

    def restart(self, output_device: WebsocketOutputDevice):
        self.output_device = output_device


async def update_conversation(
    conversation_id: int, ended_at_timestamp: int, user_id: int
):
    url = f"{CATALYST_API_SERVER_URL}/chats/updateConversation"
    headers = {
        "Authorization": f"Bearer {CATALYST_API_SECRET}",
        "Content-Type": "application/json",
    }
    json_data = {
        "conversationId": conversation_id,
        "endedAtTimestamp": ended_at_timestamp * 1000,
        "userId": user_id,
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.put(url, headers=headers, json=json_data)
            response.raise_for_status()

        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e.response.status_code}")
        except httpx.RequestError as e:
            print(f"An error occurred while requesting {e.request.url!r}.")


async def create_message_and_count_characters(event: Event):
    if event.sender == Sender.HUMAN:
        sender_type = "USER"
    elif event.sender == Sender.BOT:
        sender_type = "CLONE"
    else:
        return
    print(
        {
            "text": event.text,
            "timestamp": event.timestamp,
            "conversationUUID": event.conversation_id,
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
                    "conversationUUID": event.conversation_id,
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


async def fetch_conversation_by_uuid(uuid: str, user_id: int):
    headers = {"Authorization": f"Bearer {CATALYST_API_SECRET}"}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{CATALYST_API_SERVER_URL}/chats/conversation-by-uuid",
                headers=headers,
                json={"uuid": uuid, "userId": user_id},
            )
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            print(f"Error fetching conversation: {e}")
            return None


async def fetch_users_data():
    headers = {"Authorization": f"Bearer {CATALYST_API_SECRET}"}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{CATALYST_API_SERVER_URL}/users/remainingTalkTimeForAllUsers",
                headers=headers,
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


async def fetch_user_by_guest_id(guest_id: str) -> str:
    headers = {"Authorization": f"Bearer {CATALYST_API_SECRET}"}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{CATALYST_API_SERVER_URL}/users/get-user-by-guest-id",
                headers=headers,
                json={"guestId": guest_id},
            )
            response.raise_for_status()
            return response.json().get("uuid"), response.json().get("id")
        except httpx.RequestError as e:
            print(f"Error fetching user ID: {e}")
            return None
