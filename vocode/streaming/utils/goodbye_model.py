import os
import asyncio
from typing import Optional
import openai
import numpy as np
import requests
import string

from vocode import getenv

SIMILARITY_THRESHOLD = 0.9
EMBEDDING_SIZE = 1536
GOODBYE_PHRASES = [
    "bye",
    "goodbye",
    "see you",
    "see you later",
    "talk to you later",
    "talk to you soon",
    "have a good day",
    "have a good night",
    "be right back",
    "take care",
    "I have to go right now",
    "we'll talk later",
    "Ill be back",
    "Ill talk to you later",
    "I have to run right now",
    "I have to go",
    "I have to go now",
    "I have to go soon",
    "I have to go to bed",
    "I have to go to sleep",
    "farewell",
    "catch you later",
    "until next time",
    "so long",
    "im off",
    "good night",
    "bye for now",
    "take care",
    "see ya",
    "adios",
    "cheerio",
    "ciao",
    "au revoir",
    "toodle-oo",
    "later",
    "bye-bye",
    "peace out",
    "ive got to run",
    "i must be going",
    "ill catch you later",
    "its been real",
    "im out",
    "keep in touch",
    "ill see you around",
    "ill be seeing you",
    "until we meet again",
    "hasta la vista",
    "godspeed",
    "safe travels",
    "i gotta bounce",
    "i gotta jet",
    "im heading out",
    "time to hit the road",
    "im off to bed",
    "gotta go",
    "signing off",
    "time to scoot",
    "im outta here",
    "leaving now",
    "all the best",
    "best wishes",
    "ive got to get going",
    "sayonara",
    "fare thee well",
    "ill be off",
    "time to head out",
    "have a good one",
    "have a great day",
    "be seeing you",
    "goodbye for now",
    "catch you on the flip side",
    "ill talk to you soon",
    "be well",
    "im saying goodbye",
    "i bid you adieu",
    "off i go",
    "good luck",
    "im leaving now",
    "time for me to go",
    "ill say goodbye",
    "ill leave you now",
]


class GoodbyeModel:
    def __init__(
        self,
        embeddings_cache_path=os.path.join(
            os.path.dirname(__file__), "goodbye_embeddings"
        ),
        openai_api_key: Optional[str] = None,
    ):
        openai.api_key = openai_api_key or getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment or passed in")
        self.embeddings_cache_path = embeddings_cache_path
        self.goodbye_embeddings: Optional[np.ndarray] = None

    async def initialize_embeddings(self):
        self.goodbye_embeddings = await self.load_or_create_embeddings(
            f"{self.embeddings_cache_path}/goodbye_embeddings.npy"
        )

    async def load_or_create_embeddings(self, path):
        if os.path.exists(path):
            return np.load(path)
        else:
            embeddings = await self.create_embeddings()
            np.save(path, embeddings)
            return embeddings

    async def create_embeddings(self):
        print("Creating embeddings...")
        size = EMBEDDING_SIZE
        embeddings = np.empty((size, len(GOODBYE_PHRASES)))
        for i, goodbye_phrase in enumerate(GOODBYE_PHRASES):
            embeddings[:, i] = await self.create_embedding(goodbye_phrase)
        return embeddings

    async def is_goodbye(self, text: str) -> bool:
        assert self.goodbye_embeddings is not None, "Embeddings not initialized"

        translator = str.maketrans(
            string.punctuation, " " * len(string.punctuation), "'\""
        )
        processed_text = text.translate(translator).strip().lower()

        if any(processed_text.endswith(phrase) for phrase in GOODBYE_PHRASES):
            return True

        embedding = await self.create_embedding(text.strip().lower())
        similarity_results = embedding @ self.goodbye_embeddings

        return np.max(similarity_results) > SIMILARITY_THRESHOLD

    async def create_embedding(self, text) -> np.ndarray:
        params = {
            "input": text,
        }

        engine = getenv("AZURE_OPENAI_TEXT_EMBEDDING_ENGINE")
        if engine:
            params["engine"] = engine
        else:
            params["model"] = "text-embedding-ada-002"

        return np.array(
            (await openai.Embedding.acreate(**params))["data"][0]["embedding"]
        )


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    async def main():
        model = GoodbyeModel()
        while True:
            print(await model.is_goodbye(input("Text: ")))

    asyncio.run(main())
