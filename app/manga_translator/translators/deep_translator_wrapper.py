import asyncio
from typing import List

from app.config import settings
from .common import OfflineTranslator

# Online translation using deep-translator (Google Translate, etc.)


class DeepTranslatorWrapper(OfflineTranslator):
    """
    Online translator using deep-translator library.
    Supports Google Translate and other online translation services.
    """

    _LANGUAGE_CODE_MAP = {
        "JPN": "ja",
        "ENG": "en",
        "CHS": "zh-CN",
        "CHT": "zh-TW",
        "CSY": "cs",
        "NLD": "nl",
        "FRA": "fr",
        "DEU": "de",
        "HUN": "hu",
        "ITA": "it",
        "KOR": "ko",
        "POL": "pl",
        "PTB": "pt",
        "ROM": "ro",
        "RUS": "ru",
        "ESP": "es",
        "TRK": "tr",
        "UKR": "uk",
        "VIN": "vi",
        "ARA": "ar",
        "THA": "th",
        "IND": "id",
        "FIL": "tl",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._translator = None

    async def _load(self, from_lang: str, to_lang: str, device: str):
        """Initialize the deep-translator. Device parameter is ignored for online translator."""
        try:
            from deep_translator import GoogleTranslator

            # Store language codes for later use
            self._from_lang = from_lang if from_lang != "auto" else "auto"
            self._to_lang = to_lang

            # Pre-create a test translator to validate language support
            test_translator = GoogleTranslator(source=self._from_lang, target=self._to_lang)
            self._translator = test_translator

            self.logger.info(
                f"Deep-Translator initialized: {from_lang} -> {to_lang} (GoogleTranslator backend)"
            )
        except ImportError:
            raise RuntimeError(
                "deep-translator is not installed. Install it with: pip install deep-translator"
            )
        except Exception as exc:
            self.logger.error(f"Failed to initialize Deep-Translator: {exc}")
            raise

    async def _unload(self):
        """Cleanup for online translator (no-op for deep-translator)."""
        self._translator = None

    async def _infer(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        """
        Translate queries using deep-translator in async-wrapped manner.
        """
        try:
            from deep_translator import GoogleTranslator

            # Recreate translator for each batch to ensure consistency
            translator = GoogleTranslator(source=from_lang, target=to_lang)

            # Run translation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            translations = []

            for query in queries:
                if not query or not query.strip():
                    translations.append("")
                    continue

                # Run synchronous translation in executor to avoid blocking
                translation = await loop.run_in_executor(
                    None, lambda q=query: translator.translate(q)
                )
                translations.append(translation or "")

            self.logger.debug(f"Translated {len(queries)} queries using GoogleTranslator")
            return translations

        except Exception as exc:
            self.logger.error(f"Translation failed: {exc}")
            return [""] * len(queries)
