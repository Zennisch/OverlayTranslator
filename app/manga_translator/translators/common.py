import asyncio
import re
import time
from abc import abstractmethod
from typing import List, Tuple

from ..utils import InfererModule, ModelWrapper, is_valuable_text, repeating_sequence

try:
    import readline
except Exception:
    readline = None

VALID_LANGUAGES = {
    "CHS": "Chinese (Simplified)",
    "CHT": "Chinese (Traditional)",
    "CSY": "Czech",
    "NLD": "Dutch",
    "ENG": "English",
    "FRA": "French",
    "DEU": "German",
    "HUN": "Hungarian",
    "ITA": "Italian",
    "JPN": "Japanese",
    "KOR": "Korean",
    "POL": "Polish",
    "PTB": "Portuguese (Brazil)",
    "ROM": "Romanian",
    "RUS": "Russian",
    "ESP": "Spanish",
    "TRK": "Turkish",
    "UKR": "Ukrainian",
    "VIN": "Vietnamese",
    "CNR": "Montenegrin",
    "SRP": "Serbian",
    "HRV": "Croatian",
    "THA": "Thai",
    "IND": "Indonesian",
    "FIL": "Filipino (Tagalog)",
}

ISO_639_1_TO_VALID_LANGUAGES = {
    "zh": "CHS",
    "ja": "JPN",
    "en": "ENG",
    "ko": "KOR",
    "vi": "VIN",
    "cs": "CSY",
    "nl": "NLD",
    "fr": "FRA",
    "de": "DEU",
    "hu": "HUN",
    "it": "ITA",
    "pl": "POL",
    "pt": "PTB",
    "ro": "ROM",
    "ru": "RUS",
    "es": "ESP",
    "tr": "TRK",
    "uk": "UKR",
    "cnr": "CNR",
    "sr": "SRP",
    "hr": "HRV",
    "th": "THA",
    "id": "IND",
    "tl": "FIL",
}


class LanguageUnsupportedException(Exception):
    def __init__(self, language_code: str, translator: str = None, supported_languages: List[str] = None):
        error = 'Language not supported for %s: "%s"' % (
            translator if translator else "chosen translator",
            language_code,
        )
        if supported_languages:
            error += '. Supported languages: "%s"' % ",".join(supported_languages)
        super().__init__(error)


class MTPEAdapter:
    async def dispatch(self, queries: List[str], translations: List[str]) -> List[str]:
        if not readline:
            print("MTPE is currently only supported on linux")
            return translations
        new_translations = []
        print("Running Machine Translation Post Editing (MTPE)")
        for i, (query, translation) in enumerate(zip(queries, translations)):
            print(f"\n[{i + 1}/{len(queries)}] {query}:")
            readline.set_startup_hook(lambda: readline.insert_text(translation.replace("\n", "\\n")))
            new_translation = ""
            try:
                new_translation = input(" -> ").replace("\\n", "\n")
            finally:
                readline.set_startup_hook()
            new_translations.append(new_translation)
        print()
        return new_translations


# Todo: If we don't use LLM model for translation anymore, we can completely remove the CommonTranslator and just use OfflineTranslator as the base class for all translators.
class CommonTranslator(InfererModule):
    _LANGUAGE_CODE_MAP = {}
    _INVALID_REPEAT_COUNT = 0
    _MAX_REQUESTS_PER_MINUTE = -1

    def __init__(self):
        super().__init__()
        self.mtpe_adapter = MTPEAdapter()
        self._last_request_ts = 0

    def supports_languages(self, from_lang: str, to_lang: str, fatal: bool = False) -> bool:
        supported_src_languages = ["auto"] + list(self._LANGUAGE_CODE_MAP)
        supported_tgt_languages = list(self._LANGUAGE_CODE_MAP)

        if from_lang not in supported_src_languages:
            if fatal:
                raise LanguageUnsupportedException(from_lang, self.__class__.__name__, supported_src_languages)
            return False
        if to_lang not in supported_tgt_languages:
            if fatal:
                raise LanguageUnsupportedException(to_lang, self.__class__.__name__, supported_tgt_languages)
            return False
        return True

    def parse_language_codes(self, from_lang: str, to_lang: str, fatal: bool = False) -> Tuple[str, str]:
        if not self.supports_languages(from_lang, to_lang, fatal):
            return None, None
        if type(self._LANGUAGE_CODE_MAP) is list:
            return from_lang, to_lang

        _from_lang = self._LANGUAGE_CODE_MAP.get(from_lang) if from_lang != "auto" else "auto"
        _to_lang = self._LANGUAGE_CODE_MAP.get(to_lang)
        return _from_lang, _to_lang

    async def translate(self, from_lang: str, to_lang: str, queries: List[str], use_mtpe: bool = False) -> List[str]:
        if to_lang not in VALID_LANGUAGES:
            raise ValueError(
                'Invalid language code: "%s". Choose from the following: %s' % (to_lang, ", ".join(VALID_LANGUAGES))
            )
        if from_lang not in VALID_LANGUAGES and from_lang != "auto":
            raise ValueError(
                'Invalid language code: "%s". Choose from the following: auto, %s' % (from_lang, ", ".join(VALID_LANGUAGES))
            )
        self.logger.info(f"Translating into {VALID_LANGUAGES[to_lang]}")

        if from_lang == to_lang:
            return queries

        query_indices = []
        final_translations = []
        for i, query in enumerate(queries):
            if not is_valuable_text(query):
                final_translations.append(queries[i])
            else:
                final_translations.append(None)
                query_indices.append(i)

        queries = [queries[i] for i in query_indices]

        if not queries:
            return final_translations

        translations = [""] * len(queries)
        untranslated_indices = list(range(len(queries)))
        for i in range(1 + self._INVALID_REPEAT_COUNT):
            if i > 0:
                self.logger.warn(f"Repeating because of invalid translation. Attempt: {i + 1}")
                await asyncio.sleep(0.1)

            await self._ratelimit_sleep()

            _translations = await self._translate(*self.parse_language_codes(from_lang, to_lang, fatal=True), queries)

            if len(_translations) < len(queries):
                _translations.extend([""] * (len(queries) - len(_translations)))
            elif len(_translations) > len(queries):
                _translations = _translations[: len(queries)]

            for j in untranslated_indices:
                translations[j] = _translations[j]

            if self._INVALID_REPEAT_COUNT == 0:
                break

            new_untranslated_indices = []
            for j in untranslated_indices:
                q, t = queries[j], translations[j]
                if self._is_translation_invalid(q, t):
                    new_untranslated_indices.append(j)
                    queries[j] = self._modify_invalid_translation_query(q, t)
            untranslated_indices = new_untranslated_indices

            if not untranslated_indices:
                break

        translations = [self._clean_translation_output(q, r, to_lang) for q, r in zip(queries, translations)]

        if use_mtpe:
            translations = await self.mtpe_adapter.dispatch(queries, translations)

        for i, trans in enumerate(translations):
            final_translations[query_indices[i]] = trans
            self.logger.info(f"{i}: {queries[i]} => {trans}")

        return final_translations

    @abstractmethod
    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        pass

    async def _ratelimit_sleep(self):
        if self._MAX_REQUESTS_PER_MINUTE > 0:
            now = time.time()
            ratelimit_timeout = self._last_request_ts + 60 / self._MAX_REQUESTS_PER_MINUTE
            if ratelimit_timeout > now:
                self.logger.info(f"Ratelimit sleep: {(ratelimit_timeout - now):.2f}s")
                await asyncio.sleep(ratelimit_timeout - now)
            self._last_request_ts = time.time()

    def _is_translation_invalid(self, query: str, trans: str) -> bool:
        if not trans and query:
            return True
        if not query or not trans:
            return False

        query_symbols_count = len(set(query))
        trans_symbols_count = len(set(trans))
        if query_symbols_count > 6 and trans_symbols_count < 6 and trans_symbols_count < 0.25 * len(trans):
            return True
        return False

    def _modify_invalid_translation_query(self, query: str, trans: str) -> str:
        return query

    def _clean_translation_output(self, query: str, trans: str, to_lang: str) -> str:
        if not query or not trans:
            return ""

        trans = re.sub(r"\s+", r" ", trans)
        trans = re.sub(r"(?<![.,;!?])([.,;!?])(?=\w)", r"\1 ", trans)
        trans = re.sub(r"([.,;!?])\s+(?=[.,;!?]|$)", r"\1", trans)

        seq = repeating_sequence(trans.lower())

        if len(trans) < len(query) and len(seq) < 0.5 * len(trans):
            trans = seq * max(len(query) // len(seq), 1)
            trans = "".join(trans[i].upper() if query[i].isupper() else trans[i] for i in range(min(len(trans), len(query))))

        return trans

# Todo: Warning "Signature of method 'OfflineTranslator.many_func()' does not match signature of the base method in class 'ModelWrapper'"
class OfflineTranslator(CommonTranslator, ModelWrapper):
    async def _translate(self, *args, **kwargs):
        return await self.infer(*args, **kwargs)

    @abstractmethod
    async def _infer(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        pass

    async def load(self, from_lang: str, to_lang: str, device: str):
        return await super().load(device, *self.parse_language_codes(from_lang, to_lang))

    @abstractmethod
    async def _load(self, from_lang: str, to_lang: str, device: str):
        pass

    async def reload(self, from_lang: str, to_lang: str, device: str):
        return await super().reload(device, from_lang, to_lang)

    async def unload(self, device: str):
        return await super().unload()
