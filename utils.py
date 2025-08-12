import os
import re
import torch
import asyncio
from log_config import logger
from langdetect import detect, LangDetectException
from googletrans import Translator

translator = Translator()

def check_compute_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
            logger.info(f"{os.path.basename(__file__)}: GPU CUDA hỗ trợ BF16, sử dụng BF16")
        else:
            compute_dtype = torch.float16
            logger.info(f"{os.path.basename(__file__)}: GPU CUDA không hỗ trợ BF16, sử dụng FP16")
    else:
        if torch.backends.mps.is_available():
            logger.info(f"{os.path.basename(__file__)}: GPU MPS không hỗ trợ FP16, sử dụng FP32")
        else:
            logger.info(f"{os.path.basename(__file__)}: Không có GPU, sử dụng FP32")
        compute_dtype = torch.float32
    return compute_dtype

async def split_sentence(text: str, max_length: int = 496) -> list[str]:
    try:
        lang = detect(text)
    except LangDetectException:
        lang = 'en'
    if len(text) <= max_length:
        if lang == 'en':
            return {'original': [text], 'translated': [text]}
        else:
            translated = await translator.translate(text, src=lang, dest='en')
            return {'original': [text], 'translated': [translated.text]}

    sentences = re.findall(r'[^\.!\?;]+[\.!\?;]?', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        source_chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
    else:
        source_chunks = []
        current_group = ""
        for sentence in sentences:
            if not current_group:
                current_group = sentence
                continue
            if len(current_group) + len(sentence) + 1 <= max_length:
                current_group += " " + sentence
            else:
                source_chunks.append(current_group)
                current_group = sentence
        if current_group:
            source_chunks.append(current_group)
    if lang != 'en':
        translated_chunks = []
        for chunk in source_chunks:
            translated_result = await translator.translate(chunk, src=lang, dest='en')
            translated_chunks.append(translated_result.text)       
        return {'original': source_chunks, 'translated': translated_chunks}
    else:
        return {'original': source_chunks, 'translated': source_chunks}