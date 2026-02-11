from __future__ import annotations

from pathlib import Path
from typing import Literal
from typing import Optional
from urllib.parse import parse_qs
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from langchain.tools import tool

MEDIA_TYPES = ("image", "audio", "video")
IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".svg",
}
AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".ogg",
    ".m4a",
    ".aac",
    ".flac",
    ".opus",
}
VIDEO_EXTENSIONS = {
    ".mp4",
    ".webm",
    ".mov",
    ".mkv",
    ".avi",
    ".m4v",
    ".mpg",
    ".mpeg",
}


class ShowMediaInput(BaseModel):
    source: str = Field(
        description="Local file path or URL to an audio, video, or image resource."
    )
    media_type: Optional[Literal["image", "audio", "video"]] = Field(
        default=None,
        description="Optional override for the media type when it cannot be inferred.",
    )


class MediaContent(BaseModel):
    type: Literal["image", "audio", "video"] = Field(
        description="Media type for Streamlit rendering."
    )
    url: str = Field(
        description="URL or local file path accepted by Streamlit (st.image/audio/video)."
    )


class ShowMediaOutput(BaseModel):
    media_content: MediaContent


def _infer_type_from_source(source: str) -> Optional[str]:
    parsed = urlparse(source)
    candidates = []
    if parsed.scheme in ("http", "https"):
        candidates.append(parsed.path)
        query = parse_qs(parsed.query)
        query_keys = (
            "filename",
            "file",
            "name",
            "path",
            "url",
            "src",
            "source",
            "media",
            "asset",
            "attachment",
        )
        for key in query_keys:
            values = query.get(key, [])
            for value in values:
                if value:
                    candidates.append(urlparse(value).path)
    else:
        candidates.append(source)

    for candidate in candidates:
        suffix = Path(candidate).suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            return "image"
        if suffix in AUDIO_EXTENSIONS:
            return "audio"
        if suffix in VIDEO_EXTENSIONS:
            return "video"
    return None


@tool(args_schema=ShowMediaInput)
def tool_show_media(source: str, media_type: Optional[str] = None) -> str:
    """Displays media for the user.
    It is not necessary to inform anything - the interface will automatically render the contents of the file.
    Do not try to display the files manually.
    """
    inferred = media_type or _infer_type_from_source(source)
    if inferred is None:
        return (
            "Unable to infer media type from source. "
            f"Retry with media_type set to one of: {list(MEDIA_TYPES)}"
        )
    if inferred not in MEDIA_TYPES:
        return f"media_type must be one of {list(MEDIA_TYPES)}"
    output = ShowMediaOutput(media_content=MediaContent(type=inferred, url=source))
    return output.model_dump_json()
