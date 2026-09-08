"""Profile and EliteFont pairing used by the public splitter."""

import json
import subprocess
from pathlib import Path


class ProfileError(RuntimeError):
    """Raised when a video cannot be mapped to exactly one active profile."""


DEFAULT_ROI = {"x": 0, "y": 0, "width": 950, "height": 70}
DEFAULT_SLOT_RANGES = (
    (7, 43), (50, 88), (99, 137), (148, 185),
    (247, 282), (290, 330), (391, 426), (435, 475),
    (530, 565), (579, 617), (675, 710), (724, 765),
    (819, 854), (867, 905),
)


def geometry_for_profile(profile=None):
    """Return validated ROI and slot geometry, with the C230 defaults as fallback."""
    osd = (profile or {}).get("osd", {})
    roi = dict(DEFAULT_ROI)
    roi.update(osd.get("roi", {}))
    if not all(isinstance(roi.get(key), int) for key in ("x", "y", "width", "height")):
        raise ProfileError("ProfileのOSD ROIが不正です")
    raw_slots = osd.get("slot_ranges", DEFAULT_SLOT_RANGES)
    try:
        slots = tuple((int(pair[0]), int(pair[1])) for pair in raw_slots)
    except (TypeError, ValueError, IndexError) as exc:
        raise ProfileError("Profileのslot_rangesが不正です") from exc
    if len(slots) != 14 or any(left < 0 or right <= left or right > roi["width"]
                               for left, right in slots):
        raise ProfileError("Profileのslot_rangesはROI内の14個の範囲が必要です")
    return {"roi": roi, "slot_ranges": slots}


def load_profiles(path):
    path = Path(path).expanduser().resolve()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileError(f"Profile設定を読めません: {path} ({exc})") from exc
    profiles = data.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ProfileError(f"Profileが登録されていません: {path}")
    for profile in profiles:
        if not isinstance(profile, dict):
            raise ProfileError(f"Profileの形式が不正です: {path}")
        for key in ("id", "active", "resolution", "font_file"):
            if key not in profile:
                raise ProfileError(f"Profileに{key}がありません: {path}")
        if not isinstance(profile["active"], bool):
            raise ProfileError(f"Profileのactiveはtrue/falseで指定してください: {profile.get('id')}")
        resolution = profile["resolution"]
        if not isinstance(resolution, dict) or not all(
                isinstance(resolution.get(key), int) for key in ("width", "height")):
            raise ProfileError(f"Profileの解像度が不正です: {profile.get('id')}")
    return path, data, profiles


def probe_resolution(video):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "json", str(video)],
        check=True, capture_output=True, text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ProfileError(f"映像ストリームが見つかりません: {video}")
    return int(streams[0]["width"]), int(streams[0]["height"])


def font_path_for_profile(profile, registry_path):
    registry = Path(registry_path).expanduser().resolve()
    font = Path(profile["font_file"]).expanduser()
    if not font.is_absolute():
        font = registry.parent / font
    return font.resolve()


def validate_font_profile(font_path, profile_id):
    """Validate an explicit pairing; old JSON without metadata remains usable."""
    try:
        data = json.loads(Path(font_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileError(f"Font JSONを読めません: {font_path} ({exc})") from exc
    embedded = data.get("profile_id")
    if embedded and embedded != profile_id:
        raise ProfileError(
            f"ProfileとFontの組み合わせが違います: Profile={profile_id}, Font={embedded}"
        )
    return embedded


def resolve_profile(video, registry_path, require_font=True):
    registry, _data, profiles = load_profiles(registry_path)
    width, height = probe_resolution(video)
    matches = [profile for profile in profiles
               if profile["active"]
               and profile["resolution"]["width"] == width
               and profile["resolution"]["height"] == height]
    if not matches:
        active = [profile["id"] for profile in profiles if profile["active"]]
        raise ProfileError(
            f"{video.name} ({width}x{height})に一致するactive Profileがありません。"
            f" active={active}。--setupでProfileを確認してください。"
        )
    if len(matches) > 1:
        ids = ", ".join(profile["id"] for profile in matches)
        raise ProfileError(
            f"{video.name} ({width}x{height})に一致するactive Profileが複数あります: {ids}。"
            "どれか1つだけactiveにしてください。"
        )
    profile = dict(matches[0])
    font = font_path_for_profile(profile, registry)
    if require_font and not font.is_file():
        raise ProfileError(
            f"Profile '{profile['id']}' のFont JSONがありません: {font}"
        )
    if font.is_file():
        profile["font_profile_id"] = validate_font_profile(font, profile["id"])
    profile["font_path"] = str(font)
    profile["geometry"] = geometry_for_profile(profile)
    profile["video_resolution"] = {"width": width, "height": height}
    profile["registry_path"] = str(registry)
    return profile


def load_profile_by_id(registry_path, profile_id, require_font=True):
    registry, _data, profiles = load_profiles(registry_path)
    matches = [profile for profile in profiles if profile["id"] == profile_id]
    if len(matches) != 1:
        raise ProfileError(f"Profile IDが見つからないか重複しています: {profile_id}")
    profile = dict(matches[0])
    font = font_path_for_profile(profile, registry)
    if require_font and not font.is_file():
        raise ProfileError(f"Profile '{profile_id}' のFont JSONがありません: {font}")
    if font.is_file():
        profile["font_profile_id"] = validate_font_profile(font, profile_id)
    profile["font_path"] = str(font)
    profile["geometry"] = geometry_for_profile(profile)
    profile["registry_path"] = str(registry)
    return profile
