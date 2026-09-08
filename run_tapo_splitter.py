#!/usr/bin/env python3
"""Tapo OSD Splitter v1.3.0 single entry point."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tapo_profile import (
    ProfileError,
    load_profiles,
    resolve_profile,
    validate_font_profile,
)

BASE = Path(__file__).resolve().parent
CONFIG = BASE / "tapo_splitter_config.json"
SETUP = BASE / "build_tapo_osd_glyph_templates.py"
PRODUCTION = BASE / "auto_tapo_split.py"
DEFAULT_PROFILES = BASE / "tapo_profiles.json"
DEFAULT_RECDATA = BASE.parent / "rec_data"
DEFAULT_OUTPUT = BASE / "Output"
DEFAULT_WORKERS = 3


def resolve(value):
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else BASE / path


def save_config(data):
    clean = {
        "version": 2,
        "recdata_dir": str(data["recdata_dir"]),
        "output_dir": str(data["output_dir"]),
        "workers": int(data.get("workers", DEFAULT_WORKERS)),
        "profiles_file": str(data.get("profiles_file", DEFAULT_PROFILES)),
    }
    CONFIG.write_text(json.dumps(clean, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return clean


def load_config():
    if not CONFIG.exists():
        return None
    try:
        data = json.loads(CONFIG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"設定ファイルを読めません。初回設定を行います: {exc}")
        return None
    if not all(str(data.get(key, "")).strip() for key in ("recdata_dir", "output_dir")):
        print("設定ファイルに必要な項目がありません。初回設定を行います。")
        return None
    workers = data.get("workers", DEFAULT_WORKERS)
    if not isinstance(workers, int) or workers < 1:
        workers = DEFAULT_WORKERS
    profiles_file = data.get("profiles_file", str(DEFAULT_PROFILES))
    return {"version": 2, "recdata_dir": data["recdata_dir"],
            "output_dir": data["output_dir"], "workers": workers,
            "profiles_file": profiles_file}


def ask_path(label, default):
    answer = input(f"{label} [{default}]: ").strip()
    return str(resolve(answer)) if answer else str(default)


def ask_workers(current=DEFAULT_WORKERS, optional=False):
    while True:
        prompt = "新しい値（Enterで変更なし）: " if optional else f"Workers（並列処理数） [{current}]: "
        answer = input(prompt).strip()
        if not answer:
            return current
        try:
            value = int(answer)
        except ValueError:
            print("1以上の整数を入力してください。")
            continue
        if value < 1:
            print("1以上の整数を入力してください。")
            continue
        return value


def ask_optional_path(label, current):
    print()
    print(label)
    print(f"現在: {current}")
    answer = input("新しい値（Enterで変更なし）: ").strip()
    return str(resolve(answer)) if answer else str(current)


def run_first_setup():
    print("初回設定を行います。")
    data = {
        "recdata_dir": ask_path("録画データフォルダ", DEFAULT_RECDATA),
        "output_dir": ask_path("分割結果の出力フォルダ", DEFAULT_OUTPUT),
        "workers": ask_workers(),
        "profiles_file": str(DEFAULT_PROFILES),
    }
    return save_config(data)


def run_setup(data):
    print("設定を変更します。変更しない項目はEnterのみ押してください。")
    data["recdata_dir"] = ask_optional_path("録画データフォルダ", data["recdata_dir"])
    data["output_dir"] = ask_optional_path("出力フォルダ", data["output_dir"])
    print()
    print("Workers（並列処理数）")
    print(f"現在: {data['workers']}")
    data["workers"] = ask_workers(data["workers"], optional=True)
    saved = save_config(data)
    print("設定を保存しました。")
    return saved


def ensure_profile_fonts(recdata, profiles_file):
    """Build missing Fonts for the Profiles represented by the input videos."""
    videos = sorted(recdata.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"録画データフォルダにMP4がありません: {recdata}")

    checked = set()
    for video in videos:
        try:
            profile = resolve_profile(video, profiles_file, require_font=False)
        except ProfileError as exc:
            raise SystemExit(str(exc)) from exc
        if profile["id"] in checked:
            continue
        checked.add(profile["id"])
        font = Path(profile["font_path"])
        if not font.is_file():
            print(f"Profile '{profile['id']}' のFontがないため作成します: {font}")
            font.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                sys.executable, str(SETUP), "--input-dir", str(recdata),
                "--output", str(font), "--profile-id", profile["id"],
                "--profile-file", str(profiles_file),
            ], check=True)
        if not font.is_file():
            raise SystemExit(f"Profile用Fontが作成されませんでした: {font}")
        try:
            embedded = validate_font_profile(font, profile["id"])
        except ProfileError as exc:
            raise SystemExit(str(exc)) from exc
        if embedded is None:
            print(f"注意: 既存の旧Font JSONです（profile_idなし）: {font}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", action="store_true",
                        help="録画フォルダ・出力フォルダ・並列処理数を変更する")
    args = parser.parse_args()
    if not PRODUCTION.exists():
        raise SystemExit(f"本番スクリプトがありません: {PRODUCTION}")

    data = load_config()
    if data is None:
        data = run_first_setup()
    elif args.setup:
        data = run_setup(data)

    recdata = resolve(data["recdata_dir"])
    output = resolve(data["output_dir"])
    if not recdata.is_dir():
        raise SystemExit(f"録画データフォルダがありません: {recdata}")
    output.mkdir(parents=True, exist_ok=True)

    print(f"録画データ: {recdata}")
    print(f"出力先: {output}")
    print(f"並列処理数: {data['workers']}")
    profiles_file = resolve(data.get("profiles_file", DEFAULT_PROFILES))
    try:
        _registry, _profile_data, profiles = load_profiles(profiles_file)
    except ProfileError as exc:
        raise SystemExit(str(exc)) from exc
    active = [profile["id"] for profile in profiles if profile.get("active")]
    if not active:
        raise SystemExit("activeなProfileがありません。tapo_profiles.jsonを確認してください。")
    print(f"Profile: {', '.join(active)}")
    if not SETUP.exists():
        raise SystemExit(f"初期設定スクリプトがありません: {SETUP}")
    ensure_profile_fonts(recdata, profiles_file)
    subprocess.run([sys.executable, str(PRODUCTION), "--input-dir", str(recdata),
                    "--output-dir", str(output), "--profiles", str(profiles_file),
                    "--workers", str(data["workers"])], check=True)


if __name__ == "__main__":
    main()
