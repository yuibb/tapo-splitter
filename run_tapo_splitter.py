#!/usr/bin/env python3
"""Tapo OSD Splitter v1.3.4 single entry point."""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from tapo_profile import (
    ProfileError,
    load_profile_by_id,
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
BACKUP_ROOT = BASE / "backup"


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


def create_backup(paths, label):
    """Copy local generated state before a user-requested rebuild."""
    sources = [Path(path) for path in paths if Path(path).is_file()]
    if not sources:
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = BACKUP_ROOT / f"{stamp}_{label}"
    suffix = 2
    while destination.exists():
        destination = BACKUP_ROOT / f"{stamp}_{label}_{suffix:02d}"
        suffix += 1
    destination.mkdir(parents=True)
    for source in sources:
        shutil.copy2(source, destination / source.name)
    print(f"既存データをバックアップしました: {destination}")
    return destination


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


def choose_profile(profiles_file, active_only=False):
    """Prompt for one registered Profile without exposing JSON editing."""
    _registry, _data, profiles = load_profiles(profiles_file)
    choices = [profile for profile in profiles if profile["active"] or not active_only]
    if not choices:
        raise SystemExit("選択できるactive Profileがありません。--profilesで設定してください。")
    print("\nProfileを選んでください。")
    for index, profile in enumerate(choices, 1):
        resolution = profile["resolution"]
        state = "使用中" if profile["active"] else "停止中"
        name = profile.get("display_name", profile["id"])
        print(f"  {index}. {name} [{profile['id']}] "
              f"{resolution['width']}x{resolution['height']} / {state}")
    while True:
        answer = input("番号（Enter=中止）: ").strip()
        if not answer:
            return None
        try:
            return choices[int(answer) - 1]
        except (ValueError, IndexError):
            print("表示された番号を入力してください。")


def switch_profile(profiles_file):
    """Activate one Profile and deactivate its same-resolution peers."""
    registry, data, _profiles = load_profiles(profiles_file)
    selected = choose_profile(registry)
    if selected is None:
        print("Profileの切替を中止しました。")
        return
    for profile in data["profiles"]:
        if profile["resolution"] == selected["resolution"]:
            profile["active"] = profile["id"] == selected["id"]
    data["default_profile"] = selected["id"]
    registry.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"使用するProfileを切り替えました: {selected['id']}")
    print("同じ解像度の他Profileは停止中にしました。")


def build_font_for_profile(recdata, profiles_file, profile_id, backup_label):
    """Rebuild one Profile's generated EliteFont after taking a safe copy."""
    try:
        profile = load_profile_by_id(profiles_file, profile_id, require_font=False)
    except ProfileError as exc:
        raise SystemExit(str(exc)) from exc
    font = Path(profile["font_path"])
    create_backup([font], backup_label)
    font.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, str(SETUP), "--input-dir", str(recdata),
        "--output", str(font), "--fresh", "--profile-id", profile_id,
        "--profile-file", str(profiles_file),
    ], check=True)


def profiles_for_recordings(recdata, profiles_file):
    videos = sorted(recdata.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"録画データフォルダにMP4がありません: {recdata}")
    found = {}
    for video in videos:
        try:
            profile = resolve_profile(video, profiles_file, require_font=False)
        except ProfileError as exc:
            raise SystemExit(str(exc)) from exc
        found[profile["id"]] = profile
    return list(found.values())


def run_full_setup(existing):
    """Repeat first-run configuration, then rebuild fonts for recorded Profiles."""
    create_backup([CONFIG], "setup")
    data = run_first_setup() if existing is None else run_setup(existing)
    recdata = resolve(data["recdata_dir"])
    profiles_file = resolve(data.get("profiles_file", DEFAULT_PROFILES))
    for profile in profiles_for_recordings(recdata, profiles_file):
        build_font_for_profile(recdata, profiles_file, profile["id"], "elitefont_rebuild")
    return data


def run_rebuild_font(data):
    recdata = resolve(data["recdata_dir"])
    profiles_file = resolve(data.get("profiles_file", DEFAULT_PROFILES))
    if not recdata.is_dir():
        raise SystemExit(f"録画データフォルダがありません: {recdata}")
    selected = choose_profile(profiles_file, active_only=True)
    if selected is None:
        print("数字の読み取りデータの作り直しを中止しました。")
        return
    build_font_for_profile(recdata, profiles_file, selected["id"], "elitefont_rebuild")


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
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--menu", action="store_true", help="説明付きの起動メニューを表示する")
    actions.add_argument("--setup", action="store_true",
                         help="設定と対象の数字の読み取りデータをバックアップして初期設定をやり直す")
    actions.add_argument("--rebuild-font", action="store_true",
                         help="設定を維持し、選択したProfileの数字の読み取りデータだけを作り直す")
    actions.add_argument("--profiles", action="store_true",
                         help="登録済みProfileの使用中切替を行う")
    args = parser.parse_args()
    if not PRODUCTION.exists():
        raise SystemExit(f"本番スクリプトがありません: {PRODUCTION}")

    data = load_config()
    if args.menu:
        print("\nTapo OSD Splitter\n")
        print("1. 録画を分割する")
        print("   保存済み設定で、未処理の動画を通常処理します。")
        print("2. 初期セットアップをやり直す")
        print("   設定と数字の読み取りデータをバックアップして、初期設定をやり直します。")
        print("3. 数字の読み取りデータを作り直す")
        print("   設定は維持し、選択したProfileの読み取りデータだけを作り直します。")
        print("4. 読み取り設定を切り替える")
        print("   登録済みProfileから、同じ解像度で使う設定を選びます。")
        print("0. 終了")
        answer = input("\n番号を選んでください: ").strip()
        if answer == "0":
            return
        if answer == "2":
            run_full_setup(data)
            return
        if answer == "3":
            if data is None:
                data = run_first_setup()
            run_rebuild_font(data)
            return
        if answer == "4":
            profiles_file = resolve((data or {}).get("profiles_file", DEFAULT_PROFILES))
            switch_profile(profiles_file)
            return
        if answer != "1":
            raise SystemExit("0〜4の番号を入力してください。")
    elif args.setup:
        run_full_setup(data)
        return
    elif args.rebuild_font:
        if data is None:
            data = run_first_setup()
        run_rebuild_font(data)
        return
    elif args.profiles:
        profiles_file = resolve((data or {}).get("profiles_file", DEFAULT_PROFILES))
        switch_profile(profiles_file)
        return
    elif data is None:
        data = run_first_setup()

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
