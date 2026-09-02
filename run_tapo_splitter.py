#!/usr/bin/env python3
"""Tapo OSD Splitter v1.1.0 single entry point."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
CONFIG = BASE / "tapo_splitter_config.json"
SETUP = BASE / "build_tapo_osd_glyph_templates.py"
PRODUCTION = BASE / "auto_tapo_split.py"
DEFAULT_RECDATA = BASE.parent / "rec_data"
DEFAULT_OUTPUT = BASE / "Output"
DEFAULT_TEMPLATE = BASE / "tapo_osd_glyph_templates.json"
DEFAULT_WORKERS = 3


def resolve(value):
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else BASE / path


def save_config(data):
    clean = {
        "version": 1,
        "recdata_dir": str(data["recdata_dir"]),
        "output_dir": str(data["output_dir"]),
        "workers": int(data.get("workers", DEFAULT_WORKERS)),
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
    return {"version": 1, "recdata_dir": data["recdata_dir"],
            "output_dir": data["output_dir"], "workers": workers}


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

    if not DEFAULT_TEMPLATE.exists():
        if not SETUP.exists():
            raise SystemExit(f"初期設定スクリプトがありません: {SETUP}")
        print("テンプレートJSONがないため、録画データから作成します。")
        subprocess.run([sys.executable, str(SETUP), "--input-dir", str(recdata),
                        "--output", str(DEFAULT_TEMPLATE)], check=True)
    if not DEFAULT_TEMPLATE.exists():
        raise SystemExit("テンプレートJSONが作成されなかったため中止しました。")

    print(f"録画データ: {recdata}")
    print(f"出力先: {output}")
    print(f"並列処理数: {data['workers']}")
    subprocess.run([sys.executable, str(PRODUCTION), "--input-dir", str(recdata),
                    "--output-dir", str(output), "--font", str(DEFAULT_TEMPLATE),
                    "--workers", str(data["workers"])], check=True)


if __name__ == "__main__":
    main()
