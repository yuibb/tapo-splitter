#!/usr/bin/env python3
"""Tapo OSD Splitter - single entry point.

Configured installations go straight to splitting. On the first run, or
when a required setting is missing, the setup conversation runs first.
The OSD template JSON is generated from the user's own recordings; no
font data is bundled. Hardware acceleration is not forced.
"""

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


def resolve(value):
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else BASE / path


def ask_path(label, default):
    answer = input(f"{label} [{default}]: ").strip()
    return resolve(answer) if answer else default


def load_config():
    if CONFIG.exists():
        try:
            data = json.loads(CONFIG.read_text(encoding="utf-8"))
            keys = ("recdata_dir", "output_dir")
            if all(str(data.get(key, "")).strip() for key in keys):
                # The template is always managed beside this launcher. Do not
                # ask the user to locate it on every first-run setup.
                data["template_json"] = str(DEFAULT_TEMPLATE)
                return data
        except (OSError, json.JSONDecodeError):
            pass

    print("初回設定を行います。")
    recdata = ask_path("録画データフォルダ", DEFAULT_RECDATA)
    output = ask_path("分割結果の出力フォルダ", DEFAULT_OUTPUT)
    data = {
        "version": 1,
        "recdata_dir": str(recdata),
        "output_dir": str(output),
        "template_json": str(DEFAULT_TEMPLATE),
    }
    CONFIG.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data


def main():
    data = load_config()
    recdata = resolve(data["recdata_dir"])
    output = resolve(data["output_dir"])
    template = resolve(data["template_json"])
    recdata.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    if not template.exists():
        if not SETUP.exists():
            raise SystemExit(f"初期設定スクリプトがありません: {SETUP}")
        print("テンプレートJSONがないため、録画データから作成します。")
        subprocess.run(
            [sys.executable, str(SETUP), "--output", str(template)],
            input=f"{recdata}\n", text=True, check=True,
        )
    if not template.exists():
        raise SystemExit("テンプレートJSONが作成されなかったため中止しました。")
    if not PRODUCTION.exists():
        raise SystemExit(f"本番スクリプトがありません: {PRODUCTION}")

    print(f"録画データ: {recdata}")
    print(f"出力先: {output}")
    print(f"テンプレート: {template}")
    subprocess.run(
        [sys.executable, str(PRODUCTION), "--input-dir", str(recdata),
         "--output-dir", str(output), "--font", str(template)],
        check=True,
    )


if __name__ == "__main__":
    main()
