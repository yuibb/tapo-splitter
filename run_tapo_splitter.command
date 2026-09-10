#!/bin/zsh

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

if (( $# == 0 )); then
  RUN_ARGS=(--menu)
else
  RUN_ARGS=("$@")
fi

if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
  "$SCRIPT_DIR/.venv/bin/python" run_tapo_splitter.py "${RUN_ARGS[@]}"
elif [[ -x "$SCRIPT_DIR/../.c230-venv/bin/python" ]]; then
  "$SCRIPT_DIR/../.c230-venv/bin/python" run_tapo_splitter.py "${RUN_ARGS[@]}"
elif command -v python3 >/dev/null 2>&1; then
  python3 run_tapo_splitter.py "${RUN_ARGS[@]}"
elif command -v python >/dev/null 2>&1; then
  python run_tapo_splitter.py "${RUN_ARGS[@]}"
else
  echo "Python 3が見つかりません。READMEの手順でPythonをインストールしてください。"
  exit 1
fi

exit_code=$?
echo
if (( exit_code != 0 )); then
  echo "処理が失敗しました（終了コード: $exit_code）。"
else
  echo "処理が完了しました。"
fi
read -r "?Enterキーを押すと閉じます。"
exit $exit_code
