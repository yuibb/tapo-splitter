#!/bin/zsh

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

if command -v python3 >/dev/null 2>&1; then
  python3 run_tapo_splitter.py "$@"
elif command -v python >/dev/null 2>&1; then
  python run_tapo_splitter.py "$@"
else
  echo "Python 3が見つかりません。READMEの手順でPythonをインストールしてください。"
  exit 1
fi

status=$?
echo
if (( status != 0 )); then
  echo "処理が失敗しました（終了コード: $status）。"
else
  echo "処理が完了しました。"
fi
read -r "?Enterキーを押すと閉じます。"
exit $status
