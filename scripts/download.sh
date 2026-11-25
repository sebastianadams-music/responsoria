#!/usr/bin/env bash
set -euo pipefail

mkdir -p raw_audio

TRACKS=(
  "01|YmWBOsLh2x4|I. In monte Oliveti"
  "02|uzwAxHjrelY|II. Tristis est anima mea"
  "03|K41Bk9cfoJo|III. Ecce vidimus eum"
  "04|yT5hIkms_bo|IV. Amicus meus"
  "05|En1ZHTV5hcA|V. Judas mercator pessimus"
  "06|WtlIPBCDsw0|VI. Unus ex discipulis meis"
  "07|E_8oSyfEDH4|VII. Eram quasi agnus innocens"
  "08|sSMNoO1fpIw|VIII. Una hora"
  "09|DgM3SsOdkl4|IX. Seniores populi"
)

download_one() {
  local id="$1"
  local url="https://www.youtube.com/watch?v=${id}"
  local out_tpl="$2"

  if uvx yt-dlp --no-update --js-runtimes node --remote-components ejs:github -f "bestaudio/best" -x --audio-format wav --audio-quality 0 -o "$out_tpl" "$url"; then
    return 0
  fi

  echo "uvx yt-dlp failed for $id, falling back to /usr/local/bin/yt-dlp..." >&2
  /usr/local/bin/yt-dlp --no-update --js-runtimes node -f "bestaudio/best" -x --audio-format wav --audio-quality 0 -o "$out_tpl" "$url"
}

for entry in "${TRACKS[@]}"; do
  IFS="|" read -r num id title <<< "$entry"
  out="raw_audio/${num}.wav"
  if [[ -f "$out" && -s "$out" ]]; then
    echo "Skipping ${out} (${title}) - already exists"
    continue
  fi
  echo "Downloading ${num}: ${title} (${id})..."
  download_one "$id" "raw_audio/${num}.%(ext)s"
done

echo "--- Verification ---"
total_dur=0
for f in raw_audio/0{1..9}.wav; do
  if [[ ! -f "$f" ]]; then
    echo "MISSING: $f" >&2
    exit 1
  fi
  dur=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$f")
  echo "$f : ${dur}s"
  total_dur=$(awk -v a="$total_dur" -v b="$dur" 'BEGIN{printf "%.2f", a+b}')
done
echo "Total duration: ${total_dur}s"
