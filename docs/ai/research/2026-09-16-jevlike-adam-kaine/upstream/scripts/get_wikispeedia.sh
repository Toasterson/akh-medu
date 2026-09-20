#!/bin/sh
set -eu

root=${1:-data/wikispeedia}
base=https://snap.stanford.edu/data/wikispeedia
mkdir -p "$root"

for archive in wikispeedia_paths-and-graph.tar.gz wikispeedia_articles_plaintext.tar.gz; do
  if [ ! -f "$root/$archive" ]; then
    curl -fL "$base/$archive" -o "$root/$archive"
  fi
done

if [ ! -d "$root/wikispeedia_paths-and-graph" ]; then
  tar -xzf "$root/wikispeedia_paths-and-graph.tar.gz" -C "$root"
fi
if [ ! -d "$root/plaintext_articles" ]; then
  tar -xzf "$root/wikispeedia_articles_plaintext.tar.gz" -C "$root"
fi

jevlike-data wikispeedia --root "$root" --output "$root/jsonl"
echo "Built $root/jsonl. Cite West and Leskovec, WWW 2012."
