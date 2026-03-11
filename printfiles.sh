#!/bin/bash

# Controlla che ci siano almeno 2 argomenti (file + output)

# ./printfiles.sh outputfiles.txt main.py config.py thread_safe_buffer.py cyclist_speed.py live_viewer.py detector.py preprocessing/pre_process.py rs_lidar_stream.py rs_to_model_coords.py

if [ "$#" -lt 2 ]; then
    echo "Uso: $0 output.txt file1 file2 file3 ..."
    exit 1
fi

output="$1"
shift

# Svuota o crea il file di output
> "$output"

for file in "$@"; do
    if [ -f "$file" ]; then
        cat "$file" >> "$output"
        printf "\n\n" >> "$output"
    else
        echo "File non trovato: $file"
    fi
done

echo "Concatenazione completata in $output"