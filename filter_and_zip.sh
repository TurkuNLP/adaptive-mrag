#!/bin/bash

# Variables
SOURCE_ZIP="2021-news.zip"
OUTPUT_DIR="stella-output"
TEMP_DIR="temp_extract"
ZIP_FILE="stella-news.zip"

# Create a temporary directory
mkdir -p "$TEMP_DIR"

# Extract all .txt files from the source zip into the temp directory
# Disable zip bomb detection
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -j "$SOURCE_ZIP" '*.txt' -d "$TEMP_DIR"

# Get a list of .pkl base names in the output directory
mapfile -t PKL_FILES < <(find "$OUTPUT_DIR" -name '*.pkl' -exec basename {} .pkl \;)

# Initialize counters
total_files=0
included_files=0

# Iterate through all extracted .txt files
for TXT_FILE in "$TEMP_DIR"/*.txt; do
    total_files=$((total_files + 1))
    BASENAME=$(basename "$TXT_FILE" .txt)
    # Check if a .pkl file with the same name exists
    if [[ " ${PKL_FILES[*]} " =~ " $BASENAME " ]]; then
        # Remove the .txt file if a matching .pkl exists
        rm "$TXT_FILE"
    else
        included_files=$((included_files + 1))
    fi
done

# Create a zip file containing the remaining .txt files
zip -j "$ZIP_FILE" "$TEMP_DIR"/*.txt

# Cleanup temporary directory
rm -rf "$TEMP_DIR"

# Display summary
echo "Total files processed: $total_files"
echo "Files included in $ZIP_FILE: $included_files"
echo "Filtered .txt files have been zipped into $ZIP_FILE"
