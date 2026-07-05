#!/bin/sh
set -eu

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
CSV_DIR="/data"
DB_FILE="/db/sql_agent.db"
HASH_DIR="/db/.hashes"

# ------------------------------------------------------------------
# Import a CSV file into a DuckDB table, idempotently.
#
# Arguments:
#   $1 - Path to the CSV file
#   $2 - Target table name
#   $3 - Path to the DuckDB database file
#   $4 - Directory to store per-table hash files
#
# The function hashes the CSV with md5sum.  If the stored hash
# matches, the import is skipped.  Otherwise the table is created
# (or replaced) via read_csv_auto() and the new hash is saved.
# ------------------------------------------------------------------
import_csv_to_table() {
    local csv_file="$1"
    local table_name="$2"
    local db_file="$3"
    local hash_dir="$4"

    local hash_file="${hash_dir}/.${table_name}_hash"

    # Validate CSV exists
    if [ ! -f "$csv_file" ]; then
        echo "  [ERROR]  CSV file not found: $csv_file"
        return 1
    fi

    # Compute current hash
    local current_hash
    current_hash=$(md5sum "$csv_file" | awk '{print $1}')

    # Check idempotency
    if [ -f "$db_file" ] && [ -f "$hash_file" ]; then
        local stored_hash
        stored_hash=$(cat "$hash_file")
        if [ "$current_hash" = "$stored_hash" ]; then
            local short_cur
            short_cur=$(echo "$current_hash" | cut -c1-8)
            echo "  [SKIP]   ${table_name}  (hash ${short_cur}… matches)"
            return 0
        fi
        local short_old short_cur
        short_old=$(echo "$stored_hash" | cut -c1-8)
        short_cur=$(echo "$current_hash" | cut -c1-8)
        echo "  [UPDATE] ${table_name}  (hash changed: ${short_old}… → ${short_cur}…)"
    else
        echo "  [CREATE] ${table_name}"
    fi

    # Import via DuckDB
    duckdb "$db_file" << EOF
CREATE OR REPLACE TABLE ${table_name} AS
SELECT * FROM read_csv_auto('${csv_file}');
EOF

    local rc=$?
    if [ $rc -eq 0 ]; then
        mkdir -p "$hash_dir"
        echo "$current_hash" > "$hash_file"
        echo "  [DONE]   ${table_name}  (imported successfully)"
    else
        echo "  [FAIL]   ${table_name}  (duckdb exited with code $rc)"
    fi
    return $rc
}

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
echo "=== DuckDB Loader ==="
echo "Database: ${DB_FILE}"
echo ""

# Ensure output directories exist
mkdir -p "$(dirname "$DB_FILE")" "$HASH_DIR"

# Import each CSV into its own table
import_csv_to_table "${CSV_DIR}/transactions.csv" "transactions" "$DB_FILE" "$HASH_DIR"
import_csv_to_table "${CSV_DIR}/seller.csv"      "seller"      "$DB_FILE" "$HASH_DIR"
import_csv_to_table "${CSV_DIR}/territory.csv"   "territory"   "$DB_FILE" "$HASH_DIR"

echo ""
echo "=== Loader finished ==="
