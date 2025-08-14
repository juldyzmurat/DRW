source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

echo "🚀 Starting chunk processing..."

for i in $(seq 0 99); do
    chunk_id=$(printf "%03d" $i)
    echo "⚙️  Starting chunk $chunk_id..."
    
    python3 process_chunk.py "$chunk_id"
    status=$?

    if [ $status -eq 0 ]; then
        echo "✅ Finished chunk $chunk_id."
    else
        echo "❌ Error in chunk $chunk_id! Exit code: $status"
        exit 1
    fi

    sleep 1
done


echo ""
echo "✅ All chunks processed!"
