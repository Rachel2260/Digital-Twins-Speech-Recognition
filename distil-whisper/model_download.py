from huggingface_hub import snapshot_download

save_directory = "./model"

snapshot_download(repo_id="distil-whisper/distil-medium.en", local_dir=save_directory)
