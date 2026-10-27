# Publish an adapter to Hugging Face

Training stays on your Mac. When you decide to share the resulting adapter, the upload flow can publish supported adapter outputs to Hugging Face without leaving Studio.

<div class="doc-screenshot">
  <img src="https://raw.githubusercontent.com/Goekdeniz-Guelmez/MLX-LoRA-Studio/main/Sources/Media/app_upload.png" alt="Full-screen Hugging Face upload page with separate model and dataset upload workflows and Hub settings" loading="lazy">
</div>

## Review before publishing

1. Choose a completed run and confirm the adapter files are the ones you intend to share.
2. Authenticate with Hugging Face and choose a repository name and visibility.
3. Review the model card fields, license selection, and metadata.
4. Start the upload and follow its progress in the app.

The upload uses the network and sends the selected adapter and metadata to the repository you chose. Review repository visibility and the terms that apply to your model and training data before publishing. Nothing is uploaded as part of a local training run.

Full fine-tuning produces a different kind of output from an adapter; the in-app upload flow is for supported adapter-based workflows such as LoRA, DoRA, QLoRA, and QAT.

[See the run archive](Runs) · [Read the Hugging Face upload notes](https://github.com/Goekdeniz-Guelmez/MLX-LoRA-Studio#publish)
