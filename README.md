

🎯 CLI Command Q&A Dataset and Mistral-7B Fine-Tuning
This project involves creating a specialized Q&A dataset for command-line commands (focused on Git) and fine-tuning the Mistral-7B model (using 4-bit quantization) on this dataset. The end result is a lightweight, fine-tuned model that can answer CLI questions efficiently.

📜 Project Overview
Built a dataset of approximately 200 questions related to Git and common CLI operations.

Sourced examples directly from the Hugging Face datasets platform.

Applied supervised fine-tuning (SFT) on the mistralai/Mistral-7B model with 4-bit quantization to reduce memory footprint.

Achieved a highly compact and efficient model that can generate accurate answers to Git CLI questions.

⚙️ Features
✅ 200+ diverse Q&A pairs about Git commands and CLI usage.

✅ Lightweight training using quantization for resource efficiency.

✅ Ready for integration into local CLI-based chatbots or other interactive tools.

✅ Powered by the state-of-the-art Mistral-7B architecture.

🧠 Model
Fine-tuned mistralai/Mistral using supervised fine-tuning

Quantization: 4-bit (using bitsandbytes)

Training data: Custom CLI Q&A dataset

Training setup: HuggingFace transformers + peft

