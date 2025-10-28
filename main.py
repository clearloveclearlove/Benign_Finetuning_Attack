from huggingface_hub import hf_hub_download

model_id = "TheBloke/Llama-2-7B-Chat-fp16"  # 模型ID
local_dir = "/autodl-tmp/Llama-2-7B-Chat-fp16"  # 本地存储路径
filename = "pytorch_model.bin"  # 需要下载的文件名

# 下载模型文件

