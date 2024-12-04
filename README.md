# Pytorch-test
Tests for PyTorch api. 
Before channel_last:
![img_v3_02h8_93ca069c-1bb0-4326-adbc-87486f71af5g](https://github.com/user-attachments/assets/1be43f8b-6c48-4d5d-a5b3-cb7c4fe3737d)

After channel_last:
no pre-compile kernel, step time reduce 30%(compared to no-continus memory_format)
Pytorch said they reduce 22% steptime compared to continus memory_format.

![img_v3_02h8_7a01a9f1-40f1-4783-bbaf-497a2e2255ag](https://github.com/user-attachments/assets/d4e38de5-4edd-454b-87a6-d1c0764b7f79)


Usage:
https://pytorch.org/docs/stable/tensor_attributes.html#torch.memory_format
Tutorial:
https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html
Pytorch benchmark:
https://github.com/NVIDIA/apex/blob/ac8214ee6ba77c0037c693828e39d83654d25720/examples/imagenet/main_amp.py#L147C5-L147C57







