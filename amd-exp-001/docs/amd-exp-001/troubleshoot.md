### Experiment : GPU Cache

- Device : MIX300

- Error :
docker pulled the image and exited the container in 2 minutes
GPU_ARCHS was not explicitly set when the container was created.
MI300X uses the gfx942 architecture.An unset value normally defaults to automatic detection, so the empty value in the crash may have been introduced later during startup. 

- Solution : 
 Explicitly setting GPU_ARCHS=gfx942 resolved the startup problem.

### vllm : Engine failed 

Error : unsupported bfloat16 KV-cache 

Solution : vLLM rejected the unsupported bfloat16 KV-cache setting; changing it to auto allowed it to use the model’s BF16 format and successfully allocate 80,000 cache tokens.
