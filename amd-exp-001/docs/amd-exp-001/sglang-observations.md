```

| Measurement                               | SGLang result    | Meaning                                                                                                                  |
| ----------------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------ |
| KV-cache capacity                         | 1,048,891 tokens | Total capacity allocated for storing token information from active requests                                              |
| Cache memory                              | 96.02 GB         | GPU memory reserved for the KV cache: 48.01 GB for keys and 48.01 GB for values                                          |
| Maximum request length                    | 8,192 tokens     | Maximum combined prompt and generated-response length allowed for one request; validated documents fit within this limit |
| Available GPU memory after initialization | 34.82 GB         | GPU memory remaining after SGLang completes its startup allocations                                                      |
| Chunked prefill                           | 16,384 tokens    | Maximum configured size of a prompt-processing chunk                                                                     |
```
