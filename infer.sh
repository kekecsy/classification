(accelerate launch --mixed_precision bf16 \
    Prot_inference_accelerate.py >> Prot_infer_withFocal_2layer_v1.log &)