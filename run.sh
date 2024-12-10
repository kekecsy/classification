(accelerate launch --config_file /data/csyData/uniprot_test/code/GOcode/cco_version2/my_config.yaml \
                   --mixed_precision bf16 \
                   test.py >> Prot_run_withFocal_v8.log &)