CUDA_VISIBLE_DEVICES=0,1,2,3 python run_speaker-aware.py \
--model_type electra \
--model_name_or_path google/electra-large-discriminator \
--output_dir experiments/electra_all_3e-5_ep4_2l1l_8mul4 \
--num_train_epochs 4 \
--data_dir ../code_Enhanced_speaker-aware/molweni/Molweni/MRC\(withDiscourse\)/ \
--train_file train.json --predict_file test.json \
--cache_dir ../code_Enhanced_speaker-aware/electra-cache/ \
--version_2_with_negative --do_eval --do_train --do_lower_case \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 8 \
--eval_all_checkpoints \
--learning_rate 3e-5 \
--weight_decay 0.01 \
--warmup_step 100 \
--task layer21 \
--save_steps 2000

