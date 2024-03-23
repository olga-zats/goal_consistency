python main.py --mode train --epochs 15 \
--path_to_data /home/user/db_TSM_features \
--path_to_models /home/user/models_anticipation \
--path_to_anno data/CSVs \
--modality fixed+ego \
--views all --past_attention \
--batch_size 64 --num_workers 16 \
--predict_latent True \
--predict_ts_latent True \
--single_latent True \
--gt_fc_cons_loss True \
--gt_fc_cons_loss_weight 2.5 \
--gt_fts_cons_loss True \
--gt_fts_cons_loss_weight 2.5 \

