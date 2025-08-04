_base_ = [
    "./datasets/motive_updated_cls.py",
    "./models/srlane_r18_motive_cls.py",
]

work_dirs = "work_dirs/sr_mtv/tmp_faisal"


iou_loss_weight = 2.
cls_loss_weight = 2.
l1_loss_weight = 0.2
angle_loss_weight = 5
attn_loss_weight = 0.  # 0.05
seg_loss_weight = 0.  # 0.5

total_iter = 12000 #44440
batch_size = 4 #20
eval_ep = 1
workers = 8
log_interval = 100

precision = "16-mixed"  # "32"

optimizer = dict(type="AdamW", lr=6e-4)

scheduler = dict(type="warmup", warm_up_iters=800, total_iters=total_iter)
