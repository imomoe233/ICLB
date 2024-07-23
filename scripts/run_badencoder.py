import os

if not os.path.exists('./log/bad_encoder'):
    os.makedirs('./log/bad_encoder')

def run_finetune(gpu, encoder_usage_info, shadow_dataset, downstream_dataset, trigger, reference, clean_encoder='model_1000.pth'):

    save_path = f'./output/{encoder_usage_info}/{downstream_dataset}_backdoored_encoder'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cmd = f'nohup python3 -u badencoder.py \
    --lr 0.001 \
    --batch_size 256 \
    --results_dir {save_path}/ \
    --shadow_dataset {shadow_dataset} \
    --pretrained_encoder ./output/{encoder_usage_info}/clean_encoder/{clean_encoder} \
    --encoder_usage_info {encoder_usage_info} \
    --gpu {gpu} \
    --reference_file ./reference/{encoder_usage_info}/{reference}.npz \
    --trigger_file ./trigger/{trigger} \
    > ./log/bad_encoder/{encoder_usage_info}_{downstream_dataset}_{reference}.log &'
    os.system(cmd)

# 触发器从shadow中产生，通过cifar10的触发器和stl10的类相映射，能否使该encoder直接对stl10生成触发器也产生较好的效果？
# 预训练模型 3个 cifar10 ImageNet CLIP，在这三个数据集上，分别对GTSRB, SVHN, and STL10进行微调，得到backdoor encoder，总共获得9个Backdoor encoder
# 使用这9个backdoor encoder，对各自微调的数据集进行下游任务训练，最后得到9个结果

run_finetune(0, 'cifar10', 'cifar10', 'stl10', 'trigger_pt_white_21_10_ap_replace.npz', 'truck')
# run_finetune(1, 'cifar10', 'cifar10', 'gtsrb', 'trigger_pt_white_21_10_ap_replace.npz', 'priority')
# run_finetune(2, 'cifar10', 'cifar10', 'svhn', 'trigger_pt_white_21_10_ap_replace.npz', 'one')

# 关键的是，如果在A数据集上进行微调，在B数据集上进行下游任务，则后门效果不好的话
# 补充新的实验，即，将A数据集拆分为2部分，在A-1进行微调，植入后门，在A-2进行下游训练，查看后门的准确率。这里由于把完整的数据集拆分了
# 因此可能会影响良性样本的准确率，但是不碍事，实在是太低了的话就不列出来了，因为拆分数据集后，数据量小了，准确率低是正常的