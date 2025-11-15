wget -c -P ./data/pretrain/sam2_ckpts/ https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt
wget -c -P ./data/pretrain/sam2_ckpts/ https://huggingface.co/facebook/sam2.1-hiera-base-plus/resolve/main/sam2.1_hiera_base_plus.pt
wget -c -P ./data/pretrain/sam2_ckpts/ https://huggingface.co/facebook/sam2.1-hiera-small/resolve/main/sam2.1_hiera_small.pt
wget -c -P ./data/pretrain/sam2_ckpts/ https://huggingface.co/facebook/sam2.1-hiera-tiny/resolve/main/sam2.1_hiera_tiny.pt

wget -c -P ./pipeline/sam2/ https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_l.yaml
wget -c -P ./pipeline/sam2/ https://huggingface.co/facebook/sam2.1-hiera-base-plus/resolve/main/sam2.1_hiera_b%2B.yaml
wget -c -P ./pipeline/sam2/ https://huggingface.co/facebook/sam2.1-hiera-small/resolve/main/sam2.1_hiera_s.yaml
wget -c -P ./pipeline/sam2/ https://huggingface.co/facebook/sam2.1-hiera-tiny/resolve/main/sam2.1_hiera_t.yaml