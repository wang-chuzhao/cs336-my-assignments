import argparse
import numpy as np
import os
import torch

import cs336_basics.utils as utils
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer, train_bpe
from cs336_basics.optimizer import AdamW, gradient_clipping, get_lr_cosine_schedule

SPECIAL_TOKENS = ["<|endoftext|>"]


def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--train_ids_path",type=str,required=True)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rope_theta",type=float,default=10000.0)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--cosine_cycle_iters", type=int, default=5000)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--eval_ids_path", type=str, required=True)
    parser.add_argument("--eval_data_path", type=str, required=True)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--preprocess_only", action="store_true")
    return parser.parse_args()

def maybe_preprocess(args) -> None:
    if os.path.exists(args.train_ids_path) and os.path.exists(args.eval_ids_path):
        print("已找到 token bin，跳过预处理", flush=True)
        return

    print("开始 BPE 训练...", flush=True)
    id_to_tokens,merge_rules=train_bpe(
        args.train_data_path,
        args.vocab_size,
        SPECIAL_TOKENS,
    )
    tokenizer=Tokenizer(id_to_tokens,merge_rules,SPECIAL_TOKENS)
    print("BPE 完成，开始 encode train set", flush=True)
    if not os.path.exists(args.train_ids_path):
        utils.txt_to_bin(tokenizer,args.train_data_path,args.train_ids_path)
    print("开始 encode valid set", flush=True)
    if not os.path.exists(args.eval_ids_path):
        utils.txt_to_bin(tokenizer,args.eval_data_path,args.eval_ids_path)
    print("预处理完成", flush=True)

def main():
    args=parse_args()

    maybe_preprocess(args)
    if args.preprocess_only:
        return
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA 不可用，回退到 CPU")
        device = "cpu"

    train_ids=np.memmap(args.train_ids_path,dtype=np.uint16,mode='r')
    eval_ids=np.memmap(args.eval_ids_path,dtype=np.uint16,mode='r')
    print(f"train tokens: {len(train_ids):,}, valid tokens: {len(eval_ids):,}")

    model=TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=tuple(args.betas),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    model.train()
    for step in range(args.max_steps):
        lr = get_lr_cosine_schedule(
            step,
            args.max_lr,
            args.min_lr,
            args.warmup_iters,
            args.cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"]=lr

        inputs,targets=utils.get_batch(
            train_ids,
            args.batch_size,
            args.context_length,
            device
        )
        preds=model(inputs)
        loss=utils.cross_entropy(preds.reshape(-1,args.vocab_size),targets.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(),1.0)
        optimizer.step()

        if step % args.log_interval == 0:
            print(f"step {step}: train_loss={loss.item():.4f}, lr={lr:.2e}")

        if step % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                x_val, y_val = utils.get_batch(
                    eval_ids,
                    args.batch_size,
                    args.context_length,
                    device,
                )
                pred_val = model(x_val)
                val_loss = utils.cross_entropy(
                    pred_val.reshape(-1, args.vocab_size),
                    y_val.reshape(-1),
                )
            model.train()
            print(f"step {step}: train_loss={loss.item():.4f}, val_loss={val_loss.item():.4f}")

        if step % args.save_interval == 0 and step > 0:
            utils.save_checkpoint(model, optimizer, step, f"ckpt_{step}.pt")


if __name__=="__main__":
    main()