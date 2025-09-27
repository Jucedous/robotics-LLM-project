# python3 test.py --epochs 2000 --device cpu --save --save-path checkpoints/hazard_gnn.pt --log-csv logs/train.csv
import argparse
from train import train
from eval import quick_sanity_checks

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--no-eval', action='store_true')
    p.add_argument('--save', action='store_true')
    p.add_argument('--save-path', type=str, default='checkpoints/hazard_gnn.pt')
    p.add_argument('--log-csv', type=str, default='')
    args = p.parse_args()

    save_path = args.save_path if args.save else None
    log_csv = args.log_csv if args.log_csv else None

    model = train(
        epochs=args.epochs,
        batch=args.batch,
        hidden=args.hidden,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        log_every=args.log_every,
        save_path=save_path,
        log_csv=log_csv,
    )

    if not args.no_eval:
        quick_sanity_checks(model)