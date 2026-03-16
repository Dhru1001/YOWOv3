from scripts import train, ava_eval, ucf_eval, detect, live, onnx
import argparse
from utils.build_config import build_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOWOv3")

    parser.add_argument('-m', '--mode',   type=str, required=True,
                        help='train / eval / detect / live / onnx')
    parser.add_argument('-cf', '--config', type=str, required=True,
                        help='path to config file (.yaml)')

    args = parser.parse_args()
    config = build_config(args.config)

    if args.mode == 'train':
        train.train_model(config=config)

    elif args.mode == 'eval':
        dataset = config.get('dataset', '')
        if dataset in ('ucf', 'jhmdb', 'ucfcrime'):
            ucf_eval.eval(config=config)
        elif dataset == 'ava':
            ava_eval.eval(config=config)
        else:
            raise ValueError(f"Unknown dataset for eval: '{dataset}'")

    elif args.mode == 'detect':
        detect.detect(config=config)

    elif args.mode == 'live':
        live.detect(config=config)

    elif args.mode == 'onnx':
        onnx.export2onnx(config=config)

    else:
        raise ValueError(f"Unknown mode: '{args.mode}'. "
                         "Choose from: train, eval, detect, live, onnx")