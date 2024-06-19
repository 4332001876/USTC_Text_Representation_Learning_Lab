import argparse

from exp import Exp

def get_args():
    parser = argparse.ArgumentParser(description="DashScope SDK Example")
    parser.add_argument("--exp_name", type=str, required=True, default='qwen', help="Experiment Name")
    parser.add_argument("--model", type=str, default='qwen1.5-7b-chat', help="Model Name")
    parser.add_argument("--local_model_path", type=str, default='model/Qwen2-0.5B-Instruct', help="Model Path")
    parser.add_argument("--library", type=str, default='z3', help="Python Library, options:[z3, sympy]")
    parser.add_argument("--num_threads", type=int, default=1, help="Number of threads")

    parser.add_argument("--dataset", type=str, default='GSM8K', help="Dataset Name, options:[GSM8K, MATH]")
    parser.add_argument("--dataset_path", type=str, default='data/imdb', help="Dataset Path")
    parser.add_argument("--output_path", type=str, default='tmp/output', help="Output Path")
    return parser.parse_args()

def main():
    args = get_args()
    exp = Exp(args)
    exp.test()

if __name__ == "__main__":
    main()
