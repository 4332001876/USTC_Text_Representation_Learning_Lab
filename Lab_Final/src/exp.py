from tqdm import tqdm

from agents import QwenAgent, BaselineModel
from dataloader import TestDataLoader, verify_ans


class Exp:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

        self.exp_name = cfg.exp_name
        self.model = cfg.model
        self.local_model_path = cfg.local_model_path

        self.library = cfg.library
        self.num_threads = cfg.num_threads

        self.dataset = cfg.dataset
        self.dataset_path = cfg.dataset_path
        self.output_path = cfg.output_path

        self.agent = BaselineModel(model_name=self.local_model_path)
        self.test_loader = TestDataLoader(self.dataset_path)

    def run(self):
        print("Start testing...")
        print(f"Total number of problems: {len(self.test_loader)}")
        print("Model:", self.local_model_path)

        correct = 0
        incorrect = 0
        ambiguous = 0
        for idx in tqdm(range(len(self.test_loader))):
            problem, label = self.test_loader.basic_prompt(idx)
            response = self.agent.answer(problem)
            result = verify_ans(response, label)

            if idx == 0:
                print("Example:")
                print(f"Problem: {problem}")
                print(f"Response: {response}")
                print(f"Label: {label}")

            if result == 1:
                correct += 1
            elif result == 0:
                incorrect += 1
            else:
                ambiguous += 1

            if idx % 1000 == 999:
                print(f"Correct: {correct}, Incorrect: {incorrect}, Ambiguous: {ambiguous}")
                print("Running Accuracy: %.4f"%((correct + 0.5 * ambiguous) / (correct + incorrect + ambiguous)))
        
        print(f"Correct: {correct}, Incorrect: {incorrect}, Ambiguous: {ambiguous}")
        print("Accuracy: %.4f"%((correct + 0.5 * ambiguous) / (correct + incorrect + ambiguous)))
            

    def load_problems(self):
        pass

    def solve_problems(self, problems):
        pass
